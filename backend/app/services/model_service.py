"""Model inference seam.

This module is the *only* place allowed to know how inference works. It is
split from the API layer on purpose:

    routes.py            validates the upload, calls get_model_service(),
                        normalises the result onto AnalysisResponse.
    model_service.py     loads the model, preprocesses the image, runs
                        inference, returns a raw prediction.
    model_class_map.py   collapses the model's classes onto the binary
                        API verdict.

The real model is wired in. :class:`RealModelService` loads the checkpoint once
per process and answers every request from that single instance; the FastAPI
route neither imports PyTorch nor knows what a tensor is.

The contract this implements was taken from the AI team's reference script
``ai_model/predict.py`` and is reproduced exactly - architecture, checkpoint
keys, preprocessing, softmax, argmax, confidence. Deviations would silently
change predictions, so none are made here:

    architecture   torchvision.models.convnext_base(weights=None)
    classifier     classifier[2] replaced with Linear(in_features, n_classes)
    checkpoint     torch.load(..., weights_only=False); "classes" and
                   "model_state_dict" read from the archive
    preprocessing  PIL open -> convert("RGB") -> resize(350, 350) BILINEAR
                   -> ToTensor -> Normalize(ImageNet mean/std)
    inference      model.eval(), torch.inference_mode(), forward,
                   softmax(dim=1), max -> (confidence, index)

The only intentional difference is the *reporting* of confidence: the reference
prints a percentage, the API returns 0.0-1.0 as the frontend contract requires.

Three failure modes are kept distinct, because the API answers them
differently:

    ModelNotAvailableError  (503) the checkpoint or framework is missing or
                            unusable - the model could not be loaded at all
    ModelInferenceError     (500) the model is loaded but this request failed
    UploadRejected          (400/413) not this module's concern; raised upstream

No path fabricates a prediction. If the model cannot be loaded, every request
fails loudly rather than the backend pretending to work.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

from PIL import Image, UnidentifiedImageError

from app.core.config import settings
from app.schemas.analysis import Verdict
from app.services.model_class_map import verdict_for_class

logger = logging.getLogger(__name__)

# --- Reference preprocessing, mirrored verbatim from ai_model/predict.py ----
# These are the model's contract, not ours: changing any of them changes the
# predictions. Kept as module constants so the parity tests can assert against
# the same values the service uses.
MODEL_INPUT_SIZE: tuple[int, int] = (350, 350)
MODEL_RESAMPLE = Image.Resampling.BILINEAR
MODEL_NORMALIZE_MEAN: list[float] = [0.485, 0.456, 0.406]
MODEL_NORMALIZE_STD: list[float] = [0.229, 0.224, 0.225]


class ModelError(RuntimeError):
    """Base class for inference-layer failures."""


class ModelNotAvailableError(ModelError):
    """No usable model could be loaded.

    Surfaces as HTTP 503. Raised when the framework is missing, the checkpoint
    file is absent or unreadable, or the archive does not match the expected
    architecture. It is never raised because of a *particular image* - that is
    a different failure and gets a different status.
    """


class ModelInferenceError(ModelError):
    """A model exists but failed to produce a usable prediction.

    Surfaces as HTTP 500.
    """


@dataclass(frozen=True)
class ImagePayload:
    """Raw bytes of an upload that already passed API-level validation.

    The service layer receives bytes, not a file handle, so the inference
    implementation is free to decode however it needs to without depending
    on FastAPI or Starlette types.
    """

    data: bytes
    filename: str
    content_type: str | None = None


@dataclass(frozen=True)
class ModelPrediction:
    """A model result, before it is mapped onto the public API contract.

    ``verdict`` and ``confidence`` are the agreed minimum. ``class_name``
    carries the pre-collapse model output for logging and diagnostics; it is
    deliberately not exposed on the API, whose contract is binary.
    """

    verdict: Verdict
    confidence: float
    class_name: str | None = None


class ModelService(ABC):
    """Contract every inference implementation must satisfy."""

    @property
    @abstractmethod
    def is_available(self) -> bool:
        """Whether this service can serve a request right now."""

    @abstractmethod
    def analyze(self, image: ImagePayload) -> ModelPrediction:
        """Run inference on a validated image.

        Raises:
            ModelNotAvailableError: no model is wired up.
            ModelInferenceError: inference ran but produced no usable result.
        """


class RealModelService(ModelService):
    """Inference backed by the AI team's trained ConvNeXt checkpoint.

    The checkpoint is loaded exactly once, in ``__init__``, and held for the
    lifetime of the instance. Per-request work is limited to decode, resize,
    normalise and one forward pass - no disk I/O, no weight reloading, no
    recompilation.

    Args:
        checkpoint_path: the loadable ``.pth`` archive. Produced by
            ``backend/tools/prepare_checkpoint.py`` from the extracted
            ``ai_model/best_accuracy_model/`` directory.
        device: force a device instead of auto-detecting. ``None`` selects
            CUDA when available and CPU otherwise.

    Raises:
        ModelNotAvailableError: the model could not be loaded.
    """

    def __init__(self, checkpoint_path: Path | str | None = None, *, device: str | None = None) -> None:
        self._checkpoint_path = (
            Path(checkpoint_path) if checkpoint_path is not None else settings.resolved_model_path
        )
        self._requested_device = device if device is not None else settings.model_device

        self._model = None
        self._transform = None
        self._device = None
        self._class_names: list[str] = []
        self._load_error: str | None = None

        self._load()

    # -- loading ----------------------------------------------------------

    def _load(self) -> None:
        """Load the checkpoint and build the network, once.

        Everything that can go wrong here is a *model availability* problem,
        not a request problem, so it is normalised into one exception type.
        The underlying cause is logged and kept as ``_load_error`` for
        diagnostics; neither is ever returned to the API consumer.
        """
        try:
            self._import_torch()
            self._resolve_device()
            checkpoint = self._torch.load(
                self._checkpoint_path,
                map_location=self._device,
                weights_only=False,
            )
            self._read_class_names(checkpoint)
            model = self._build_network(len(self._class_names))
            model.load_state_dict(checkpoint["model_state_dict"])
            self._model = model.to(self._device).eval()
            self._build_transform()
        except ModelNotAvailableError:
            self._load_error = "model unavailable"
            raise
        except Exception as exc:  # noqa: BLE001 - normalised below
            # Framework import failure, missing file, corrupt archive, shape
            # mismatch: all of them mean "no model to serve".
            self._load_error = f"{type(exc).__name__}: {exc}"
            logger.error(
                "Failed to load model from %s: %s", self._checkpoint_path, self._load_error
            )
            raise ModelNotAvailableError(
                "The analysis model could not be loaded. The inference layer "
                "is not available."
            ) from exc

        logger.info(
            "Loaded ConvNeXt model from %s (classes=%s, device=%s, parameters=%s)",
            self._checkpoint_path,
            self._class_names,
            self._device,
            f"{sum(p.numel() for p in self._model.parameters()):,}",
        )

    def _import_torch(self) -> None:
        """Import the inference stack, reporting absence as unavailability."""
        try:
            import torch
            from torchvision import transforms
            from torchvision.models import convnext_base
        except ImportError as exc:
            raise ModelNotAvailableError(
                "The analysis model could not be loaded: the PyTorch "
                "inference dependencies are not installed."
            ) from exc

        self._torch = torch
        self._transforms = transforms
        self._convnext_base = convnext_base

    def _resolve_device(self) -> None:
        """Pick the device: explicit request, else CUDA when present, else CPU.

        Mirrors the reference script's ``torch.device("cuda" if
        torch.cuda.is_available() else "cpu")``.
        """
        if self._requested_device:
            device = self._torch.device(self._requested_device)
            if device.type == "cuda" and not self._torch.cuda.is_available():
                raise ModelNotAvailableError(
                    "MODEL_DEVICE requests CUDA, but no CUDA device is available."
                )
        else:
            device = self._torch.device(
                "cuda" if self._torch.cuda.is_available() else "cpu"
            )

        self._device = device

    def _read_class_names(self, checkpoint: object) -> None:
        """Take the class order from the checkpoint, never from code.

        ``checkpoint["classes"]`` is the authoritative index -> name ordering.
        The provisional verdict mapping is keyed by these names.
        """
        if not isinstance(checkpoint, dict):
            raise ModelNotAvailableError(
                "Checkpoint archive does not contain a state dictionary."
            )
        if "model_state_dict" not in checkpoint:
            raise ModelNotAvailableError(
                "Checkpoint archive has no 'model_state_dict' entry."
            )

        classes = checkpoint.get("classes")
        if not classes:
            raise ModelNotAvailableError(
                "Checkpoint archive has no 'classes' entry; the class order "
                "cannot be established."
            )

        self._class_names = list(classes)

    def _build_network(self, num_classes: int):
        """Build ConvNeXt Base with the classifier replaced, as the reference does."""
        model = self._convnext_base(weights=None)
        model.classifier[2] = self._torch.nn.Linear(
            model.classifier[2].in_features,
            num_classes,
        )
        return model

    def _build_transform(self) -> None:
        """Build the reference evaluation transform.

        ``ToTensor`` then ``Normalize``; the resize to 350x350 with BILINEAR
        sampling happens per request (see :meth:`_preprocess`) because the
        reference applies it to the opened image before the transform, in that
        order.
        """
        self._transform = self._transforms.Compose(
            [
                self._transforms.ToTensor(),
                self._transforms.Normalize(
                    mean=MODEL_NORMALIZE_MEAN,
                    std=MODEL_NORMALIZE_STD,
                ),
            ]
        )

    # -- inference --------------------------------------------------------

    @property
    def is_available(self) -> bool:
        return self._model is not None and self._transform is not None

    @property
    def device(self) -> str:
        """Name of the device inference runs on."""
        return str(self._device)

    @property
    def class_names(self) -> tuple[str, ...]:
        """Class order reported by the checkpoint."""
        return tuple(self._class_names)

    def _preprocess(self, image: ImagePayload):
        """Decode, resize and normalise an upload into a batched tensor.

        Step for step with the reference script. Returns a ``(1, 3, 350, 350)``
        tensor already on the model device.
        """
        try:
            with Image.open(BytesIO(image.data)) as opened:
                rgb = opened.convert("RGB")

                resized = rgb.resize(MODEL_INPUT_SIZE, MODEL_RESAMPLE)

                tensor = self._transform(resized)
        except (UnidentifiedImageError, OSError, ValueError, SyntaxError) as exc:
            raise ModelInferenceError(
                "The uploaded bytes could not be decoded as an image."
            ) from exc

        return tensor.unsqueeze(0).to(self._device)

    def analyze(self, image: ImagePayload) -> ModelPrediction:
        """Run inference and map the result onto the binary verdict.

        Raises:
            ModelNotAvailableError: the model is not loaded.
            ModelInferenceError: decoding or the forward pass failed, or the
                model reported a class the provisional mapping does not cover.
        """
        if not self.is_available:
            raise ModelNotAvailableError(
                "The analysis model is not loaded."
            )

        try:
            tensor = self._preprocess(image)
        except ModelInferenceError:
            raise
        except Exception as exc:  # noqa: BLE001 - request-scoped failure
            logger.exception("Preprocessing failed for %s", image.filename)
            raise ModelInferenceError(
                "The image could not be prepared for analysis."
            ) from exc

        try:
            with self._torch.inference_mode():
                output = self._model(tensor)
                probabilities = self._torch.softmax(output, dim=1)
                confidence, predicted = self._torch.max(probabilities, dim=1)

            index = predicted.item()
            class_name = self._class_names[index]
            # The reference prints confidence * 100; the API contract is 0.0-1.0.
            confidence_fraction = confidence.item()
        except Exception as exc:  # noqa: BLE001 - request-scoped failure
            logger.exception("Inference failed for %s", image.filename)
            raise ModelInferenceError(
                "The analysis model failed to produce a prediction."
            ) from exc

        try:
            verdict = verdict_for_class(class_name)
        except KeyError as exc:
            # The checkpoint's class list changed. Guessing a verdict here
            # would make the endpoint answer with a decision nobody made.
            logger.error(
                "Model reported class %r, which the provisional verdict "
                "mapping does not cover. Known classes: %s. Update "
                "app/services/model_class_map.py.",
                class_name,
                sorted(self._class_names),
            )
            raise ModelInferenceError(
                "The analysis model reported a class this API cannot map to a "
                "verdict."
            ) from exc

        logger.info(
            "%s -> %s (%s%%, conf=%.4f) -> %s",
            image.filename,
            class_name,
            f"{confidence_fraction * 100:.2f}",
            confidence_fraction,
            verdict.value,
        )

        return ModelPrediction(
            verdict=verdict,
            confidence=confidence_fraction,
            class_name=class_name,
        )


class UnavailableModelService(ModelService):
    """A service that has no model and refuses to pretend otherwise.

    Returned by :func:`get_model_service` only when model loading is disabled
    outright. It never fabricates a result: loading a model, decoding pixels,
    or emitting a default verdict would misrepresent the backend as working,
    so every call fails loudly instead.
    """

    def __init__(self, reason: str = "The inference layer is not configured.") -> None:
        self._reason = reason

    @property
    def is_available(self) -> bool:
        return False

    def analyze(self, image: ImagePayload) -> ModelPrediction:
        raise ModelNotAvailableError(self._reason)


# --- Process-wide singleton ------------------------------------------------
# The checkpoint is ~544 MiB and takes seconds to load. Constructing a service
# per request would reload it every time, so exactly one instance is built and
# reused. Built lazily on first use rather than at import time so that starting
# the API (and answering GET /health) is not blocked by model loading, and so a
# missing checkpoint surfaces as a 503 on /analyze rather than an import error.

_model_service: ModelService | None = None


def get_model_service() -> ModelService:
    """Return the configured inference implementation.

    Returns the cached :class:`RealModelService`, building it on first call. A
    load failure is not cached as a success or a fallback: the exception
    propagates so the route answers 503, and the next call retries.
    """
    global _model_service

    if _model_service is None:
        _model_service = RealModelService()

    return _model_service


def reset_model_service() -> None:
    """Discard the cached instance.

    Test hook, and the way to force a reload after replacing the checkpoint on
    disk.
    """
    global _model_service
    _model_service = None
