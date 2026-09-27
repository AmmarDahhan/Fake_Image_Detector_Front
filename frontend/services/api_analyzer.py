"""Real backend integration.

The only place in the frontend that performs network I/O. It POSTs the uploaded
image bytes to the FastAPI backend and maps the response onto
:class:`~core.models.AnalysisResult`, which is all the UI ever consumes - so
the result card, error state and copy work unchanged.

Endpoint contract (``POST {API_BASE_URL}{API_ANALYZE_PATH}``)::

    request   multipart/form-data, one part named "file"
              -> {"verdict": "real" | "fake", "confidence": 0.0-1.0}

Confidence needs no conversion in either direction: the backend already returns
a 0.0-1.0 fraction, which is exactly what ``AnalysisResult`` stores, and
``AnalysisResult.confidence_percent`` / ``format_confidence`` render it as a
percentage for display.

Error handling maps every outcome onto the two error states the UI already
renders, so no UI change was needed:

    AnalyzerUnavailableError  -> "Analysis service unavailable"
      transport failure, timeout, or HTTP 503 (model not loaded)
    ApiAnalyzerError           -> "Analysis could not be completed"
      HTTP 400/413/422/500/other, and any response that is not the agreed
      schema (unparseable JSON, missing/ill-typed fields)

Both are plain exceptions carrying a sanitised message; the UI catches them,
logs them, and shows fixed copy. A traceback can never reach the user.
"""

from __future__ import annotations

import logging

import requests

from core.analyzer import Analyzer, AnalyzerUnavailableError
from core.config import API_ANALYZE_PATH, API_BASE_URL, API_TIMEOUT_SECONDS
from core.models import AnalysisResult, ImageInfo, Verdict

logger = logging.getLogger(__name__)

#: Multipart field name the backend expects. Must match the route signature.
UPLOAD_FIELD_NAME = "file"

#: Sent when the uploader gave us no usable filename, so the backend's own
#: extension check has nothing to work from. The frontend validates the
#: extension before analysis, so this is only a last-resort fallback.
FALLBACK_FILENAME = "upload"

#: Longest backend ``detail`` echoed into a log message. The detail is never
#: rendered in the UI, but it should still not be able to flood the log.
_MAX_DETAIL_CHARS = 300


class ApiAnalyzerError(RuntimeError):
    """The backend answered, but not with something usable.

    Covers non-success statuses other than 503, and responses that do not match
    the agreed schema. The UI surfaces these as a generic analysis failure.
    """


class ApiAnalyzer(Analyzer):
    """Analyzer backed by the real FastAPI service.

    Args:
        base_url: backend origin, e.g. ``http://localhost:8000``. Defaults to
            ``core.config.API_BASE_URL``.
        timeout: seconds to wait for the whole request. Defaults to
            ``core.config.API_TIMEOUT_SECONDS``.
        endpoint: analysis path, normally ``/analyze``.

    No network call happens in ``__init__``; the backend is contacted lazily on
    the first :meth:`analyze`, so constructing the analyzer (which the UI does
    on every Streamlit rerun) stays instant and costs nothing.
    """

    def __init__(
        self,
        base_url: str | None = None,
        timeout: float | None = None,
        endpoint: str = API_ANALYZE_PATH,
    ) -> None:
        self.base_url = (base_url if base_url is not None else API_BASE_URL).rstrip("/")
        self.timeout = API_TIMEOUT_SECONDS if timeout is None else timeout
        self.endpoint = endpoint

    @property
    def url(self) -> str:
        """Full URL of the analysis endpoint."""
        return f"{self.base_url}{self.endpoint}"

    @property
    def is_available(self) -> bool:
        """Whether this analyzer is implemented and able to issue a request.

        Deliberately not a reachability probe: the backend's liveness is a
        property of the network, and answering it here would mean a blocking
        request on every Streamlit rerun. An unreachable backend surfaces as
        :class:`AnalyzerUnavailableError` from :meth:`analyze` instead, which
        is where it can be handled meaningfully.
        """
        return True

    def analyze(
        self, image_bytes: bytes, image_info: ImageInfo | None = None
    ) -> AnalysisResult:
        """POST the image to the backend and return the mapped result.

        Args:
            image_bytes: the exact validated bytes selected by the user. Sent
                unmodified, so the backend classifies the same pixels the user
                chose.
            image_info: used only for the multipart filename and content type.
                The backend applies the same format checks the frontend did.

        Raises:
            AnalyzerUnavailableError: the backend could not be reached, timed
                out, or reported 503 (model not loaded).
            ApiAnalyzerError: any other HTTP failure, or a response that does
                not match the agreed schema.
        """
        if not image_bytes:
            raise ApiAnalyzerError("No image bytes were provided to analyze.")

        files = {
            UPLOAD_FIELD_NAME: (
                self._filename(image_info),
                image_bytes,
                self._content_type(image_info),
            )
        }

        try:
            response = requests.post(self.url, files=files, timeout=self.timeout)
        except requests.RequestException as exc:
            # Connection refused, DNS failure, timeout, too many redirects:
            # the service is not answering. Reported as unavailable rather
            # than as a failed analysis, because nothing is known about the
            # image.
            logger.warning("Backend request to %s failed: %s", self.url, exc)
            raise AnalyzerUnavailableError(
                f"The analysis backend at {self.base_url} is not reachable."
            ) from exc

        return self._to_result(response)

    # -- response mapping --------------------------------------------------

    def _to_result(self, response: requests.Response) -> AnalysisResult:
        """Turn an HTTP response into an :class:`AnalysisResult`."""
        status = response.status_code

        if status == 503:
            # The API is up but the model could not be loaded. The user's
            # retry may well succeed, so it reads as "try again shortly".
            logger.error("Backend reported 503: %s", self._detail(response))
            raise AnalyzerUnavailableError("The analysis model is not available.")

        if status != 200:
            logger.error(
                "Backend returned HTTP %s: %s", status, self._detail(response)
            )
            raise ApiAnalyzerError(
                f"The analysis backend returned HTTP {status}: "
                f"{self._detail(response)}"
            )

        payload = self._parse_payload(response)
        verdict = self._parse_verdict(payload)
        confidence = self._parse_confidence(payload)

        logger.info(
            "Backend result: verdict=%s confidence=%.4f", verdict.value, confidence
        )

        return AnalysisResult(
            verdict=verdict,
            confidence=confidence,
            metadata={"source": "api", "provider": "ApiAnalyzer"},
        )

    def _parse_payload(self, response: requests.Response) -> dict:
        """Return the response body as a JSON object.

        A 200 with a body that is not the agreed object shape is a broken
        contract, not a verdict - it is refused rather than guessed at.
        """
        try:
            payload = response.json()
        except ValueError as exc:
            logger.error("Backend response was not valid JSON: %s", exc)
            raise ApiAnalyzerError(
                "The analysis backend returned a response that is not valid JSON."
            ) from exc

        if not isinstance(payload, dict):
            logger.error("Backend JSON body was %s, expected an object", type(payload))
            raise ApiAnalyzerError(
                "The analysis backend returned an unexpected response shape."
            )

        return payload

    def _parse_verdict(self, payload: dict) -> Verdict:
        raw = payload.get("verdict")
        try:
            return Verdict.parse(raw)
        except ValueError as exc:
            logger.error("Backend returned an unknown verdict: %r", raw)
            raise ApiAnalyzerError(
                "The analysis backend returned an unrecognised verdict."
            ) from exc

    def _parse_confidence(self, payload: dict) -> float:
        """Return confidence as a 0.0-1.0 float, matching ``AnalysisResult``.

        The backend already speaks in fractions, so this is a pass-through plus
        a range check. Anything else is rejected instead of clamped: a
        confidence of 1.7 or "0.9" means the contract has been broken, and
        quietly adjusting it would report a number the model never produced.
        """
        raw = payload.get("confidence")

        # bool is an int subclass; True would otherwise become 1.0.
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            logger.error("Backend returned a non-numeric confidence: %r", raw)
            raise ApiAnalyzerError(
                "The analysis backend returned an invalid confidence value."
            )

        confidence = float(raw)
        if not 0.0 <= confidence <= 1.0:
            logger.error("Backend returned an out-of-range confidence: %r", raw)
            raise ApiAnalyzerError(
                "The analysis backend returned a confidence outside 0.0-1.0."
            )

        return confidence

    # -- multipart helpers -------------------------------------------------

    def _filename(self, image_info: ImageInfo | None) -> str:
        if image_info and image_info.filename:
            return image_info.filename
        return FALLBACK_FILENAME

    def _content_type(self, image_info: ImageInfo | None) -> str:
        if image_info and image_info.mime_type:
            return image_info.mime_type
        # The backend accepts a file on either its extension or its MIME type.
        # With the real filename present that check still passes, so an empty
        # content type here is safe.
        return "application/octet-stream"

    @staticmethod
    def _detail(response: requests.Response) -> str:
        """Best-effort, bounded extraction of the backend's ``detail`` field.

        Logging only - never shown to the user, and never assumed to be present
        or even JSON.
        """
        try:
            body = response.json()
        except ValueError:
            return (response.text or "").strip()[:_MAX_DETAIL_CHARS]

        if isinstance(body, dict):
            detail = body.get("detail")
            if isinstance(detail, str):
                return detail[:_MAX_DETAIL_CHARS]
            if detail is not None:
                return str(detail)[:_MAX_DETAIL_CHARS]

        return (response.text or "").strip()[:_MAX_DETAIL_CHARS]
