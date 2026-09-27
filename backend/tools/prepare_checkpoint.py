"""Reconstruct a loadable ``.pth`` archive from an extracted PyTorch archive.

WHY THIS EXISTS
---------------
The AI team delivered the checkpoint as ``ai_model/best_accuracy_model/``,
which is an *unzipped* PyTorch serialization archive. It contains the exact
members a ``.pth`` file is made of - ``data.pkl``, ``data/<n>`` storage blobs,
``version``, ``byteorder``, ``.format_version``, ``.storage_alignment`` and
``.data/serialization_id`` - but spread over a directory instead of a single
zip stream.

``torch.load`` opens its argument with Python's ``zipfile`` (via
``torch._C.PyTorchFileReader``), which requires a seekable *file*. It cannot
open a directory, and it has no "load from extracted directory" mode. So the
artefact as shipped is not directly loadable by the reference script
``ai_model/predict.py``, which searches for a file named
``best_accuracy_model.pth``.

A second wrinkle: PyTorch's zip reader does not accept those members at the
archive root. It resolves member names as ``<archive stem>/<member>`` and fails
with ``file in archive is not in a subdirectory`` when the entries sit at the
top level. That is because ``torch.save`` always nests a checkpoint under a
directory named after the file it writes, so ``best_accuracy_model.pth``
normally contains ``best_accuracy_model/data.pkl``, ``best_accuracy_model/version``
and so on. The shipped directory is the *contents* of that prefix directory,
with the prefix stripped.

The fix is therefore mechanical, not semantic: re-zip the directory, byte for
byte, back under a top-level prefix named after the destination file, exactly
reproducing the layout ``torch.save`` emits. ``data.pkl`` and every
``data/<n>`` storage blob are copied verbatim, so the resulting object graph -
tensor names, shapes, dtypes, storage offsets and the ``classes`` list - is
exactly what the AI team trained. No tensor is converted, re-ordered,
re-typed or re-normalised.

The source directory is only ever READ. The derived file is written outside
``ai_model/`` to ``backend/artifacts/`` so the AI team's folder stays byte
identical to what they delivered.

Usage (from ``backend/``)::

    python tools/prepare_checkpoint.py

Defaults point at the repository-relative extracted archive; override with
``--source`` / ``--destination``.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

# backend/tools/prepare_checkpoint.py -> backend/ -> project root
BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = BACKEND_DIR.parent

DEFAULT_SOURCE = PROJECT_DIR / "ai_model" / "best_accuracy_model"
DEFAULT_DESTINATION = BACKEND_DIR / "artifacts" / "best_accuracy_model.pth"

# Members that make up a PyTorch ``.pth`` archive. ``data.pkl`` is the pickle
# that references the rest; the others are the small header files the reader
# uses to interpret storage layout.
REQUIRED_MEMBERS = ("data.pkl", "version", "byteorder", ".format_version")


class ExtractionError(RuntimeError):
    """The source directory is not an extracted PyTorch archive."""


def _iter_members(source: Path, prefix: str) -> list[tuple[Path, str]]:
    """Return ``(absolute_path, archive_name)`` for every member.

    ``prefix`` is the top-level directory the members are nested under, with a
    trailing slash. It must be non-empty: PyTorch's zip reader resolves member
    names relative to a subdirectory of the archive and refuses an archive
    whose entries live at the root.
    """
    members: list[tuple[Path, str]] = []

    for path in sorted(source.rglob("*")):
        if not path.is_file():
            continue
        members.append((path, prefix + path.relative_to(source).as_posix()))

    return members


def _relative_members(source: Path) -> list[str]:
    """Archive-relative names, without the top-level prefix."""
    return [
        path.relative_to(source).as_posix()
        for path in sorted(source.rglob("*"))
        if path.is_file()
    ]


def inspect(source: Path) -> None:
    """Fail loudly if ``source`` is not an extracted PyTorch archive."""
    if not source.is_dir():
        raise ExtractionError(f"Not a directory: {source}")

    names = set(_relative_members(source))

    missing = [name for name in REQUIRED_MEMBERS if name not in names]
    if missing:
        raise ExtractionError(
            f"{source} is missing PyTorch archive members: {', '.join(missing)}"
        )

    if not any(name.startswith("data/") for name in names):
        raise ExtractionError(f"{source} contains no 'data/' storage blobs.")

    print(f"Source looks like an extracted PyTorch archive: {source}")


def build(source: Path, destination: Path) -> Path:
    """Zip ``source`` into ``destination`` without altering any content."""
    inspect(source)

    # torch.save names the inner directory after the file it writes; mirroring
    # that here keeps the derived artefact indistinguishable from a freshly
    # saved checkpoint.
    prefix = f"{destination.stem}/"

    destination.parent.mkdir(parents=True, exist_ok=True)

    # ZIP_STORED, not deflate: torch's own writer stores tensor payloads
    # uncompressed so they can be memory-mapped. Compressing would also work
    # for correctness, but it would make the derived file diverge from how
    # torch.save produces one, for no benefit.
    with zipfile.ZipFile(
        destination, mode="w", compression=zipfile.ZIP_STORED, allowZip64=True
    ) as archive:
        for path, name in _iter_members(source, prefix):
            archive.write(path, arcname=name)

    return destination


def verify(destination: Path) -> dict[str, object]:
    """Load the derived archive with torch and report what is inside."""
    import torch

    # The stored tensors were serialised on a CUDA device, so map_location is
    # mandatory for a CPU load - exactly as ai_model/predict.py does.
    checkpoint = torch.load(destination, map_location="cpu", weights_only=False)

    if not isinstance(checkpoint, dict):
        raise ExtractionError(
            f"Expected a dict at the archive root, got {type(checkpoint).__name__}."
        )

    if "model_state_dict" not in checkpoint:
        raise ExtractionError("Archive has no 'model_state_dict' entry.")

    state_dict = checkpoint["model_state_dict"]

    print(f"Derived archive: {destination}")
    print(f"  size: {destination.stat().st_size / (1024 * 1024):.1f} MiB")
    print(f"  top-level keys: {sorted(checkpoint)}")
    if "classes" in checkpoint:
        print(f"  classes: {checkpoint['classes']}")
    if "epoch" in checkpoint:
        print(f"  epoch: {checkpoint['epoch']}")
    print(f"  state_dict tensors: {len(state_dict)}")

    return checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=DEFAULT_SOURCE,
        help="Extracted PyTorch archive directory (read-only).",
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help="Where to write the reconstructed .pth file.",
    )
    args = parser.parse_args(argv)

    source = args.source.resolve()
    destination = args.destination.resolve()

    if destination.exists():
        # Rebuilding is cheap relative to a wrong artefact, but say so rather
        # than silently overwriting a file the user may have curated.
        print(f"Overwriting existing derived archive: {destination}")

    build(source, destination)
    verify(destination)
    return 0


if __name__ == "__main__":
    sys.exit(main())
