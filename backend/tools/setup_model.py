"""Download the runtime checkpoint from a release asset.

WHY THIS EXISTS
---------------
The trained ConvNeXt checkpoint is ~544 MiB. That is far too large to sit in
normal Git history: it would bloat every clone forever, and the repository's
``.gitignore`` already excludes ``backend/artifacts/`` to keep it out.

The consequence is that a fresh clone has *no* model, and therefore cannot
start the backend. This script closes that gap: it fetches the checkpoint from
a release asset into the exact path the backend already loads by default,

    backend/artifacts/best_accuracy_model.pth

so no code change, no environment variable and no extra step is needed once it
has run.

WHAT THIS IS NOT
----------------
This is a *transport* for the artefact, not a producer of it. It does not
train, convert, re-zip or otherwise derive the checkpoint, and it never calls
``prepare_checkpoint.py``. The bytes it writes must be byte-identical to the
ones the AI team delivered. ``prepare_checkpoint.py`` remains the separate,
documented fallback for developers who hold the extracted archive instead of a
released ``.pth`` (see the root README).

The release URL and checksum are intentionally **not** hardcoded. They are
supplied per-environment through ``--url``/``FID_MODEL_URL`` and
``--sha256``/``FID_MODEL_SHA256`` until the Release actually exists, at which
point the responsible developer records the real values in the README and in
the release notes.

TRUST MODEL
-----------
A partially written or truncated 544 MiB file is a *misleading* artefact: the
backend would load it, or fail to, long after the download that produced it
has scrolled out of view. Three properties prevent that:

1. Bytes go to a temporary ``.part`` file in the destination directory, never
   to the destination itself.
2. The destination is only ever replaced by ``os.replace``, an atomic rename
   on the same filesystem, and only after the bytes have been verified.
3. Every failure path - bad status, dropped connection, short read, disk full,
   Ctrl-C, checksum mismatch - unlinks the temporary file.

So the final checkpoint is either the previous good file or the new verified
file. It is never a partial one.

SHA-256 is recommended but not mandatory, because a checksum is only useful if
it is published alongside the asset. With no checksum the download is verified
by HTTP status and length only; the script says so plainly rather than implying
the file was integrity-checked.

Usage (from ``backend/``)::

    python tools/setup_model.py --url <release asset url> --sha256 <hex digest>

Or, once the real values are known::

    set FID_MODEL_URL=https://github.com/<org>/<repo>/releases/download/<tag>/best_accuracy_model.pth
    set FID_MODEL_SHA256=<hex digest>
    python tools/setup_model.py
"""

from __future__ import annotations

import argparse
import errno
import hashlib
import os
import shutil
import sys
import tempfile
import time
import urllib.error
import urllib.request
from pathlib import Path

# backend/tools/setup_model.py -> backend/tools -> backend/
BACKEND_DIR = Path(__file__).resolve().parents[1]

#: The path the backend already loads by default (see app/core/config.py and
#: app/services/model_service.py). Writing here means zero further setup.
DEFAULT_DESTINATION = BACKEND_DIR / "artifacts" / "best_accuracy_model.pth"

URL_ENV_VAR = "FID_MODEL_URL"
SHA256_ENV_VAR = "FID_MODEL_SHA256"

#: 1 MiB. Large enough that syscall overhead is irrelevant, small enough that
#: memory use is flat regardless of how large the checkpoint is.
CHUNK_SIZE = 1024 * 1024

#: Free space required beyond the download itself, to absorb filesystem
#: metadata and the temporary file the atomic rename needs.
DISK_HEADROOM_BYTES = 16 * 1024 * 1024

_HEX_DIGITS = set("0123456789abcdef")


class ModelSetupError(RuntimeError):
    """An expected, user-actionable failure.

    Every message this class carries is meant to be read by a person who just
    ran the script, so each one names the thing that went wrong and the thing
    to do next. Nothing here is a bug in the script.
    """


class MissingUrlError(ModelSetupError):
    """No release URL was provided by flag or environment."""


class InvalidChecksumError(ModelSetupError):
    """The supplied checksum is not a usable SHA-256 digest."""


class DownloadFailedError(ModelSetupError):
    """The transfer itself failed: HTTP error, connection lost, short read."""


class InsufficientDiskSpaceError(ModelSetupError):
    """There is not enough free space to hold the download."""


class ChecksumMismatchError(ModelSetupError):
    """The downloaded bytes do not match the expected digest."""


class DestinationExistsError(ModelSetupError):
    """A checkpoint is already in place and --force was not given."""


def human_bytes(count: float) -> str:
    """Format a byte count for a human reader.

    Mirrors the "543.6 MiB" style used elsewhere in this project, and is
    deliberately a one-liner: the only job here is to make a progress line
    skimmable, not to be a units library.
    """
    if count < 1024:
        return f"{int(count)} B"
    if count < 1024 * 1024:
        return f"{count / 1024:.1f} KiB"
    if count < 1024 * 1024 * 1024:
        return f"{count / (1024 * 1024):.1f} MiB"
    return f"{count / (1024 * 1024 * 1024):.2f} GiB"


def normalize_sha256(value: str) -> str:
    """Return ``value`` as a bare lowercase 64-character hex digest.

    Accepts the shapes a checksum arrives in from the wild: uppercase, and
    prefixed with ``sha256:`` (the format GitHub shows next to an asset's
    digest). Rejects anything else loudly, because a mistyped digest that is
    silently ignored would turn an integrity check into a no-op.
    """
    candidate = value.strip().lower()
    if candidate.startswith("sha256:"):
        candidate = candidate[len("sha256:") :].strip()

    if len(candidate) != 64 or not set(candidate) <= _HEX_DIGITS:
        raise InvalidChecksumError(
            f"{SHA256_ENV_VAR}/--sha256 must be a 64-character SHA-256 hex "
            f"digest; got {value!r}."
        )

    return candidate


def sha256_of(path: Path) -> str:
    """Return the SHA-256 hex digest of a file, reading it in chunks.

    Streaming rather than ``path.read_bytes()`` for the same reason the
    download streams: a 544 MiB file must not become a 544 MiB allocation.
    """
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_free_space(destination: Path, needed: int) -> None:
    """Fail before writing if the volume cannot hold the download."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(destination.parent).free

    if free < needed + DISK_HEADROOM_BYTES:
        raise InsufficientDiskSpaceError(
            f"Not enough free space at {destination.parent}: "
            f"{human_bytes(free)} available, "
            f"{human_bytes(needed + DISK_HEADROOM_BYTES)} required "
            f"({human_bytes(needed)} download plus "
            f"{human_bytes(DISK_HEADROOM_BYTES)} headroom). "
            "Free space and run this again."
        )


def _content_length(response: object) -> int | None:
    """Return the declared body size, or ``None`` if the server omitted it.

    Chunked transfer responses legitimately have no Content-Length, so its
    absence is not an error - it only means the progress line cannot show a
    percentage or pre-check disk space.
    """
    headers = getattr(response, "headers", None)
    if headers is None:
        return None

    raw = headers.get("Content-Length")
    if raw is None:
        return None

    try:
        declared = int(raw)
    except (TypeError, ValueError):
        return None

    return declared if declared >= 0 else None


def _report(
    downloaded: int, total: int | None, started: float, destination: Path
) -> None:
    """    Overwrite one line with transfer progress.

    Rate is measured from a monotonic start time rather than a rolling
    average, because for a single large file the only number that matters is
    "is this going to finish", and elapsed-based rate answers that.
    """
    elapsed = max(time.monotonic() - started, 1e-6)
    rate = downloaded / elapsed
    size = human_bytes(downloaded)

    if total:
        percent = min(downloaded * 100 / total, 100.0)
        filled = int(percent // 2.5)
        bar = "#" * filled + "-" * (40 - filled)
        detail = (
            f"  {bar}  {percent:5.1f}%  {size} / {human_bytes(total)}"
            f"  at {human_bytes(rate)}/s"
        )
    else:
        detail = f"  {size}  at {human_bytes(rate)}/s  (size not advertised)"

    print(f"\r{detail}", end="", file=sys.stdout, flush=True)


def _stream_to_disk(
    response: object, handle: object, total: int | None, destination: Path
) -> int:
    """Copy the response body into ``handle``, returning bytes written.

    Split out from the caller so the transfer loop can be exercised on its
    own, and so the ENOSPC translation sits next to the write that can raise
    it.
    """
    started = time.monotonic()
    downloaded = 0
    last_report = 0
    last_reported_at = -1

    while True:
        try:
            chunk = response.read(CHUNK_SIZE)
        except OSError as exc:
            raise DownloadFailedError(
                f"Connection lost after {human_bytes(downloaded)}: {exc}"
            ) from exc

        if not chunk:
            break

        try:
            handle.write(chunk)
        except OSError as exc:
            if exc.errno == errno.ENOSPC:
                raise InsufficientDiskSpaceError(
                    f"Ran out of disk space while writing to {destination}. "
                    "Free space and run this again."
                ) from exc
            raise DownloadFailedError(
                f"Failed to write to {destination} after "
                f"{human_bytes(downloaded)}: {exc}"
            ) from exc

        downloaded += len(chunk)

        # Repainting on every 1 MiB would flood a log file; once every 8 MiB is
        # enough to show that the transfer is alive. Servers that advertise no
        # size report on every chunk so something is always visible.
        if downloaded - last_report >= 8 * CHUNK_SIZE or not total:
            _report(downloaded, total, started, destination)
            last_report = downloaded
            last_reported_at = downloaded

    # The loop above may already have painted the final position - a file that
    # is an exact multiple of the report interval lands on it. Repainting the
    # same bytes would print the finished bar twice.
    if downloaded != last_reported_at:
        _report(downloaded, total, started, destination)
    print("", file=sys.stdout, flush=True)

    return downloaded


def _download_to_temporary(url: str, destination: Path, total: int | None) -> Path:
    """Fetch ``url`` into a ``.part`` file beside ``destination``.

    Returns the path of the temporary file on success. The caller owns it and
    is responsible for either promoting or deleting it - nothing else cleans
    up after a failure, so a crashed run can never leave a file that a later
    run would mistake for a finished download.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)

    handle = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=destination.parent,
        prefix=f"{destination.name}.",
        suffix=".part",
        delete=False,
    )
    temporary = Path(handle.name)

    try:
        request = urllib.request.Request(
            url, headers={"User-Agent": "fake-image-detector-setup-model"}
        )

        try:
            response = urllib.request.urlopen(request, timeout=60)
        except urllib.error.HTTPError as exc:
            raise DownloadFailedError(
                f"Server returned HTTP {exc.code} {exc.reason} for {url}."
            ) from exc
        except urllib.error.URLError as exc:
            raise DownloadFailedError(
                f"Could not reach {url}: {exc.reason}"
            ) from exc
        except OSError as exc:
            raise DownloadFailedError(
                f"Could not reach {url}: {exc}"
            ) from exc

        with response:
            declared = _content_length(response)
            _stream_to_disk(response, handle, declared, destination)

    except BaseException:
        # KeyboardInterrupt and SystemExit land here too, which is the point:
        # an interrupted run must not leave a .part file behind either.
        handle.close()
        temporary.unlink(missing_ok=True)
        raise

    handle.close()
    return temporary


def _promote(temporary: Path, destination: Path) -> None:
    """Atomically move a verified temporary file onto the destination.

    ``os.replace`` is a rename, so a reader either sees the old file or the new
    one and never a half-written target. It is why the temporary file has to
    live in the destination's directory - a rename across filesystems is a
    copy, and a copy can be interrupted.
    """
    try:
        os.replace(temporary, destination)
    except OSError as exc:
        temporary.unlink(missing_ok=True)
        raise ModelSetupError(
            f"Downloaded and verified, but could not move it into place at "
            f"{destination}: {exc}"
        ) from exc


def setup_model(
    url: str,
    destination: Path = DEFAULT_DESTINATION,
    expected_sha256: str | None = None,
    force: bool = False,
) -> Path:
    """Ensure ``destination`` holds the release checkpoint.

    Returns the destination path on success. Raises :class:`ModelSetupError`
    for every expected failure, so the caller never has to inspect a return
    value to know whether the model is in place.
    """
    if not url or not url.strip():
        raise MissingUrlError(
            "No download URL supplied. Pass --url <release asset url> or set "
            f"the {URL_ENV_VAR} environment variable."
        )

    url = url.strip()

    # `None` means "not supplied" and is allowed. An empty string is not: it
    # reaches here from `--sha256 ""` or an empty FID_MODEL_SHA256, and both
    # are a misconfiguration that would otherwise silently disable the check.
    expected = (
        normalize_sha256(expected_sha256) if expected_sha256 is not None else None
    )

    if destination.exists():
        if not force:
            if expected is None:
                raise DestinationExistsError(
                    f"A checkpoint already exists at {destination} "
                    f"({human_bytes(destination.stat().st_size)}). "
                    f"No {SHA256_ENV_VAR}/--sha256 was supplied, so it cannot "
                    "be verified; leaving it untouched. Run with --force to "
                    "replace it."
                )

            actual = sha256_of(destination)
            if actual == expected:
                print(f"Model already present and verified: {destination}")
                print(f"  sha256: {actual}")
                print("  Nothing to download.")
                return destination

            raise ChecksumMismatchError(
                f"The existing checkpoint at {destination} does not match the "
                f"expected digest.\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}\n"
                "Leaving it untouched. Run with --force to replace it."
            )

        print(f"--force: replacing the existing checkpoint at {destination}")

    print(f"Downloading runtime checkpoint from:\n  {url}")
    print(f"Destination: {destination}")
    if expected is None:
        print(
            f"WARNING: no SHA-256 supplied, so the download cannot be checked "
            f"for corruption.\n"
            f"         Set {SHA256_ENV_VAR} or pass --sha256 to enable "
            f"verification."
        )

    try:
        head = urllib.request.urlopen(
            urllib.request.Request(
                url, method="HEAD", headers={"User-Agent": "setup_model"}
            ),
            timeout=60,
        )
        with head:
            total = _content_length(head)
    except (urllib.error.HTTPError, urllib.error.URLError, OSError):
        # HEAD is an optimisation, not a requirement: plenty of servers reject
        # it or omit Content-Length. Let the GET decide the truth.
        total = None

    if total:
        _require_free_space(destination, total)

    temporary: Path | None = None
    try:
        temporary = _download_to_temporary(url, destination, total)
        written = temporary.stat().st_size

        if total and written != total:
            raise DownloadFailedError(
                f"Incomplete download: server declared "
                f"{human_bytes(total)} but {human_bytes(written)} arrived. "
                "The transfer was cut short. Run this again."
            )

        actual = sha256_of(temporary)
        if expected is not None and actual != expected:
            raise ChecksumMismatchError(
                f"Checksum mismatch - the download is corrupt and was "
                f"discarded.\n"
                f"  expected: {expected}\n"
                f"  actual:   {actual}\n"
                f"  url:      {url}\n"
                f"{destination} was not modified."
            )

        _promote(temporary, destination)
    except BaseException:
        # Belt and braces: _download_to_temporary and _promote already clean up
        # their own failures, but this guarantees the invariant holds for every
        # exit path, including one added later.
        if temporary is not None and temporary.exists():
            temporary.unlink(missing_ok=True)
        raise

    print(f"\nModel ready: {destination}")
    print(f"  size:   {human_bytes(destination.stat().st_size)}")
    print(f"  sha256: {actual}")
    if expected is None:
        print("  (unverified - no checksum was supplied)")

    return destination


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Download the runtime checkpoint from a release asset into "
            "backend/artifacts/best_accuracy_model.pth. This script fetches "
            "the artefact; it does not build it - see prepare_checkpoint.py "
            "for the extracted-archive fallback."
        )
    )
    parser.add_argument(
        "--url",
        default=os.environ.get(URL_ENV_VAR),
        help=(
            f"Release asset URL to download. Defaults to ${URL_ENV_VAR}."
        ),
    )
    parser.add_argument(
        "--sha256",
        default=os.environ.get(SHA256_ENV_VAR),
        help=(
            f"Expected SHA-256 hex digest of the asset. Defaults to "
            f"${SHA256_ENV_VAR}. Strongly recommended."
        ),
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help="Where to write the checkpoint. Defaults to the path the "
        "backend already loads.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing checkpoint instead of leaving it in place.",
    )
    args = parser.parse_args(argv)

    try:
        setup_model(
            url=args.url or "",
            destination=args.destination,
            expected_sha256=args.sha256,
            force=args.force,
        )
    except ModelSetupError as exc:
        print(f"\nerror: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\ninterrupted - no checkpoint was written", file=sys.stderr)
        return 130

    return 0


if __name__ == "__main__":
    sys.exit(main())
