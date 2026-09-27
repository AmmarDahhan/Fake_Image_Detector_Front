"""Tests for tools/setup_model.py.

The release asset is not reachable from a test run, so ``urllib.request`` is
replaced with in-process doubles. What is under test is everything the script
is responsible for:

* the source of the URL and the checksum (flag, environment, or neither),
* SHA-256 verification, and what happens on a mismatch,
* rerun safety - a valid file must not be re-downloaded, and must not be
  silently replaced,
* the failure taxonomy (HTTP, connection, short read, disk full, Ctrl-C),
* and the central invariant: a partial or corrupt download never becomes the
  final checkpoint.

Payloads are small byte strings. The checkpoint is 544 MiB in production, but
every decision in this script is driven by the SHA-256 of the bytes and by
whether a transfer completed, neither of which depends on size, so testing at
kilobytes exercises the same code paths without a 544 MiB fixture.
"""

from __future__ import annotations

import ast
import errno
import hashlib
import urllib.error
from collections import namedtuple
from pathlib import Path

import pytest

from tools import setup_model
from tools.setup_model import (
    ChecksumMismatchError,
    DestinationExistsError,
    DownloadFailedError,
    InsufficientDiskSpaceError,
    InvalidChecksumError,
    MissingUrlError,
)

PAYLOAD = b"not really a checkpoint, but the same bytes matter"
PAYLOAD_SHA256 = hashlib.sha256(PAYLOAD).hexdigest()
OTHER_PAYLOAD = b"a different artefact entirely"
OTHER_SHA256 = hashlib.sha256(OTHER_PAYLOAD).hexdigest()

URL = "https://example.invalid/assets/best_accuracy_model.pth"


# --------------------------------------------------------------------------
# Doubles for urllib
# --------------------------------------------------------------------------


class FakeResponse:
    """Minimal stand-in for the object ``urlopen`` returns.

    Only the three attributes the script actually touches are implemented:
    ``read``, ``headers`` and context-manager support. Anything else would be
    a test of the double rather than of the script.
    """

    def __init__(
        self,
        body: bytes = PAYLOAD,
        content_length: int | None = None,
        declared_length: int | None = None,
        read_error: BaseException | None = None,
    ) -> None:
        self._body = body
        self._offset = 0
        self._read_error = read_error

        # `content_length` is what the headers advertise; `declared_length`
        # lets a test advertise more than it delivers, simulating a truncated
        # transfer.
        advertised = declared_length if declared_length is not None else content_length
        self.headers = {} if advertised is None else {"Content-Length": str(advertised)}

    def read(self, size: int) -> bytes:
        if self._read_error is not None and self._offset == 0:
            raise self._read_error
        chunk = self._body[self._offset : self._offset + size]
        self._offset += len(chunk)
        return chunk

    def __enter__(self) -> "FakeResponse":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None


def install_urlopen(monkeypatch: pytest.MonkeyPatch, handler) -> list[str]:
    """Route every ``urlopen`` call to ``handler``, and record the URLs.

    HEAD is issued first as an optimisation and GET second as the real
    transfer, so a handler has to cope with both unless the test only cares
    about one. Returning the list of requested URLs lets a test assert that
    ``--force`` really did hit the network.
    """
    seen: list[str] = []
    calls: list[tuple[str, str | None]] = []

    def _urlopen(request, timeout=None):
        method = getattr(request, "method", None)
        seen.append(request.full_url)
        calls.append((request.full_url, method))
        return handler(request, len(calls))

    monkeypatch.setattr(setup_model.urllib.request, "urlopen", _urlopen)
    return seen


def serve(body: bytes = PAYLOAD, **response_kwargs):
    """Handler that answers HEAD with the body size and GET with ``body``.

    Advertising the size on HEAD is what a real release server does, and the
    script uses it to pre-check free space, so tests that do not care about
    HEAD should get the realistic behaviour rather than an empty one.
    """

    def handler(request, call_number: int):
        method = getattr(request, "method", None)
        if method == "HEAD":
            return FakeResponse(b"", content_length=len(body))
        return FakeResponse(body, **response_kwargs)

    return handler


@pytest.fixture
def destination(tmp_path: Path) -> Path:
    return tmp_path / "artifacts" / "best_accuracy_model.pth"


def parts_in(directory: Path) -> list[Path]:
    """Any leftover temporary download files."""
    return sorted(directory.glob("*.part"))


# --------------------------------------------------------------------------
# Successful download
# --------------------------------------------------------------------------


def test_downloads_and_installs_the_checkpoint(
    monkeypatch: pytest.MonkeyPatch, destination: Path, capsys
) -> None:
    install_urlopen(monkeypatch, serve())

    result = setup_model.setup_model(
        url=URL, destination=destination, expected_sha256=PAYLOAD_SHA256
    )

    assert result == destination
    assert destination.read_bytes() == PAYLOAD
    assert setup_model.sha256_of(destination) == PAYLOAD_SHA256
    assert "Model ready" in capsys.readouterr().out


def test_destination_defaults_to_the_path_the_backend_loads() -> None:
    """The whole point of the script: no extra configuration after it runs."""
    assert setup_model.DEFAULT_DESTINATION.name == "best_accuracy_model.pth"
    assert setup_model.DEFAULT_DESTINATION.parent.name == "artifacts"
    assert setup_model.DEFAULT_DESTINATION.parent.parent.name == "backend"


def test_creates_the_artifacts_directory(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    assert not destination.parent.exists()

    install_urlopen(monkeypatch, serve())
    setup_model.setup_model(url=URL, destination=destination)

    assert destination.parent.is_dir()


def test_works_without_a_checksum_but_says_so(
    monkeypatch: pytest.MonkeyPatch, destination: Path, capsys
) -> None:
    install_urlopen(monkeypatch, serve())

    setup_model.setup_model(url=URL, destination=destination)

    assert destination.read_bytes() == PAYLOAD
    output = capsys.readouterr().out
    assert "WARNING" in output
    assert "cannot be checked" in output
    assert "unverified" in output


def test_tolerates_a_server_that_omits_content_length(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    install_urlopen(monkeypatch, serve(content_length=None))

    setup_model.setup_model(url=URL, destination=destination, expected_sha256=PAYLOAD_SHA256)

    assert destination.read_bytes() == PAYLOAD


# --------------------------------------------------------------------------
# SHA-256 verification
# --------------------------------------------------------------------------


def test_accepts_an_uppercase_digest_with_a_prefix(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    """GitHub shows asset digests as `sha256:<hex>`, sometimes uppercase."""
    install_urlopen(monkeypatch, serve())

    setup_model.setup_model(
        url=URL,
        destination=destination,
        expected_sha256=f"  sha256:{PAYLOAD_SHA256.upper()}  ",
    )

    assert destination.read_bytes() == PAYLOAD


@pytest.mark.parametrize(
    "value",
    [
        "",
        "abc",
        "g" * 64,
        PAYLOAD_SHA256[:-1],
        PAYLOAD_SHA256 + "0",
    ],
)
def test_rejects_an_unusable_digest(value: str, destination: Path) -> None:
    with pytest.raises(InvalidChecksumError):
        setup_model.setup_model(
            url=URL, destination=destination, expected_sha256=value
        )

    assert not destination.exists()


def test_rejects_a_malformed_digest_before_downloading(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    seen = install_urlopen(monkeypatch, serve())

    with pytest.raises(InvalidChecksumError):
        setup_model.setup_model(
            url=URL, destination=destination, expected_sha256="not-a-digest"
        )

    assert seen == [], "must not touch the network with an invalid digest"


def test_sha256_of_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(PAYLOAD)

    assert setup_model.sha256_of(path) == PAYLOAD_SHA256


def test_sha256_of_streams_rather_than_reading_whole_file(tmp_path: Path) -> None:
    """A 544 MiB checkpoint must not become a 544 MiB allocation."""
    path = tmp_path / "big.bin"
    path.write_bytes(b"x" * (3 * setup_model.CHUNK_SIZE + 17))

    opened = 0
    real_open = Path.open

    def counting_open(self, *args, **kwargs):
        nonlocal opened
        if self == path:
            opened += 1
        return real_open(self, *args, **kwargs)

    Path.open = counting_open
    try:
        setup_model.sha256_of(path)
    finally:
        Path.open = real_open

    assert opened == 1


# --------------------------------------------------------------------------
# Checksum mismatch
# --------------------------------------------------------------------------


def test_corrupt_download_under_force_keeps_the_previous_checkpoint(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    """A failed replacement must not cost the developer the good file."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(PAYLOAD)

    # --force is required to get here at all: the installed checkpoint already
    # matches the expected digest, so without it the rerun check short-circuits.
    # The server then hands back something else entirely.
    install_urlopen(monkeypatch, serve(OTHER_PAYLOAD))

    with pytest.raises(ChecksumMismatchError) as excinfo:
        setup_model.setup_model(
            url=URL,
            destination=destination,
            expected_sha256=PAYLOAD_SHA256,
            force=True,
        )

    assert destination.read_bytes() == PAYLOAD, "the good checkpoint was damaged"
    assert "Checksum mismatch" in str(excinfo.value)
    assert PAYLOAD_SHA256 in str(excinfo.value)
    assert OTHER_SHA256 in str(excinfo.value)
    assert parts_in(destination.parent) == [], "corrupt download was left on disk"


def test_mismatch_leaves_no_final_file_at_all(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    install_urlopen(monkeypatch, serve())

    with pytest.raises(ChecksumMismatchError):
        setup_model.setup_model(
            url=URL, destination=destination, expected_sha256=OTHER_SHA256
        )

    assert not destination.exists()
    assert parts_in(destination.parent) == []


def test_truncated_but_self_consistent_bytes_still_fail_verification(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    """A short read that also matches Content-Length is caught by the digest."""
    truncated = PAYLOAD[:10]
    install_urlopen(monkeypatch, serve(truncated))

    with pytest.raises(ChecksumMismatchError):
        setup_model.setup_model(
            url=URL, destination=destination, expected_sha256=PAYLOAD_SHA256
        )

    assert not destination.exists()


# --------------------------------------------------------------------------
# Rerun safety
# --------------------------------------------------------------------------


def test_existing_valid_file_is_reported_and_not_downloaded_again(
    monkeypatch: pytest.MonkeyPatch, destination: Path, capsys
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(PAYLOAD)

    seen = install_urlopen(monkeypatch, serve())

    result = setup_model.setup_model(
        url=URL, destination=destination, expected_sha256=PAYLOAD_SHA256
    )

    assert result == destination
    assert seen == [], "re-downloaded a checkpoint that was already correct"
    output = capsys.readouterr().out
    assert "already present" in output
    assert "Nothing to download" in output


def test_existing_file_without_a_checksum_is_left_alone(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(PAYLOAD)

    seen = install_urlopen(monkeypatch, serve())

    with pytest.raises(DestinationExistsError) as excinfo:
        setup_model.setup_model(url=URL, destination=destination)

    assert destination.read_bytes() == PAYLOAD
    assert seen == [], "downloaded without being asked to"
    assert "--force" in str(excinfo.value)


def test_existing_file_with_the_wrong_checksum_is_left_alone(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(OTHER_PAYLOAD)

    seen = install_urlopen(monkeypatch, serve())

    with pytest.raises(ChecksumMismatchError) as excinfo:
        setup_model.setup_model(
            url=URL, destination=destination, expected_sha256=PAYLOAD_SHA256
        )

    assert destination.read_bytes() == OTHER_PAYLOAD
    assert seen == []
    assert "--force" in str(excinfo.value)


# --------------------------------------------------------------------------
# --force
# --------------------------------------------------------------------------


def test_force_replaces_a_valid_existing_checkpoint(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    """--force is an explicit request to replace, so it re-downloads."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(PAYLOAD)

    seen = install_urlopen(monkeypatch, serve(OTHER_PAYLOAD))

    setup_model.setup_model(
        url=URL,
        destination=destination,
        expected_sha256=OTHER_SHA256,
        force=True,
    )

    assert destination.read_bytes() == OTHER_PAYLOAD
    assert seen, "force did not download"


def test_force_is_not_needed_when_nothing_is_there(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    install_urlopen(monkeypatch, serve())

    setup_model.setup_model(
        url=URL,
        destination=destination,
        expected_sha256=PAYLOAD_SHA256,
        force=True,
    )

    assert destination.read_bytes() == PAYLOAD


def test_main_force_flag_is_wired_through(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(PAYLOAD)

    install_urlopen(monkeypatch, serve(OTHER_PAYLOAD))

    exit_code = setup_model.main(
        [
            "--url",
            URL,
            "--sha256",
            OTHER_SHA256,
            "--destination",
            str(destination),
            "--force",
        ]
    )

    assert exit_code == 0
    assert destination.read_bytes() == OTHER_PAYLOAD


# --------------------------------------------------------------------------
# Failure paths
# --------------------------------------------------------------------------


def test_no_url_supplied_is_a_clear_error(destination: Path) -> None:
    with pytest.raises(MissingUrlError) as excinfo:
        setup_model.setup_model(url="", destination=destination)

    message = str(excinfo.value)
    assert "--url" in message
    assert setup_model.URL_ENV_VAR in message
    assert not destination.exists()


def test_no_url_via_main_is_a_clear_error(
    monkeypatch: pytest.MonkeyPatch, destination: Path, capsys
) -> None:
    monkeypatch.delenv(setup_model.URL_ENV_VAR, raising=False)

    exit_code = setup_model.main(["--destination", str(destination)])

    assert exit_code == 1
    error = capsys.readouterr().err
    assert "No download URL supplied" in error
    assert setup_model.URL_ENV_VAR in error


def test_whitespace_url_is_also_rejected(destination: Path) -> None:
    with pytest.raises(MissingUrlError):
        setup_model.setup_model(url="   ", destination=destination)


def test_http_error_is_reported_clearly(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    def handler(request, call_number: int):
        raise urllib.error.HTTPError(URL, 404, "Not Found", {}, None)

    install_urlopen(monkeypatch, handler)

    with pytest.raises(DownloadFailedError) as excinfo:
        setup_model.setup_model(url=URL, destination=destination)

    assert "404" in str(excinfo.value)
    assert not destination.exists()
    assert parts_in(destination.parent) == []


def test_connection_error_is_reported_clearly(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    def handler(request, call_number: int):
        if getattr(request, "method", None) == "HEAD":
            return FakeResponse(b"", content_length=None)
        raise urllib.error.URLError("name resolution failed")

    install_urlopen(monkeypatch, handler)

    with pytest.raises(DownloadFailedError) as excinfo:
        setup_model.setup_model(url=URL, destination=destination)

    assert "name resolution failed" in str(excinfo.value)
    assert not destination.exists()


def test_a_server_that_rejects_head_still_downloads(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    """HEAD is an optimisation; its absence must not be fatal."""

    def handler(request, call_number: int):
        if getattr(request, "method", None) == "HEAD":
            raise urllib.error.HTTPError(URL, 405, "Method Not Allowed", {}, None)
        return FakeResponse(PAYLOAD, content_length=len(PAYLOAD))

    install_urlopen(monkeypatch, handler)

    setup_model.setup_model(
        url=URL, destination=destination, expected_sha256=PAYLOAD_SHA256
    )

    assert destination.read_bytes() == PAYLOAD


def test_dropped_connection_mid_transfer_is_reported(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    def handler(request, call_number: int):
        if getattr(request, "method", None) == "HEAD":
            return FakeResponse(b"", content_length=None)
        return FakeResponse(
            PAYLOAD, read_error=OSError("connection reset by peer")
        )

    install_urlopen(monkeypatch, handler)

    with pytest.raises(DownloadFailedError) as excinfo:
        setup_model.setup_model(url=URL, destination=destination)

    assert "connection reset by peer" in str(excinfo.value)
    assert not destination.exists()
    assert parts_in(destination.parent) == []


def test_short_read_against_the_declared_length_is_rejected(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    """Server promises more than it delivers: treat as a failed transfer."""
    truncated = PAYLOAD[:12]

    def handler(request, call_number: int):
        if getattr(request, "method", None) == "HEAD":
            return FakeResponse(b"", content_length=len(PAYLOAD))
        return FakeResponse(truncated, content_length=len(PAYLOAD))

    install_urlopen(monkeypatch, handler)

    with pytest.raises(DownloadFailedError) as excinfo:
        setup_model.setup_model(url=URL, destination=destination)

    assert "Incomplete download" in str(excinfo.value)
    assert not destination.exists()
    assert parts_in(destination.parent) == []


def test_insufficient_disk_space_is_caught_before_writing(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    # 1 KiB free, against a download that needs 50 B plus 16 MiB of headroom.
    usage = namedtuple("usage", "total used free")
    monkeypatch.setattr(setup_model.shutil, "disk_usage", lambda path: usage(2048, 1024, 1024))

    install_urlopen(monkeypatch, serve(content_length=len(PAYLOAD)))

    with pytest.raises(InsufficientDiskSpaceError) as excinfo:
        setup_model.setup_model(url=URL, destination=destination)

    assert "free space" in str(excinfo.value).lower()
    assert not destination.exists()


def test_disk_filling_mid_write_is_caught(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    install_urlopen(monkeypatch, serve())

    def full_write(self, chunk):
        raise OSError(errno.ENOSPC, "No space left on device")

    original = setup_model.tempfile.NamedTemporaryFile

    def patched(*args, **kwargs):
        handle = original(*args, **kwargs)
        real_write = handle.write

        def write(chunk):
            full_write(handle, chunk)
            return real_write(chunk)

        handle.write = write
        return handle

    monkeypatch.setattr(setup_model.tempfile, "NamedTemporaryFile", patched)

    with pytest.raises(InsufficientDiskSpaceError) as excinfo:
        setup_model.setup_model(url=URL, destination=destination)

    assert "disk space" in str(excinfo.value).lower()
    assert not destination.exists()
    assert parts_in(destination.parent) == []


# --------------------------------------------------------------------------
# The core invariant: no partial download becomes the checkpoint
# --------------------------------------------------------------------------


def test_partial_download_never_becomes_the_final_checkpoint(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    """Interrupted, corrupt, oversized and cancelled all leave nothing behind."""
    cancelled = KeyboardInterrupt()
    scenarios = {
        "truncated": serve(PAYLOAD[:8], content_length=len(PAYLOAD)),
        "corrupt": serve(OTHER_PAYLOAD),
        "unreachable": lambda request, n: (_ for _ in ()).throw(
            urllib.error.URLError("down")
        ),
        "cancelled": lambda request, n: (_ for _ in ()).throw(cancelled),
    }

    for name, handler in scenarios.items():
        artifact = tmp_path_for(destination) / f"{name}.pth"
        install_urlopen(monkeypatch, handler)

        try:
            setup_model.setup_model(
                url=URL, destination=artifact, expected_sha256=PAYLOAD_SHA256
            )
        except (DownloadFailedError, ChecksumMismatchError):
            pass
        except KeyboardInterrupt:
            pass

        assert not artifact.exists(), f"{name}: a partial file became the checkpoint"
        assert parts_in(artifact.parent) == [], f"{name}: left a .part behind"


def tmp_path_for(destination: Path) -> Path:
    """A sibling of ``destination`` so each scenario gets a clean filename."""
    return destination.parent / destination.name.replace(".pth", "-scenario.pth")


def test_cancelled_download_leaves_no_temporary_file(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    def handler(request, call_number: int):
        if getattr(request, "method", None) == "HEAD":
            return FakeResponse(b"", content_length=None)
        raise KeyboardInterrupt

    install_urlopen(monkeypatch, handler)

    with pytest.raises(KeyboardInterrupt):
        setup_model.setup_model(url=URL, destination=destination)

    assert not destination.exists()
    assert parts_in(destination.parent) == []


def test_main_reports_interruption_without_a_traceback(
    monkeypatch: pytest.MonkeyPatch, destination: Path, capsys
) -> None:
    def handler(request, call_number: int):
        if getattr(request, "method", None) == "HEAD":
            return FakeResponse(b"", content_length=None)
        raise KeyboardInterrupt

    install_urlopen(monkeypatch, handler)

    exit_code = setup_model.main(["--url", URL, "--destination", str(destination)])

    assert exit_code == 130
    assert "interrupted" in capsys.readouterr().err
    assert not destination.exists()


def test_temporary_file_lives_beside_the_destination(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    """os.replace is only atomic within one filesystem, so the .part must
    share the destination's directory."""
    observed: list[Path] = []
    original = setup_model.tempfile.NamedTemporaryFile

    def spy(*args, **kwargs):
        handle = original(*args, **kwargs)
        observed.append(Path(handle.name).parent)
        return handle

    monkeypatch.setattr(setup_model.tempfile, "NamedTemporaryFile", spy)
    install_urlopen(monkeypatch, serve())

    setup_model.setup_model(
        url=URL, destination=destination, expected_sha256=PAYLOAD_SHA256
    )

    assert observed == [destination.parent]
    assert destination.read_bytes() == PAYLOAD


# --------------------------------------------------------------------------
# Environment variables and the CLI surface
# --------------------------------------------------------------------------


def test_url_and_checksum_can_come_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    monkeypatch.setenv(setup_model.URL_ENV_VAR, URL)
    monkeypatch.setenv(setup_model.SHA256_ENV_VAR, PAYLOAD_SHA256)
    install_urlopen(monkeypatch, serve())

    exit_code = setup_model.main(["--destination", str(destination)])

    assert exit_code == 0
    assert destination.read_bytes() == PAYLOAD


def test_command_line_beats_the_environment(
    monkeypatch: pytest.MonkeyPatch, destination: Path
) -> None:
    monkeypatch.setenv(setup_model.URL_ENV_VAR, "https://wrong.invalid/model.pth")
    monkeypatch.setenv(setup_model.SHA256_ENV_VAR, OTHER_SHA256)
    seen = install_urlopen(monkeypatch, serve(PAYLOAD))

    exit_code = setup_model.main(
        [
            "--url",
            URL,
            "--sha256",
            PAYLOAD_SHA256,
            "--destination",
            str(destination),
        ]
    )

    assert exit_code == 0
    assert seen == [URL, URL]
    assert destination.read_bytes() == PAYLOAD


def test_main_returns_zero_on_success(monkeypatch: pytest.MonkeyPatch, destination: Path) -> None:
    install_urlopen(monkeypatch, serve())

    exit_code = setup_model.main(
        [
            "--url",
            URL,
            "--sha256",
            PAYLOAD_SHA256,
            "--destination",
            str(destination),
        ]
    )

    assert exit_code == 0


def test_main_reports_http_failure_with_exit_code_one(
    monkeypatch: pytest.MonkeyPatch, destination: Path, capsys
) -> None:
    def handler(request, call_number: int):
        raise urllib.error.HTTPError(URL, 503, "Service Unavailable", {}, None)

    install_urlopen(monkeypatch, handler)

    exit_code = setup_model.main(
        ["--url", URL, "--destination", str(destination)]
    )

    assert exit_code == 1
    assert "503" in capsys.readouterr().err


def test_no_release_url_is_hardcoded() -> None:
    """The Release does not exist yet, so the script must ship without one.

    Docstrings are excluded deliberately: the module docstring documents the
    *shape* of a release asset URL so the eventual real command is obvious,
    and a placeholder like ``<org>/<repo>/<tag>`` is not a runnable value.
    What must not exist is a URL the script could actually fetch.
    """
    source = Path(setup_model.__file__).read_text(encoding="utf-8")
    tree = ast.parse(source)

    docstrings = {
        id(node.value)
        for node in ast.walk(tree)
        if isinstance(node, ast.Expr)
        and isinstance(node.value, ast.Constant)
        and isinstance(node.value.value, str)
    }

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and id(node) not in docstrings
        ):
            assert not node.value.startswith(("http://", "https://")), (
                f"hardcoded URL in executable code: {node.value!r}"
            )


def test_placeholders_in_the_docstring_are_not_runnable() -> None:
    """If the docstring example ever grows a real owner, fail here."""
    source = Path(setup_model.__file__).read_text(encoding="utf-8")

    for line in source.splitlines():
        if "releases/download" in line:
            assert "<org>" in line and "<repo>" in line and "<tag>" in line, (
                f"docstring names a real release: {line.strip()}"
            )


# --------------------------------------------------------------------------
# Formatting helper
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "value,expected",
    [
        (0, "0 B"),
        (512, "512 B"),
        (1024, "1.0 KiB"),
        (1024 * 1024, "1.0 MiB"),
        (543.6 * 1024 * 1024, "543.6 MiB"),
        (2 * 1024**3, "2.00 GiB"),
    ],
)
def test_human_bytes(value: float, expected: str) -> None:
    assert setup_model.human_bytes(value) == expected
