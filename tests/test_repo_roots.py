"""Repository authorization regressions using temporary, non-secret files only."""

import asyncio
import errno
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from unittest.mock import AsyncMock, Mock, patch

import pytest
from pydantic import ValidationError

from orchestrai.config import settings as config
from orchestrai.execution import shell
from orchestrai.execution.shell import run_command
from orchestrai.orchestrator import intake
from orchestrai.orchestrator.intake import scan_repo
from orchestrai.orchestrator.orchestrator import Orchestrator
from orchestrai.policies.paths import PathPolicy, PathPolicyError
from orchestrai.registry.registry import CapabilityRegistry
from orchestrai.server.tools import _reload_config, handle_tool


@pytest.fixture(autouse=True)
def clean_environment():
    with patch.dict(os.environ, {}, clear=True):
        yield


def test_settings_omitted_roots_snapshot_startup_before_first_load(tmp_path, monkeypatch):
    startup = Path.cwd().resolve()
    monkeypatch.chdir(tmp_path)
    first = config.Settings.load()
    assert first.policy.allowed_roots == (startup,)
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    assert config.reload_settings().policy.allowed_roots == (startup,)


def test_settings_explicit_empty_roots_deny_all():
    assert config.Settings(policy={"allowed_roots": []}).policy.allowed_roots == ()


@pytest.mark.asyncio
async def test_execution_rejects_unauthorized_cwd_before_spawn(tmp_path, monkeypatch):
    allowed = tmp_path / "allowed"
    denied = tmp_path / "denied"
    allowed.mkdir()
    denied.mkdir()
    monkeypatch.setattr(
        config, "_settings", config.Settings(policy={"allowed_roots": [str(allowed)]}),
    )
    spawn = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    with pytest.raises(PathPolicyError):
        await run_command([sys.executable, "-c", "pass"], cwd=str(denied))

    spawn.assert_not_awaited()


@pytest.mark.asyncio
async def test_execution_rejects_explicit_empty_cwd_before_spawn(tmp_path, monkeypatch):
    publish_roots([tmp_path])
    monkeypatch.setattr(shell, "STARTUP_WORKSPACE", tmp_path)
    process = AsyncMock()
    process.returncode = 0
    process.communicate.return_value = (b"", b"")
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    with pytest.raises(PathPolicyError, match="^Repository path is not permitted$"):
        await run_command([sys.executable, "-c", "pass"], cwd="")

    spawn.assert_not_awaited()


@pytest.mark.asyncio
async def test_execution_without_cwd_respects_deny_all(monkeypatch):
    monkeypatch.setattr(
        config, "_settings", config.Settings(policy={"allowed_roots": []}),
    )
    spawn = AsyncMock()
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)

    with pytest.raises(PathPolicyError):
        await run_command([sys.executable, "-c", "pass"])

    spawn.assert_not_awaited()


async def test_execution_pins_authorized_cwd_across_rename_swap(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "marker.txt").write_text("inside dummy")
    (outside / "marker.txt").write_text("outside dummy")
    publish_roots([root])
    spawn = asyncio.create_subprocess_exec
    inherited = ()

    async def swap_then_spawn(*args, **kwargs):
        nonlocal inherited
        inherited = kwargs.get("pass_fds", ())
        root.rename(tmp_path / "saved-repo")
        root.symlink_to(outside, target_is_directory=True)
        return await spawn(*args, **kwargs)

    monkeypatch.setattr(asyncio, "create_subprocess_exec", swap_then_spawn)
    output = await run_command(
        [sys.executable, "-c", "from pathlib import Path; print(Path('marker.txt').read_text())"],
        cwd=str(root),
    )

    assert output.success
    assert output.stdout.strip() == "inside dummy"
    assert "outside dummy" not in output.stdout + output.stderr
    assert inherited
    for descriptor in inherited:
        with pytest.raises(OSError) as error:
            os.fstat(descriptor)
        assert error.value.errno == errno.EBADF


@pytest.mark.parametrize("outcome", ["success", "missing", "error", "cancelled"])
async def test_execution_closes_pinned_cwd_with_mock_spawn(tmp_path, monkeypatch, outcome):
    publish_roots([tmp_path])
    process = AsyncMock()
    process.returncode = 0
    process.communicate.return_value = (b"inside dummy", b"")
    inherited = ()

    async def spawn(*args, **kwargs):
        nonlocal inherited
        inherited = kwargs["pass_fds"]
        assert os.path.samestat(os.stat(kwargs["cwd"]), os.stat(tmp_path))
        os.fstat(inherited[0])
        if outcome == "missing":
            raise FileNotFoundError("dummy executable missing")
        if outcome == "error":
            raise OSError("dummy spawn failure")
        if outcome == "cancelled":
            raise asyncio.CancelledError
        return process

    mocked_spawn = AsyncMock(side_effect=spawn)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", mocked_spawn)
    if outcome in ("error", "cancelled"):
        with pytest.raises(OSError if outcome == "error" else asyncio.CancelledError):
            await run_command(["dummy"], cwd=str(tmp_path))
    else:
        output = await run_command(["dummy"], cwd=str(tmp_path))
        assert output.success is (outcome == "success")
    mocked_spawn.assert_awaited_once()
    assert inherited
    for descriptor in inherited:
        with pytest.raises(OSError) as error:
            os.fstat(descriptor)
        assert error.value.errno == errno.EBADF


@pytest.mark.parametrize("reference", ["missing", "wrong-directory", "dev-fd-fallback"])
async def test_execution_requires_supported_descriptor_cwd(tmp_path, monkeypatch, reference):
    publish_roots([tmp_path])
    real_stat = os.stat
    references = []

    def stat_reference(path, *args, **kwargs):
        if isinstance(path, str) and path.startswith(("/proc/self/fd/", "/dev/fd/")):
            references.append(path)
            if reference == "wrong-directory":
                return real_stat(tmp_path.parent)
            if reference == "dev-fd-fallback" and path.startswith("/dev/fd/"):
                return os.fstat(int(path.rsplit("/", 1)[1]))
            raise FileNotFoundError("dummy descriptor reference unavailable")
        return real_stat(path, *args, **kwargs)

    process = AsyncMock()
    process.returncode = 0
    process.communicate.return_value = (b"", b"")
    spawn = AsyncMock(return_value=process)
    monkeypatch.setattr(os, "stat", stat_reference)
    monkeypatch.setattr(asyncio, "create_subprocess_exec", spawn)
    if reference == "dev-fd-fallback":
        assert (await run_command(["dummy"], cwd=str(tmp_path))).success
        assert spawn.call_args.kwargs["cwd"].startswith("/dev/fd/")
    else:
        with pytest.raises(PathPolicyError, match="^Repository path is not permitted$"):
            await run_command(["dummy"], cwd=str(tmp_path))
        spawn.assert_not_awaited()
    assert len(references) == 2
    for path in references:
        with pytest.raises(OSError) as error:
            os.fstat(int(path.rsplit("/", 1)[1]))
        assert error.value.errno == errno.EBADF


@pytest.mark.parametrize("runner", ["run_tests", "run_lint", "run_typecheck"])
async def test_shell_detection_authorizes_before_any_probe(tmp_path, runner):
    publish_roots([])
    with (
        patch.object(PathPolicy, "repository", side_effect=PathPolicyError) as authorize,
        patch.object(Path, "exists", side_effect=AssertionError("early probe")) as exists,
        patch.object(Path, "lstat", side_effect=AssertionError("lstat before admission")) as lstat,
        patch.object(Path, "read_text", side_effect=AssertionError("early read")) as read,
        patch("os.lstat", side_effect=AssertionError("OS lstat before admission")) as os_lstat,
        patch("os.listdir", side_effect=AssertionError("listing before admission")) as listing,
        patch.object(asyncio, "create_subprocess_exec", new_callable=AsyncMock) as spawn,
        pytest.raises(PathPolicyError, match="^Repository path is not permitted$"),
    ):
        await getattr(shell, runner)(str(tmp_path))
    authorize.assert_called_once_with(str(tmp_path))
    for effect in (exists, lstat, read, os_lstat, listing, spawn):
        effect.assert_not_called()


@pytest.mark.parametrize("content", [
    b'{"jest":', b'["jest"]', b'{"jest": "\xff"}',
    pytest.param(b'{"jest":' + b'9' * 5000 + b'}', id="oversized-number"),
])
async def test_shell_detection_rejects_malformed_package(tmp_path, content):
    (tmp_path / "package.json").write_bytes(content)
    publish_roots([tmp_path])
    with (
        patch.object(shell, "run_command", new_callable=AsyncMock) as command,
        pytest.raises(PathPolicyError, match="^Repository path is not permitted$"),
    ):
        command.return_value = Mock(success=True, stdout="", stderr="")
        await shell.run_tests(str(tmp_path))
    command.assert_not_awaited()


@pytest.fixture
def observed_package_reads(monkeypatch):
    """Observe real descriptor reads and explicit or implicit UTF-8 decoding."""
    fdopen = os.fdopen
    opened = []
    decoded = Mock()

    class ObservedBytes(bytes):
        def decode(self, *args, **kwargs):
            decoded()
            return super().decode(*args, **kwargs)

    def observe(descriptor, *args, **kwargs):
        stream = fdopen(descriptor, *args, **kwargs)
        read = stream.read

        def read_content(*args, **kwargs):
            if isinstance(stream, io.TextIOBase):
                decoded()
            content = read(*args, **kwargs)
            return ObservedBytes(content) if isinstance(content, bytes) else content

        stream.read = Mock(side_effect=read_content)
        opened.append((stream, descriptor))
        return stream

    monkeypatch.setattr(os, "fdopen", observe)
    return opened, decoded


@pytest.mark.parametrize("content", [
    pytest.param(b'"' + b'x' * (64 * 1024) + b'"', id="large-valid-string"),
    pytest.param(b'{"jest":"' + b'x' * (64 * 1024) + b'"}', id="large-valid-object"),
    pytest.param(b'["' + b'x' * (64 * 1024) + b'"]', id="large-valid-array"),
    pytest.param(b'{"jest":"' + b'x' * (64 * 1024), id="large-malformed-document"),
])
async def test_shell_rejects_oversized_package_before_read_decode_parse_or_spawn(
    tmp_path, content, observed_package_reads,
):
    (tmp_path / "package.json").write_bytes(content)
    publish_roots([tmp_path])
    opened, decoded = observed_package_reads
    process = AsyncMock()
    process.returncode = 0
    process.communicate.return_value = (b"", b"")
    with (
        patch.object(json, "loads", wraps=json.loads) as parse,
        patch.object(asyncio, "create_subprocess_exec", new_callable=AsyncMock) as spawn,
    ):
        spawn.return_value = process
        with pytest.raises(PathPolicyError, match="^Repository path is not permitted$"):
            await shell.run_tests(str(tmp_path))

    decoded.assert_not_called()
    parse.assert_not_called()
    spawn.assert_not_awaited()
    assert opened
    for stream, descriptor in opened:
        stream.read.assert_not_called()
        assert stream.closed
        with pytest.raises(OSError) as error:
            os.fstat(descriptor)
        assert error.value.errno == errno.EBADF


async def test_shell_rejects_package_growth_after_fstat_before_decode_parse_or_spawn(
    tmp_path, monkeypatch, observed_package_reads,
):
    maximum = 64 * 1024
    package = tmp_path / "package.json"
    initial = b'{"jest":"dummy"}'
    package.write_bytes(initial)
    identity = package.stat()
    publish_roots([tmp_path])
    opened, decoded = observed_package_reads
    fstat = os.fstat
    grew = False

    def grow_after_stat(descriptor):
        nonlocal grew
        metadata = fstat(descriptor)
        if not grew and os.path.samestat(metadata, identity):
            assert metadata.st_size == len(initial)
            with package.open("ab") as writer:
                writer.write(b" " * (maximum + 1 - len(initial)))
            grew = True
        return metadata

    monkeypatch.setattr(os, "fstat", grow_after_stat)
    process = AsyncMock()
    process.returncode = 0
    process.communicate.return_value = (b"", b"")
    with (
        patch.object(json, "loads", wraps=json.loads) as parse,
        patch.object(asyncio, "create_subprocess_exec", new_callable=AsyncMock) as spawn,
    ):
        spawn.return_value = process
        with pytest.raises(PathPolicyError, match="^Repository path is not permitted$"):
            await shell.run_tests(str(tmp_path))

    assert grew
    decoded.assert_not_called()
    parse.assert_not_called()
    spawn.assert_not_awaited()
    assert len(opened) == 1
    stream, descriptor = opened[0]
    stream.read.assert_called_once_with(maximum + 1)
    assert stream.closed
    with pytest.raises(OSError) as error:
        fstat(descriptor)
    assert error.value.errno == errno.EBADF


@pytest.mark.parametrize("size", [64 * 1024 - 1, 64 * 1024])
async def test_shell_accepts_package_up_to_byte_limit(tmp_path, size, observed_package_reads):
    content = b'{"jest":"' + b'x' * (size - len(b'{"jest":""}')) + b'"}'
    (tmp_path / "package.json").write_bytes(content)
    publish_roots([tmp_path])
    opened, decoded = observed_package_reads
    with patch.object(shell, "run_command", new_callable=AsyncMock) as command:
        command.return_value = Mock(success=True, stdout="", stderr="")
        result = await shell.run_tests(str(tmp_path))

    assert result.framework == "jest"
    command.assert_awaited_once()
    decoded.assert_called_once()
    assert len(opened) == 1
    assert opened[0][0].closed


@pytest.mark.parametrize("content", [
    pytest.param(b'"' + b'x' * (64 * 1024) + b'"', id="large-valid-string"),
    pytest.param(b'{"jest":"' + b'x' * (64 * 1024) + b'"}', id="large-valid-object"),
    pytest.param(b'["' + b'x' * (64 * 1024) + b'"]', id="large-valid-array"),
    pytest.param(b'{"jest":"' + b'x' * (64 * 1024), id="large-malformed-document"),
])
def test_scan_rejects_oversized_package_before_read_decode_parse_or_side_effects(
    tmp_path, content, observed_package_reads, capsys, caplog,
):
    package = tmp_path / "package.json"
    package.write_bytes(content)
    publish_roots([tmp_path])
    opened, decoded = observed_package_reads
    with (
        patch.object(json, "loads", wraps=json.loads) as parse,
        patch.object(intake, "make_artifact_id", return_value="unused-artifact") as artifact_id,
        patch.object(asyncio, "create_subprocess_exec", new_callable=AsyncMock) as spawn,
        pytest.raises(PathPolicyError, match="^Repository path is not permitted$"),
    ):
        scan_repo(str(tmp_path))

    decoded.assert_not_called()
    parse.assert_not_called()
    artifact_id.assert_not_called()
    spawn.assert_not_awaited()
    assert list(tmp_path.iterdir()) == [package]
    assert capsys.readouterr() == ("", "")
    assert caplog.text == ""
    assert len(opened) == 1
    stream, descriptor = opened[0]
    stream.read.assert_not_called()
    assert stream.closed
    with pytest.raises(OSError) as error:
        os.fstat(descriptor)
    assert error.value.errno == errno.EBADF


def test_scan_rejects_package_growth_after_fstat_before_decode_parse_or_side_effects(
    tmp_path, monkeypatch, observed_package_reads, capsys, caplog,
):
    maximum = 64 * 1024
    package = tmp_path / "package.json"
    initial = b'{"jest":"dummy"}'
    package.write_bytes(initial)
    identity = package.stat()
    publish_roots([tmp_path])
    opened, decoded = observed_package_reads
    fstat = os.fstat
    grew = False

    def grow_after_stat(descriptor):
        nonlocal grew
        metadata = fstat(descriptor)
        if not grew and os.path.samestat(metadata, identity):
            assert metadata.st_size == len(initial)
            with package.open("ab") as writer:
                writer.write(b" " * (maximum + 1 - len(initial)))
            grew = True
        return metadata

    monkeypatch.setattr(os, "fstat", grow_after_stat)
    with (
        patch.object(json, "loads", wraps=json.loads) as parse,
        patch.object(intake, "make_artifact_id", return_value="unused-artifact") as artifact_id,
        patch.object(asyncio, "create_subprocess_exec", new_callable=AsyncMock) as spawn,
        pytest.raises(PathPolicyError, match="^Repository path is not permitted$"),
    ):
        scan_repo(str(tmp_path))

    assert grew
    decoded.assert_not_called()
    parse.assert_not_called()
    artifact_id.assert_not_called()
    spawn.assert_not_awaited()
    assert list(tmp_path.iterdir()) == [package]
    assert capsys.readouterr() == ("", "")
    assert caplog.text == ""
    assert len(opened) == 1
    stream, descriptor = opened[0]
    stream.read.assert_called_once_with(maximum + 1)
    assert stream.closed
    with pytest.raises(OSError) as error:
        fstat(descriptor)
    assert error.value.errno == errno.EBADF


@pytest.mark.parametrize("size", [64 * 1024 - 1, 64 * 1024])
def test_scan_accepts_package_up_to_byte_limit(tmp_path, size, observed_package_reads):
    prefix = b'{"react":"\xc3\xa9","next":"dummy","vitest":"'
    content = prefix + b'x' * (size - len(prefix) - len(b'"}')) + b'"}'
    (tmp_path / "package.json").write_bytes(content)
    publish_roots([tmp_path])
    opened, decoded = observed_package_reads

    summary = scan_repo(str(tmp_path))

    assert summary.language == "typescript/javascript"
    assert summary.frameworks == ["react", "nextjs"]
    assert summary.test_framework == "vitest"
    decoded.assert_called_once()
    assert len(opened) == 1
    opened[0][0].read.assert_called_once_with(64 * 1024 + 1)
    assert opened[0][0].closed


def test_scan_rejects_invalid_utf8_package(tmp_path, observed_package_reads):
    (tmp_path / "package.json").write_bytes(b'{"react":"\xff"}')
    publish_roots([tmp_path])
    opened, decoded = observed_package_reads
    with (
        patch.object(intake, "make_artifact_id", return_value="unused-artifact") as artifact_id,
        pytest.raises(PathPolicyError, match="^Repository path is not permitted$") as error,
    ):
        scan_repo(str(tmp_path))

    assert error.value.__cause__ is None
    assert error.value.__suppress_context__
    artifact_id.assert_not_called()
    decoded.assert_called_once()
    assert len(opened) == 1
    assert opened[0][0].closed


def test_general_read_text_preserves_unbounded_replacement_and_newlines(tmp_path):
    content = b'x' * (64 * 1024 + 1) + b'\r\n\xff'
    (tmp_path / "notes.txt").write_bytes(content)
    policy = PathPolicy((tmp_path,))

    assert policy.read_text("notes.txt", tmp_path) == "x" * (64 * 1024 + 1) + "\n\ufffd"


@pytest.mark.parametrize("reader", ["read_text", "read_bytes"])
@pytest.mark.parametrize("failure_type", [
    OSError, MemoryError, PathPolicyError, asyncio.CancelledError,
])
@pytest.mark.parametrize("cleanup_fails", [False, True], ids=["cleanup-ok", "cleanup-error"])
def test_read_closes_raw_descriptor_on_fdopen_failure(
    tmp_path, monkeypatch, reader, failure_type, cleanup_fails,
):
    target = tmp_path / "notes.txt"
    target.write_text("dummy content")
    policy = PathPolicy((tmp_path,))
    read = getattr(policy, reader)
    kwargs = {} if reader == "read_text" else {"max_bytes": 100}
    failure = failure_type() if failure_type is PathPolicyError else failure_type("dummy OS detail")
    opened = []
    raw_closes = []
    close = os.close

    def fail_fdopen(descriptor, *args, **kwargs):
        opened.append(descriptor)
        assert os.path.samestat(os.fstat(descriptor), target.stat())
        raise failure

    def observe_close(descriptor):
        if descriptor in opened:
            raw_closes.append(descriptor)
        close(descriptor)
        if cleanup_fails and descriptor in opened:
            # A close error need not leave the descriptor open; never retry it.
            raise OSError("dummy cleanup detail")

    monkeypatch.setattr(os, "fdopen", fail_fdopen)
    monkeypatch.setattr(os, "close", observe_close)
    expected = PathPolicyError if failure_type is OSError else failure_type
    try:
        with pytest.raises(expected) as error:
            read("notes.txt", tmp_path, **kwargs)

        if failure_type is OSError:
            assert str(error.value) == "Repository path is not permitted"
            assert error.value.__cause__ is None
            assert error.value.__suppress_context__
            assert error.value.__context__ is failure
        else:
            assert error.value is failure
        assert len(opened) == 1
        with pytest.raises(OSError) as closed:
            os.fstat(opened[0])
        assert closed.value.errno == errno.EBADF
        assert raw_closes == opened
    finally:
        # Keep the deliberately failing RED run from leaking its test descriptor.
        for descriptor in opened:
            try:
                os.fstat(descriptor)
            except OSError as error:
                assert error.errno == errno.EBADF
            else:
                close(descriptor)


@pytest.mark.parametrize("reader", ["read_text", "read_bytes"])
@pytest.mark.parametrize("failure_type", [
    None, OSError, UnicodeError, RuntimeError, MemoryError, PathPolicyError, asyncio.CancelledError,
])
def test_read_preserves_stream_ownership(tmp_path, monkeypatch, reader, failure_type):
    (tmp_path / "notes.txt").write_text("dummy content")
    policy = PathPolicy((tmp_path,))
    read = getattr(policy, reader)
    kwargs = {} if reader == "read_text" else {"max_bytes": 100}
    sanitized = (OSError, UnicodeError) if reader == "read_text" else (OSError,)
    failure = None if failure_type is None else (
        failure_type() if failure_type is PathPolicyError else failure_type("dummy read detail")
    )
    opened = []
    raw_closes = []
    fdopen = os.fdopen
    close = os.close

    def observe_fdopen(descriptor, *args, **kwargs):
        stream = fdopen(descriptor, *args, **kwargs)
        stream.close = Mock(wraps=stream.close)
        if failure is not None:
            stream.read = Mock(side_effect=failure)
        opened.append((stream, descriptor))
        return stream

    def observe_close(descriptor):
        if opened and descriptor == opened[0][1]:
            raw_closes.append(descriptor)
        close(descriptor)

    monkeypatch.setattr(os, "fdopen", observe_fdopen)
    monkeypatch.setattr(os, "close", observe_close)
    if failure_type is None:
        expected_content = "dummy content" if reader == "read_text" else b"dummy content"
        assert read("notes.txt", tmp_path, **kwargs) == expected_content
    else:
        expected = PathPolicyError if failure_type in sanitized else failure_type
        with pytest.raises(expected) as error:
            read("notes.txt", tmp_path, **kwargs)
        if failure_type in sanitized:
            assert str(error.value) == "Repository path is not permitted"
            assert error.value.__cause__ is None
            assert error.value.__suppress_context__
        else:
            assert error.value is failure
    assert len(opened) == 1
    stream, descriptor = opened[0]
    stream.close.assert_called_once_with()
    assert stream.closed
    assert raw_closes == []
    with pytest.raises(OSError) as closed:
        os.fstat(descriptor)
    assert closed.value.errno == errno.EBADF


@pytest.mark.parametrize("runner", ["run_tests", "run_lint", "run_typecheck"])
async def test_shell_detection_denied_cwd_never_inspects_candidates(tmp_path, runner):
    allowed = tmp_path / "allowed"
    denied = tmp_path / "denied"
    allowed.mkdir()
    denied.mkdir()
    publish_roots([allowed])
    with (
        patch.object(PathPolicy, "file", side_effect=AssertionError("file probe")) as probe,
        patch.object(Path, "exists", side_effect=AssertionError("raw probe")) as exists,
        patch.object(Path, "lstat", side_effect=AssertionError("raw lstat")) as lstat,
        patch.object(Path, "read_text", side_effect=AssertionError("raw read")) as read,
        patch("os.listdir", side_effect=AssertionError("listing")) as listing,
        patch.object(asyncio, "create_subprocess_exec", new_callable=AsyncMock) as spawn,
        pytest.raises(PathPolicyError, match="^Repository path is not permitted$"),
    ):
        await getattr(shell, runner)(str(denied))
    for effect in (probe, exists, lstat, read, listing, spawn):
        effect.assert_not_called()


async def test_shell_explicit_framework_still_authorizes_first(tmp_path):
    publish_roots([])
    with (
        patch.object(shell, "_detect_test_framework") as detection,
        patch.object(shell, "run_command", new_callable=AsyncMock) as command,
        pytest.raises(PathPolicyError),
    ):
        await shell.run_tests(str(tmp_path), framework="pytest")
    detection.assert_not_called()
    command.assert_not_awaited()


@pytest.mark.parametrize("runner,name,content,field,expected", [
    ("run_tests", "pytest.ini", "", "framework", "pytest"),
    ("run_tests", "pyproject.toml", "", "framework", "pytest"),
    ("run_tests", "package.json", '{"vitest":"dummy"}', "framework", "vitest"),
    ("run_tests", "package.json", '{"jest":"dummy"}', "framework", "jest"),
    ("run_tests", "package.json", '{"mocha":"dummy"}', "framework", "mocha"),
    ("run_tests", "package.json", '{}', "framework", "npm test"),
    ("run_tests", "Cargo.toml", "", "framework", "cargo test"),
    ("run_tests", "go.mod", "", "framework", "go test"),
    ("run_lint", ".ruff.toml", "", "tool", "ruff"),
    ("run_lint", ".eslintrc.json", "", "tool", "eslint"),
    ("run_lint", ".eslintrc.js", "", "tool", "eslint"),
    ("run_lint", "Cargo.toml", "", "tool", "clippy"),
    ("run_lint", "go.mod", "", "tool", "golint"),
    ("run_typecheck", "tsconfig.json", "", "tool", "tsc"),
    ("run_typecheck", "pyproject.toml", "", "tool", "mypy"),
    ("run_typecheck", "go.mod", "", "tool", "go build"),
])
async def test_shell_detection_uses_authorized_files_and_canonical_cwd(
    tmp_path, runner, name, content, field, expected,
):
    root = tmp_path / "repo"
    root.mkdir()
    (root / name).write_text(content)
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    publish_roots([root])
    with (
        patch.object(Path, "exists", side_effect=AssertionError("raw probe")),
        patch.object(Path, "read_text", side_effect=AssertionError("raw read")),
        patch.object(shell, "run_command", new_callable=AsyncMock) as command,
    ):
        command.return_value = Mock(success=True, stdout="", stderr="")
        result = await getattr(shell, runner)(str(alias))
    assert getattr(result, field) == expected
    assert command.call_args.kwargs["cwd"] == str(root)


@pytest.mark.parametrize("runner,name", [
    ("run_tests", "package.json"), ("run_lint", ".eslintrc.json"),
    ("run_typecheck", "tsconfig.json"),
])
@pytest.mark.parametrize("denial", ["symlink", "sensitive", "directory"])
async def test_shell_detection_skips_unauthorized_candidates(tmp_path, runner, name, denial):
    root = tmp_path / "repo"
    root.mkdir()
    if denial == "symlink":
        outside = tmp_path / "outside-dummy"
        outside.write_text('{"jest":"outside dummy"}')
        (root / name).symlink_to(outside)
    elif denial == "directory":
        (root / name).mkdir()
    else:
        (root / name).write_text('{"jest":"inside dummy"}')
    publish_roots([root], sensitive_path_patterns=[name] if denial == "sensitive" else [])
    with (
        patch.object(PathPolicy, "read_text", side_effect=AssertionError("denied read")),
        patch.object(shell, "run_command", new_callable=AsyncMock) as command,
    ):
        command.return_value = Mock(success=True, stdout="", stderr="")
        result = await getattr(shell, runner)(str(root))
    if runner == "run_tests":
        assert result.framework == "pytest"
    else:
        assert result.tool == ("ruff" if runner == "run_lint" else "mypy")


@pytest.mark.parametrize("swap", ["file", "parent"])
async def test_shell_detection_read_rejects_swap_after_file_authorization(
    tmp_path, monkeypatch, swap,
):
    root = tmp_path / "repo"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (root / "package.json").write_text('{"jest":"inside dummy"}')
    (outside / "package.json").write_text('{"vitest":"outside dummy"}')
    publish_roots([root])
    authorize = PathPolicy.file
    calls = 0

    def authorize_then_swap(self, value, repository, **kwargs):
        nonlocal calls
        result = authorize(self, value, repository, **kwargs)
        if value == "package.json":
            calls += 1
            if calls == 2:  # Detection authorized presence; read must pin its own descriptor.
                if swap == "file":
                    result.unlink()
                    result.symlink_to(outside / "package.json")
                else:
                    root.rename(tmp_path / "saved-repo")
                    root.symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(PathPolicy, "file", authorize_then_swap)
    with (
        patch.object(shell, "run_command", new_callable=AsyncMock) as command,
        pytest.raises(PathPolicyError, match="^Repository path is not permitted$"),
    ):
        await shell.run_tests(str(root))
    assert calls == 2
    command.assert_not_awaited()


@pytest.mark.parametrize("runner,helper", [
    ("run_tests", "_detect_test_framework"), ("run_lint", "_detect_lint_command"),
    ("run_typecheck", "_detect_typecheck_command"),
])
async def test_shell_detection_retains_run_command_as_last_gate(
    tmp_path, monkeypatch, runner, helper,
):
    publish_roots([tmp_path])
    detect = getattr(shell, helper)

    def detect_then_revoke(*args, **kwargs):
        result = detect(*args, **kwargs)
        publish_roots([])
        return result

    monkeypatch.setattr(shell, helper, detect_then_revoke)
    with (
        patch.object(asyncio, "create_subprocess_exec", new_callable=AsyncMock) as spawn,
        pytest.raises(PathPolicyError),
    ):
        await getattr(shell, runner)(str(tmp_path))
    spawn.assert_not_awaited()


def test_settings_roots_are_canonical_and_immutable(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    link = tmp_path / "alias"
    link.symlink_to(root, target_is_directory=True)
    supplied = [str(link)]
    settings = config.Settings(policy={"allowed_roots": supplied})
    supplied.clear()
    assert settings.policy.allowed_roots == (root.resolve(),)
    with pytest.raises(ValidationError):
        settings.policy.allowed_roots = ()


@pytest.mark.parametrize("kind", ["root", "root-link", "missing", "file", "traversal"])
def test_settings_reject_invalid_roots(tmp_path, kind):
    selected = tmp_path / "private-dummy-location"
    if kind == "root":
        selected = Path("/")
    elif kind == "root-link":
        selected.symlink_to("/", target_is_directory=True)
    elif kind == "file":
        selected.write_text("dummy")
    elif kind == "traversal":
        selected = tmp_path / ".." / tmp_path.name
    with pytest.raises(ValueError) as error:
        config.Settings(policy={"allowed_roots": [str(selected)]})
    assert "private-dummy-location" not in str(error.value)


def test_settings_implicit_filesystem_root_fails_at_construction():
    code = """
from orchestrai.config.settings import Settings
try:
    Settings(_env_file=None)
except ValueError:
    print('denied')
else:
    print('accepted')
"""
    result = subprocess.run(
        [sys.executable, "-c", code], cwd="/", capture_output=True, text=True, check=True,
    )
    assert result.stdout.strip() == "denied"


def test_settings_invalid_yaml_roots_cannot_hide_under_environment(tmp_path, monkeypatch):
    path = tmp_path / "settings.yaml"
    path.write_text('policy: {allowed_roots: ["/"]}')
    monkeypatch.setenv("ORCHESTRAI__POLICY__ALLOWED_ROOTS", json.dumps([str(tmp_path)]))
    with pytest.raises(config.ConfigurationError, match="^Invalid configuration: settings$"):
        config.Settings.load(path)


@pytest.mark.parametrize("value", ['["/"]', "null", "not-json", '"/tmp"', "[17]"])
def test_settings_invalid_environment_roots_fail_closed(tmp_path, monkeypatch, value):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("ORCHESTRAI__POLICY__ALLOWED_ROOTS", value)
    with pytest.raises(config.ConfigurationError, match="^Invalid configuration: settings$"):
        config.Settings.load()


def test_settings_valid_environment_roots_override_yaml(tmp_path, monkeypatch):
    path = tmp_path / "settings.yaml"
    path.write_text(f"policy: {{allowed_roots: [{json.dumps(str(tmp_path))}]}}")
    monkeypatch.setenv("ORCHESTRAI__POLICY__ALLOWED_ROOTS", "[]")
    assert config.Settings.load(path).policy.allowed_roots == ()


def test_settings_invalid_root_reload_keeps_published_instance(tmp_path, monkeypatch):
    prior = config.Settings(policy={"allowed_roots": [str(tmp_path)]})
    config.publish_settings(prior)
    monkeypatch.setenv("ORCHESTRAI__POLICY__ALLOWED_ROOTS", '["/"]')
    with pytest.raises(config.ConfigurationError):
        config.reload_settings()
    assert config.get_settings() is prior


@pytest.mark.parametrize("value", [None, "", "/tmp", [""], [None]])
def test_settings_malformed_root_values_fail_closed(value):
    with pytest.raises(ValueError):
        config.Settings(policy={"allowed_roots": value})


def publish_roots(roots, **policy):
    settings = config.Settings(policy={"allowed_roots": roots, **policy})
    config.publish_settings(settings)
    return settings


@pytest.mark.parametrize("case", ["empty", "sibling", "traversal", "relative-escape", "link"])
def test_scan_admission_rejects_escape_before_listing(tmp_path, case):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "repo-sibling"
    outside.mkdir()
    publish_roots([] if case == "empty" else [root])
    requested = root
    if case == "sibling":
        requested = outside
    elif case == "traversal":
        requested = root / ".." / root.name
    elif case == "relative-escape":
        requested = Path("../OrchestrAI")
    elif case == "link":
        requested = root / "alias"
        requested.symlink_to(outside, target_is_directory=True)
    with (
        patch.object(Path, "glob", side_effect=AssertionError("unauthorized listing")),
        pytest.raises(ValueError, match="^Repository path is not permitted$"),
    ):
        scan_repo(str(requested))


def test_scan_admission_accepts_root_descendant_and_resolved_alias(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    child = root / "child"
    child.mkdir()
    link = tmp_path / "alias"
    link.symlink_to(child, target_is_directory=True)
    publish_roots([root])
    for requested in (root, child, link):
        assert scan_repo(str(requested)).total_files == 0


@pytest.mark.parametrize("kind", ["missing", "file", "blank"])
def test_scan_admission_rejects_invalid_directory(tmp_path, kind):
    publish_roots([tmp_path])
    requested = tmp_path / "missing"
    if kind == "file":
        requested.write_text("dummy")
    with pytest.raises(ValueError, match="^Repository path is not permitted$"):
        scan_repo("" if kind == "blank" else str(requested))


@pytest.mark.parametrize("name", [
    ".git/objects", ".ssh", ".aws", ".gnupg", ".kube", ".azure", ".docker",
    ".config", ".env", ".env.local", "secrets", "dummy.pem", "dummy.key",
    ".npmrc", ".pypirc", ".netrc", ".git-credentials", "id_rsa", "id_ed25519",
])
def test_sensitive_paths_denied_even_with_empty_custom_patterns(tmp_path, name):
    requested = tmp_path / name
    requested.mkdir(parents=True)
    publish_roots([tmp_path], sensitive_path_patterns=[])
    with pytest.raises(ValueError, match="^Repository path is not permitted$"):
        PathPolicy.current().repository(requested)


@pytest.mark.parametrize("kind", ["lexical", "resolved"])
def test_sensitive_symlink_names_and_targets_are_denied(tmp_path, kind):
    safe = tmp_path / "ordinary"
    safe.mkdir()
    sensitive = tmp_path / ".ssh"
    if kind == "lexical":
        sensitive.symlink_to(safe, target_is_directory=True)
        requested = sensitive
    else:
        sensitive.mkdir()
        requested = safe / "alias"
        requested.symlink_to(sensitive, target_is_directory=True)
    publish_roots([tmp_path])
    with pytest.raises(ValueError, match="^Repository path is not permitted$"):
        PathPolicy.current().repository(requested)


def test_sensitive_patterns_extend_minimum_without_broad_name_blocks(tmp_path):
    publish_roots([tmp_path], sensitive_path_patterns=["private/", "*.vault"])
    for name in ("private", "account.vault", "credentials_parser", "config", "configuration"):
        path = tmp_path / name
        path.mkdir()
        if name in ("private", "account.vault"):
            with pytest.raises(ValueError):
                PathPolicy.current().repository(path)
        else:
            assert PathPolicy.current().repository(path) == path


@pytest.mark.parametrize("name,patterns", [
    (".ENV", []), (".EnV.Local", []), ("Secrets", []),
    ("dummy.PEM", []), ("dummy.KEY", []),
    ("\uff0eENV", []), (".\uff45\uff4e\uff56", []), ("\uff33ecrets", []),
    ("dummy.\uff30\uff25\uff2d", []), ("dummy.\u212aey", []),
    (".\uff33\uff33\uff28", []),
    ("Data.vault", ["*.\uff36\uff21\uff35\uff2c\uff34"]),
    ("CAFE\u0301", ["Caf\u00e9/"]),
    ("Protected/NEW.TXT", ["protected/new.txt"]),
])
def test_sensitive_matching_normalizes_names_and_patterns(tmp_path, name, patterns):
    requested = tmp_path / name
    requested.mkdir(parents=True)
    policy = PathPolicy((tmp_path,), tuple(patterns))
    with pytest.raises(PathPolicyError, match="^Repository path is not permitted$"):
        policy.repository(requested)


@pytest.mark.parametrize("name", [
    ".ENV_notes", "Secrets_parser.py", "cert.PEM.txt", "Caf\u00e9.py",
])
def test_sensitive_normalization_preserves_legitimate_files(tmp_path, name):
    target = tmp_path / name
    target.write_text("dummy")
    policy = PathPolicy((tmp_path,))
    assert policy.file(name, tmp_path) == target
    assert policy.read_text(name, tmp_path) == "dummy"


@pytest.mark.parametrize("outside_name", ["REPO", "\uff52\uff45\uff50\uff4f"])
def test_sensitive_normalization_does_not_expand_root_containment(tmp_path, outside_name):
    root = tmp_path / "repo"
    outside = tmp_path / outside_name
    root.mkdir()
    if outside.exists():
        pytest.skip("filesystem treats these directory names as identical")
    outside.mkdir()
    policy = PathPolicy((root,))
    with pytest.raises(PathPolicyError):
        policy.repository(outside)


@pytest.mark.parametrize("tree", ["/etc", "/proc", "/sys", "/dev", "/usr", "/var", "/run"])
def test_sensitive_system_trees_denied_before_resolution(tree):
    # Construct a policy directly to assert deny-first ordering without touching system files.
    policy = PathPolicy((Path(tree),))
    with (
        patch.object(Path, "resolve", side_effect=AssertionError("system path inspected")),
        pytest.raises(ValueError, match="^Repository path is not permitted$"),
    ):
        policy.repository(tree)


@pytest.mark.parametrize("tree", [
    "/System", "/Library", "/private", "/SYSTEM", "/library", "/PRIVATE",
    "/\uff33ystem", "/\uff2cibrary", "/\uff50rivate",
])
@pytest.mark.parametrize("suffix", ["", "/dummy-child"])
def test_macos_system_trees_denied_without_host_inspection(tree, suffix):
    policy = PathPolicy((Path(tree),))
    with (
        patch.object(Path, "resolve", side_effect=AssertionError("system resolution")) as resolve,
        patch.object(Path, "lstat", side_effect=AssertionError("system path inspected")) as lstat,
        pytest.raises(PathPolicyError, match="^Repository path is not permitted$"),
    ):
        policy.repository(tree + suffix)
    resolve.assert_not_called()
    lstat.assert_not_called()


def make_orchestrator(tmp_path, roots):
    settings = config.Settings(
        policy={"allowed_roots": roots},
        observability={
            "artifact_dir": str(tmp_path / "artifacts"),
            "trace_dir": str(tmp_path / "traces"),
        },
    )
    config.publish_settings(settings)
    return Orchestrator(CapabilityRegistry())


@pytest.mark.parametrize("case", ["denied", "empty", "blank"])
async def test_submit_root_validation_precedes_every_side_effect(tmp_path, case):
    root = tmp_path / "repo"
    root.mkdir()
    orch = make_orchestrator(tmp_path, [root] if case == "denied" else [])
    requested = "" if case == "blank" else str(tmp_path)
    with (
        patch(
            "orchestrai.orchestrator.orchestrator.TaskContext.resolve",
            side_effect=AssertionError("eligibility ran before path admission"),
        ) as context,
        patch("orchestrai.orchestrator.orchestrator.make_task_id") as task_id,
        patch("orchestrai.orchestrator.orchestrator.make_trace_id") as trace_id,
        patch("orchestrai.orchestrator.orchestrator.ArtifactStore") as store,
        patch("orchestrai.orchestrator.orchestrator.Tracer") as tracer,
        patch("orchestrai.orchestrator.orchestrator.scan_repo") as scan,
        patch.object(orch._router, "route") as route,
        patch.object(orch, "_run", new_callable=AsyncMock) as execute,
    ):
        with pytest.raises(ValueError, match="^Repository path is not permitted$"):
            await orch.submit("Review dummy code", repo_root=requested)
        for effect in (context, task_id, trace_id, store, tracer, scan, route, execute):
            effect.assert_not_called()
    assert not orch._active and not orch._stores and not orch._events
    assert not (tmp_path / "artifacts").exists()
    assert not (tmp_path / "traces").exists()


async def test_submit_stores_canonical_repo_root(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    alias = tmp_path / "alias"
    alias.symlink_to(root, target_is_directory=True)
    orch = make_orchestrator(tmp_path, [root])
    with patch.object(orch, "_run", new_callable=AsyncMock):
        task = await orch.submit("Review dummy code", repo_root=str(alias))
    assert task.brief.repo_root == str(root)


async def test_submit_omitted_repo_remains_none_without_scanning(tmp_path):
    orch = make_orchestrator(tmp_path, [])
    with (
        patch("orchestrai.orchestrator.orchestrator.scan_repo") as scan,
        patch.object(orch, "_run", new_callable=AsyncMock),
    ):
        task = await orch.submit("Explain this algorithm")
    assert task.brief.repo_root is None
    scan.assert_not_called()


@pytest.mark.parametrize("case", [
    "absolute-outside", "traversal", "nested-link", "sensitive", "directory", "missing-link",
    "without-repo", "blank", "malformed",
])
async def test_submit_target_validation_precedes_ids_and_scan(tmp_path, case):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "dummy.py").write_text("pass")
    orch = make_orchestrator(tmp_path, [tmp_path])
    target = str(outside / "dummy.py")
    if case == "traversal":
        target = "../outside/dummy.py"
    elif case == "nested-link":
        (root / "linked").symlink_to(outside, target_is_directory=True)
        target = "linked/dummy.py"
    elif case == "sensitive":
        target = ".env"
        (root / target).write_text("dummy")
    elif case == "directory":
        target = "."
    elif case == "missing-link":
        target = "broken.py"
        (root / target).symlink_to(root / "missing.py")
    elif case == "blank":
        target = ""
    elif case == "malformed":
        target = None
    with (
        patch("orchestrai.orchestrator.orchestrator.make_task_id",
              side_effect=AssertionError("IDs before target admission")) as ids,
        patch("orchestrai.orchestrator.orchestrator.scan_repo") as scan,
    ):
        with pytest.raises(ValueError, match="^Repository path is not permitted$"):
            await orch.submit(
                "Add dummy code", repo_root=None if case == "without-repo" else str(root),
                target_files=[target],
            )
        ids.assert_not_called()
        scan.assert_not_called()
    assert not orch._active and not orch._stores
    assert not (tmp_path / "artifacts").exists()


async def test_submit_targets_canonicalize_existing_and_future_files(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    source = root / "source.py"
    source.write_text("pass")
    (root / "alias.py").symlink_to(source)
    orch = make_orchestrator(tmp_path, [root])
    with patch.object(orch, "_run", new_callable=AsyncMock):
        task = await orch.submit("Add dummy code", repo_root=str(root), target_files=[
            "alias.py", "future/module.py",
        ])
    assert task.brief.target_files == [str(source), str(root / "future/module.py")]


@pytest.mark.parametrize("name", ["package.json", "pyproject.toml", "README.md"])
def test_scan_rejects_nested_file_symlink_escape_before_read_or_probe(tmp_path, name):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside-dummy"
    outside.write_text('{"dependencies": {"react": "dummy"}}')
    (root / name).symlink_to(outside)
    publish_roots([tmp_path])  # Scope is still this repository, not every allowed sibling.
    with patch.object(Path, "read_text", side_effect=AssertionError("escaped content read")):
        summary = scan_repo(str(root))
    assert summary.total_files == 0
    assert summary.language == "unknown"
    assert summary.key_files == []


@pytest.mark.parametrize("name", [".git", ".ssh", "secrets", ".config", "private"])
def test_scan_skips_sensitive_trees_before_listing(tmp_path, name):
    hidden = tmp_path / name
    hidden.mkdir()
    (hidden / "dummy.py").write_text("pass")
    (tmp_path / "README.md").write_text("Public dummy documentation")
    publish_roots([tmp_path], sensitive_path_patterns=["private/"])
    summary = scan_repo(str(tmp_path))
    assert summary.language == "unknown"
    assert summary.total_files == 1
    assert summary.key_files == ["README.md"]


def test_scan_respects_custom_sensitive_file_patterns(tmp_path):
    (tmp_path / "package.json").write_text('{"dependencies": {"react": "dummy"}}')
    publish_roots([tmp_path], sensitive_path_patterns=["package.json"])
    with patch.object(Path, "read_text", side_effect=AssertionError("sensitive content read")):
        summary = scan_repo(str(tmp_path))
    assert summary.total_files == 0


@pytest.mark.parametrize("kind", ["python", "javascript", "rust", "go"])
def test_scan_preserves_authorized_repository_detection(tmp_path, kind):
    filename, content, language, framework = {
        "python": ("pyproject.toml", "[project]", "python", "pytest"),
        "javascript": ("package.json", '{"react":"dummy","vitest":"dummy","next":"dummy"}',
                       "typescript/javascript", "vitest"),
        "rust": ("Cargo.toml", "[package]", "rust", "cargo test"),
        "go": ("go.mod", "module dummy", "go", "go test"),
    }[kind]
    (tmp_path / filename).write_text(content)
    (tmp_path / "README.md").write_text("Dummy documentation")
    publish_roots([tmp_path])
    summary = scan_repo(str(tmp_path))
    assert summary.language == language
    assert summary.test_framework == framework
    assert summary.key_files == [filename, "README.md"]
    assert summary.total_files == 2


def test_future_target_sensitive_pattern_checks_resolved_full_path(tmp_path):
    protected = tmp_path / "protected"
    protected.mkdir()
    (tmp_path / "alias").symlink_to(protected, target_is_directory=True)
    publish_roots([tmp_path], sensitive_path_patterns=["protected/new.py"])
    with pytest.raises(ValueError, match="^Repository path is not permitted$"):
        PathPolicy.current().task_paths(str(tmp_path), ["alias/new.py"])


@pytest.mark.parametrize("targets", ["", "f", {}, {"f": "ignored"}, 0, False, ()])
def test_malformed_target_collections_fail_closed(tmp_path, targets):
    (tmp_path / "f").write_text("dummy")
    publish_roots([tmp_path])
    with pytest.raises(ValueError, match="^Repository path is not permitted$"):
        PathPolicy.current().task_paths(str(tmp_path), targets)


@pytest.mark.parametrize("revocation", ["roots", "targets", "target-link"])
async def test_rerun_revalidates_current_published_policy_before_side_effects(tmp_path, revocation):
    root = tmp_path / "repo"
    root.mkdir()
    target = root / "dummy.py"
    target.write_text("pass")
    orch = make_orchestrator(tmp_path, [root])
    with patch.object(orch, "_run", new_callable=AsyncMock):
        original = await orch.submit("Review dummy code", repo_root=str(root),
                                     target_files=["dummy.py"],
                                     user_preferences={"local_only": True})
    old_settings = orch._settings
    if revocation == "target-link":
        outside = tmp_path / "outside.py"
        outside.write_text("pass")
        target.unlink()
        target.symlink_to(outside)
    else:
        publish_roots([] if revocation == "roots" else [root],
                      sensitive_path_patterns=["dummy.py"] if revocation == "targets" else [])
        assert config.get_settings() is not old_settings
        assert orch._settings is old_settings  # independently published policy can outpace runtime
    before = set(orch._active)
    with (
        patch("orchestrai.orchestrator.orchestrator.make_task_id",
              side_effect=AssertionError("rerun IDs before revalidation")),
        pytest.raises(ValueError, match="^Repository path is not permitted$"),
    ):
        await orch.rerun(original.id, {"local_only": False})
    assert set(orch._active) == before


async def test_rerun_uses_stored_canonical_paths_after_alias_retarget(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    (root / "original.py").write_text("pass")
    (root / "other.py").write_text("pass")
    alias = root / "alias.py"
    alias.symlink_to(root / "original.py")
    orch = make_orchestrator(tmp_path, [root])
    with patch.object(orch, "_run", new_callable=AsyncMock):
        original = await orch.submit("Review dummy code", repo_root=str(root),
                                     target_files=["alias.py"],
                                     user_preferences={"local_only": True})
        alias.unlink()
        alias.symlink_to(root / "other.py")
        rerun = await orch.rerun(original.id, {"local_only": False})
    assert rerun.brief.target_files == [str(root / "original.py")]
    assert rerun._context.eligibility.local_only is True


@pytest.mark.parametrize("entry", ["submit_task", "rerun_with_policy"])
async def test_mcp_uses_same_path_admission(tmp_path, entry):
    orch = make_orchestrator(tmp_path, [tmp_path])
    with patch.object(orch, "_run", new_callable=AsyncMock):
        original = await orch.submit("Review dummy code", repo_root=str(tmp_path))
    publish_roots([])
    args = {"request": "Review dummy code", "repo_root": str(tmp_path)}
    if entry == "rerun_with_policy":
        args = {"task_id": original.id, "policy_overrides": {}}
    before = set(orch._active)
    result = await handle_tool(entry, args, orch, orch._registry)
    assert result["error"] == "Repository path is not permitted"
    assert set(orch._active) == before


async def test_invalid_root_reload_config_does_not_discover_or_publish(tmp_path, monkeypatch):
    orch = make_orchestrator(tmp_path, [tmp_path])
    before = (config.get_settings(), orch._settings, orch._registry, orch._router._policy)
    yaml = tmp_path / "settings.yaml"
    yaml.write_text('policy: {allowed_roots: ["/"]}')
    monkeypatch.setenv("ORCHESTRAI_CONFIG", str(yaml))
    with patch(
        "orchestrai.providers.discovery.discover_providers", new_callable=AsyncMock,
    ) as probe:
        result = await _reload_config({}, orch, orch._registry)
    assert "error" in result
    probe.assert_not_called()
    assert (config.get_settings(), orch._settings, orch._registry, orch._router._policy) == before


def test_relative_roots_use_startup_snapshot_after_cwd_change(tmp_path, monkeypatch):
    startup = Path.cwd().resolve()
    monkeypatch.chdir(tmp_path)
    settings = config.Settings(policy={"allowed_roots": ["src"]})
    config.publish_settings(settings)
    assert settings.policy.allowed_roots == (startup / "src",)
    assert PathPolicy.current().repository("src") == startup / "src"


@pytest.mark.parametrize("swap", ["file", "parent"])
def test_read_text_rejects_symlink_swap_after_authorization(tmp_path, monkeypatch, swap):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (root / "package.json").write_text("safe dummy")
    (outside / "package.json").write_text("outside dummy")
    publish_roots([tmp_path])
    original = PathPolicy.file

    def authorize_then_swap(self, value, repository, **kwargs):
        path = original(self, value, repository, **kwargs)
        if swap == "file":
            path.unlink()
            path.symlink_to(outside / "package.json")
        else:
            root.rename(tmp_path / "saved-repo")
            root.symlink_to(outside, target_is_directory=True)
        return path

    monkeypatch.setattr(PathPolicy, "file", authorize_then_swap)
    with pytest.raises(ValueError, match="^Repository path is not permitted$"):
        PathPolicy.current().read_text("package.json", root)


def test_enumeration_rejects_parent_symlink_swap_after_authorization(tmp_path, monkeypatch):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "dummy.py").write_text("pass")
    publish_roots([tmp_path])
    authorize = PathPolicy._authorize
    calls = 0

    def authorize_then_swap(self, value, base, kind, repository=None, allow_missing=False):
        nonlocal calls
        result = authorize(self, value, base, kind, repository, allow_missing)
        if kind == "directory":
            calls += 1
            if calls == 2:  # First authorizes admission; second precedes actual enumeration.
                root.rename(tmp_path / "saved-repo")
                root.symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(PathPolicy, "_authorize", authorize_then_swap)
    with pytest.raises(ValueError, match="^Repository path is not permitted$"):
        list(PathPolicy.current().iter_files(root))


@pytest.mark.parametrize("name", ["outside.py", "pyproject.toml", "README.md"])
def test_enumeration_leaf_swap_cannot_leak_file_metadata(tmp_path, monkeypatch, name):
    root = tmp_path / "repo"
    root.mkdir()
    entry = root / name
    entry.write_text("safe dummy")
    outside = tmp_path / "outside-dummy"
    outside.write_text("outside dummy")
    publish_roots([tmp_path])
    authorize = PathPolicy._authorize

    def authorize_then_swap(self, value, base, kind, repository=None, allow_missing=False):
        result = authorize(self, value, base, kind, repository, allow_missing)
        if kind == "any" and result == entry:
            entry.unlink()
            entry.symlink_to(outside)
        return result

    monkeypatch.setattr(PathPolicy, "_authorize", authorize_then_swap)
    summary = scan_repo(str(root))
    assert summary.total_files == 0
    assert summary.language == "unknown"
    assert summary.key_files == []


def test_enumeration_preserves_static_in_repo_symlink_files(tmp_path):
    real = tmp_path / "data" / "original"
    real.parent.mkdir()
    real.write_text("safe dummy")
    (tmp_path / "README.md").symlink_to(real)
    (tmp_path / "alias").symlink_to(real.parent, target_is_directory=True)
    (real.parent / "cycle").symlink_to(tmp_path, target_is_directory=True)
    publish_roots([tmp_path])
    summary = scan_repo(str(tmp_path))
    assert summary.total_files == 2
    assert summary.language == "unknown"
    assert summary.key_files == ["README.md"]


@pytest.mark.parametrize("kind", ["file", "directory", "future-ancestor", "configured-root"])
def test_path_kind_checks_reject_post_resolution_symlink_swap(tmp_path, monkeypatch, kind):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "dummy.py").write_text("outside dummy")
    publish_roots([tmp_path])
    selected = root
    if kind == "file":
        selected = root / "dummy.py"
        selected.write_text("safe dummy")
    elif kind == "future-ancestor":
        selected = root / "future"
        selected.mkdir()
    resolve = Path.resolve

    def resolve_then_swap(self, *args, **kwargs):
        result = resolve(self, *args, **kwargs)
        if self == selected:
            if kind == "file":
                selected.unlink()
                selected.symlink_to(outside / "dummy.py")
            else:
                selected.rename(tmp_path / "saved-directory")
                selected.symlink_to(outside, target_is_directory=True)
        return result

    monkeypatch.setattr(Path, "resolve", resolve_then_swap)
    with pytest.raises(ValueError):
        if kind == "file":
            PathPolicy.current().file("dummy.py", root)
        elif kind == "future-ancestor":
            PathPolicy.current().file("future/new.py", root, allow_missing=True)
        elif kind == "configured-root":
            config.Settings(policy={"allowed_roots": [str(root)]})
        else:
            PathPolicy.current().repository(root)
