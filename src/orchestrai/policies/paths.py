"""Canonical repository paths, independent of configuration loading."""

import os
import stat
from collections.abc import Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Literal
from unicodedata import normalize

# Captured before any Settings load, cwd change, or runtime reload.
STARTUP_WORKSPACE = Path.cwd().resolve()
MAX_PACKAGE_METADATA_BYTES = 64 * 1024


def _sensitive_text(value: str) -> str:
    return normalize("NFKC", value).casefold()


_SYSTEM_TREES = tuple(Path(_sensitive_text(name)) for name in (
    "/etc", "/proc", "/sys", "/dev", "/root", "/boot", "/usr", "/bin", "/sbin",
    "/lib", "/lib64", "/var", "/run", "/System", "/Library", "/private",
))
_SENSITIVE_NAMES = frozenset(_sensitive_text(name) for name in (
    ".git", ".ssh", ".aws", ".gnupg", ".kube", ".azure", ".docker", ".config",
    ".npmrc", ".pypirc", ".netrc", ".git-credentials", ".gitconfig",
    "id_rsa", "id_dsa", "id_ecdsa", "id_ed25519",
))
_MINIMUM_PATTERNS = (".env", ".env.*", "secrets/", "*.pem", "*.key")


def canonical_roots(roots: tuple[Path, ...]) -> tuple[Path, ...]:
    try:
        canonical = []
        for root in roots:
            if ".." in root.parts:
                raise ValueError
            path = (root if root.is_absolute() else STARTUP_WORKSPACE / root).resolve(strict=True)
            if path == Path(path.anchor) or not stat.S_ISDIR(_file_mode(path)):
                raise ValueError
            canonical.append(path)
        return tuple(canonical)
    except (OSError, RuntimeError, ValueError):
        raise ValueError("Invalid repository roots") from None


class PathPolicyError(ValueError):
    """A path rejection that never includes the submitted path or OS error."""

    def __init__(self) -> None:
        super().__init__("Repository path is not permitted")


@dataclass(frozen=True)
class PathPolicy:
    allowed_roots: tuple[Path, ...]
    sensitive_patterns: tuple[str, ...] = ()

    @classmethod
    def current(cls) -> "PathPolicy":
        # Local import keeps Settings construction independent of runtime publication.
        from orchestrai.config.settings import get_settings

        policy = get_settings().policy
        return cls(policy.allowed_roots, policy.sensitive_path_patterns)

    def repository(self, value: str | Path) -> Path:
        """Authorize an existing directory, relative to the startup workspace."""
        return self._authorize(value, STARTUP_WORKSPACE, "directory")

    @contextmanager
    def command_cwd(self, value: str | Path) -> Iterator[tuple[str, int]]:
        """Keep the authorized directory pinned until the child has spawned."""
        path = self.repository(value)
        with ExitStack() as stack:
            try:
                descriptor = stack.enter_context(_open_directory(path))
                pinned = os.fstat(descriptor)
                for prefix in ("/proc/self/fd", "/dev/fd"):
                    reference = f"{prefix}/{descriptor}"
                    try:
                        if os.path.samestat(pinned, os.stat(reference)):
                            break
                    except OSError:
                        continue
                else:
                    raise PathPolicyError
            except OSError:
                raise PathPolicyError from None
            # Do not translate subprocess errors raised through the caller's yield.
            yield reference, descriptor

    def file(self, value: str | Path, repository: Path, *, allow_missing: bool = False) -> Path:
        """Authorize a file within the repository, including a proposed new file."""
        return self._authorize(value, repository, "file", repository, allow_missing)

    def has_file(self, value: str | Path, repository: Path) -> bool:
        """Detect only authorized regular files; skip denied or absent candidates."""
        try:
            self.file(value, repository)
        except PathPolicyError:
            return False
        return True

    def task_paths(
        self, repo_root: str | None, targets: list[str] | None,
    ) -> tuple[Path | None, tuple[Path, ...]]:
        """Admit task inputs without inferring a repository for a repo-free task."""
        if targets is not None and not isinstance(targets, list):
            raise PathPolicyError
        if repo_root is None:
            if targets:
                raise PathPolicyError
            return None, ()
        repository = self.repository(repo_root)
        return repository, tuple(
            self.file(target, repository, allow_missing=True) for target in (targets or [])
        )

    def iter_files(self, repository: Path) -> Iterator[Path]:
        """List only authorized entries, pruning denied directories before descent."""
        pending = [self.repository(repository)]
        visited: set[Path] = set()
        while pending:
            directory = self._authorize(pending.pop(), repository, "directory", repository)
            if directory in visited:
                continue
            visited.add(directory)
            try:
                with _open_directory(directory) as descriptor:
                    for name in os.listdir(descriptor):
                        entry = directory / name
                        try:
                            resolved = self._authorize(entry, repository, "any", repository)
                        except PathPolicyError:
                            continue
                        parent = descriptor if resolved.parent == directory else None
                        mode = _file_mode(resolved, parent=parent)
                        if stat.S_ISDIR(mode):
                            pending.append(resolved)
                        elif stat.S_ISREG(mode):
                            yield entry
            except OSError:
                raise PathPolicyError from None

    def read_text(
        self, value: str | Path, repository: Path, *,
        errors: Literal["strict", "replace"] = "replace",
    ) -> str:
        """Reauthorize immediately before reading an existing file."""
        path = self.file(value, repository)
        try:
            with _open_directory(path.parent) as parent:
                descriptor = os.open(
                    path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent,
                )
                try:
                    stream = os.fdopen(descriptor, encoding="utf-8", errors=errors)
                except BaseException:
                    try:  # noqa: SIM105 - Avoid allocating a cleanup context after MemoryError.
                        os.close(descriptor)
                    except OSError:
                        pass  # Preserve the original fdopen failure, including cancellation.
                    raise
                with stream:
                    if not stat.S_ISREG(os.fstat(stream.fileno()).st_mode):
                        raise PathPolicyError
                    return stream.read()
        except (OSError, UnicodeError):
            raise PathPolicyError from None

    def read_bytes(self, value: str | Path, repository: Path, *, max_bytes: int) -> bytes:
        """Read bounded content from an authorized descriptor without decoding."""
        if max_bytes < 0:
            raise PathPolicyError
        path = self.file(value, repository)
        try:
            with _open_directory(path.parent) as parent:
                descriptor = os.open(
                    path.name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent,
                )
                try:
                    stream = os.fdopen(descriptor, "rb")
                except BaseException:
                    try:  # noqa: SIM105 - Avoid allocating a cleanup context after MemoryError.
                        os.close(descriptor)
                    except OSError:
                        pass  # Preserve the original fdopen failure, including cancellation.
                    raise
                with stream:
                    metadata = os.fstat(stream.fileno())
                    if not stat.S_ISREG(metadata.st_mode) or metadata.st_size > max_bytes:
                        raise PathPolicyError
                    content = stream.read(max_bytes + 1)
                    if len(content) > max_bytes:
                        raise PathPolicyError
                    return content
        except OSError:
            raise PathPolicyError from None

    def _authorize(
        self, value: str | Path, base: Path, kind: Literal["directory", "file", "any"],
        repository: Path | None = None, allow_missing: bool = False,
    ) -> Path:
        try:
            if not isinstance(value, (str, Path)) or not value:
                raise PathPolicyError
            path = Path(value)
            if ".." in path.parts:
                raise PathPolicyError
            path = path if path.is_absolute() else base / path
            self._deny_sensitive(path)
            missing: list[str] = []
            if allow_missing:
                # Locate an existing ancestor without treating broken symlinks as
                # new files. Resolve that ancestor strictly before adding new names.
                while True:
                    try:
                        path.lstat()
                        break
                    except FileNotFoundError:
                        missing.append(path.name)
                        path = path.parent
            path = path.resolve(strict=True)
            self._deny_sensitive(path)
            if not any(path == root or root in path.parents for root in self.allowed_roots):
                raise PathPolicyError
            if repository is not None and path != repository and repository not in path.parents:
                raise PathPolicyError
            if missing:
                if not stat.S_ISDIR(_file_mode(path)):
                    raise PathPolicyError
                path = path.joinpath(*reversed(missing))
                self._deny_sensitive(path)
                return path
            if (kind == "directory" and not stat.S_ISDIR(_file_mode(path))) or (
                kind == "file" and not stat.S_ISREG(_file_mode(path))
            ):
                raise PathPolicyError
            return path
        except (OSError, RuntimeError, ValueError):
            raise PathPolicyError from None

    def _deny_sensitive(self, path: Path) -> None:
        # Normalize only the deny matcher, never the filesystem path or containment checks.
        matching_path = Path(*(_sensitive_text(part) for part in path.parts))
        if any(tree == matching_path or tree in matching_path.parents for tree in _SYSTEM_TREES):
            raise PathPolicyError
        if any(part in _SENSITIVE_NAMES for part in matching_path.parts):
            raise PathPolicyError
        for ancestor in (matching_path, *matching_path.parents):
            for pattern in (*_MINIMUM_PATTERNS, *self.sensitive_patterns):
                pattern = _sensitive_text(pattern).rstrip("/")
                if pattern and (
                    fnmatchcase(ancestor.name, pattern)
                    or ancestor.match(pattern)
                ):
                    raise PathPolicyError


def _file_mode(path: Path, *, parent: int | None = None) -> int:
    """Classify a canonical leaf without following a replacement symlink."""
    if parent is None:
        with _open_directory(path.parent) as descriptor:
            return _file_mode(path, parent=descriptor)
    return os.stat(path.name, dir_fd=parent, follow_symlinks=False).st_mode


@contextmanager
def _open_directory(path: Path) -> Iterator[int]:
    """Pin each canonical component without following a replacement symlink.

    POSIX descriptor-relative access is required; unsupported platforms fail
    closed. Authorization still runs first, before enumeration or content I/O.
    """
    if os.open not in os.supports_dir_fd or not hasattr(os, "O_NOFOLLOW"):
        raise PathPolicyError
    flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
    descriptor = os.open(path.anchor, flags)
    try:
        for component in path.parts[1:]:
            child = os.open(component, flags, dir_fd=descriptor)
            os.close(descriptor)
            descriptor = child
        yield descriptor
    finally:
        os.close(descriptor)
