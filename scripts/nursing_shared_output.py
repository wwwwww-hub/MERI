"""Small filesystem primitives for nursing workers sharing an output directory.

The directory creation and replacement operations must be atomic on the shared
filesystem. Locks never expire automatically: a slow or disconnected worker may
still own its output. Remove a leftover lock only after its worker has stopped.
"""

from __future__ import annotations

import json
import math
import os
import socket
import tempfile
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def read_json(path: Path) -> dict[str, Any] | None:
    """Read a JSON object, returning None for missing, unreadable, or invalid data."""
    try:
        payload = Path(path).read_text(encoding="utf-8")
        value = json.loads(payload)
    except (OSError, UnicodeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def atomic_replace(temporary: Path, path: Path) -> None:
    """Publish a same-filesystem staging file, tolerating brief Windows sharing conflicts.

    Callers own staging-file creation, flushing, and cleanup. Keep both paths
    on the same filesystem so the successful replacement remains atomic.
    """
    temporary, path = Path(temporary), Path(path)
    deadline = time.monotonic() + 2.0
    while True:
        try:
            os.replace(temporary, path)
            return
        except PermissionError as exc:
            # Concurrent renames or open readers can briefly deny replacement
            # on Windows. Genuine permissions failures still surface promptly.
            if getattr(exc, "winerror", None) not in {5, 32, 33} or time.monotonic() >= deadline:
                raise
            time.sleep(0.01)


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    """Publish complete JSON using a unique, flushed temporary file beside it."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".nso-", suffix=".tmp", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        atomic_replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class SharedDirectoryLock:
    """Claim one shared path with atomic mkdir, without any stale-lock takeover.

    All workers must use the same logical lock path on the shared filesystem.
    Keep the lock for the entire write operation, and do not delete active locks.
    Metadata is diagnostic; timestamps and remote PIDs never justify takeover.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self._token = uuid.uuid4().hex
        self._acquired = False

    @property
    def owner_path(self) -> Path:
        return self.path / "owner.json"

    def acquire(self) -> bool:
        """Acquire immediately, returning False if another owner has this path."""
        if self._acquired:
            raise RuntimeError(f"This lock object already owns {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        try:
            self.path.mkdir()
        except FileExistsError:
            return False
        try:
            owner = {
                "hostname": socket.gethostname(),
                "pid": os.getpid(),
                "token": self._token,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "created_unix": time.time(),
            }
            atomic_json(self.owner_path, owner)
        except BaseException:
            # The mkdir belongs to this acquire. Never recursively remove it:
            # unexpected files must remain visible for manual investigation.
            try:
                self.owner_path.unlink(missing_ok=True)
                self.path.rmdir()
            except OSError:
                pass
            raise
        self._acquired = True
        return True

    def release(self) -> bool:
        """Release only our matching owner token; return False if ownership changed."""
        if not self._acquired:
            return False
        owner = read_json(self.owner_path)
        if owner is None or owner.get("token") != self._token:
            self._acquired = False
            return False
        self.owner_path.unlink()
        self._acquired = False
        self.path.rmdir()
        return True

    def describe_owner(self) -> str:
        """Return readable diagnostics even during metadata publication or corruption."""
        owner = read_json(self.owner_path)
        if owner is None:
            return f"{self.path}: owner metadata is missing or unreadable"
        return (
            f"{self.path}: hostname={owner.get('hostname', '?')} "
            f"pid={owner.get('pid', '?')} created_at={owner.get('created_at', '?')} "
            f"token={owner.get('token', '?')}"
        )

    def wait_acquire(self, poll_sec: float = 1.0, timeout_sec: float | None = None) -> bool:
        """Wait using a monotonic deadline, retaining remote locks regardless of age."""
        if not math.isfinite(poll_sec) or poll_sec <= 0:
            raise ValueError("poll_sec must be finite and positive")
        if timeout_sec is not None and (not math.isfinite(timeout_sec) or timeout_sec < 0):
            raise ValueError("timeout_sec must be finite and nonnegative, or None")
        deadline = None if timeout_sec is None else time.monotonic() + timeout_sec
        while True:
            if self.acquire():
                return True
            delay = poll_sec
            if deadline is not None:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError(f"Timed out waiting for shared lock; {self.describe_owner()}")
                delay = min(delay, remaining)
            time.sleep(delay)

    def __enter__(self) -> SharedDirectoryLock:
        self.wait_acquire()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.release()
