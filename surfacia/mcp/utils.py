from __future__ import annotations

import contextlib
import importlib
import io
import json
import os
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator

try:
    import numpy as np
except Exception:  # pragma: no cover - optional in some environments
    np = None


class ToolExecutionError(RuntimeError):
    """Raised when a tool cannot complete its requested work."""


@contextlib.contextmanager
def working_directory(path: str | os.PathLike[str]) -> Iterator[Path]:
    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    original = Path.cwd()
    os.chdir(target)
    try:
        yield target
    finally:
        os.chdir(original)


def capture_python_output(func: Callable[..., Any], *args: Any, **kwargs: Any) -> tuple[Any, str, str]:
    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        result = func(*args, **kwargs)
    return result, stdout_buffer.getvalue(), stderr_buffer.getvalue()


def serialize_jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value.resolve())
    if isinstance(value, dict):
        return {str(key): serialize_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [serialize_jsonable(item) for item in value]
    if np is not None:
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    if hasattr(value, "tolist") and callable(getattr(value, "tolist")):
        try:
            return value.tolist()
        except Exception:
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def result_payload(
    tool_name: str,
    summary: str,
    *,
    ok: bool = True,
    working_dir: str | None = None,
    artifacts: dict[str, Any] | None = None,
    metrics: dict[str, Any] | None = None,
    logs: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "ok": ok,
        "tool": tool_name,
        "summary": summary,
    }
    if working_dir is not None:
        payload["working_dir"] = str(Path(working_dir).expanduser().resolve())
    if artifacts:
        payload["artifacts"] = serialize_jsonable(artifacts)
    if metrics:
        payload["metrics"] = serialize_jsonable(metrics)
    if logs:
        payload["logs"] = serialize_jsonable(logs)
    if extra:
        payload.update(serialize_jsonable(extra))
    return payload


def ensure_directory(path: str | os.PathLike[str]) -> Path:
    directory = Path(path).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def command_available(command: str) -> bool:
    return shutil.which(command) is not None


def import_available(module_name: str) -> bool:
    try:
        importlib.import_module(module_name)
        return True
    except Exception:
        return False


def normalize_sample_names(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [item.strip() for item in value.split(",") if item.strip()]
    if isinstance(value, (list, tuple, set)):
        return [str(item) for item in value]
    return [str(value)]


def collect_paths(
    base_dir: str | os.PathLike[str],
    patterns: Iterable[str],
    *,
    recursive: bool = False,
) -> list[Path]:
    root = Path(base_dir).expanduser().resolve()
    results: list[Path] = []
    iterator_name = "rglob" if recursive else "glob"
    for pattern in patterns:
        iterator = getattr(root, iterator_name)
        results.extend(iterator(pattern))
    deduped = sorted({path.resolve() for path in results if path.exists()})
    return deduped


def find_latest_path(
    base_dir: str | os.PathLike[str],
    patterns: Iterable[str],
    *,
    recursive: bool = False,
) -> Path | None:
    matches = collect_paths(base_dir, patterns, recursive=recursive)
    if not matches:
        return None
    return max(matches, key=lambda path: path.stat().st_mtime)


def file_info(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "name": path.name,
        "size": path.stat().st_size,
        "modified_at": path.stat().st_mtime,
    }


def compact_log(text: str, *, limit: int = 12000) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    head = cleaned[: limit - 80].rstrip()
    return f"{head}\n...[truncated {len(cleaned) - len(head)} chars]"


def to_pretty_json(value: Any) -> str:
    return json.dumps(serialize_jsonable(value), indent=2, ensure_ascii=False, sort_keys=True)
