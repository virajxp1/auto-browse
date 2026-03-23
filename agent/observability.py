from __future__ import annotations

import configparser
import logging
import os
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator

logger = logging.getLogger(__name__)

try:
    import braintrust
except Exception:  # pragma: no cover - dependency import guard
    braintrust = None


class _NoopSpan:
    def log(self, **event: Any) -> None:
        return None

    def export(self) -> str:
        return ""


_NOOP_SPAN = _NoopSpan()

_FALSE_VALUES = {"0", "false", "no", "off"}
_DEFAULT_CONFIG_PATH = Path("config/config.ini")


def _env(name: str) -> str:
    return os.getenv(name, "").strip()


@lru_cache(maxsize=1)
def _read_config_project_id(path: Path = _DEFAULT_CONFIG_PATH) -> str | None:
    if not path.is_file():
        return None

    parser = configparser.ConfigParser()
    try:
        parser.read(path, encoding="utf-8")
    except configparser.Error:
        return None

    if "braintrust" not in parser:
        return None
    return parser.get("braintrust", "project_id", fallback="").strip() or None


def _should_enable() -> bool:
    override = _env("AUTO_BROWSE_BRAINTRUST_ENABLED").lower()
    if override in _FALSE_VALUES:
        return False
    return bool(_env("BRAINTRUST_API_KEY") and _read_config_project_id())


@lru_cache(maxsize=1)
def _ensure_initialized() -> bool:
    if braintrust is None:
        return False
    if not _should_enable():
        return False

    project_id = _read_config_project_id()
    if not project_id:
        return False

    init_kwargs: dict[str, Any] = {
        "api_key": _env("BRAINTRUST_API_KEY"),
        "project_id": project_id,
        "set_current": True,
    }

    org_name = _env("BRAINTRUST_ORG_NAME")
    if org_name:
        init_kwargs["org_name"] = org_name

    app_url = _env("BRAINTRUST_APP_URL")
    if app_url:
        init_kwargs["app_url"] = app_url

    try:
        braintrust.init_logger(**init_kwargs)
        return True
    except Exception:
        logger.exception("Braintrust initialization failed; instrumentation disabled.")
        return False


def is_enabled() -> bool:
    return bool(_ensure_initialized())


@contextmanager
def start_span(
    name: str,
    *,
    span_type: str | None = None,
    parent: str | None = None,
    **event: Any,
) -> Iterator[Any]:
    if not is_enabled() or braintrust is None:
        yield _NOOP_SPAN
        return

    kwargs = dict(event)
    if span_type is not None:
        kwargs["type"] = span_type
    if parent:
        kwargs["parent"] = parent

    try:
        span_context = braintrust.start_span(name=name, set_current=True, **kwargs)
    except Exception:
        logger.exception("Braintrust start_span failed for '%s'; continuing without span.", name)
        yield _NOOP_SPAN
        return

    with span_context as span:
        yield span


def span_log(span: Any, **event: Any) -> None:
    if span is _NOOP_SPAN:
        return
    try:
        span.log(**event)
    except Exception:
        logger.debug("Braintrust span.log failed", exc_info=True)


def flush() -> None:
    if not is_enabled() or braintrust is None:
        return
    try:
        braintrust.flush()
    except Exception:
        logger.debug("Braintrust flush failed", exc_info=True)


def export_current_span_parent() -> str | None:
    if not is_enabled() or braintrust is None:
        return None
    try:
        return braintrust.current_span().export() or None
    except Exception:
        return None
