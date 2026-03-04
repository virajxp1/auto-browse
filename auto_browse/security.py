from __future__ import annotations

import asyncio
import ipaddress
import logging
import os
import time
import tomllib
from collections import deque
from dataclasses import dataclass
from pathlib import Path

from fastapi import Request
from fastapi.responses import JSONResponse
from starlette.types import ASGIApp, Message

DEFAULT_API_TOKEN_HEADER = "x-api-token"
DEFAULT_RATE_LIMIT_MAX_REQUESTS = 30
DEFAULT_RATE_LIMIT_WINDOW_SECONDS = 60
DEFAULT_MAX_CONCURRENT_REQUESTS_PER_IP = 4
DEFAULT_MAX_REQUEST_BODY_BYTES = 64 * 1024
DEFAULT_SECURITY_CONFIG_PATH = "config/security.toml"
DEFAULT_TRUST_X_FORWARDED_FOR = False
DEFAULT_TRUSTED_PROXY_CIDRS: tuple[str, ...] = ()

logger = logging.getLogger("uvicorn.error")


def _parse_int_value(raw_value: int | str, *, name: str, minimum: int) -> int:
    if isinstance(raw_value, bool):
        raise ValueError(f"{name} must be an integer value")
    if isinstance(raw_value, int):
        value = raw_value
    elif isinstance(raw_value, str):
        trimmed = raw_value.strip()
        if not trimmed:
            raise ValueError(f"{name} cannot be empty")
        try:
            value = int(trimmed)
        except ValueError as exc:
            raise ValueError(f"{name} must be an integer value") from exc
    else:
        raise ValueError(f"{name} must be an integer value")

    if value < minimum:
        raise ValueError(f"{name} must be >= {minimum}")
    return value


def _parse_bool_value(raw_value: bool | str, *, name: str) -> bool:
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        lowered = raw_value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    raise ValueError(f"{name} must be a boolean value")


def _parse_proxy_cidrs_value(raw_value: list[object] | str, *, name: str) -> tuple[str, ...]:
    if isinstance(raw_value, str):
        parts = [part.strip() for part in raw_value.split(",")]
        cidr_items = [part for part in parts if part]
    elif isinstance(raw_value, list):
        cidr_items = []
        for item in raw_value:
            if not isinstance(item, str):
                raise ValueError(f"{name} must contain string CIDR entries")
            trimmed = item.strip()
            if trimmed:
                cidr_items.append(trimmed)
    else:
        raise ValueError(f"{name} must be a list of CIDR strings")

    validated: list[str] = []
    for cidr in cidr_items:
        try:
            ipaddress.ip_network(cidr, strict=False)
        except ValueError as exc:
            raise ValueError(f"{name} contains invalid CIDR: {cidr}") from exc
        validated.append(cidr)

    return tuple(validated)


def _load_env_file_if_present(path: Path = Path(".env")) -> None:
    if not path.is_file():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if not key:
            continue
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]

        os.environ.setdefault(key, value)


def _read_security_config(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}

    data = tomllib.loads(path.read_text(encoding="utf-8"))
    security_section = data.get("security", {})
    if not isinstance(security_section, dict):
        raise ValueError("security config file must contain a [security] table")
    return security_section


def _resolve_security_config_path() -> Path:
    configured_path = os.getenv("AUTO_BROWSE_SECURITY_CONFIG_PATH", DEFAULT_SECURITY_CONFIG_PATH).strip()
    if not configured_path:
        raise ValueError("AUTO_BROWSE_SECURITY_CONFIG_PATH cannot be empty")
    return Path(configured_path)


def _string_setting(
    *,
    env_name: str,
    config: dict[str, object],
    config_key: str,
    default: str,
) -> str:
    env_value = os.getenv(env_name)
    if env_value is not None and env_value.strip():
        return env_value.strip()

    config_value = config.get(config_key)
    if config_value is None:
        return default
    if not isinstance(config_value, str):
        raise ValueError(f"{config_key} in security config must be a string value")

    trimmed = config_value.strip()
    if not trimmed:
        raise ValueError(f"{config_key} in security config cannot be empty")
    return trimmed


def _bool_setting(
    *,
    env_name: str,
    config: dict[str, object],
    config_key: str,
    default: bool,
) -> bool:
    env_value = os.getenv(env_name)
    if env_value is not None and env_value.strip():
        return _parse_bool_value(env_value, name=env_name)

    config_value = config.get(config_key)
    if config_value is None:
        return default
    if not isinstance(config_value, (bool, str)):
        raise ValueError(f"{config_key} in security config must be a boolean value")

    return _parse_bool_value(config_value, name=f"{config_key} in security config")


def _int_setting(
    *,
    env_name: str,
    config: dict[str, object],
    config_key: str,
    default: int,
    minimum: int,
) -> int:
    env_value = os.getenv(env_name)
    if env_value is not None and env_value.strip():
        return _parse_int_value(env_value, name=env_name, minimum=minimum)

    config_value = config.get(config_key)
    if config_value is None:
        return default

    return _parse_int_value(
        config_value,
        name=f"{config_key} in security config",
        minimum=minimum,
    )


def _proxy_cidrs_setting(
    *,
    env_name: str,
    config: dict[str, object],
    config_key: str,
    default: tuple[str, ...],
) -> tuple[str, ...]:
    env_value = os.getenv(env_name)
    if env_value is not None and env_value.strip():
        return _parse_proxy_cidrs_value(env_value, name=env_name)

    config_value = config.get(config_key)
    if config_value is None:
        return default

    if isinstance(config_value, (str, list)):
        return _parse_proxy_cidrs_value(
            config_value,
            name=f"{config_key} in security config",
        )

    raise ValueError(f"{config_key} in security config must be a list of CIDR strings")


def _get_client_ip(request: Request) -> str:
    if request.client and request.client.host:
        return request.client.host
    return "unknown"


def _resolve_rate_limit_ip(request: Request, settings: "SecuritySettings") -> str:
    client_ip = _get_client_ip(request)
    if not settings.trust_x_forwarded_for:
        return client_ip

    try:
        source_ip = ipaddress.ip_address(client_ip)
    except ValueError:
        return client_ip

    if not settings.trusted_proxy_networks:
        return client_ip

    if not any(source_ip in network for network in settings.trusted_proxy_networks):
        return client_ip

    forwarded_for = request.headers.get("x-forwarded-for", "")
    for candidate in (item.strip() for item in forwarded_for.split(",")):
        if not candidate:
            continue
        try:
            ipaddress.ip_address(candidate)
        except ValueError:
            continue
        return candidate

    return client_ip


@dataclass(frozen=True)
class SecuritySettings:
    api_token: str
    api_token_header: str
    rate_limit_max_requests: int
    rate_limit_window_seconds: int
    max_concurrent_requests_per_ip: int
    max_request_body_bytes: int
    trust_x_forwarded_for: bool = DEFAULT_TRUST_X_FORWARDED_FOR
    trusted_proxy_networks: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = ()

    @classmethod
    def from_env(cls) -> "SecuritySettings":
        _load_env_file_if_present()
        config_path = _resolve_security_config_path()
        config = _read_security_config(config_path)

        api_token = os.getenv("AUTO_BROWSE_API_TOKEN", "").strip()
        if not api_token:
            raise ValueError("Missing required environment variable: AUTO_BROWSE_API_TOKEN")

        header_name = _string_setting(
            env_name="AUTO_BROWSE_API_TOKEN_HEADER",
            config=config,
            config_key="api_token_header",
            default=DEFAULT_API_TOKEN_HEADER,
        ).lower()
        if not header_name:
            raise ValueError("AUTO_BROWSE_API_TOKEN_HEADER cannot be empty")
        trusted_proxy_cidrs = _proxy_cidrs_setting(
            env_name="AUTO_BROWSE_TRUSTED_PROXY_CIDRS",
            config=config,
            config_key="trusted_proxy_cidrs",
            default=DEFAULT_TRUSTED_PROXY_CIDRS,
        )

        return cls(
            api_token=api_token,
            api_token_header=header_name,
            rate_limit_max_requests=_int_setting(
                env_name="AUTO_BROWSE_RATE_LIMIT_MAX_REQUESTS",
                config=config,
                config_key="rate_limit_max_requests",
                default=DEFAULT_RATE_LIMIT_MAX_REQUESTS,
                minimum=1,
            ),
            rate_limit_window_seconds=_int_setting(
                env_name="AUTO_BROWSE_RATE_LIMIT_WINDOW_SECONDS",
                config=config,
                config_key="rate_limit_window_seconds",
                default=DEFAULT_RATE_LIMIT_WINDOW_SECONDS,
                minimum=1,
            ),
            max_concurrent_requests_per_ip=_int_setting(
                env_name="AUTO_BROWSE_MAX_CONCURRENT_REQUESTS_PER_IP",
                config=config,
                config_key="max_concurrent_requests_per_ip",
                default=DEFAULT_MAX_CONCURRENT_REQUESTS_PER_IP,
                minimum=1,
            ),
            max_request_body_bytes=_int_setting(
                env_name="AUTO_BROWSE_MAX_REQUEST_BODY_BYTES",
                config=config,
                config_key="max_request_body_bytes",
                default=DEFAULT_MAX_REQUEST_BODY_BYTES,
                minimum=1,
            ),
            trust_x_forwarded_for=_bool_setting(
                env_name="AUTO_BROWSE_TRUST_X_FORWARDED_FOR",
                config=config,
                config_key="trust_x_forwarded_for",
                default=DEFAULT_TRUST_X_FORWARDED_FOR,
            ),
            trusted_proxy_networks=tuple(
                ipaddress.ip_network(cidr, strict=False) for cidr in trusted_proxy_cidrs
            ),
        )


class ApiSecurityMiddleware:
    def __init__(self, app: ASGIApp, *, settings: SecuritySettings) -> None:
        self.app = app
        self.settings = settings
        self._recent_requests_by_ip: dict[str, deque[float]] = {}
        self._in_flight_requests_by_ip: dict[str, int] = {}
        self._request_lock = asyncio.Lock()

    async def _buffer_request_body(
        self,
        receive,  # type: ignore[no-untyped-def]
    ) -> tuple[list[Message], bool]:
        buffered: list[Message] = []
        total_bytes = 0
        while True:
            message = await receive()
            buffered.append(message)
            if message["type"] != "http.request":
                if message["type"] == "http.disconnect":
                    return buffered, False
                continue

            body_chunk = message.get("body", b"")
            total_bytes += len(body_chunk)
            if total_bytes > self.settings.max_request_body_bytes:
                return buffered, True

            if not message.get("more_body", False):
                return buffered, False

    async def __call__(
        self,
        scope,  # type: ignore[no-untyped-def]
        receive,  # type: ignore[no-untyped-def]
        send,  # type: ignore[no-untyped-def]
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        request = Request(scope, receive=receive)
        request_method = request.method
        request_path = request.url.path
        source_ip = _get_client_ip(request)

        async def _reject(
            *,
            status_code: int,
            detail: str,
            reason: str,
            headers: dict[str, str] | None = None,
            client_ip: str | None = None,
        ) -> None:
            rejected_ip = client_ip or source_ip
            logger.info(
                "[security] rejected method=%s path=%s client_ip=%s status=%s reason=%s detail=%s",
                request_method,
                request_path,
                rejected_ip,
                status_code,
                reason,
                detail,
            )
            response = JSONResponse(status_code=status_code, content={"detail": detail}, headers=headers)
            await response(scope, receive, send)

        provided_token = request.headers.get(self.settings.api_token_header, "")
        if provided_token != self.settings.api_token:
            await _reject(
                status_code=401,
                detail="Missing or invalid API token",
                reason="invalid_api_token",
            )
            return

        content_length_header = request.headers.get("content-length")
        if content_length_header:
            try:
                content_length = int(content_length_header)
            except ValueError:
                await _reject(
                    status_code=400,
                    detail="Invalid Content-Length header",
                    reason="invalid_content_length",
                )
                return

            if content_length > self.settings.max_request_body_bytes:
                await _reject(
                    status_code=413,
                    detail="Request body too large",
                    reason="content_length_too_large",
                )
                return

        client_ip = _resolve_rate_limit_ip(request, self.settings)
        now = time.monotonic()
        window_start = now - self.settings.rate_limit_window_seconds

        async with self._request_lock:
            recent_requests = self._recent_requests_by_ip.setdefault(client_ip, deque())
            while recent_requests and recent_requests[0] < window_start:
                recent_requests.popleft()

            if len(recent_requests) >= self.settings.rate_limit_max_requests:
                await _reject(
                    status_code=429,
                    detail="Rate limit exceeded",
                    reason="rate_limit_exceeded",
                    headers={"Retry-After": str(self.settings.rate_limit_window_seconds)},
                    client_ip=client_ip,
                )
                return

            current_in_flight = self._in_flight_requests_by_ip.get(client_ip, 0)
            if current_in_flight >= self.settings.max_concurrent_requests_per_ip:
                await _reject(
                    status_code=429,
                    detail="Too many concurrent requests",
                    reason="concurrency_limit_exceeded",
                    client_ip=client_ip,
                )
                return

            recent_requests.append(now)
            self._in_flight_requests_by_ip[client_ip] = current_in_flight + 1

        try:
            buffered_body_messages, body_too_large = await self._buffer_request_body(receive)
            if body_too_large:
                await _reject(
                    status_code=413,
                    detail="Request body too large",
                    reason="buffered_body_too_large",
                    client_ip=client_ip,
                )
                return

            message_index = 0

            async def replay_receive() -> Message:
                nonlocal message_index
                if message_index < len(buffered_body_messages):
                    message = buffered_body_messages[message_index]
                    message_index += 1
                    return message
                return {"type": "http.request", "body": b"", "more_body": False}

            await self.app(scope, replay_receive, send)
        finally:
            async with self._request_lock:
                current_in_flight = self._in_flight_requests_by_ip.get(client_ip, 0)
                if current_in_flight <= 1:
                    self._in_flight_requests_by_ip.pop(client_ip, None)
                else:
                    self._in_flight_requests_by_ip[client_ip] = current_in_flight - 1
