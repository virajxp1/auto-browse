from __future__ import annotations

import ipaddress
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from auto_browse.security import SecuritySettings, _resolve_rate_limit_ip


class SecuritySettingsTest(unittest.TestCase):
    def test_from_env_requires_api_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "security.toml").write_text("[security]\n", encoding="utf-8")

            old_cwd = os.getcwd()
            try:
                os.chdir(root)
                with patch.dict(os.environ, {}, clear=True):
                    with self.assertRaisesRegex(
                        ValueError,
                        "Missing required environment variable: AUTO_BROWSE_API_TOKEN",
                    ):
                        SecuritySettings.from_env()
            finally:
                os.chdir(old_cwd)

    def test_from_env_reads_non_secret_settings_from_config_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".env").write_text("AUTO_BROWSE_API_TOKEN=test-token\n", encoding="utf-8")
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "security.toml").write_text(
                "[security]\n"
                'api_token_header = "x-auth-token"\n'
                "rate_limit_max_requests = 10\n"
                "rate_limit_window_seconds = 15\n"
                "max_concurrent_requests_per_ip = 3\n"
                "max_request_body_bytes = 2048\n"
                "trust_x_forwarded_for = true\n"
                'trusted_proxy_cidrs = ["127.0.0.1/32"]\n',
                encoding="utf-8",
            )

            old_cwd = os.getcwd()
            try:
                os.chdir(root)
                with patch.dict(os.environ, {}, clear=True):
                    settings = SecuritySettings.from_env()
            finally:
                os.chdir(old_cwd)

        self.assertEqual(settings.api_token, "test-token")
        self.assertEqual(settings.api_token_header, "x-auth-token")
        self.assertEqual(settings.rate_limit_max_requests, 10)
        self.assertEqual(settings.rate_limit_window_seconds, 15)
        self.assertEqual(settings.max_concurrent_requests_per_ip, 3)
        self.assertEqual(settings.max_request_body_bytes, 2048)
        self.assertTrue(settings.trust_x_forwarded_for)
        self.assertEqual(len(settings.trusted_proxy_networks), 1)

    def test_from_env_prefers_env_over_config_for_non_secret_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / ".env").write_text("AUTO_BROWSE_API_TOKEN=test-token\n", encoding="utf-8")
            (root / "config").mkdir(parents=True, exist_ok=True)
            (root / "config" / "security.toml").write_text(
                "[security]\n"
                "rate_limit_max_requests = 10\n",
                encoding="utf-8",
            )

            old_cwd = os.getcwd()
            try:
                os.chdir(root)
                with patch.dict(
                    os.environ,
                    {"AUTO_BROWSE_RATE_LIMIT_MAX_REQUESTS": "25"},
                    clear=True,
                ):
                    settings = SecuritySettings.from_env()
            finally:
                os.chdir(old_cwd)

        self.assertEqual(settings.rate_limit_max_requests, 25)

    def test_resolve_rate_limit_ip_ignores_forwarded_for_from_untrusted_source(self) -> None:
        settings = SecuritySettings(
            api_token="test-token",
            api_token_header="x-api-token",
            rate_limit_max_requests=10,
            rate_limit_window_seconds=60,
            max_concurrent_requests_per_ip=5,
            max_request_body_bytes=2048,
            trust_x_forwarded_for=True,
            trusted_proxy_networks=(),
        )
        request = SimpleNamespace(
            headers={"x-forwarded-for": "198.51.100.44"},
            client=SimpleNamespace(host="203.0.113.9"),
        )

        self.assertEqual(_resolve_rate_limit_ip(request, settings), "203.0.113.9")

    def test_resolve_rate_limit_ip_uses_forwarded_for_from_trusted_proxy(self) -> None:
        settings = SecuritySettings(
            api_token="test-token",
            api_token_header="x-api-token",
            rate_limit_max_requests=10,
            rate_limit_window_seconds=60,
            max_concurrent_requests_per_ip=5,
            max_request_body_bytes=2048,
            trust_x_forwarded_for=True,
            trusted_proxy_networks=(ipaddress.ip_network("203.0.113.0/24", strict=False),),
        )
        request = SimpleNamespace(
            headers={"x-forwarded-for": "198.51.100.44"},
            client=SimpleNamespace(host="203.0.113.9"),
        )

        self.assertEqual(_resolve_rate_limit_ip(request, settings), "198.51.100.44")
