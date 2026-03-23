from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.openrouter_client import OpenRouterClient


class EnvLoadingTest(unittest.TestCase):
    def test_from_env_loads_dotenv_file_and_config_ini(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "OPENROUTER_API_KEY=test_key\n",
                encoding="utf-8",
            )
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "config.ini").write_text(
                "[openrouter]\nmodel=test_model\n",
                encoding="utf-8",
            )

            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch.dict(os.environ, {}, clear=True):
                    client = OpenRouterClient.from_env()
            finally:
                os.chdir(old_cwd)

        self.assertEqual(client.api_key, "test_key")
        self.assertEqual(client.model_name, "test_model")

    def test_from_env_accepts_open_router_api_key_alias(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "OPEN_ROUTER_API_KEY=test_key_alias\n",
                encoding="utf-8",
            )
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "config.ini").write_text(
                "[openrouter]\nmodel=test_model\n",
                encoding="utf-8",
            )

            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch.dict(os.environ, {}, clear=True):
                    client = OpenRouterClient.from_env()
            finally:
                os.chdir(old_cwd)

        self.assertEqual(client.api_key, "test_key_alias")
        self.assertEqual(client.model_name, "test_model")

    def test_from_env_respects_openrouter_config_path_override_env_var(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                (
                    "OPENROUTER_API_KEY=test_key\n"
                    "AUTO_BROWSE_OPENROUTER_CONFIG_PATH=settings/custom.ini\n"
                ),
                encoding="utf-8",
            )
            config_dir = Path(tmpdir) / "config"
            config_dir.mkdir(parents=True, exist_ok=True)
            (config_dir / "config.ini").write_text(
                "[openrouter]\nmodel=from-default-config\n",
                encoding="utf-8",
            )
            settings_dir = Path(tmpdir) / "settings"
            settings_dir.mkdir(parents=True, exist_ok=True)
            (settings_dir / "custom.ini").write_text(
                "[openrouter]\nmodel=openai/gpt-4o-mini\n",
                encoding="utf-8",
            )

            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch.dict(os.environ, {}, clear=True):
                    client = OpenRouterClient.from_env()
            finally:
                os.chdir(old_cwd)

        self.assertEqual(client.api_key, "test_key")
        self.assertEqual(client.model_name, "openai/gpt-4o-mini")

    def test_from_env_falls_back_to_packaged_default_when_config_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "OPENROUTER_API_KEY=test_key\n",
                encoding="utf-8",
            )

            old_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)
                with patch.dict(os.environ, {}, clear=True):
                    client = OpenRouterClient.from_env()
            finally:
                os.chdir(old_cwd)

        self.assertEqual(client.api_key, "test_key")
        self.assertEqual(client.model_name, "gpt-oss-20b")
