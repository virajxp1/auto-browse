from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from agent.openrouter_client import OpenRouterClient


class EnvLoadingTest(unittest.TestCase):
    def test_from_env_loads_dotenv_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            env_path = Path(tmpdir) / ".env"
            env_path.write_text(
                "OPENROUTER_API_KEY=test_key\nOPENROUTER_MODEL=test_model\n",
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
