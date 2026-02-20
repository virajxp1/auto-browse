from __future__ import annotations

import unittest

import auto_browse
from auto_browse import OpenRouterClient, run_agent


class PublicApiTest(unittest.TestCase):
    def test_public_exports_are_available(self) -> None:
        self.assertTrue(callable(run_agent))
        self.assertTrue(hasattr(OpenRouterClient, "from_env"))
        self.assertIn("run_agent", auto_browse.__all__)
        self.assertIn("OpenRouterClient", auto_browse.__all__)


if __name__ == "__main__":
    unittest.main()
