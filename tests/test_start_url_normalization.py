from __future__ import annotations

import unittest

from agent.browser import _normalize_start_url


class StartUrlNormalizationTest(unittest.TestCase):
    def test_adds_https_scheme_for_bare_domain(self) -> None:
        self.assertEqual(_normalize_start_url("www.google.com"), "https://www.google.com")

    def test_preserves_existing_scheme(self) -> None:
        self.assertEqual(_normalize_start_url("http://example.com"), "http://example.com")

    def test_handles_protocol_relative_url(self) -> None:
        self.assertEqual(_normalize_start_url("//example.com"), "https://example.com")

    def test_rejects_empty_url(self) -> None:
        with self.assertRaises(ValueError):
            _normalize_start_url("   ")

    def test_rejects_missing_host(self) -> None:
        with self.assertRaises(ValueError):
            _normalize_start_url("https:///path")

    def test_rejects_host_with_whitespace(self) -> None:
        with self.assertRaises(ValueError):
            _normalize_start_url("https://exa mple.com")

    def test_rejects_host_with_missing_name_and_port_only(self) -> None:
        with self.assertRaises(ValueError):
            _normalize_start_url("http://:80")

    def test_rejects_host_with_invalid_label(self) -> None:
        with self.assertRaises(ValueError):
            _normalize_start_url("https://-bad.example.com")

    def test_rejects_mailto_scheme(self) -> None:
        with self.assertRaises(ValueError):
            _normalize_start_url("mailto:test@example.com")

    def test_rejects_non_http_scheme(self) -> None:
        with self.assertRaises(ValueError):
            _normalize_start_url("ftp://example.com/file.txt")

    def test_rejects_non_numeric_port(self) -> None:
        with self.assertRaises(ValueError):
            _normalize_start_url("https://example.com:abc")

    def test_rejects_out_of_range_port(self) -> None:
        with self.assertRaises(ValueError):
            _normalize_start_url("https://example.com:70000")

    def test_preserves_host_port_shorthand(self) -> None:
        self.assertEqual(_normalize_start_url("localhost:8000/path"), "https://localhost:8000/path")
