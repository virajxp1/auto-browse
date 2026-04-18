#!/usr/bin/env python3
"""Seed diverse eval cases into the Braintrust dataset.

Anti-overfitting protocol: these cases cover non-Wikipedia sites,
different interaction patterns, and varied extraction types.
"""
from __future__ import annotations

import configparser
import os
from pathlib import Path


def _load_env() -> None:
    env_path = Path(".env")
    if not env_path.is_file():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        os.environ.setdefault(key, value)


DIVERSE_CASES = [
    # --- Category: Direct navigation + title extraction (non-Wikipedia) ---
    {
        "id": "python_org_homepage_title",
        "start_url": "https://www.python.org",
        "target_prompt": "Navigate to python.org and report the page title.",
        "max_steps": 5,
        "max_runtime_seconds": 60,
        "expected_contains": ["Python"],
    },
    {
        "id": "mdn_css_grid_title",
        "start_url": "https://developer.mozilla.org/en-US/docs/Web/CSS/CSS_grid_layout",
        "target_prompt": "Navigate to the MDN CSS Grid Layout documentation page and report the article title.",
        "max_steps": 5,
        "max_runtime_seconds": 60,
        "expected_contains": ["CSS grid layout"],
    },
    {
        "id": "w3_html_spec_title",
        "start_url": "https://www.w3.org/TR/html/",
        "target_prompt": "Navigate to the W3C HTML specification page and report the document title.",
        "max_steps": 5,
        "max_runtime_seconds": 60,
        "expected_contains": ["HTML"],
    },
    # --- Category: Google search → non-Wikipedia destination ---
    {
        "id": "search_python_org_downloads",
        "start_url": "https://www.google.com",
        "target_prompt": "Use Google to search for 'python download' and find the official Python downloads page on python.org. Report the page title.",
        "max_steps": 8,
        "max_runtime_seconds": 90,
        "min_trace_steps": 2,
        "required_actions": ["type_and_submit"],
        "expected_contains": ["Python", "Download"],
    },
    {
        "id": "search_rfc_2616_title",
        "start_url": "https://www.google.com",
        "target_prompt": "Use Google to search for 'RFC 2616' and open the IETF RFC page. Report the document title.",
        "max_steps": 8,
        "max_runtime_seconds": 90,
        "min_trace_steps": 2,
        "required_actions": ["type_and_submit"],
        "expected_contains": ["RFC 2616"],
    },
    {
        "id": "search_mdn_javascript_title",
        "start_url": "https://www.google.com",
        "target_prompt": "Use Google to search for 'MDN JavaScript guide' and open the Mozilla Developer Network JavaScript guide. Report the page title.",
        "max_steps": 8,
        "max_runtime_seconds": 90,
        "min_trace_steps": 2,
        "required_actions": ["type_and_submit"],
        "expected_contains": ["JavaScript"],
    },
    # --- Category: Schema extraction from non-Wikipedia pages ---
    {
        "id": "schema_python_org_version",
        "start_url": "https://www.python.org",
        "target_prompt": "Navigate to python.org and extract the latest stable Python version number shown on the homepage.",
        "max_steps": 5,
        "max_runtime_seconds": 60,
        "extraction_schema": {
            "latest_version": "The latest stable Python version number (e.g. 3.x.x)"
        },
        "expected_contains": ["3."],
    },
    {
        "id": "schema_httpbin_ip",
        "start_url": "https://httpbin.org/ip",
        "target_prompt": "Navigate to httpbin.org/ip and extract the origin IP address shown.",
        "max_steps": 5,
        "max_runtime_seconds": 60,
        "extraction_schema": {
            "origin_ip": "The IP address shown in the JSON response"
        },
    },
    # --- Category: Multi-step click navigation ---
    {
        "id": "python_org_click_to_docs",
        "start_url": "https://www.python.org",
        "target_prompt": "Go to python.org, click on the 'Documentation' link in the top navigation, and report the page title of the documentation page.",
        "max_steps": 6,
        "max_runtime_seconds": 60,
        "min_trace_steps": 2,
        "required_actions": ["click"],
        "expected_contains": ["Documentation"],
    },
    {
        "id": "hacker_news_top_story",
        "start_url": "https://news.ycombinator.com",
        "target_prompt": "Navigate to Hacker News and report the title of the first (top) story on the front page.",
        "max_steps": 5,
        "max_runtime_seconds": 60,
    },
    # --- Category: Direct URL navigation + content extraction ---
    {
        "id": "jsonplaceholder_first_post",
        "start_url": "https://jsonplaceholder.typicode.com/posts/1",
        "target_prompt": "Navigate to jsonplaceholder.typicode.com/posts/1 and report the title of the post.",
        "max_steps": 5,
        "max_runtime_seconds": 60,
        "expected_contains": ["sunt aut facere"],
    },
    {
        "id": "github_readme_heading",
        "start_url": "https://github.com/python/cpython",
        "target_prompt": "Navigate to the CPython repository on GitHub and report the main heading from the README.",
        "max_steps": 6,
        "max_runtime_seconds": 90,
        "expected_contains": ["CPython"],
    },
    # --- Category: Search + multi-step on specific sites ---
    {
        "id": "search_github_flask_stars",
        "start_url": "https://www.google.com",
        "target_prompt": "Use Google to search for 'Flask GitHub repository' and navigate to the Flask repository on GitHub. Report the repository description.",
        "max_steps": 8,
        "max_runtime_seconds": 90,
        "min_trace_steps": 2,
        "required_actions": ["type_and_submit"],
        "expected_contains": ["Flask"],
    },
]


def main() -> None:
    _load_env()

    config_path = Path("config/config.ini")
    parser = configparser.ConfigParser()
    parser.read(config_path)

    project_name = parser.get("braintrust", "project_name")
    project_id = parser.get("braintrust", "project_id")
    dataset_name = parser.get("braintrust", "dataset_name", fallback="auto-browse-eval-cases")

    from braintrust import init_dataset

    dataset = init_dataset(
        project=project_name,
        project_id=project_id,
        name=dataset_name,
    )

    # Fetch existing IDs to avoid duplicates
    existing_ids = set()
    for record in dataset.fetch():
        if isinstance(record, dict):
            existing_ids.add(record.get("id", ""))

    added = 0
    skipped = 0
    for case in DIVERSE_CASES:
        case_id = case["id"]
        if case_id in existing_ids:
            print(f"  SKIP (exists): {case_id}")
            skipped += 1
            continue

        input_payload = dict(case)
        expected = {}

        if case.get("expected_contains"):
            expected["expected_contains"] = case["expected_contains"]
            expected["reference_text"] = ", ".join(case["expected_contains"])
        if case.get("min_trace_steps"):
            expected["min_trace_steps"] = case["min_trace_steps"]
        if case.get("required_actions"):
            expected["required_actions"] = case["required_actions"]

        metadata = {
            "task_id": case_id,
            "source": "scripts/seed_diverse_cases.py",
            "is_complex": bool(case.get("required_actions") or (case.get("min_trace_steps") or 0) > 1),
        }

        tags = ["diverse"]
        if metadata["is_complex"]:
            tags.append("complex")

        dataset.insert(
            id=case_id,
            input=input_payload,
            expected=expected or None,
            metadata=metadata,
            tags=tags,
        )
        print(f"  ADDED: {case_id}")
        added += 1

    # Flush
    dataset.flush()

    print(f"\nDone: {added} added, {skipped} skipped (already existed)")
    print(f"Total cases in dataset: {len(existing_ids) + added}")


if __name__ == "__main__":
    main()
