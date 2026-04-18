---
name: search-results
description: Work with ranked result pages and listings. Use when the task involves searching, opening the best or first result, or extracting data from a results page rather than global site navigation.
allowed-tools: read_file
---

# Search Results

## When to Use
- The page shows many links or cards and the user wants one specific result.
- The task says "click the first result" or "find the best matching result."
- The page mixes result links with unrelated site navigation.

## How to Work
1. Treat result pages as ranked lists and prefer links in `region=main` over site chrome.
2. Use `context=` text to tell result cards apart when link labels are short or repetitive.
3. If the task asks for the first result, choose the first relevant body result, not the first link in DOM order.
4. After opening a result, verify that the destination page title or body text matches the expected item.

## What to Avoid
- Do not click logo links, category links, or top navigation unless the body lacks results.
- Do not assume a result is relevant from selector shape alone.
