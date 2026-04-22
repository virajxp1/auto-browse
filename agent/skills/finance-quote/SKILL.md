---
name: finance-quote
description: Extract visible quote information from finance pages. Use when the user wants the current price of a stock or asset from a finance website.
allowed-tools: read_file
---

# Finance Quote

## When to Use
- The page is a quote page or finance summary page.
- The user wants the current price of a stock, ETF, or asset.

## How to Work
1. Look for the visible headline quote module first.
2. Prefer evidence near quote terms such as `previous close`, `open`, `market cap`, or exchange/ticker context.
3. If the current page already shows a clear headline quote, extract it directly.
4. Verify that the visible ticker/company matches the requested asset before returning a price.

## What to Avoid
- Do not confuse chart axes, percent change, or secondary metrics with the headline price.
- Do not return a price if the page does not clearly identify the asset.
