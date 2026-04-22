---
name: commerce-page
description: Understand retail product pages and shopping result pages. Use when the task involves finding a product, identifying the right product card, or extracting a visible price from a commerce site.
allowed-tools: read_file
---

# Commerce Page

## When to Use
- The page contains product cards, prices, buy buttons, or shopping filters.
- The user wants the price of a product on a retail site.
- The current page has both global navigation and product links, and you need to choose the right body content.

## How to Work
1. Prefer `region=main` and `region=form` interactables over `region=nav`, `region=header`, or `region=footer`.
2. Distinguish between a product-detail page and a search/results page.
3. Treat price text as valid only if it is visibly associated with the requested product name or model.
4. If the current page already shows a visible product price, prefer extracting or verifying it instead of navigating again.
5. If there are multiple product cards, prefer the one whose label or context most closely matches the requested item.

## What to Avoid
- Do not use header navigation when a body product card already advances the goal.
- Do not assume the first visible price belongs to the right product.
- Do not treat accessory, financing, or trade-in prices as the main product price unless the page makes that explicit.
