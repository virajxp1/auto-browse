---
name: travel-search
description: Work through travel search forms and result pages. Use when the task involves routes, dates, passengers, hotels, or fare/result filtering.
allowed-tools: read_file
---

# Travel Search

## When to Use
- The task involves flights, hotels, dates, travellers, or fare/result filters.
- The page contains destination inputs, date controls, traveller selectors, or booking results.

## How to Work
1. Identify whether the page is still in form-entry mode or already in results mode.
2. Map each requested trip field to a visible control before acting.
3. Expect autocompletes, date pickers, and passenger selectors to require multiple actions.
4. After submission, wait for results and verify that the visible route/date context matches the request.

## What to Avoid
- Do not claim success while still on the search form.
- Do not treat ads, flexible-date suggestions, or unrelated destinations as the requested result.
