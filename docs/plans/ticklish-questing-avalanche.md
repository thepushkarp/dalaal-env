# DalaalEnv v2 — Redesign Plan

## Context

v1 is complete and passes all validation (structural 8/8, runtime 6/6). But Codex (GPT-5.4 on xhigh) identified five fundamental design issues that would hurt scoring:

1. **`run_check` is an oracle** — agent directly queries the same grading functions used by `compute_ground_truth()`. Optimal policy is trivially "run all 10 checks, submit all failures." No genuine RL exploration.
2. **No discoverable DOM** — `inspect_element` requires a selector the agent must guess. No `list_elements` action. `page_summary` only shows tag frequency counts.
3. **page_summary disappears** — bug at `environment.py:256` hardcodes `page_summary=""` for all steps after reset.
4. **Dataset too small** — 5 pages, no fully-responsive (zero-issue) pages, no hard adversarial pages.
5. **Blank README** — `# dalaal-env` only. Critical for Round 1 LLM scoring.

**Deadline**: April 8 (2 days). This plan prioritizes changes by Round 1 impact.

---

## v2 Action Space Redesign

### Remove `run_check` — Replace with Evidence-Based Exploration

The agent should see the PAGE, not the GRADING FUNCTION's opinion of the page.

**v1 actions (oracle):**
```
set_viewport, inspect_element, run_check, submit_report
```

**v2 actions (evidence-based):**
```
set_viewport, list_elements, inspect_element, get_page_info, submit_report
```

### New Action: `list_elements`

Returns all elements at the current viewport with basic metadata — enough to decide what to inspect, but NOT enough to determine responsiveness issues.

```python
# Action
action_type: "list_elements"
element_filter: Optional[str]  # "interactive", "images", "tables", "layout", or None (all)

# Returns in observation.elements_list (new field)
[
    {"selector": "div.container", "tag": "div", "classes": ["container"], "children": 5, "text_preview": "LaunchPad Ship Products..."},
    {"selector": "img", "tag": "img", "has_src": true, "classes": []},
    {"selector": "a.cta-btn", "tag": "a", "is_interactive": true, "text_preview": "Get Started"},
    ...
]
```

Why this isn't an oracle: it shows element STRUCTURE but not CSS properties or responsiveness issues. The agent must inspect elements and reason about what it finds.

### New Action: `get_page_info`

Returns page-level metadata that persists across steps (fixes the page_summary bug).

```python
# Action
action_type: "get_page_info"

# Returns in observation
page_summary: "Title: ... Elements: ... CSS rules: ... Media queries: ... Viewport meta: yes/no"
```

### Redesigned `submit_report`

Richer format with per-issue evidence (CONFIRMED — user chose this):

```python
# Action fields for submit_report
identified_issues: list[dict]  # each dict has: issue, affected_selectors, affected_viewports, description
overall_assessment: Literal["responsive", "partially_responsive", "non_responsive"]

# Example payload:
{
    "action_type": "submit_report",
    "identified_issues": [
        {
            "issue": "fixed_width_elements",
            "affected_selectors": [".wrapper"],
            "affected_viewports": [320, 375],
            "description": "The .wrapper element has width: 960px which overflows on mobile"
        }
    ],
    "overall_assessment": "non_responsive"
}
```

### Timeout Behavior (CONFIRMED — user chose zero reward)

When step limit is hit without submit:
```python
reward = 0.0
done = True
error = "Step limit reached without submit."
```
No oracle fallback. Agent learns that not submitting = 0 reward.

### Reward Changes

```
R = F1 * 0.85 + evidence_bonus (max 0.05) + coverage_bonus (max 0.05) + efficiency_bonus (max 0.05)
```

- **F1 (85%)** — comparing issue names from `identified_issues[*].issue` against ground truth
- **Evidence bonus (5%)** — for each true-positive issue, check if `affected_selectors` actually have the problem and `affected_viewports` are viewports where it manifests. Score = avg correctness across TP issues.
- **Coverage bonus (5%)** — proportion of standard viewports tested
- **Efficiency bonus (5%)** — (max_steps - steps_used) / max_steps

Evidence bonus implementation:
```python
for each TP issue:
    gt_check = ground_truth[issue_name]
    cited_selectors = issue_dict["affected_selectors"]
    cited_viewports = issue_dict["affected_viewports"]
    # Check if cited selectors are in gt evidence
    # Check if cited viewports are ones where the check actually fails
    evidence_correct += (selector_match + viewport_match) / 2
evidence_bonus = 0.05 * evidence_correct / max(tp_count, 1)
```

---

## Files to Modify

### 1. `models.py` — Updated Action/Observation Models

Changes:
- Remove `CheckName` type, `check_name` field from `DalaalAction`
- Change `ActionType` to: `set_viewport | list_elements | inspect_element | get_page_info | submit_report`
- Add `element_filter: Optional[Literal["interactive", "images", "tables", "layout"]]` field
- Change `identified_issues` from `Optional[list[str]]` to `Optional[list[dict]]` (rich evidence)
- Add `overall_assessment: Optional[Literal["responsive", "partially_responsive", "non_responsive"]]`
- Add `elements_list: Optional[list[dict]]` to `DalaalObservation` (for list_elements response)
- Remove `check_result` from `DalaalObservation`
- Remove `checks_run` and `checks_failed` from `DalaalState` and `DalaalObservation`

### 2. `server/environment.py` — Core Episode Logic

Changes:
- Remove `_handle_run_check` method entirely
- Add `_handle_list_elements(action, analysis)`:
  - Build metadata list from `analysis.elements`
  - Apply filter if `action.element_filter` is set
  - Return: `[{selector, tag, classes, children_count, text_preview (first 60 chars), is_interactive}]`
  - Cap at 50 elements to keep observation size manageable
- Add `_handle_get_page_info(analysis)`:
  - Returns `analysis.summary()` at any step (fixes page_summary bug)
- Update `_handle_submit_report`:
  - Accept `identified_issues` as `list[dict]` (each with issue, affected_selectors, affected_viewports, description)
  - Extract issue names for F1, pass full dicts for evidence bonus
- Timeout: return `reward=0.0, done=True, error="Step limit reached"` — no auto-submit
- Remove `checks_run` / `checks_failed` tracking from state
- Remove `page_summary` from `_obs()` builder (agent uses `get_page_info` explicitly)

### 3. `analyzer/scorer.py` — Updated Reward Function

Changes:
- Update `compute_reward()` signature: `identified` becomes `list[dict]`, add `ground_truth_evidence` param
- Extract issue names from `item["issue"]` for F1 computation
- Add `_compute_evidence_bonus(identified_dicts, ground_truth)`:
  - For each TP issue, check if cited selectors appear in GT evidence strings
  - For each TP issue, check if cited viewports are ones where check actually fails
  - Return average correctness * 0.05
- Remove `checks_run` param (no longer tracked)
- Adjust weights: 0.85 F1 + 0.05 evidence + 0.05 coverage + 0.05 efficiency

### 4. `analyzer/parser.py` — Add Element Listing Support

Changes:
- Add `PageAnalysis.list_elements(element_filter=None, limit=50)`:
  - Returns `list[dict]` with keys: `selector, tag, classes, children_count, text_preview, is_interactive`
  - `text_preview`: first 60 chars of element's text content
  - Filters:
    - `"interactive"` → a, button, input, select, textarea
    - `"images"` → img
    - `"tables"` → table
    - `"layout"` → div, section, main, article, aside, nav, header, footer (with children > 0)
    - `None` → all elements
  - Uses existing `ElementInfo` dataclass — minimal new code
  - Capped at `limit` elements to keep observation compact

### 5. `client.py` — Updated Client

Changes:
- Update `_step_payload` and `_parse_result` for new action/observation fields
- Remove check-related code

### 6. `data/` — Expanded Dataset (15-20 pages)

Target: 20 pages total (15 new + 5 existing).

**Fully responsive (5 new, zero issues):**
- `responsive_portfolio.html` — CSS Grid gallery, clamp() fonts, responsive images
- `responsive_form.html` — contact form with flexbox layout, rem spacing
- `responsive_docs.html` — documentation page with sidebar that collapses on mobile
- `responsive_gallery.html` — image grid with auto-fit, responsive breakpoints
- `responsive_nav.html` — hamburger menu pattern, mobile-first media queries

**Partially responsive (3 new, 2-4 issues):**
- `partial_news.html` — news site with fixed-width sidebar, px fonts, but has viewport meta
- `partial_portfolio.html` — portfolio with fixed image sizes but flexible layout
- `partial_table_heavy.html` — data dashboard with responsive layout but fixed-width tables

**Non-responsive (2 new, 6+ issues):**
- `legacy_restaurant.html` — 800px fixed layout, no viewport meta, table-based
- `legacy_forum.html` — nested tables, fixed widths, px everything

**Adversarial (3 new):**
- `tricky_desktop_first.html` — desktop-first design, no explicit desktop media query (should pass — tests media_queries check isn't overly strict)
- `tricky_hidden_overflow.html` — uses overflow:hidden to mask issues (looks clean but content is clipped)
- `tricky_inline_styles.html` — all styles are inline, no `<style>` blocks (tests parser handles inline-only pages)

Regenerate `manifest.json` with ground truth after all pages are created.

### 7. `README.md` — Comprehensive Documentation

Structure:
```
# DalaalEnv — Website Responsiveness Audit RL Environment

## Overview
What this environment does and why it's an RL problem.

## Quick Start
Install, run server, connect client, play an episode.

## Action Space
Table: action_type, required fields, description, example payload.

## Observation Space
Table: field, type, description, when populated.

## Reward Function
Formula, explanation of each component, edge cases.

## Example Episode
5-step JSON transcript: reset → list_elements → set_viewport(375) → inspect_element(.wrapper) → submit_report.

## Dataset
How pages are structured, quality tiers, how to add pages.

## What Makes This a Good RL Problem
- Exploration: agent must test multiple viewports and inspect elements
- Reasoning: agent must interpret raw CSS properties, not check results
- Tradeoff: gather evidence vs submit efficiently
- Delayed reward: only terminal reward via GRPO

## Technical Details
CSS-cascade-lite analysis engine, how ground truth is computed.
```

---

## Implementation Order (2 days)

### Block 1: Core Redesign (3-4 hours)
1. Update `models.py` — new action types, remove run_check, richer submit
2. Update `server/environment.py` — remove `_handle_run_check`, add `_handle_list_elements` + `_handle_get_page_info`, fix page_summary bug, update timeout
3. Update `analyzer/scorer.py` — richer reward with evidence bonus
4. Add `list_elements()` method to `PageAnalysis`
5. Update `client.py` for new fields
6. Run `openenv validate` — must still pass

### Block 2: Dataset Expansion (2-3 hours)
7. Write 10-15 new HTML pages across all tiers
8. Regenerate `manifest.json` with ground truth
9. Verify ground truth distribution makes sense

### Block 3: README (1-2 hours)
10. Write comprehensive README with all sections above
11. Include example episode transcript with actual JSON

### Block 4: Verify & Deploy (1 hour)
12. Run `openenv validate` (structural)
13. Start server, run `openenv validate --url` (runtime)
14. Test full episode via client
15. Commit and push

---

## What We Keep from v1

- **Analysis engine** (`analyzer/parser.py`, `analyzer/checks.py`) — the 10 checks stay as INTERNAL grading functions. They just aren't exposed to the agent anymore.
- **CSS-cascade-lite** — `PageAnalysis`, `get_element_css()`, all working correctly after v1 fixes
- **F1-based reward** — proven to work, with small refinements
- **OpenEnv wiring** — `create_app()`, Dockerfile, `openenv.yaml`, all passing validation
- **Unique selector generation** — `_build_selector()` with `:nth(n)` dedup (fixed in v1)

## Verification

1. `openenv validate` — 8/8 structural checks pass
2. `openenv validate --url http://localhost:8000` — 6/6 runtime checks pass
3. Manual episode test: reset → list_elements → set_viewport(375) → inspect_element(selector) → submit_report → verify reward
4. Edge case: submit on fully responsive page with no issues → reward ~1.0
5. Edge case: submit all 10 issues on responsive page → reward ~0.0 (false positives penalized)
6. Edge case: timeout without submit → zero reward (not oracle-based fallback)
7. `GET /schema` returns updated action/observation schemas
