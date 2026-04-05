# DalaalEnv — Website Responsiveness Audit RL Environment

An RL environment where an agent audits web pages for responsiveness issues across different screen sizes — like Google Lighthouse, but as an interactive environment for training RL agents.

Built with [OpenEnv](https://github.com/meta-pytorch/OpenEnv) for the Meta PyTorch OpenEnv Hackathon.

## Overview

**The task**: Given an HTML page, the agent must explore its structure and CSS properties across viewport sizes, identify responsiveness issues, and submit an audit report with evidence.

**Why it's an RL problem**:
- **Exploration has value** — the agent must test multiple viewport sizes and inspect elements to gather evidence. Simply guessing issues without evidence scores poorly.
- **Genuine reasoning required** — the agent sees raw CSS properties, not pass/fail checks. It must interpret `width: 960px` at a 375px viewport and reason that this causes overflow.
- **Exploration-exploitation tradeoff** — gather more evidence for accuracy vs submit early for efficiency bonus.
- **Delayed reward** — reward is only computed at episode end (ideal for GRPO training).

## Quick Start

```bash
# Install
uv sync

# Run server
uv run server
# Server at http://localhost:8000

# Play an episode (Python)
from client import DalaalEnv
from dalaal_env.models import DalaalAction

with DalaalEnv(base_url="http://localhost:8000").sync() as env:
    result = env.reset()
    print(result.observation.page_summary)

    # Discover page elements
    result = env.step(DalaalAction(action_type="list_elements", element_filter="layout"))
    for el in result.observation.elements_list[:5]:
        print(f"  {el['selector']} ({el['tag']})")

    # Test at mobile viewport
    result = env.step(DalaalAction(action_type="set_viewport", viewport_width=375))

    # Inspect a suspicious element
    result = env.step(DalaalAction(action_type="inspect_element", css_selector="div#wrapper"))
    print(result.observation.element_info)
    # → {'selector': 'div#wrapper', 'viewport_width': 375, 'computed_styles': {'width': '960px'}}
    # Agent reasons: 960px at 375px viewport = overflow!

    # Submit audit report with evidence
    result = env.step(DalaalAction(
        action_type="submit_report",
        identified_issues=[
            {
                "issue": "fixed_width_elements",
                "affected_selectors": ["div#wrapper"],
                "affected_viewports": [375, 320],
                "description": "960px fixed wrapper overflows on mobile viewports"
            },
            {
                "issue": "viewport_meta",
                "affected_selectors": [],
                "affected_viewports": [375],
                "description": "Missing <meta name='viewport'> tag"
            }
        ],
        overall_assessment="non_responsive",
    ))
    print(f"Reward: {result.reward:.4f}")
    print(f"Scores: {result.observation.final_scores}")
```

## Action Space

| Action | Required Fields | Description |
|--------|----------------|-------------|
| `set_viewport` | `viewport_width` (320-2560) | Change the simulated viewport width |
| `list_elements` | `element_filter` (optional: "interactive", "images", "tables", "layout") | Discover page elements with metadata |
| `inspect_element` | `css_selector` | Get computed CSS properties at current viewport |
| `get_page_info` | — | Get page-level summary (title, element counts, CSS stats) |
| `submit_report` | `identified_issues`, `overall_assessment` | End episode with audit findings |

### `identified_issues` format

Each issue is a dict with:
- `issue` — one of: `viewport_meta`, `media_queries`, `fixed_width_elements`, `responsive_images`, `font_sizing`, `flexible_layouts`, `touch_targets`, `horizontal_scroll`, `text_readability`, `responsive_tables`
- `affected_selectors` — list of CSS selectors with the issue
- `affected_viewports` — list of viewport widths where issue manifests
- `description` — explanation of the issue

## Observation Space

| Field | Type | When Populated |
|-------|------|----------------|
| `page_id` | str | Always |
| `current_viewport_width` | int | Always |
| `viewports_tested` | list[int] | Always |
| `step_budget_remaining` | int | Always |
| `page_summary` | str | On reset and `get_page_info` |
| `elements_list` | list[dict] | After `list_elements` |
| `element_info` | dict | After `inspect_element` |
| `final_scores` | dict | On `done=True` |
| `error` | str | On invalid actions |

## Reward Function

```
R = F1 × 0.85 + evidence_bonus (max 0.05) + coverage_bonus (max 0.05) + efficiency_bonus (max 0.05)
```

| Component | Weight | How Computed |
|-----------|--------|-------------|
| **F1 score** | 0.85 | Precision × Recall over identified issue names vs ground truth |
| **Evidence bonus** | 0.05 | Avg correctness of cited selectors and viewports per true-positive |
| **Coverage bonus** | 0.05 | Proportion of standard viewports (320, 375, 768, 1024, 1280, 1440) tested |
| **Efficiency bonus** | 0.05 | (max_steps − steps_used) / max_steps |

**Edge cases**:
- Fully responsive page + agent submits empty report → F1 = 1.0 (correct!)
- Fully responsive page + agent hallucinated issues → F1 = 0.0 (penalized!)
- Step limit without submit → reward = 0.0 (agent must learn to submit)

## Example Episode

```
Step 0: reset()
  → page_id: "legacy_corporate", viewport: 1280px
  → summary: "Title: Acme Corp | 68 elements | 0 media queries | viewport meta: no"

Step 1: list_elements(filter="layout")
  → 11 layout containers: div#wrapper, div#header, div#content, ...

Step 2: set_viewport(375)
  → viewports_tested: [1280, 375]

Step 3: inspect_element("div#wrapper")
  → computed_styles: {width: "960px"}
  → Agent reasoning: 960px at 375px = overflows! Fixed width issue.

Step 4: inspect_element("img.banner-img")
  → computed_styles: {width: "960px", height: "300px", display: "block"}
  → Agent reasoning: Fixed 960px image, no max-width: 100%. Image issue.

Step 5: submit_report(
    issues=[viewport_meta, fixed_width_elements, responsive_images, media_queries],
    evidence per issue,
    assessment="non_responsive"
  )
  → reward: 0.52 (4/10 issues found, precision=1.0, recall=0.4)
```

## Dataset

20 HTML pages with embedded CSS across quality tiers:

| Tier | Count | Description |
|------|-------|-------------|
| Fully responsive | 7 | Modern CSS Grid, flexbox, rem fonts, media queries — zero issues |
| Partially responsive | 5 | Mix of responsive and fixed elements — 1-4 issues |
| Non-responsive | 5 | Legacy fixed-width layouts — 6-10 issues |
| Adversarial | 3 | Tricky edge cases (desktop-first, hidden overflow, inline-only) |

Pages cover diverse archetypes: blogs, e-commerce, corporate sites, dashboards, landing pages, portfolios, forms, documentation, galleries, restaurants, forums.

## 10 Responsiveness Checks (Internal Grading)

These checks are used to compute ground truth — the agent never sees them directly.

| Check | What It Detects |
|-------|----------------|
| `viewport_meta` | Missing `<meta name="viewport">` tag |
| `media_queries` | Missing CSS breakpoints for mobile/tablet/desktop |
| `fixed_width_elements` | Elements with fixed px widths exceeding viewport |
| `responsive_images` | Images without `max-width: 100%` |
| `font_sizing` | Font sizes using fixed px instead of rem/em/% |
| `flexible_layouts` | Layout containers using fixed positioning instead of flex/grid |
| `touch_targets` | Interactive elements < 44px at mobile viewports |
| `horizontal_scroll` | Elements causing horizontal overflow |
| `text_readability` | Text blocks with problematic line lengths |
| `responsive_tables` | Tables without responsive handling (overflow-x, etc.) |

## Technical Details

**Analysis engine**: Pure CSS/HTML parsing using `beautifulsoup4` + `tinycss2`. No browser needed.
- Parses `<style>` blocks and extracts `@media` rules
- Resolves CSS properties at any viewport width via cascade-lite (last matching rule wins)
- Tracks: width, max-width, min-width, height, min-height, font-size, display, position, overflow-x

**Server**: FastAPI + WebSocket via OpenEnv `create_app()`. Supports concurrent sessions for GRPO parallel rollouts.

**Deployment**: `openenv push --repo-id <username>/dalaal-env` deploys to HF Spaces.

## Validation

```bash
# Structural validation
uv run openenv validate
# [OK] dalaal-env: Ready for multi-mode deployment

# Runtime validation
uv run server &
uv run openenv validate --url http://localhost:8000
# passed: 6/6 criteria
```
