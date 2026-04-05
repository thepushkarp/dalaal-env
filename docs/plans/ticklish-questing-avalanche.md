# DalaalEnv — Website Responsiveness Audit RL Environment

## Context

**Hackathon**: Meta PyTorch OpenEnv Hackathon x SST (scaler.com). Build RL environments using Meta's OpenEnv framework.
- Round 1 (remote, deadline Apr 8): automated `openenv validate` checks + LLM-based scoring of README/design
- Round 2 (in-person Apr 25-26): code review by Meta engineers, $10K prizes, Meta/HF interview opportunities
- Framework: [openenv-course](https://github.com/huggingface/openenv-course), environments are Docker microservices with FastAPI/WebSocket

**Problem**: No RL environment exists for evaluating website responsiveness. Google Lighthouse does this as a static tool — we make it an interactive RL problem where an agent learns to systematically audit web pages across viewports.

**Why this is a good RL problem**:
1. **Exploration has value** — the agent must test multiple viewport sizes and inspect elements to gather evidence
2. **Genuine tradeoff** — submit early (high efficiency, low accuracy) vs gather all evidence (high accuracy, low efficiency)
3. **Delayed reward** — reward=0 for all intermediate steps, only computed at episode end (ideal for GRPO)
4. **LLM-native** — actions are structured JSON, naturally expressible by language models

---

## Architecture

### Episode Flow

```
RESET → load random HTML page, return page summary at 1280px viewport
  │
  ├── set_viewport(width)     → re-analyze at new breakpoint, return layout data
  ├── inspect_element(selector) → return computed CSS properties of element
  ├── run_check(check_name)   → run one of 10 responsiveness checks
  └── submit_report(issues)   → end episode, compute F1 reward
  │
  └── max 20 steps, forced submit if budget exhausted
```

### Analysis Engine (no browser — pure CSS/HTML parsing)

Uses `tinycss2` + `beautifulsoup4` to parse HTML/CSS and detect responsive patterns programmatically:
- Parse `<style>` blocks and extract `@media` rules with tinycss2
- At any viewport width, filter rules by media condition to resolve applied styles
- Resolve 6-8 key properties: `width`, `max-width`, `font-size`, `display`, `position`, `overflow-x`
- Runs in milliseconds per page, zero browser infrastructure needed

### 10 Responsiveness Checks

| Check | Logic |
|-------|-------|
| `viewport_meta` | `<meta name="viewport" content="width=device-width">` present |
| `media_queries` | CSS covers mobile (<=480), tablet (481-1024), desktop (>1024) ranges |
| `fixed_width_elements` | Elements with fixed px width > 90% of viewport width |
| `responsive_images` | `<img>` tags have `max-width: 100%` or `width: 100%` |
| `font_sizing` | >80% of font-size declarations use rem/em/%/vw/clamp() not fixed px |
| `flexible_layouts` | Major containers use flex/grid, not fixed positioning |
| `touch_targets` | Clickable elements >= 44px at mobile viewports (<=480px) |
| `horizontal_scroll` | No elements with fixed width exceeding viewport |
| `text_readability` | Estimated chars-per-line between 20-90 |
| `responsive_tables` | Tables have overflow-x: auto wrapper or responsive display rules |

### Reward Function

```
R = F1 * 0.85 + coverage_bonus (max 0.10) + efficiency_bonus (max 0.10)

Where:
  F1         = 2 * Precision * Recall / (P + R)  over identified vs ground truth issues
  coverage   = 0.10 * (viewports_tested ∩ {320,375,768,1024,1280,1440}) / 6
  efficiency = 0.10 * max(0, (max_steps - step_count) / max_steps)
```

Edge: GT empty (fully responsive page) + agent submits empty → F1=1. GT empty + agent reports issues → F1=0. Correctly penalizes hallucination.

### Dataset

30 HTML pages, all CSS embedded inline, across 3 tiers:
- **10 fully responsive** (modern CSS Grid, media queries, rem fonts) — tests agent doesn't hallucinate
- **12 partially responsive** (2-5 issues each) — the learning zone
- **8 non-responsive** (legacy fixed-width, no media queries) — easy to identify

5 archetypes (blog, e-commerce, landing page, dashboard, corporate) x 3 variants (good/partial/legacy) + extras.

---

## File Structure

```
dalaal_env/
├── __init__.py                    # exports DalaalAction, DalaalObservation, DalaalEnv
├── models.py                      # Pydantic: DalaalAction, DalaalObservation, DalaalState
├── client.py                      # DalaalEnv(EnvClient) — WebSocket client
├── analyzer/
│   ├── __init__.py
│   ├── parser.py                  # PageParser → PageAnalysis dataclass, CSS-cascade-lite
│   ├── checks.py                  # 10 check functions, each → CheckResult
│   └── scorer.py                  # compute_ground_truth() + compute_reward()
├── data/
│   ├── pages/                     # 30 HTML files with embedded CSS
│   └── manifest.json              # page metadata + pre-computed ground truth
├── server/
│   ├── __init__.py
│   ├── environment.py             # DalaalEnvironment(Environment) — core RL logic
│   ├── app.py                     # create_app() + main() entry point
│   └── Dockerfile
├── openenv.yaml
├── pyproject.toml                 # [project.scripts] server = "dalaal_env.server.app:main"
└── uv.lock
```

### Key Models (models.py)

**DalaalAction** — flat structure with `action_type: Literal[...]` discriminator + optional fields per action type. LLM sees one clean JSON schema. Pydantic `extra="forbid"` from base catches hallucinated fields.

**DalaalObservation** — always has: `page_id`, `current_viewport_width`, `viewports_tested`, `checks_run`, `step_budget_remaining`, `page_summary`. Conditionally has: `element_info` (after inspect), `check_result` (after run_check), `final_scores` (on done), `error` (on invalid action).

**DalaalState** — extends base State with: `page_id`, `current_viewport_width`, `viewports_tested`, `checks_run`, `max_steps`, `submitted`.

### Dependencies (minimal)

```toml
dependencies = [
    "openenv-core>=0.2.3",     # framework (FastAPI, uvicorn, pydantic, websockets)
    "beautifulsoup4>=4.12.0",  # HTML parsing
    "tinycss2>=1.3.0",         # CSS tokenization + rule parsing
    "lxml>=5.0.0",             # faster BS4 parser backend
]
```

No browser. No numpy. No requests. Pure Python analysis running in milliseconds.

---

## Implementation Plan

### Day 1 — Analysis Engine (no server)

Build and test the CSS/HTML analysis in isolation:

1. **`analyzer/parser.py`** — `PageParser.parse(html, page_id) → PageAnalysis`
   - BeautifulSoup for element tree + inline styles
   - tinycss2 for `<style>` blocks + `@media` rule extraction
   - `PageAnalysis.get_element_css(selector, viewport_width)` for viewport-aware property resolution
2. **`analyzer/checks.py`** — implement all 10 checks, start with simplest (viewport_meta, media_queries, responsive_images = pure string matching)
3. **`analyzer/scorer.py`** — `compute_ground_truth()` and `compute_reward()`
4. **5 test HTML pages** (1 per archetype, covering all 3 tiers)
5. **`data/manifest.json`** — run scorer offline to populate ground_truth_issues
6. Manual test: `python -c "from dalaal_env.analyzer.parser import PageParser; ..."`

### Day 2 — OpenEnv Integration

Wire analysis engine into the framework:

1. **`models.py`** — DalaalAction, DalaalObservation, DalaalState (exact Pydantic classes from architecture doc)
2. **`server/environment.py`** — DalaalEnvironment with `reset()`, `step()`, `state`. 4 action handlers. Error obs on invalid actions (never crash GRPO rollouts).
3. **`server/app.py`** — `create_app(DalaalEnvironment, DalaalAction, DalaalObservation, env_name="dalaal_env", max_concurrent_envs=8)`
4. **`client.py`** — DalaalEnv(EnvClient) with `_step_payload`, `_parse_result`, `_parse_state`
5. **`__init__.py`**, **`openenv.yaml`**, update **`pyproject.toml`**, run `uv lock`
6. Run `openenv validate` — fix structural issues
7. Start server: `uv run server` → test endpoints with curl
8. Run `openenv validate --url http://localhost:8000` — all runtime checks must pass

### Day 3 — Dataset, Polish, Deploy

1. Expand dataset to 30 HTML pages (5 archetypes x 3 variants + 15 extras)
2. Re-run scorer to fill manifest.json ground truth for all pages
3. Test `server/Dockerfile` locally with `openenv build`
4. Write README with: problem statement, action space table, observation schema, reward formula, example episode transcript (critical for Round 1 LLM scoring)
5. `openenv push --repo-id <username>/dalaal-env`
6. `openenv validate --url https://<username>-dalaal-env.hf.space` — final verification

---

## What Will Impress Judges

**Code quality signals**:
- Full type annotations everywhere
- `SUPPORTS_CONCURRENT_SESSIONS = True` (shows GRPO awareness)
- Page cache avoids re-parsing identical HTML across episodes
- Deterministic seeding (`reset(seed=42)` always picks same page)
- Graceful error handling — invalid actions return error obs, never crash
- Ground truth never leaked to agent in any observation

**RL design quality**:
- Non-trivial reward — cannot get F1=1 without gathering evidence
- Exploration incentivized — more viewports tested = higher coverage bonus
- Graded difficulty — manifest has easy/medium/hard pages for curriculum learning
- `final_scores` breakdown shows exactly where reward came from

**README for LLM scoring**:
- Clear problem statement
- Action space + observation tables
- Reward formula written out
- 5-step example episode transcript with actual JSON payloads and rewards
- Why this is a good RL problem (exploration, delayed reward, real-world applicability)

---

## Verification

1. `openenv validate` — all 8 structural checks pass
2. `openenv validate --url http://localhost:8000` — all 6 runtime checks pass
3. Manual episode: reset → set_viewport(320) → run_check("viewport_meta") → inspect_element("img") → submit_report(["fixed_width_elements"]) → verify reward makes sense
4. Edge case: submit empty report on fully responsive page → reward should be ~1.0
5. Edge case: submit all 10 issues on fully responsive page → reward should be ~0.0
6. Edge case: hit step limit without submitting → forced submit, low reward
7. `openenv push` + live validation on HF Spaces

---

## Existing Resources to Reuse

- **OpenEnv template code**: `.venv/lib/python3.12/site-packages/openenv/cli/templates/openenv_env/` — exact patterns for app.py, client.py, models.py, Dockerfile, import try/except pattern
- **Detailed architecture doc**: `docs/plans/ticklish-questing-avalanche-agent-aea280722fd1c7c13.md` — full Pydantic class definitions, environment pseudocode, check implementations, reward formula edge cases

---

## Unresolved Questions

1. Should we also support an "agent fixes the CSS" mode (beyond audit)? → **Defer to v2** — audit mode is simpler and clearer for the hackathon deadline
2. Should `severity_scores` in submit_report be included in v1? → **Include the field but make severity_bonus optional** — shows design forethought without adding implementation complexity
3. Should ground truth be pre-computed in manifest.json or computed at reset time? → **Both** — pre-compute for fast resets, but also validate by re-computing at startup
4. How to handle external CSS (`<link>` tags)? → **All test pages embed CSS inline** — avoids HTTP fetching, keeps the environment self-contained
