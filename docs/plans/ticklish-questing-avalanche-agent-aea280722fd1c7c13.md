# DalaalEnv — Architecture Plan
*Website Responsiveness Audit RL Environment for Meta OpenEnv Hackathon*  
*Deadline: April 8, 2026 | Author: Architecture Planning Agent*

---

## 1. Architecture Validation & Overall Assessment

The proposed file structure is sound and matches the OpenEnv 3-component pattern exactly. A few
improvements are noted below before diving into the full design.

**What to keep as-is:**
- The `analyzer/` sub-package as a pure Python module with no server coupling — correct.
- Separate `checks.py` and `scorer.py` — good separation of concerns.
- `data/pages/` + `data/manifest.json` — clean dataset layout.

**Changes to make:**
1. Rename `server/dalaal_environment.py` → `server/environment.py` (matches OpenEnv convention).
2. Add `dalaal_env/__init__.py` that exports `DalaalAction`, `DalaalObservation`, `DalaalEnv`.
3. The `models.py` must live at the package root (`dalaal_env/models.py`), not inside `server/`.
4. Add a `DalaalState` subclass of `State` in `models.py` — don't use bare `State`.
5. `SUPPORTS_CONCURRENT_SESSIONS = True` on the Environment class — needed for GRPO parallel rollouts.

**Final file structure:**
```
dalaal_env/
├── __init__.py                    # exports DalaalAction, DalaalObservation, DalaalEnv
├── models.py                      # DalaalAction, DalaalObservation, DalaalState
├── client.py                      # DalaalEnv (EnvClient subclass)
├── analyzer/
│   ├── __init__.py
│   ├── parser.py                  # HTML/CSS parsing — PageAnalysis dataclass
│   ├── checks.py                  # 10 individual responsiveness check functions
│   └── scorer.py                  # ground truth computation + F1 scoring
├── data/
│   ├── pages/                     # 30 HTML files
│   └── manifest.json              # page metadata + ground truth issue lists
├── server/
│   ├── __init__.py
│   ├── environment.py             # DalaalEnvironment(Environment) — core logic
│   ├── app.py                     # create_app() wiring + main()
│   └── Dockerfile
├── openenv.yaml
├── pyproject.toml
└── uv.lock
```

---

## 2. Pydantic Models — Exact Class Definitions

### `dalaal_env/models.py`

```python
from __future__ import annotations
from typing import Literal, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation, State


# ── Action ────────────────────────────────────────────────────────────────────

ActionType = Literal[
    "set_viewport",
    "inspect_element",
    "run_check",
    "submit_report",
]

class DalaalAction(Action):
    """
    One action the agent can take during a responsiveness audit.

    Fields are union-style: which fields are used depends on action_type.
    All optional fields default to None so the Pydantic schema stays flat
    (better for LLM structured-output parsing).
    """
    action_type: ActionType = Field(
        ...,
        description=(
            "set_viewport: change the simulated viewport width. "
            "inspect_element: get CSS properties of a CSS selector. "
            "run_check: run one named responsiveness check. "
            "submit_report: end the episode and submit audit findings."
        ),
    )

    # --- set_viewport ---
    viewport_width: Optional[int] = Field(
        default=None,
        ge=320,
        le=2560,
        description="Viewport width in pixels (320–2560). Required for set_viewport.",
    )

    # --- inspect_element ---
    css_selector: Optional[str] = Field(
        default=None,
        max_length=256,
        description="CSS selector of element to inspect. Required for inspect_element.",
    )

    # --- run_check ---
    check_name: Optional[Literal[
        "viewport_meta",
        "media_queries",
        "fixed_width_elements",
        "responsive_images",
        "font_sizing",
        "flexible_layouts",
        "touch_targets",
        "horizontal_scroll",
        "text_readability",
        "responsive_tables",
    ]] = Field(
        default=None,
        description="Which check to run. Required for run_check.",
    )

    # --- submit_report ---
    identified_issues: Optional[list[str]] = Field(
        default=None,
        description=(
            "List of issue keys the agent believes are present. "
            "Must be a subset of the 10 check names. Required for submit_report."
        ),
    )
    severity_scores: Optional[dict[str, float]] = Field(
        default=None,
        description=(
            "Optional per-issue severity (0.0–1.0). Keys must match identified_issues. "
            "Used to compute partial-credit reward."
        ),
    )


# ── Observation ───────────────────────────────────────────────────────────────

class DalaalObservation(Observation):
    """
    What the agent receives after each action.
    `done` and `reward` are inherited from Observation base class.
    """

    # Episode context (always present)
    page_id: str = Field(
        default="",
        description="Identifier of the HTML page being audited.",
    )
    current_viewport_width: int = Field(
        default=1280,
        description="Current simulated viewport width in pixels.",
    )
    viewports_tested: list[int] = Field(
        default_factory=list,
        description="All viewport widths tested so far this episode.",
    )
    checks_run: list[str] = Field(
        default_factory=list,
        description="Names of checks that have been explicitly run.",
    )
    step_budget_remaining: int = Field(
        default=20,
        description="How many steps remain before forced termination.",
    )

    # Page structure (populated on reset and inspect_element)
    page_summary: str = Field(
        default="",
        description=(
            "Human-readable summary of the page: title, element counts, "
            "stylesheet count, inline style count."
        ),
    )
    element_info: Optional[dict] = Field(
        default=None,
        description=(
            "CSS properties of the last inspected element at current viewport. "
            "Keys: selector, tag, computed_width, display, position, "
            "font_size, max_width, overflow."
        ),
    )

    # Check results (populated after run_check)
    check_result: Optional[dict] = Field(
        default=None,
        description=(
            "Result of the last run_check. Keys: check_name, passed (bool), "
            "detail (str), evidence (list[str])."
        ),
    )

    # Episode outcome (only on done=True)
    final_scores: Optional[dict] = Field(
        default=None,
        description=(
            "Only present when done=True. Keys: precision, recall, f1, "
            "coverage_bonus, efficiency_bonus, total."
        ),
    )

    # Error feedback
    error: Optional[str] = Field(
        default=None,
        description="Human-readable error message if the action was invalid.",
    )


# ── State ─────────────────────────────────────────────────────────────────────

class DalaalState(State):
    """
    Episode metadata (available via GET /state at any time).
    episode_id and step_count are inherited from State base class.
    """
    page_id: str = Field(default="")
    current_viewport_width: int = Field(default=1280)
    viewports_tested: list[int] = Field(default_factory=list)
    checks_run: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=20)
    submitted: bool = Field(default=False)
```

**Key design decisions:**
- Flat action structure (all fields optional, discriminated by `action_type`) is better for LLM
  JSON generation than a union type, because the LLM sees one schema with clear descriptions.
- `Action` has `extra="forbid"` from the base — every field name must be exact. The `Literal`
  type on `action_type` enforces the action space boundary automatically.
- Severity scores in submit_report enable partial-credit reward without complicating F1 scoring.
- `error` field in observation gives the LLM textual feedback on malformed actions rather than
  a crash — critical for stable GRPO rollouts.

---

## 3. Episode Flow — Step by Step

```
RESET
│
│   DalaalEnvironment.reset():
│     1. Sample page_id from manifest (random or seeded)
│     2. Load HTML from data/pages/<page_id>.html
│     3. Run PageParser.parse(html) → PageAnalysis (cached per page_id)
│     4. Compute ground_truth = scorer.compute_ground_truth(analysis)
│     5. Initialize DalaalState(episode_id, step_count=0, page_id, ...)
│     6. Return DalaalObservation with page_summary, viewport=1280, done=False, reward=0
│
▼
STEP LOOP (up to max_steps=20)
│
│   Agent sends DalaalAction with action_type ∈ {set_viewport, inspect_element,
│                                                 run_check, submit_report}
│
│   DalaalEnvironment.step(action):
│     1. Validate action fields for its action_type → if invalid, return error obs, reward=0
│     2. Increment state.step_count
│     3. Branch on action_type:
│
│       set_viewport(viewport_width):
│         - state.current_viewport_width = viewport_width
│         - if viewport_width not in state.viewports_tested: append it
│         - return obs with updated current_viewport_width
│         - reward = 0 (exploration has no immediate reward)
│
│       inspect_element(css_selector):
│         - analysis = page_cache[page_id]
│         - element_info = analysis.get_element_css(css_selector, viewport_width)
│         - return obs with element_info populated
│         - reward = 0
│
│       run_check(check_name):
│         - analysis = page_cache[page_id]
│         - result = checks.run(check_name, analysis, state.current_viewport_width)
│         - append check_name to state.checks_run (idempotent)
│         - return obs with check_result populated
│         - reward = 0 (reward only comes at episode end)
│
│       submit_report(identified_issues, severity_scores):
│         - state.submitted = True
│         - reward = scorer.compute_reward(
│               identified=identified_issues,
│               severity=severity_scores,
│               ground_truth=ground_truth,
│               viewports_tested=state.viewports_tested,
│               checks_run=state.checks_run,
│               step_count=state.step_count,
│               max_steps=max_steps,
│           )
│         - return obs with done=True, reward=reward, final_scores=breakdown
│
│     4. If state.step_count >= max_steps and not submitted:
│         - Force submit with identified_issues=checks_run (partial fallback)
│         - Compute reward as above (likely low — no explicit report = penalty)
│         - Return obs with done=True
│
▼
EPISODE END
```

**Why this flow is a good RL environment:**
- There is a meaningful exploration-exploitation tradeoff: the agent can spend steps gathering
  evidence (inspect elements, run checks, test viewports) or submit early for an efficiency bonus.
- The action space is small enough for an LLM to reliably sample from but rich enough that
  naive strategies (submit immediately) score poorly.
- The reward is *only* revealed at episode end — this is a bandit-style delayed reward, which
  GRPO handles well because it optimizes at the trajectory level.

---

## 4. Reward Function — Full Formula

### Ground Truth

The ground truth for each page is a set of issue keys that are *actually present*:

```
GT = {check_name : checks.run(check_name, analysis, all_viewports) returns passed=False}
```

Ground truth is computed once per page at reset time (or precomputed in manifest.json) and
stored server-side. The agent never sees it directly.

### At submit_report time:

```
Let:
  GT       = ground truth issue set  (e.g., {"fixed_width_elements", "font_sizing"})
  P        = agent's identified_issues set
  TP       = |P ∩ GT|
  FP       = |P \ GT|
  FN       = |GT \ P|

Precision  = TP / |P|           (0 if P is empty → 0)
Recall     = TP / |GT|          (1 if GT is empty → 1)
F1         = 2 * Precision * Recall / (Precision + Recall)   (0 if both 0)
```

### Bonus Terms

```
STANDARD_VIEWPORTS = {320, 375, 768, 1024, 1280, 1440}

coverage_bonus = 0.1 * (|viewports_tested ∩ STANDARD_VIEWPORTS| / |STANDARD_VIEWPORTS|)
    # Max +0.1 for testing all 6 standard breakpoints

efficiency_bonus = 0.1 * max(0, (max_steps - step_count) / max_steps)
    # Max +0.1 for submitting on step 1 (but F1 would be 0, so the bonus doesn't dominate)
    # Roughly +0.05 for submitting at step 10 out of 20

severity_bonus = 0.05 * pearson_correlation(severity_scores_over_GT, GT_severity_weights)
    # Only if agent provided severity_scores; GT_severity_weights are precomputed
    # Max +0.05 for perfectly calibrated severity assessment
    # Omit this in v1 if complexity is too high
```

### Total Reward

```
R_total = F1 * 0.85 + coverage_bonus + efficiency_bonus [+ severity_bonus]
```

Weights are chosen so:
- F1 dominates (85% of max reward) — the agent must correctly identify issues.
- Coverage bonus encourages testing multiple breakpoints, creating the exploration incentive.
- Efficiency bonus is small (10%) so the agent isn't incentivized to skip gathering evidence.
- Maximum possible reward = 1.0 (F1=1, all viewports tested, submitted on step 1 — impossible
  in practice, but the scale is correct).

### Edge Cases in Reward

| Situation | Handling |
|-----------|---------|
| GT is empty (fully responsive page) | Recall=1. If agent submits empty list: F1=1. If agent submits any issue: Precision=0, F1=0. This correctly penalizes false positives. |
| Agent submits empty list on non-responsive page | TP=0, F1=0. Only bonuses apply (~0.2 max). |
| Agent never submits (timeout) | Forced submit with identified_issues=checks_run. Reward computed normally but efficiency_bonus is 0 (step_count = max_steps). |
| Duplicate check runs | `checks_run` is a set; idempotent. No reward for re-running same check. |
| Invalid action (wrong fields) | reward=0 for that step, error message in obs, episode continues. |

---

## 5. The Analysis Engine — Implementation Design

### `analyzer/parser.py` — `PageAnalysis` Dataclass

```python
@dataclass
class ElementInfo:
    tag: str
    selector: str          # simplified CSS selector (tag#id.class)
    classes: list[str]
    inline_style: dict[str, str]
    # resolved CSS properties at a given viewport (computed lazily per viewport)
    # stored as: resolved_props[viewport_width][selector] = {...}

@dataclass
class StylesheetInfo:
    source: str            # "inline" | "linked:<url>" | "embedded"
    rules: list           # tinycss2 parsed rules

@dataclass
class PageAnalysis:
    page_id: str
    html: str
    title: str
    elements: list[ElementInfo]
    stylesheets: list[StylesheetInfo]
    media_queries: list[dict]  # [{min_width, max_width, rules: [...]}]
    # Derived helpers:
    def get_element_css(self, selector: str, viewport_width: int) -> dict: ...
    def get_elements_by_tag(self, tag: str) -> list[ElementInfo]: ...
    def get_all_widths_in_px(self) -> list[tuple[str, int]]: ...
    def get_font_sizes(self) -> list[tuple[str, str]]: ...
```

**Parsing strategy:**
1. Use `beautifulsoup4` to build element tree and extract inline styles.
2. Use `tinycss2` to parse all `<style>` blocks and extract `<link rel="stylesheet">` hrefs
   (for bundled HTML, these will be embedded; for the test dataset we embed all CSS inline).
3. For each `@media` rule: extract `min-width` / `max-width` from the condition, store the
   declarations that apply within that breakpoint range.
4. "Applying CSS at a viewport" = walk rules in cascade order, apply only rules where the media
   query condition is satisfied by the given viewport_width, compute the final property value.
   This is CSS-cascade-lite — we don't need full CSSOM, just width/font-size/display/overflow.

### `analyzer/checks.py` — 10 Check Functions

Each check has the signature:
```python
def check_<name>(analysis: PageAnalysis, viewport_width: int) -> CheckResult:
    ...

@dataclass
class CheckResult:
    check_name: str
    passed: bool
    detail: str           # 1-2 sentence human-readable explanation
    evidence: list[str]   # specific selectors/values that caused pass/fail
    severity: float       # 0.0–1.0 (used for GT severity weights)
```

**Implementation notes per check:**

| Check | Key Logic |
|-------|-----------|
| `viewport_meta` | BeautifulSoup: `soup.find("meta", attrs={"name": "viewport"})`. Passed if content contains `width=device-width`. Viewport-agnostic. |
| `media_queries` | Count media query blocks covering mobile (≤480), tablet (481–1024), desktop (>1024). Passed if all 3 ranges covered. |
| `fixed_width_elements` | Walk all elements; resolve `width` property at given viewport. Flag any element where width is a fixed px value > viewport_width * 0.9. |
| `responsive_images` | Find all `<img>` and elements with `background-image`. Check that `max-width: 100%` or `width: 100%` is set in CSS. |
| `font_sizing` | Collect all `font-size` declarations. Passed if > 80% use `rem`, `em`, `%`, `vw`, `clamp()`. Failed if majority use fixed `px`. |
| `flexible_layouts` | Check that major layout containers (direct children of body, or elements with class containing "container"/"wrapper"/"layout") use `display: flex` or `display: grid`, or `float`+`width:%`. |
| `touch_targets` | At viewport_width ≤ 480: find all `<a>`, `<button>`, `<input>`, `<select>`. Check height/min-height ≥ 44px (or no explicit height, defaulting to pass). |
| `horizontal_scroll` | At given viewport: find elements where resolved `width` in px > viewport_width, or `overflow-x` is not `hidden`/`scroll` and content would overflow. Use heuristic: any fixed-width element > viewport width is a flag. |
| `text_readability` | Estimate characters per line using: `element_width_px / (font_size_px * 0.5)`. Flag blocks where estimated CPL < 20 or > 90. |
| `responsive_tables` | Find all `<table>` elements. Passed if they have `overflow-x: auto` on a wrapper, or `display: block` in a media query, or `table-layout: auto` with `max-width: 100%`. |

### `analyzer/scorer.py`

```python
def compute_ground_truth(analysis: PageAnalysis) -> dict[str, CheckResult]:
    """
    Run all 10 checks across all standard viewports.
    An issue is in GT if it fails at ANY standard viewport.
    Returns: {check_name: CheckResult} for all failing checks.
    """

def compute_reward(
    identified: list[str],
    severity: dict[str, float] | None,
    ground_truth: dict[str, CheckResult],
    viewports_tested: list[int],
    checks_run: list[str],
    step_count: int,
    max_steps: int,
) -> tuple[float, dict]:  # (total_reward, breakdown_dict)
    """
    Computes the full reward breakdown.
    Returns total reward and a dict with all sub-scores for the final_scores observation field.
    """
```

---

## 6. Dataset Design — `data/manifest.json`

**Target: 30 HTML pages across 3 quality tiers.**

```json
{
  "pages": [
    {
      "id": "fully_responsive_01",
      "filename": "fully_responsive_01.html",
      "tier": "fully_responsive",
      "description": "Modern blog with CSS Grid, media queries, rem fonts",
      "ground_truth_issues": [],
      "difficulty": "easy"
    },
    {
      "id": "partially_responsive_01",
      "filename": "partially_responsive_01.html",
      "tier": "partially_responsive",
      "description": "E-commerce page with flexbox layout but fixed-width images and px fonts",
      "ground_truth_issues": ["responsive_images", "font_sizing"],
      "difficulty": "medium"
    },
    {
      "id": "non_responsive_01",
      "filename": "non_responsive_01.html",
      "tier": "non_responsive",
      "description": "Legacy 960px fixed layout, no media queries, table-based",
      "ground_truth_issues": [
        "viewport_meta", "media_queries", "fixed_width_elements",
        "responsive_images", "font_sizing", "flexible_layouts",
        "touch_targets", "horizontal_scroll", "text_readability", "responsive_tables"
      ],
      "difficulty": "easy"
    }
  ]
}
```

**Distribution: 10 fully responsive, 12 partially responsive (2–5 issues), 8 non-responsive.**

**HTML page generation strategy (to do quickly):**
1. Write 5 hand-crafted archetypes (blog, e-commerce, landing page, dashboard, corporate).
2. For each archetype, create 3 variants: good CSS, partial CSS, legacy CSS.
3. Embed all CSS inline in `<style>` blocks to avoid external HTTP requests during analysis.
4. Pages should be 200–500 lines of HTML — realistic but not gigantic.
5. Pre-compute and store `ground_truth_issues` in manifest.json (run the scorer once offline).

---

## 7. Server — `server/environment.py`

```python
import uuid
import json
from pathlib import Path
from openenv.core.env_server.interfaces import Environment
from ..models import DalaalAction, DalaalObservation, DalaalState
from ..analyzer.parser import PageParser
from ..analyzer.checks import run_check
from ..analyzer.scorer import compute_ground_truth, compute_reward

DATA_DIR = Path(__file__).parent.parent / "data"
MAX_STEPS = 20
STANDARD_VIEWPORTS = [320, 375, 768, 1024, 1280, 1440]

class DalaalEnvironment(Environment[DalaalAction, DalaalObservation, DalaalState]):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        self._state = DalaalState()
        self._manifest = self._load_manifest()
        self._page_cache: dict[str, PageAnalysis] = {}
        self._ground_truth: dict[str, CheckResult] = {}
        self._parser = PageParser()

    def reset(self, seed=None, episode_id=None, **kwargs) -> DalaalObservation:
        # Sample page (seeded or random)
        rng = random.Random(seed)
        page_meta = rng.choice(self._manifest["pages"])
        page_id = page_meta["id"]

        # Parse HTML (with caching)
        if page_id not in self._page_cache:
            html = (DATA_DIR / "pages" / page_meta["filename"]).read_text()
            self._page_cache[page_id] = self._parser.parse(html, page_id)
        analysis = self._page_cache[page_id]

        # Compute ground truth (with caching)
        if page_id not in self._ground_truth:
            self._ground_truth[page_id] = compute_ground_truth(analysis)

        # Initialize state
        self._state = DalaalState(
            episode_id=episode_id or str(uuid.uuid4()),
            step_count=0,
            page_id=page_id,
            current_viewport_width=1280,
            viewports_tested=[1280],
            checks_run=[],
            max_steps=MAX_STEPS,
            submitted=False,
        )

        return DalaalObservation(
            done=False,
            reward=0.0,
            page_id=page_id,
            current_viewport_width=1280,
            viewports_tested=[1280],
            checks_run=[],
            step_budget_remaining=MAX_STEPS,
            page_summary=analysis.summary(),
        )

    def step(self, action: DalaalAction, timeout_s=None, **kwargs) -> DalaalObservation:
        # Guard: already done
        if self._state.submitted or self._state.step_count >= MAX_STEPS:
            return self._make_obs(done=True, reward=0.0,
                                  error="Episode is already complete.")

        self._state.step_count += 1
        analysis = self._page_cache[self._state.page_id]

        if action.action_type == "set_viewport":
            return self._handle_set_viewport(action, analysis)
        elif action.action_type == "inspect_element":
            return self._handle_inspect_element(action, analysis)
        elif action.action_type == "run_check":
            return self._handle_run_check(action, analysis)
        elif action.action_type == "submit_report":
            return self._handle_submit_report(action)
        else:
            # Should never reach here due to Literal validation, but be defensive
            return self._make_obs(error=f"Unknown action_type: {action.action_type}")

    @property
    def state(self) -> DalaalState:
        return self._state

    # ... private _handle_* methods below ...
```

**The `_handle_submit_report` method is the most important:**
```python
    def _handle_submit_report(self, action: DalaalAction) -> DalaalObservation:
        if action.identified_issues is None:
            return self._make_obs(error="submit_report requires identified_issues.")

        self._state.submitted = True
        gt = self._ground_truth[self._state.page_id]

        reward, breakdown = compute_reward(
            identified=action.identified_issues,
            severity=action.severity_scores,
            ground_truth=gt,
            viewports_tested=self._state.viewports_tested,
            checks_run=self._state.checks_run,
            step_count=self._state.step_count,
            max_steps=MAX_STEPS,
        )

        return DalaalObservation(
            done=True,
            reward=reward,
            page_id=self._state.page_id,
            current_viewport_width=self._state.current_viewport_width,
            viewports_tested=self._state.viewports_tested,
            checks_run=self._state.checks_run,
            step_budget_remaining=max(0, MAX_STEPS - self._state.step_count),
            page_summary="",
            final_scores=breakdown,
        )
```

---

## 8. Client — `client.py`

```python
from openenv.core.env_client import EnvClient
from openenv.core.client_types import StepResult
from .models import DalaalAction, DalaalObservation, DalaalState

class DalaalEnv(EnvClient[DalaalAction, DalaalObservation, DalaalState]):

    def _step_payload(self, action: DalaalAction) -> dict:
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: dict) -> StepResult[DalaalObservation]:
        obs_data = payload.get("observation", {})
        obs = DalaalObservation(**obs_data)
        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: dict) -> DalaalState:
        return DalaalState(**payload)
```

Note: `action.model_dump(exclude_none=True)` cleanly serializes the flat action to JSON,
dropping any None fields, which is what the server's `DalaalAction` Pydantic model expects.

---

## 9. Server App — `server/app.py`

```python
from openenv.core.env_server.http_server import create_app
try:
    from ..models import DalaalAction, DalaalObservation
    from .environment import DalaalEnvironment
except ImportError:
    from models import DalaalAction, DalaalObservation
    from server.environment import DalaalEnvironment

app = create_app(
    DalaalEnvironment,
    DalaalAction,
    DalaalObservation,
    env_name="dalaal_env",
    max_concurrent_envs=8,   # enough for GRPO parallel rollouts
)

def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    main()
```

---

## 10. Configuration Files

### `openenv.yaml`
```yaml
spec_version: 1
name: dalaal_env
type: space
runtime: fastapi
app: server.app:app
port: 8000
```

### `pyproject.toml`
```toml
[project]
name = "openenv-dalaal-env"
version = "0.1.0"
description = "Website responsiveness audit RL environment for OpenEnv"
readme = "README.md"
requires-python = ">=3.12"
dependencies = [
    "openenv-core>=0.2.3",
    "beautifulsoup4>=4.12.0",
    "tinycss2>=1.3.0",
    "lxml>=5.0.0",
]

[project.scripts]
server = "dalaal_env.server.app:main"

[tool.setuptools.packages.find]
where = ["."]
```

### `server/Dockerfile`
```dockerfile
FROM python:3.12-slim
WORKDIR /app
RUN pip install uv
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev
COPY . .
EXPOSE 8000
CMD ["uv", "run", "server"]
```

---

## 11. Dependencies Analysis

| Package | Why | Notes |
|---------|-----|-------|
| `openenv-core>=0.2.3` | Framework base classes + FastAPI + uvicorn + WebSocket | Already in pyproject.toml |
| `beautifulsoup4>=4.12.0` | HTML parsing, element tree, attribute extraction | Fast, pure Python |
| `tinycss2>=1.3.0` | CSS tokenization and rule parsing | Lightweight (no browser needed) |
| `lxml>=5.0.0` | Optional but recommended: faster BS4 HTML parser | Add as `beautifulsoup4` extra |

**What we explicitly do NOT need:**
- `playwright` / `selenium` / `puppeteer` — too heavy for HF Spaces, requires browser binary
- `requests` / `httpx` — pages are loaded from disk, not fetched over HTTP
- `numpy` / `scipy` — all math is pure Python (F1, Pearson correlation are simple formulas)
- `cssselect` — BS4's built-in selector support is sufficient for our element lookup

---

## 12. What Will Impress Meta Engineers in Code Review

### Code quality signals to include:
1. **Full type annotations everywhere** — `from __future__ import annotations`, every method typed.
2. **`SUPPORTS_CONCURRENT_SESSIONS = True`** — shows understanding of GRPO's parallel rollout needs.
3. **Episode result caching** — the `_page_cache` avoids re-parsing identical HTML across episodes.
4. **Ground truth pre-computation** — scoring doesn't re-run checks at submit time.
5. **Graceful error handling in `step()`** — invalid actions return a descriptive error observation
   rather than raising an exception that would crash a training rollout.
6. **Deterministic seeding** — `reset(seed=42)` always picks the same page, enabling reproducible
   eval and the `openenv validate` test suite.
7. **Scored severity** — the optional `severity_scores` field in submit_report shows the design
   anticipates nuanced LLM output, not just binary classifications.
8. **Ground truth not leaked** — the server never includes `ground_truth` in any observation;
   it's server-side only.
9. **Dataset includes all 3 tiers** — fully responsive pages (reward=1.0 for empty report) are
   important; they test that the agent doesn't hallucinate issues.
10. **Clean `final_scores` breakdown** — the agent (and the human evaluator) can see exactly
    where reward came from: `{precision, recall, f1, coverage_bonus, efficiency_bonus, total}`.

### What makes this a *good* RL environment:
- **Non-trivial reward** — you cannot get F1=1 without running checks or inspecting elements.
- **Exploration has value** — testing more viewports increases coverage_bonus AND gives more
  evidence for accurate issue identification.
- **Efficiency matters** — submitting early without gathering evidence gives low F1.
- **LLM-native action space** — the flat JSON action with a `Literal` discriminator is naturally
  expressible in structured-output mode, but is also parseable from free-text JSON generation.
- **Graded difficulty** — the manifest has easy/medium/hard pages, enabling curriculum learning.

---

## 13. Edge Cases and Failure Modes

| Failure Mode | Mitigation |
|---|---|
| LLM generates invalid `action_type` | Pydantic `Literal` validation rejects it; `step()` returns error obs, episode continues. |
| LLM provides invalid CSS selector | `get_element_css()` returns `None`; obs has `error` field; no exception. |
| LLM submits same issue twice in list | `identified_issues` is deduplicated in scorer before F1 computation. |
| HTML page has no CSS at all | PageParser returns empty stylesheet list; all CSS-dependent checks return `passed=False`; ground truth reflects this accurately. |
| HTML has malformed CSS | `tinycss2` is error-tolerant by design — it skips invalid tokens and continues. |
| LLM submits with viewport_width=0 | Pydantic `ge=320` constraint rejects it at the Action model level. |
| Agent hits step limit without submitting | Forced submit in `step()` after increment; uses `checks_run` as `identified_issues`. |
| Concurrent sessions share state | `SUPPORTS_CONCURRENT_SESSIONS=True` means the server creates a fresh `DalaalEnvironment` instance per WebSocket session. The page cache can be a class-level dict with a threading.Lock for safe concurrent access. |
| Empty GT (fully responsive page) | F1 formula: if GT empty and P empty → F1=1. If GT empty and P non-empty → Precision=0, F1=0. Both cases handled correctly. |

---

## 14. Implementation Order (3-Day Plan)

### Day 1 — Core logic (no server yet)

**Priority: Get the analysis engine working and tested in isolation.**

1. `dalaal_env/analyzer/parser.py` — `PageParser` and `PageAnalysis` with minimal CSS resolution.
2. `dalaal_env/analyzer/checks.py` — implement all 10 checks (start with simple ones:
   viewport_meta, media_queries, responsive_images — these are pure string matching).
3. `dalaal_env/analyzer/scorer.py` — `compute_ground_truth()` and `compute_reward()`.
4. `dalaal_env/data/pages/` — write 5 HTML pages (one per tier), test the scorer on them.
5. `dalaal_env/data/manifest.json` — fill in ground_truth_issues by running the scorer.
6. Quick manual test: `python -c "from dalaal_env.analyzer.parser import PageParser; ..."`

### Day 2 — Models, server, client

**Priority: Get the OpenEnv interface working.**

1. `dalaal_env/models.py` — exact Pydantic classes as designed above.
2. `dalaal_env/server/environment.py` — all action handlers.
3. `dalaal_env/server/app.py` — create_app wiring.
4. `dalaal_env/client.py` — EnvClient subclass.
5. `dalaal_env/__init__.py` — exports.
6. `pyproject.toml` updates + `uv lock`.
7. Run `openenv validate` — fix any structural issues.
8. Run server locally: `openenv serve`. Test all endpoints with curl.
9. Run `openenv validate --url http://localhost:8000` — all criteria must pass.

### Day 3 — Dataset, polish, deploy

**Priority: Expand dataset to 30 pages, deploy to HF Spaces.**

1. Write remaining 25 HTML pages (use archetypes + variants).
2. Re-run scorer to fill manifest.json ground truth.
3. Write `server/Dockerfile` and test Docker build.
4. Write README with: description, action space table, observation schema, reward formula,
   example episode transcript (critical for LLM scoring in Round 1).
5. `openenv push --repo-id <username>/dalaal-env`
6. Run `openenv validate --url https://<username>-dalaal-env.hf.space` — final check.

---

## 15. README Content for LLM Scoring (Round 1)

The Round 1 automated judge uses an LLM to score the environment's README. Include:

- **Clear problem statement**: "DalaalEnv is an RL environment where an agent audits web pages
  for responsiveness issues across different screen sizes."
- **Action space table** with all 4 action types and their parameters.
- **Observation description** listing all fields.
- **Reward formula** written out explicitly.
- **Example episode** — a 5-step JSON trace showing reset → set_viewport → run_check →
  inspect_element → submit_report with actual rewards.
- **Motivation** — why responsiveness auditing is a good RL problem (exploration, delayed reward,
  real-world applicability).

---

## Summary

| Component | Key Class / File | Status |
|---|---|---|
| Action model | `DalaalAction` in `models.py` | Designed |
| Observation model | `DalaalObservation` in `models.py` | Designed |
| State model | `DalaalState` in `models.py` | Designed |
| HTML/CSS parser | `PageParser` in `analyzer/parser.py` | To build |
| 10 responsiveness checks | `analyzer/checks.py` | To build |
| Reward computation | `analyzer/scorer.py` | Designed |
| Environment logic | `DalaalEnvironment` in `server/environment.py` | Designed |
| HTTP server | `server/app.py` | Designed |
| Client | `DalaalEnv` in `client.py` | Designed |
| Dataset | `data/pages/` + `data/manifest.json` | To build |
| Config files | `openenv.yaml`, `pyproject.toml`, `Dockerfile` | Designed |
