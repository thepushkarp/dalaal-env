"""DalaalEnv v2 — Website Responsiveness Audit RL Environment.

The agent audits web pages for responsiveness issues by exploring
the page structure and CSS properties across viewport sizes.

v2: removed run_check oracle, added list_elements + get_page_info,
    richer submit_report with evidence, zero-reward timeout.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from dalaal_env.analyzer.parser import PageAnalysis, PageParser
from dalaal_env.analyzer.scorer import (
    compute_ground_truth,
    compute_ground_truth_by_viewport,
    compute_reward,
)
from dalaal_env.models import DalaalAction, DalaalObservation, DalaalState

DATA_DIR = Path(__file__).parent.parent / "data"
MAX_STEPS = 20


class DalaalEnvironment(
    Environment[DalaalAction, DalaalObservation, DalaalState],
):
    """RL environment for website responsiveness auditing.

    Each episode: agent receives a web page, discovers its elements,
    inspects CSS properties across viewports, and submits a
    responsiveness audit report. Reward is based on accuracy of
    identified issues vs ground truth, plus evidence quality.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = DalaalState()
        self._manifest = self._load_manifest()
        self._parser = PageParser()
        self._page_cache: dict[str, PageAnalysis] = {}
        self._gt_cache: dict[str, dict] = {}
        self._gt_viewports_cache: dict[str, dict] = {}
        self._current_analysis: PageAnalysis | None = None
        self._current_gt: dict = {}
        self._current_gt_viewports: dict = {}

    def _load_manifest(self) -> dict:
        manifest_path = DATA_DIR / "manifest.json"
        with manifest_path.open() as f:
            return json.load(f)

    def _load_page(self, page_meta: dict) -> PageAnalysis:
        page_id = page_meta["id"]
        if page_id not in self._page_cache:
            html_path = DATA_DIR / "pages" / page_meta["filename"]
            html = html_path.read_text()
            self._page_cache[page_id] = self._parser.parse(html, page_id)
        return self._page_cache[page_id]

    def _get_ground_truth(self, page_id: str, analysis: PageAnalysis) -> tuple:
        if page_id not in self._gt_cache:
            self._gt_cache[page_id] = compute_ground_truth(analysis)
            self._gt_viewports_cache[page_id] = compute_ground_truth_by_viewport(analysis)
        return self._gt_cache[page_id], self._gt_viewports_cache[page_id]

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs: object,
    ) -> DalaalObservation:
        """Start a new audit episode with a random web page."""
        rng = random.Random(seed)
        page_meta = rng.choice(self._manifest["pages"])

        analysis = self._load_page(page_meta)
        gt, gt_vp = self._get_ground_truth(page_meta["id"], analysis)

        self._current_analysis = analysis
        self._current_gt = gt
        self._current_gt_viewports = gt_vp

        self._state = DalaalState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            page_id=page_meta["id"],
            current_viewport_width=1280,
            viewports_tested=[1280],
            max_steps=MAX_STEPS,
            submitted=False,
        )

        return self._obs(page_summary=analysis.summary())

    def step(
        self,
        action: DalaalAction,
        timeout_s: float | None = None,
        **kwargs: object,
    ) -> DalaalObservation:
        """Process one agent action."""
        if self._state.submitted:
            return self._obs(done=True, error="Episode already complete.")

        self._state.step_count += 1
        analysis = self._current_analysis

        if analysis is None:
            return self._obs(error="No page loaded. Call reset() first.")

        # Check step budget AFTER incrementing
        if self._state.step_count > MAX_STEPS:
            self._state.submitted = True
            return self._obs(
                done=True,
                reward=0.0,
                error="Step limit reached without submit.",
            )

        if action.action_type == "set_viewport":
            return self._handle_set_viewport(action)
        if action.action_type == "list_elements":
            return self._handle_list_elements(action, analysis)
        if action.action_type == "inspect_element":
            return self._handle_inspect_element(action, analysis)
        if action.action_type == "get_page_info":
            return self._handle_get_page_info(analysis)
        if action.action_type == "submit_report":
            return self._handle_submit_report(action)

        return self._obs(error=f"Unknown action_type: {action.action_type}")

    @property
    def state(self) -> DalaalState:
        return self._state

    # --- Action handlers ---

    def _handle_set_viewport(self, action: DalaalAction) -> DalaalObservation:
        if action.viewport_width is None:
            return self._obs(error="set_viewport requires viewport_width (320-2560).")

        vw = action.viewport_width
        self._state.current_viewport_width = vw
        if vw not in self._state.viewports_tested:
            self._state.viewports_tested.append(vw)

        return self._obs()

    def _handle_list_elements(
        self, action: DalaalAction, analysis: PageAnalysis,
    ) -> DalaalObservation:
        elements = analysis.list_elements(
            element_filter=action.element_filter,
            limit=50,
        )
        return self._obs(elements_list=elements)

    def _handle_inspect_element(
        self, action: DalaalAction, analysis: PageAnalysis,
    ) -> DalaalObservation:
        if not action.css_selector:
            return self._obs(error="inspect_element requires css_selector.")

        vw = self._state.current_viewport_width
        css = analysis.get_element_css(action.css_selector, vw)

        if css is None:
            return self._obs(
                error=f"Selector '{action.css_selector}' matched no elements. Use list_elements to discover valid selectors.",
            )

        return self._obs(element_info={
            "selector": action.css_selector,
            "viewport_width": vw,
            "computed_styles": css,
        })

    def _handle_get_page_info(
        self, analysis: PageAnalysis,
    ) -> DalaalObservation:
        return self._obs(page_summary=analysis.summary())

    def _handle_submit_report(self, action: DalaalAction) -> DalaalObservation:
        if action.identified_issues is None:
            return self._obs(error="submit_report requires identified_issues list.")

        self._state.submitted = True

        reward, breakdown = compute_reward(
            identified=action.identified_issues,
            ground_truth=self._current_gt,
            gt_viewports=self._current_gt_viewports,
            viewports_tested=self._state.viewports_tested,
            step_count=self._state.step_count,
            max_steps=MAX_STEPS,
        )

        return self._obs(done=True, reward=reward, final_scores=breakdown)

    # --- Observation builder ---

    def _obs(
        self,
        done: bool = False,
        reward: float = 0.0,
        error: str | None = None,
        page_summary: str = "",
        elements_list: list[dict] | None = None,
        element_info: dict | None = None,
        final_scores: dict | None = None,
    ) -> DalaalObservation:
        """Build an observation from current state."""
        budget = max(0, MAX_STEPS - self._state.step_count)

        return DalaalObservation(
            done=done,
            reward=reward,
            page_id=self._state.page_id,
            current_viewport_width=self._state.current_viewport_width,
            viewports_tested=list(self._state.viewports_tested),
            step_budget_remaining=budget,
            page_summary=page_summary,
            elements_list=elements_list,
            element_info=element_info,
            final_scores=final_scores,
            error=error,
        )
