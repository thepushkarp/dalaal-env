"""DalaalEnv — Website Responsiveness Audit RL Environment.

The agent audits web pages for responsiveness issues across viewports.
Actions: set_viewport, inspect_element, run_check, submit_report.
Reward: F1 score of identified vs ground truth issues + bonuses.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

from dalaal_env.analyzer.checks import ALL_CHECK_NAMES, run_check
from dalaal_env.analyzer.parser import PageAnalysis, PageParser
from dalaal_env.analyzer.scorer import compute_ground_truth, compute_reward
from dalaal_env.models import DalaalAction, DalaalObservation, DalaalState

DATA_DIR = Path(__file__).parent.parent / "data"
MAX_STEPS = 20


class DalaalEnvironment(
    Environment[DalaalAction, DalaalObservation, DalaalState],
):
    """RL environment for website responsiveness auditing.

    Each episode: agent receives a web page, explores it across
    viewports, runs checks, and submits a responsiveness report.
    Reward is based on accuracy of identified issues vs ground truth.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self) -> None:
        self._state = DalaalState()
        self._manifest = self._load_manifest()
        self._parser = PageParser()
        self._page_cache: dict[str, PageAnalysis] = {}
        self._gt_cache: dict[str, dict] = {}
        self._current_analysis: PageAnalysis | None = None
        self._current_gt: dict = {}

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

    def _get_ground_truth(self, page_id: str, analysis: PageAnalysis) -> dict:
        if page_id not in self._gt_cache:
            self._gt_cache[page_id] = compute_ground_truth(analysis)
        return self._gt_cache[page_id]

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
        gt = self._get_ground_truth(page_meta["id"], analysis)

        self._current_analysis = analysis
        self._current_gt = gt

        self._state = DalaalState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            page_id=page_meta["id"],
            current_viewport_width=1280,
            viewports_tested=[1280],
            checks_run=[],
            checks_failed=[],
            max_steps=MAX_STEPS,
            submitted=False,
        )

        return DalaalObservation(
            done=False,
            reward=0.0,
            page_id=page_meta["id"],
            current_viewport_width=1280,
            viewports_tested=[1280],
            checks_run=[],
            step_budget_remaining=MAX_STEPS,
            page_summary=analysis.summary(),
        )

    def step(
        self,
        action: DalaalAction,
        timeout_s: float | None = None,
        **kwargs: object,
    ) -> DalaalObservation:
        """Process one agent action."""
        if self._state.submitted or self._state.step_count >= MAX_STEPS:
            return self._obs(done=True, reward=0.0, error="Episode already complete.")

        self._state.step_count += 1
        analysis = self._current_analysis

        if analysis is None:
            return self._obs(error="No page loaded. Call reset() first.")

        if action.action_type == "set_viewport":
            return self._handle_set_viewport(action)
        if action.action_type == "inspect_element":
            return self._handle_inspect_element(action, analysis)
        if action.action_type == "run_check":
            return self._handle_run_check(action, analysis)
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

    def _handle_inspect_element(
        self, action: DalaalAction, analysis: PageAnalysis,
    ) -> DalaalObservation:
        if not action.css_selector:
            return self._obs(error="inspect_element requires css_selector.")

        vw = self._state.current_viewport_width
        css = analysis.get_element_css(action.css_selector, vw)

        if css is None:
            return self._obs(
                error=f"Selector '{action.css_selector}' matched no elements.",
            )

        return self._obs(element_info={
            "selector": action.css_selector,
            "viewport_width": vw,
            "computed_styles": css,
        })

    def _handle_run_check(
        self, action: DalaalAction, analysis: PageAnalysis,
    ) -> DalaalObservation:
        if not action.check_name:
            return self._obs(
                error=f"run_check requires check_name. Valid: {ALL_CHECK_NAMES}",
            )

        vw = self._state.current_viewport_width
        try:
            result = run_check(action.check_name, analysis, vw)
        except ValueError as e:
            return self._obs(error=str(e))

        if action.check_name not in self._state.checks_run:
            self._state.checks_run.append(action.check_name)
        if not result.passed and action.check_name not in self._state.checks_failed:
            self._state.checks_failed.append(action.check_name)

        return self._obs(check_result={
            "check_name": result.check_name,
            "passed": result.passed,
            "detail": result.detail,
            "evidence": result.evidence,
        })

    def _handle_submit_report(self, action: DalaalAction) -> DalaalObservation:
        if action.identified_issues is None:
            return self._obs(error="submit_report requires identified_issues list.")

        self._state.submitted = True

        reward, breakdown = compute_reward(
            identified=action.identified_issues,
            ground_truth=self._current_gt,
            viewports_tested=self._state.viewports_tested,
            checks_run=self._state.checks_run,
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
        element_info: dict | None = None,
        check_result: dict | None = None,
        final_scores: dict | None = None,
    ) -> DalaalObservation:
        """Build an observation from current state."""
        # Check if step limit reached without submit
        budget = max(0, MAX_STEPS - self._state.step_count)
        if budget == 0 and not self._state.submitted:
            self._state.submitted = True
            reward_val, breakdown = compute_reward(
                identified=self._state.checks_failed,
                ground_truth=self._current_gt,
                viewports_tested=self._state.viewports_tested,
                checks_run=self._state.checks_run,
                step_count=self._state.step_count,
                max_steps=MAX_STEPS,
            )
            return DalaalObservation(
                done=True,
                reward=reward_val,
                page_id=self._state.page_id,
                current_viewport_width=self._state.current_viewport_width,
                viewports_tested=list(self._state.viewports_tested),
                checks_run=list(self._state.checks_run),
                step_budget_remaining=0,
                page_summary="",
                final_scores=breakdown,
                error="Step limit reached. Auto-submitted using failed checks only.",
            )

        return DalaalObservation(
            done=done,
            reward=reward,
            page_id=self._state.page_id,
            current_viewport_width=self._state.current_viewport_width,
            viewports_tested=list(self._state.viewports_tested),
            checks_run=list(self._state.checks_run),
            step_budget_remaining=budget,
            page_summary="" if self._state.step_count > 0 else "",
            element_info=element_info,
            check_result=check_result,
            final_scores=final_scores,
            error=error,
        )
