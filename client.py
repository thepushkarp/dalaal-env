"""DalaalEnv WebSocket client for training and evaluation."""

from __future__ import annotations

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult

from dalaal_env.models import DalaalAction, DalaalObservation, DalaalState


class DalaalEnv(
    EnvClient[DalaalAction, DalaalObservation, DalaalState],
):
    """Client for the DalaalEnv responsiveness audit environment.

    Connects via WebSocket for persistent episode sessions.

    Example:
        >>> with DalaalEnv(base_url="http://localhost:8000").sync() as env:
        ...     result = env.reset()
        ...     print(result.observation.page_summary)
        ...
        ...     result = env.step(DalaalAction(
        ...         action_type="run_check", check_name="viewport_meta",
        ...     ))
        ...     print(result.observation.check_result)
    """

    def _step_payload(self, action: DalaalAction) -> Dict:
        """Serialize action to JSON payload."""
        return action.model_dump(exclude_none=True)

    def _parse_result(
        self, payload: Dict,
    ) -> StepResult[DalaalObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})
        obs = DalaalObservation(
            page_id=obs_data.get("page_id", ""),
            current_viewport_width=obs_data.get("current_viewport_width", 1280),
            viewports_tested=obs_data.get("viewports_tested", []),
            checks_run=obs_data.get("checks_run", []),
            step_budget_remaining=obs_data.get("step_budget_remaining", 0),
            page_summary=obs_data.get("page_summary", ""),
            element_info=obs_data.get("element_info"),
            check_result=obs_data.get("check_result"),
            final_scores=obs_data.get("final_scores"),
            error=obs_data.get("error"),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=obs,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> DalaalState:
        """Parse state response."""
        return DalaalState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            page_id=payload.get("page_id", ""),
            current_viewport_width=payload.get("current_viewport_width", 1280),
            viewports_tested=payload.get("viewports_tested", []),
            checks_run=payload.get("checks_run", []),
            max_steps=payload.get("max_steps", 20),
            submitted=payload.get("submitted", False),
        )
