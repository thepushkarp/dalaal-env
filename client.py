"""Dalaal Browser-Use Environment Client."""

from typing import Any, Dict, Optional

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import DalaalEnvAction, DalaalEnvObservation


class DalaalEnvEnv(EnvClient[DalaalEnvAction, DalaalEnvObservation, State]):
    """
    Client for the Dalaal Browser-Use Environment.

    This client maintains a persistent WebSocket connection to the environment
    server, enabling efficient multi-step browser interactions.

    Example:
        >>> async with DalaalEnvEnv(base_url="http://localhost:8000") as env:
        ...     result = await env.reset(task="todo_add")
        ...     print(result.observation.accessibility_tree)
        ...     result = await env.step(DalaalEnvAction(action_type="click", element_id=3))

    Example with Docker:
        >>> env = await DalaalEnvEnv.from_docker_image("dalaal-env:latest")
        >>> result = await env.reset(task="login")
    """

    def _step_payload(self, action: DalaalEnvAction) -> Dict[str, Any]:
        """Convert DalaalEnvAction to JSON payload."""
        payload: Dict[str, Any] = {"action_type": action.action_type}
        if action.element_id is not None:
            payload["element_id"] = action.element_id
        if action.text is not None:
            payload["text"] = action.text
        if action.key is not None:
            payload["key"] = action.key
        if action.direction is not None:
            payload["direction"] = action.direction
        return payload

    def _parse_result(self, payload: Dict) -> StepResult[DalaalEnvObservation]:
        """Parse server response into StepResult."""
        obs_data = payload.get("observation", {})
        observation = DalaalEnvObservation(
            url=obs_data.get("url", ""),
            title=obs_data.get("title", ""),
            accessibility_tree=obs_data.get("accessibility_tree", ""),
            task_description=obs_data.get("task_description", ""),
            last_action_error=obs_data.get("last_action_error"),
            step_count=obs_data.get("step_count", 0),
            max_steps=obs_data.get("max_steps", 20),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )
        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """Parse server response into State."""
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
