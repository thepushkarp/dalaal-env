"""Pydantic models for DalaalEnv.

Defines the Action, Observation, and State types for the
website responsiveness audit RL environment.
"""

from __future__ import annotations

from typing import Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


ActionType = Literal[
    "set_viewport",
    "inspect_element",
    "run_check",
    "submit_report",
]

CheckName = Literal[
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
]


class DalaalAction(Action):
    """One action the agent takes during a responsiveness audit.

    Flat structure: which fields are used depends on action_type.
    """

    action_type: ActionType = Field(
        ...,
        description=(
            "set_viewport: change simulated viewport width. "
            "inspect_element: get CSS properties of an element. "
            "run_check: run a named responsiveness check. "
            "submit_report: end episode with audit findings."
        ),
    )

    # set_viewport
    viewport_width: Optional[int] = Field(
        default=None,
        ge=320,
        le=2560,
        description="Viewport width in px (320-2560). Required for set_viewport.",
    )

    # inspect_element
    css_selector: Optional[str] = Field(
        default=None,
        max_length=256,
        description="CSS selector of element to inspect. Required for inspect_element.",
    )

    # run_check
    check_name: Optional[CheckName] = Field(
        default=None,
        description="Which responsiveness check to run. Required for run_check.",
    )

    # submit_report
    identified_issues: Optional[list[str]] = Field(
        default=None,
        description="List of check names the agent believes are failing. Required for submit_report.",
    )


class DalaalObservation(Observation):
    """What the agent receives after each action.

    `done` and `reward` are inherited from Observation base.
    """

    page_id: str = Field(default="", description="ID of the HTML page being audited.")
    current_viewport_width: int = Field(default=1280, description="Current viewport width in px.")
    viewports_tested: list[int] = Field(default_factory=list, description="Viewports tested so far.")
    checks_run: list[str] = Field(default_factory=list, description="Checks explicitly run so far.")
    step_budget_remaining: int = Field(default=20, description="Steps remaining before forced termination.")
    page_summary: str = Field(default="", description="Human-readable page summary.")

    element_info: Optional[dict] = Field(
        default=None,
        description="CSS properties of the last inspected element at current viewport.",
    )
    check_result: Optional[dict] = Field(
        default=None,
        description="Result of the last run_check: check_name, passed, detail, evidence.",
    )
    final_scores: Optional[dict] = Field(
        default=None,
        description="Reward breakdown (only when done=True): precision, recall, f1, bonuses, total.",
    )
    error: Optional[str] = Field(
        default=None,
        description="Error message if the action was invalid.",
    )


class DalaalState(State):
    """Episode metadata available via GET /state.

    episode_id and step_count are inherited from State base.
    """

    page_id: str = Field(default="")
    current_viewport_width: int = Field(default=1280)
    viewports_tested: list[int] = Field(default_factory=list)
    checks_run: list[str] = Field(default_factory=list)
    max_steps: int = Field(default=20)
    submitted: bool = Field(default=False)
