"""Pydantic models for DalaalEnv v2.

Defines the Action, Observation, and State types for the
website responsiveness audit RL environment.

v2 changes: removed run_check oracle, added list_elements and
get_page_info actions, richer submit_report with evidence.
"""

from __future__ import annotations

from typing import Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import Field


ActionType = Literal[
    "set_viewport",
    "list_elements",
    "inspect_element",
    "get_page_info",
    "submit_report",
]

ElementFilter = Literal["interactive", "images", "tables", "layout"]

OverallAssessment = Literal["responsive", "partially_responsive", "non_responsive"]


class DalaalAction(Action):
    """One action the agent takes during a responsiveness audit.

    Flat structure: which fields are used depends on action_type.
    """

    action_type: ActionType = Field(
        ...,
        description=(
            "set_viewport: change simulated viewport width. "
            "list_elements: discover page elements with metadata. "
            "inspect_element: get CSS properties of an element. "
            "get_page_info: get page-level summary and structure. "
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

    # list_elements
    element_filter: Optional[ElementFilter] = Field(
        default=None,
        description=(
            "Filter for list_elements: 'interactive' (links/buttons), "
            "'images', 'tables', 'layout' (containers). None = all elements."
        ),
    )

    # inspect_element
    css_selector: Optional[str] = Field(
        default=None,
        max_length=256,
        description="CSS selector from list_elements. Required for inspect_element.",
    )

    # submit_report
    identified_issues: Optional[list[dict]] = Field(
        default=None,
        description=(
            "List of identified responsiveness issues. Each dict should have: "
            "'issue' (str, e.g. 'fixed_width_elements'), "
            "'affected_selectors' (list[str], elements with the issue), "
            "'affected_viewports' (list[int], viewport widths where issue occurs), "
            "'description' (str, explanation of the issue). "
            "Required for submit_report."
        ),
    )
    overall_assessment: Optional[OverallAssessment] = Field(
        default=None,
        description="Overall responsiveness assessment. Required for submit_report.",
    )


class DalaalObservation(Observation):
    """What the agent receives after each action.

    `done` and `reward` are inherited from Observation base.
    """

    page_id: str = Field(default="", description="ID of the HTML page being audited.")
    current_viewport_width: int = Field(default=1280, description="Current viewport width in px.")
    viewports_tested: list[int] = Field(default_factory=list, description="Viewports tested so far.")
    step_budget_remaining: int = Field(default=20, description="Steps remaining before forced termination.")

    page_summary: str = Field(
        default="",
        description="Page summary (populated on reset and get_page_info).",
    )
    elements_list: Optional[list[dict]] = Field(
        default=None,
        description="Element metadata list (populated after list_elements).",
    )
    element_info: Optional[dict] = Field(
        default=None,
        description="CSS properties of inspected element at current viewport.",
    )
    final_scores: Optional[dict] = Field(
        default=None,
        description="Reward breakdown (only when done=True).",
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
    max_steps: int = Field(default=20)
    submitted: bool = Field(default=False)
