"""
Data models for the Dalaal Browser-Use Environment.

The dalaal_env environment provides a browser interaction environment
where agents learn to navigate and interact with web pages using
an accessibility tree representation.
"""

from typing import Literal, Optional

from openenv.core.env_server.types import Action, Observation
from pydantic import Field


class DalaalEnvAction(Action):
    """Action for the Browser-Use environment.

    The agent interacts with the browser by specifying an action type
    and relevant parameters. Elements are referenced by their ID in
    the accessibility tree.
    """

    action_type: Literal[
        "click",
        "type",
        "select_option",
        "press_key",
        "scroll",
        "go_back",
        "done",
    ] = Field(..., description="Type of browser action to perform")

    element_id: Optional[int] = Field(
        default=None,
        description="ID of the element in the accessibility tree (for click, type, select_option)",
    )
    text: Optional[str] = Field(
        default=None,
        description="Text to type or option to select",
    )
    key: Optional[str] = Field(
        default=None,
        description="Key to press (e.g. 'Enter', 'Tab', 'Escape')",
    )
    direction: Optional[Literal["up", "down"]] = Field(
        default=None,
        description="Scroll direction",
    )


class DalaalEnvObservation(Observation):
    """Observation from the Browser-Use environment.

    Contains the accessibility tree of the current page, which provides
    a structured text representation of all interactive elements with
    assigned IDs that the agent can reference in actions.
    """

    url: str = Field(default="", description="Current page URL")
    title: str = Field(default="", description="Current page title")
    accessibility_tree: str = Field(
        default="", description="Text accessibility tree with element IDs"
    )
    task_description: str = Field(
        default="", description="Natural language description of the current task"
    )
    last_action_error: Optional[str] = Field(
        default=None, description="Error message from last action, if any"
    )
    step_count: int = Field(default=0, description="Current step number")
    max_steps: int = Field(default=20, description="Maximum steps for this task")
