"""
Accessibility tree extraction from Playwright pages using Chrome DevTools Protocol.

Converts the browser DOM into a numbered text representation that LLM agents
can reason about and reference by element ID.

Example output:
    [1] heading "My Todo List"
    [2] textbox "Add a new todo..." value=""
    [3] button "Add"
    [4] checkbox "Buy groceries" checked=false
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from playwright.async_api import Page


SKIP_ROLES = frozenset({
    "none", "generic", "RootWebArea", "LineBreak",
    "InlineTextBox", "StaticText", "paragraph",
    "MenuListPopup", "group",
})

INTERACTIVE_ROLES = frozenset({
    "button", "link", "textbox", "checkbox", "radio",
    "combobox", "searchbox", "option", "tab", "menuitem",
    "switch", "slider", "spinbutton",
})


@dataclass
class AccessibilityNode:
    """A node in the parsed accessibility tree."""
    id: int
    role: str
    name: str
    value: Optional[str] = None
    checked: Optional[str] = None
    selected: Optional[bool] = None
    expanded: Optional[bool] = None
    disabled: Optional[bool] = None
    focused: Optional[bool] = None
    level: Optional[int] = None


class AccessibilityTree:
    """Manages accessibility tree extraction and element ID mapping."""

    def __init__(self):
        self._nodes: dict[int, AccessibilityNode] = {}
        self._counter = 0

    def clear(self):
        self._nodes.clear()
        self._counter = 0

    def get_node(self, element_id: int) -> Optional[AccessibilityNode]:
        return self._nodes.get(element_id)

    async def extract(self, page: Page) -> str:
        """Extract accessibility tree from page via CDP and return text representation."""
        self.clear()

        cdp = await page.context.new_cdp_session(page)
        try:
            result = await cdp.send("Accessibility.getFullAXTree")
        finally:
            await cdp.detach()

        raw_nodes = result.get("nodes", [])
        lines = []

        for raw in raw_nodes:
            role = raw.get("role", {}).get("value", "")
            if role in SKIP_ROLES:
                continue

            name = raw.get("name", {}).get("value", "")
            props = {}
            for p in raw.get("properties", []):
                val = p.get("value", {})
                if "value" in val:
                    props[p["name"]] = val["value"]

            # Skip nodes with no name and non-interactive roles
            if not name and role not in INTERACTIVE_ROLES:
                continue

            self._counter += 1
            node = AccessibilityNode(
                id=self._counter,
                role=role,
                name=name,
                value=props.get("value"),
                checked=props.get("checked"),
                selected=props.get("selected"),
                expanded=props.get("expanded"),
                disabled=props.get("disabled"),
                focused=props.get("focused"),
                level=props.get("level"),
            )
            self._nodes[node.id] = node
            lines.append(self._render_node(node))

        return "\n".join(lines) if lines else "[empty page]"

    def _render_node(self, node: AccessibilityNode) -> str:
        """Render a single node as text."""
        parts = [f"[{node.id}] {node.role}"]

        if node.name:
            parts.append(f'"{node.name}"')
        if node.value is not None:
            parts.append(f'value="{node.value}"')
        if node.checked is not None:
            parts.append(f"checked={node.checked}")
        if node.selected is not None:
            parts.append(f"selected={str(node.selected).lower()}")
        if node.expanded is not None:
            parts.append(f"expanded={str(node.expanded).lower()}")
        if node.disabled is not None and node.disabled:
            parts.append("disabled")
        if node.focused is not None and node.focused:
            parts.append("focused")
        if node.level is not None:
            parts.append(f"level={node.level}")

        return " ".join(parts)
