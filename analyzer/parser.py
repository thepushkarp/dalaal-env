"""HTML/CSS parsing engine for responsiveness analysis.

Parses HTML with BeautifulSoup and CSS with tinycss2 to build a
structured representation of a web page that can be queried at
any viewport width.

This is a CSS-cascade-lite: we resolve only the 8 properties
needed for responsiveness checks, not the full CSSOM.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import tinycss2
from bs4 import BeautifulSoup, Tag


# Properties we care about for responsiveness analysis
TRACKED_PROPERTIES = frozenset({
    "width", "max-width", "min-width",
    "height", "min-height",
    "font-size",
    "display", "position",
    "overflow-x", "overflow",
})


@dataclass
class CSSRule:
    """A single CSS declaration block with its selector and media condition."""

    selector: str
    declarations: dict[str, str]
    media_min_width: float | None = None
    media_max_width: float | None = None
    source_order: int = 0

    def applies_at(self, viewport_width: int) -> bool:
        """Check if this rule applies at the given viewport width."""
        if self.media_min_width is not None and viewport_width < self.media_min_width:
            return False
        if self.media_max_width is not None and viewport_width > self.media_max_width:
            return False
        return True


@dataclass
class ElementInfo:
    """Parsed info about a single HTML element."""

    tag: str
    selector: str
    element_id: str
    classes: list[str]
    inline_styles: dict[str, str]
    children_count: int
    text_length: int
    is_interactive: bool


@dataclass
class MediaQueryInfo:
    """Extracted media query breakpoint."""

    min_width: float | None
    max_width: float | None
    rule_count: int
    raw_query: str


@dataclass
class PageAnalysis:
    """Complete analysis of an HTML page's structure and styles.

    This is the core data structure passed to all check functions.
    Call get_element_css() to resolve properties at a specific viewport.
    """

    page_id: str
    title: str
    html: str
    elements: list[ElementInfo]
    css_rules: list[CSSRule]
    media_queries: list[MediaQueryInfo]
    has_viewport_meta: bool
    viewport_meta_content: str

    # Lazy cache: resolved styles per (selector, viewport_width)
    _style_cache: dict[tuple[str, int], dict[str, str]] = field(
        default_factory=dict, repr=False,
    )

    def summary(self) -> str:
        """Human-readable summary for the agent's initial observation."""
        tag_counts: dict[str, int] = {}
        for el in self.elements:
            tag_counts[el.tag] = tag_counts.get(el.tag, 0) + 1

        top_tags = sorted(tag_counts.items(), key=lambda x: -x[1])[:8]
        tag_str = ", ".join(f"{tag}({n})" for tag, n in top_tags)

        interactive = sum(1 for el in self.elements if el.is_interactive)
        img_count = tag_counts.get("img", 0)
        table_count = tag_counts.get("table", 0)

        return (
            f"Title: {self.title or '(untitled)'}\n"
            f"Elements: {len(self.elements)} total [{tag_str}]\n"
            f"Interactive elements: {interactive}\n"
            f"Images: {img_count}, Tables: {table_count}\n"
            f"CSS rules: {len(self.css_rules)}\n"
            f"Media queries: {len(self.media_queries)}\n"
            f"Viewport meta: {'yes' if self.has_viewport_meta else 'no'}"
        )

    def get_element_css(
        self, selector: str, viewport_width: int,
    ) -> dict[str, str] | None:
        """Resolve tracked CSS properties for a selector at a viewport width.

        Returns dict of {property: value} or None if selector matches nothing.
        Uses cascade-lite: last matching rule wins (no specificity calc).
        Matches by element identity (our unique selector), not re-parsing HTML.
        """
        cache_key = (selector, viewport_width)
        if cache_key in self._style_cache:
            return self._style_cache[cache_key]

        # Find element by our unique selector
        target_el = _find_element_info(self.elements, selector)
        if target_el is None:
            return None

        resolved: dict[str, str] = {}

        # Apply CSS rules in source order (last wins for same property)
        for rule in self.css_rules:
            if not rule.applies_at(viewport_width):
                continue
            if _rule_matches_element(rule.selector, target_el):
                for prop, val in rule.declarations.items():
                    if prop in TRACKED_PROPERTIES:
                        resolved[prop] = val

        # Inline styles override cascade
        for prop, val in target_el.inline_styles.items():
            if prop in TRACKED_PROPERTIES:
                resolved[prop] = val

        self._style_cache[cache_key] = resolved
        return resolved

    def get_all_rules_for_property(self, prop: str) -> list[tuple[str, str, CSSRule]]:
        """Get all rules that set a specific property. Returns (selector, value, rule)."""
        results = []
        for rule in self.css_rules:
            if prop in rule.declarations:
                results.append((rule.selector, rule.declarations[prop], rule))
        return results

    def get_selectors_for_tag(self, tag: str) -> list[str]:
        """Get selectors of all elements with a given tag."""
        return [el.selector for el in self.elements if el.tag == tag]


def _find_element_info(
    elements: list[ElementInfo], selector: str,
) -> ElementInfo | None:
    """Find an ElementInfo by its unique selector."""
    for el in elements:
        if el.selector == selector:
            return el
    return None


# Simple pattern to split a CSS selector into tag, id, classes
_SELECTOR_PART_RE = re.compile(r"([a-zA-Z][a-zA-Z0-9]*)?(?:#([a-zA-Z0-9_-]+))?((?:\.[a-zA-Z0-9_-]+)*)")


def _rule_matches_element(css_selector: str, element: ElementInfo) -> bool:
    """Check if a CSS rule selector matches an element.

    Handles simple selectors: tag, .class, #id, tag.class, tag#id.
    For compound selectors (descendant, child combinators), matches
    only on the rightmost segment as an approximation.
    """
    # Use the rightmost segment for descendant/child selectors
    parts = css_selector.strip().split()
    rightmost = parts[-1] if parts else css_selector.strip()

    # Remove pseudo-classes/elements for matching
    rightmost = rightmost.split(":")[0]

    m = _SELECTOR_PART_RE.match(rightmost)
    if not m:
        return False

    tag_part = m.group(1)
    id_part = m.group(2)
    class_str = m.group(3)

    # Tag must match (if specified)
    if tag_part and tag_part != element.tag:
        return False

    # ID must match (if specified)
    if id_part and id_part != element.element_id:
        return False

    # All classes in the selector must be present on the element
    if class_str:
        required_classes = [c for c in class_str.split(".") if c]
        for cls in required_classes:
            if cls not in element.classes:
                return False

    return True


# --- Media query parsing ---

_MEDIA_WIDTH_RE = re.compile(
    r"\(\s*(min|max)-width\s*:\s*(\d+(?:\.\d+)?)\s*(px|em|rem)\s*\)",
)


def _parse_media_condition(prelude_text: str) -> tuple[float | None, float | None]:
    """Extract min-width and max-width from a media query prelude.

    Only handles px values (em/rem converted assuming 16px base).
    Returns (min_width_px, max_width_px).
    """
    min_w: float | None = None
    max_w: float | None = None

    for match in _MEDIA_WIDTH_RE.finditer(prelude_text):
        bound = match.group(1)  # "min" or "max"
        value = float(match.group(2))
        unit = match.group(3)

        if unit in ("em", "rem"):
            value *= 16.0

        if bound == "min":
            min_w = value
        else:
            max_w = value

    return min_w, max_w


# --- CSS parsing ---

def _parse_declarations(content: Any) -> dict[str, str]:
    """Parse CSS declarations from a rule's content block.

    Only keeps properties we track.
    """
    decls: dict[str, str] = {}
    try:
        parsed = tinycss2.parse_blocks_contents(
            content, skip_whitespace=True, skip_comments=True,
        )
    except Exception:
        return decls

    for item in parsed:
        if item.type == "declaration" and item.name in TRACKED_PROPERTIES:
            decls[item.name] = tinycss2.serialize(item.value).strip()

    return decls


def _parse_css(css_text: str) -> tuple[list[CSSRule], list[MediaQueryInfo]]:
    """Parse a CSS stylesheet string into CSSRules and MediaQueryInfo."""
    rules: list[CSSRule] = []
    media_queries: list[MediaQueryInfo] = []
    order = 0

    try:
        parsed_rules = tinycss2.parse_stylesheet(
            css_text, skip_whitespace=True, skip_comments=True,
        )
    except Exception:
        return rules, media_queries

    for rule in parsed_rules:
        if rule.type == "qualified-rule":
            selector = tinycss2.serialize(rule.prelude).strip()
            decls = _parse_declarations(rule.content)
            if decls:
                rules.append(CSSRule(
                    selector=selector,
                    declarations=decls,
                    source_order=order,
                ))
                order += 1

        elif rule.type == "at-rule" and rule.lower_at_keyword == "media":
            prelude_text = tinycss2.serialize(rule.prelude).strip()
            min_w, max_w = _parse_media_condition(prelude_text)

            nested_count = 0
            if rule.content is not None:
                try:
                    nested_rules = tinycss2.parse_rule_list(
                        rule.content, skip_whitespace=True,
                    )
                except Exception:
                    nested_rules = []

                for nested in nested_rules:
                    if nested.type == "qualified-rule":
                        nested_count += 1
                        selector = tinycss2.serialize(nested.prelude).strip()
                        decls = _parse_declarations(nested.content)
                        if decls:
                            rules.append(CSSRule(
                                selector=selector,
                                declarations=decls,
                                media_min_width=min_w,
                                media_max_width=max_w,
                                source_order=order,
                            ))
                            order += 1

            media_queries.append(MediaQueryInfo(
                min_width=min_w,
                max_width=max_w,
                rule_count=nested_count,
                raw_query=prelude_text,
            ))

    return rules, media_queries


# --- HTML parsing ---

_INTERACTIVE_TAGS = frozenset({"a", "button", "input", "select", "textarea"})


def _build_selector(tag: Tag, seen_counts: dict[str, int]) -> str:
    """Build a unique CSS selector for a BeautifulSoup Tag.

    Uses element index tracking to disambiguate repeated selectors.
    Format: tag#id or tag.class1.class2[n] where [n] is the occurrence index.
    """
    parts = [tag.name]

    tag_id = tag.get("id")
    if tag_id:
        parts.append(f"#{tag_id}")
        base = "".join(parts)
        # IDs should be unique, but track anyway
        seen_counts[base] = seen_counts.get(base, 0) + 1
        return base

    classes = tag.get("class", [])
    if classes:
        parts.extend(f".{cls}" for cls in classes[:3])

    base = "".join(parts)
    seen_counts[base] = seen_counts.get(base, 0) + 1
    count = seen_counts[base]

    # Always append occurrence index for non-ID selectors to ensure uniqueness
    if count > 1:
        return f"{base}:nth({count})"
    return base


def _parse_inline_style(style_attr: str | None) -> dict[str, str]:
    """Parse an inline style attribute into tracked properties."""
    if not style_attr:
        return {}

    decls: dict[str, str] = {}
    try:
        parsed = tinycss2.parse_blocks_contents(
            style_attr, skip_whitespace=True, skip_comments=True,
        )
    except Exception:
        return decls

    for item in parsed:
        if item.type == "declaration" and item.name in TRACKED_PROPERTIES:
            decls[item.name] = tinycss2.serialize(item.value).strip()

    return decls


def _extract_elements(soup: BeautifulSoup) -> list[ElementInfo]:
    """Extract all meaningful elements from parsed HTML."""
    elements: list[ElementInfo] = []
    seen_counts: dict[str, int] = {}

    for tag in soup.find_all(True):
        if tag.name in ("html", "head", "meta", "link", "script", "style", "br", "hr"):
            continue

        text = tag.get_text(strip=True)
        children = [c for c in tag.children if isinstance(c, Tag)]

        elements.append(ElementInfo(
            tag=tag.name,
            selector=_build_selector(tag, seen_counts),
            element_id=tag.get("id", ""),
            classes=tag.get("class", []),
            inline_styles=_parse_inline_style(tag.get("style")),
            children_count=len(children),
            text_length=len(text),
            is_interactive=tag.name in _INTERACTIVE_TAGS,
        ))

    return elements


# --- Main parser ---

class PageParser:
    """Parses HTML pages into PageAnalysis objects."""

    def parse(self, html: str, page_id: str) -> PageAnalysis:
        """Parse an HTML string into a PageAnalysis.

        Extracts all <style> blocks, parses CSS rules and media queries,
        builds element tree, and checks for viewport meta tag.
        """
        soup = BeautifulSoup(html, "lxml")

        # Extract title
        title_tag = soup.find("title")
        title = title_tag.get_text(strip=True) if title_tag else ""

        # Extract and parse all CSS from <style> blocks
        all_css = ""
        for style_tag in soup.find_all("style"):
            all_css += style_tag.get_text() + "\n"

        css_rules, media_queries = _parse_css(all_css)

        # Extract elements
        elements = _extract_elements(soup)

        # Check viewport meta
        viewport_meta = soup.find("meta", attrs={"name": "viewport"})
        has_viewport_meta = viewport_meta is not None
        viewport_meta_content = ""
        if viewport_meta:
            viewport_meta_content = viewport_meta.get("content", "")

        return PageAnalysis(
            page_id=page_id,
            title=title,
            html=html,
            elements=elements,
            css_rules=css_rules,
            media_queries=media_queries,
            has_viewport_meta=has_viewport_meta,
            viewport_meta_content=viewport_meta_content,
        )
