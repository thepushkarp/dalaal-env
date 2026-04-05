"""10 responsiveness check functions.

Each check takes a PageAnalysis and viewport width, returning a CheckResult.
Ground truth is determined by running all checks across standard viewports.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from dalaal_env.analyzer.parser import PageAnalysis, CSSRule

# Standard breakpoints used for ground truth evaluation
STANDARD_VIEWPORTS = [320, 375, 768, 1024, 1280, 1440]

ALL_CHECK_NAMES = [
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


@dataclass
class CheckResult:
    """Result of a single responsiveness check."""

    check_name: str
    passed: bool
    detail: str
    evidence: list[str] = field(default_factory=list)
    severity: float = 0.0  # 0.0 (minor) to 1.0 (critical)


# --- Utility: parse px value from CSS ---

_PX_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*px$")
_PERCENT_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*%$")
_RESPONSIVE_UNIT_RE = re.compile(
    r"(?:rem|em|%|vw|vh|vmin|vmax|clamp\()",
)


def _parse_px(value: str) -> float | None:
    """Extract numeric px value from a CSS value string."""
    m = _PX_RE.match(value.strip())
    return float(m.group(1)) if m else None


def _is_responsive_unit(value: str) -> bool:
    """Check if a CSS value uses responsive units."""
    v = value.strip()
    return bool(_RESPONSIVE_UNIT_RE.search(v)) or v == "auto" or v == "100%"


def _is_fixed_px(value: str) -> bool:
    """Check if a CSS value is a fixed pixel value."""
    return _parse_px(value) is not None


# --- Check implementations ---


def check_viewport_meta(
    analysis: PageAnalysis, viewport_width: int,
) -> CheckResult:
    """Check for proper viewport meta tag."""
    if not analysis.has_viewport_meta:
        return CheckResult(
            check_name="viewport_meta",
            passed=False,
            detail="Missing <meta name='viewport'> tag. Mobile browsers will render at desktop width.",
            evidence=["No <meta name='viewport'> found in <head>"],
            severity=0.9,
        )

    content = analysis.viewport_meta_content.lower()
    has_device_width = "width=device-width" in content

    if not has_device_width:
        return CheckResult(
            check_name="viewport_meta",
            passed=False,
            detail=f"Viewport meta exists but missing 'width=device-width'. Content: '{analysis.viewport_meta_content}'",
            evidence=[f"<meta name='viewport' content='{analysis.viewport_meta_content}'>"],
            severity=0.7,
        )

    return CheckResult(
        check_name="viewport_meta",
        passed=True,
        detail="Viewport meta tag is correctly configured.",
    )


def check_media_queries(
    analysis: PageAnalysis, viewport_width: int,
) -> CheckResult:
    """Check if CSS has media queries covering mobile, tablet, and desktop."""
    covers_mobile = False   # <= 480px
    covers_tablet = False   # 481-1024px
    covers_desktop = False  # > 1024px

    for mq in analysis.media_queries:
        if mq.rule_count == 0:
            continue

        min_w = mq.min_width or 0
        max_w = mq.max_width or float("inf")

        # Does this query affect mobile range?
        if max_w <= 480 or (min_w == 0 and max_w >= 320):
            covers_mobile = True
        if max_w <= 768 or (min_w <= 480 and max_w >= 480):
            covers_mobile = True

        # Tablet range
        if (min_w >= 481 and min_w <= 1024) or (max_w >= 481 and max_w <= 1024):
            covers_tablet = True
        if min_w <= 768 and max_w >= 768:
            covers_tablet = True

        # Desktop range
        if min_w >= 1024:
            covers_desktop = True

    missing = []
    if not covers_mobile:
        missing.append("mobile (<=480px)")
    if not covers_tablet:
        missing.append("tablet (481-1024px)")
    if not covers_desktop:
        missing.append("desktop (>1024px)")

    if missing:
        return CheckResult(
            check_name="media_queries",
            passed=False,
            detail=f"CSS media queries don't cover: {', '.join(missing)}.",
            evidence=[
                f"Found {len(analysis.media_queries)} media query blocks",
                f"Missing breakpoints for: {', '.join(missing)}",
            ],
            severity=0.8 if len(missing) >= 2 else 0.5,
        )

    return CheckResult(
        check_name="media_queries",
        passed=True,
        detail=f"CSS has media queries covering all 3 breakpoint ranges ({len(analysis.media_queries)} queries found).",
    )


def check_fixed_width_elements(
    analysis: PageAnalysis, viewport_width: int,
) -> CheckResult:
    """Check for elements with fixed px widths that exceed the viewport."""
    threshold = viewport_width * 0.9
    offenders: list[str] = []

    for rule in analysis.css_rules:
        if not rule.applies_at(viewport_width):
            continue
        width_val = rule.declarations.get("width", "")
        px = _parse_px(width_val)
        if px is not None and px > threshold:
            offenders.append(f"{rule.selector} (width: {width_val})")

    # Also check inline styles
    for el in analysis.elements:
        width_val = el.inline_styles.get("width", "")
        px = _parse_px(width_val)
        if px is not None and px > threshold:
            offenders.append(f"{el.selector} inline (width: {width_val})")

    if offenders:
        return CheckResult(
            check_name="fixed_width_elements",
            passed=False,
            detail=f"{len(offenders)} element(s) have fixed widths exceeding {threshold:.0f}px at {viewport_width}px viewport.",
            evidence=offenders[:10],
            severity=min(1.0, 0.3 + len(offenders) * 0.15),
        )

    return CheckResult(
        check_name="fixed_width_elements",
        passed=True,
        detail=f"No elements with fixed widths exceeding viewport at {viewport_width}px.",
    )


def check_responsive_images(
    analysis: PageAnalysis, viewport_width: int,
) -> CheckResult:
    """Check that images have max-width: 100% or width: 100%."""
    img_selectors = analysis.get_selectors_for_tag("img")
    if not img_selectors:
        return CheckResult(
            check_name="responsive_images",
            passed=True,
            detail="No <img> elements found on the page.",
        )

    # Check if there's a global img rule with max-width: 100%
    global_responsive = False
    for rule in analysis.css_rules:
        if rule.selector.strip() == "img" and not rule.media_min_width and not rule.media_max_width:
            mw = rule.declarations.get("max-width", "")
            w = rule.declarations.get("width", "")
            if mw == "100%" or w == "100%":
                global_responsive = True
                break

    if global_responsive:
        return CheckResult(
            check_name="responsive_images",
            passed=True,
            detail=f"Images have global responsive rule (max-width: 100%). {len(img_selectors)} image(s) covered.",
        )

    # Check individual images
    non_responsive: list[str] = []
    for sel in img_selectors:
        css = analysis.get_element_css(sel, viewport_width)
        if css is None:
            continue
        mw = css.get("max-width", "")
        w = css.get("width", "")
        if mw != "100%" and w != "100%":
            non_responsive.append(sel)

    if non_responsive:
        return CheckResult(
            check_name="responsive_images",
            passed=False,
            detail=f"{len(non_responsive)} of {len(img_selectors)} images lack max-width: 100%.",
            evidence=[f"{sel} (no max-width: 100%)" for sel in non_responsive[:5]],
            severity=0.6,
        )

    return CheckResult(
        check_name="responsive_images",
        passed=True,
        detail=f"All {len(img_selectors)} images have responsive sizing.",
    )


def check_font_sizing(
    analysis: PageAnalysis, viewport_width: int,
) -> CheckResult:
    """Check that font sizes use responsive units (rem/em/%) not fixed px."""
    all_fonts = analysis.get_all_rules_for_property("font-size")

    if not all_fonts:
        return CheckResult(
            check_name="font_sizing",
            passed=True,
            detail="No explicit font-size declarations found (browser defaults apply).",
        )

    responsive_count = 0
    fixed_count = 0
    fixed_examples: list[str] = []

    for selector, value, rule in all_fonts:
        if _is_responsive_unit(value):
            responsive_count += 1
        elif _is_fixed_px(value):
            fixed_count += 1
            fixed_examples.append(f"{selector} (font-size: {value})")

    total = responsive_count + fixed_count
    if total == 0:
        return CheckResult(
            check_name="font_sizing",
            passed=True,
            detail="No measurable font-size declarations.",
        )

    responsive_ratio = responsive_count / total

    if responsive_ratio >= 0.8:
        return CheckResult(
            check_name="font_sizing",
            passed=True,
            detail=f"{responsive_ratio:.0%} of font sizes use responsive units ({responsive_count}/{total}).",
        )

    return CheckResult(
        check_name="font_sizing",
        passed=False,
        detail=f"Only {responsive_ratio:.0%} of font sizes use responsive units. {fixed_count} use fixed px.",
        evidence=fixed_examples[:5],
        severity=0.4,
    )


def check_flexible_layouts(
    analysis: PageAnalysis, viewport_width: int,
) -> CheckResult:
    """Check that major layout containers use flexbox/grid."""
    # Find layout containers: elements with "container", "wrapper", "layout",
    # "row", "grid", "flex" in their class, or direct children of body
    layout_selectors: list[str] = []
    for el in analysis.elements:
        class_str = " ".join(el.classes).lower()
        if any(kw in class_str for kw in ("container", "wrapper", "layout", "row", "grid", "flex", "main", "content")):
            layout_selectors.append(el.selector)

    # Also check direct body children
    for el in analysis.elements:
        if el.tag == "body":
            continue
        # Heuristic: top-level divs/sections/mains are layout containers
        if el.tag in ("div", "section", "main", "article", "aside", "nav", "header", "footer"):
            if el.selector not in layout_selectors and el.children_count > 0:
                layout_selectors.append(el.selector)

    if not layout_selectors:
        return CheckResult(
            check_name="flexible_layouts",
            passed=True,
            detail="No major layout containers identified.",
        )

    flexible_count = 0
    rigid_containers: list[str] = []

    for sel in layout_selectors:
        css = analysis.get_element_css(sel, viewport_width)
        if css is None:
            continue
        display = css.get("display", "")
        if display in ("flex", "grid", "inline-flex", "inline-grid"):
            flexible_count += 1
        else:
            # Check if it has percentage/auto width (still somewhat flexible)
            width = css.get("width", "")
            if width == "100%" or width == "auto" or _is_responsive_unit(width):
                flexible_count += 1
            elif _is_fixed_px(width):
                rigid_containers.append(f"{sel} (display: {display or 'block'}, width: {width})")
            else:
                # No explicit width — could be flexible by default
                flexible_count += 1

    if rigid_containers:
        return CheckResult(
            check_name="flexible_layouts",
            passed=False,
            detail=f"{len(rigid_containers)} layout containers use rigid positioning instead of flexbox/grid.",
            evidence=rigid_containers[:5],
            severity=0.6,
        )

    return CheckResult(
        check_name="flexible_layouts",
        passed=True,
        detail=f"All {flexible_count} layout containers use flexible layouts.",
    )


def check_touch_targets(
    analysis: PageAnalysis, viewport_width: int,
) -> CheckResult:
    """Check that interactive elements are >= 44px at mobile viewports."""
    if viewport_width > 480:
        return CheckResult(
            check_name="touch_targets",
            passed=True,
            detail=f"Touch target check only applies at mobile viewports (<=480px). Current: {viewport_width}px.",
        )

    interactive = [el for el in analysis.elements if el.is_interactive]
    if not interactive:
        return CheckResult(
            check_name="touch_targets",
            passed=True,
            detail="No interactive elements found.",
        )

    small_targets: list[str] = []
    for el in interactive:
        css = analysis.get_element_css(el.selector, viewport_width)
        if css is None:
            continue
        height_val = css.get("height", "") or css.get("min-height", "")
        if height_val:
            px = _parse_px(height_val)
            if px is not None and px < 44:
                small_targets.append(f"{el.selector} (height: {height_val})")

    if small_targets:
        return CheckResult(
            check_name="touch_targets",
            passed=False,
            detail=f"{len(small_targets)} interactive elements have touch targets < 44px at {viewport_width}px.",
            evidence=small_targets[:5],
            severity=0.5,
        )

    return CheckResult(
        check_name="touch_targets",
        passed=True,
        detail=f"All {len(interactive)} interactive elements have adequate touch targets at {viewport_width}px.",
    )


def check_horizontal_scroll(
    analysis: PageAnalysis, viewport_width: int,
) -> CheckResult:
    """Check for elements that would cause horizontal scrolling."""
    offenders: list[str] = []

    for rule in analysis.css_rules:
        if not rule.applies_at(viewport_width):
            continue
        width_val = rule.declarations.get("width", "")
        px = _parse_px(width_val)
        if px is not None and px > viewport_width:
            offenders.append(f"{rule.selector} (width: {width_val})")

    # Check inline styles
    for el in analysis.elements:
        width_val = el.inline_styles.get("width", "")
        px = _parse_px(width_val)
        if px is not None and px > viewport_width:
            offenders.append(f"{el.selector} inline (width: {width_val})")

    if offenders:
        return CheckResult(
            check_name="horizontal_scroll",
            passed=False,
            detail=f"{len(offenders)} element(s) would cause horizontal scrolling at {viewport_width}px viewport.",
            evidence=offenders[:10],
            severity=0.8,
        )

    return CheckResult(
        check_name="horizontal_scroll",
        passed=True,
        detail=f"No elements cause horizontal scrolling at {viewport_width}px viewport.",
    )


_REM_EM_RE = re.compile(r"^(-?\d+(?:\.\d+)?)\s*(rem|em)$")
_BASE_FONT_PX = 16.0


def _css_value_to_px(value: str, viewport_width: int) -> float | None:
    """Convert a CSS length value to px. Returns None if unparseable."""
    v = value.strip()
    px = _parse_px(v)
    if px is not None:
        return px

    m = _PERCENT_RE.match(v)
    if m:
        return viewport_width * float(m.group(1)) / 100

    m = _REM_EM_RE.match(v)
    if m:
        return float(m.group(1)) * _BASE_FONT_PX

    return None


def _estimate_element_width(css: dict[str, str], viewport_width: int) -> float:
    """Estimate element width in px, honoring max-width."""
    el_width = float(viewport_width)

    width_val = css.get("width", "")
    if width_val:
        px = _css_value_to_px(width_val, viewport_width)
        if px is not None:
            el_width = px

    # max-width constrains the computed width
    max_width_val = css.get("max-width", "")
    if max_width_val:
        max_px = _css_value_to_px(max_width_val, viewport_width)
        if max_px is not None:
            el_width = min(el_width, max_px)

    return el_width


def _estimate_font_size_px(value: str) -> float:
    """Convert font-size value to px estimate."""
    if not value:
        return _BASE_FONT_PX

    px = _parse_px(value.strip())
    if px is not None:
        return px

    m = _REM_EM_RE.match(value.strip())
    if m:
        return float(m.group(1)) * _BASE_FONT_PX

    # clamp(), calc(), or other complex values — use default
    return _BASE_FONT_PX


def check_text_readability(
    analysis: PageAnalysis, viewport_width: int,
) -> CheckResult:
    """Check that line lengths are readable (20-90 chars per line estimate)."""
    # Estimate: chars per line ~ element_width / (font_size * 0.5)
    # We approximate element_width from CSS width or viewport
    issues: list[str] = []

    text_elements = [
        el for el in analysis.elements
        if el.tag in ("p", "li", "td", "th", "blockquote")
        and el.text_length > 50
    ]

    for el in text_elements:
        css = analysis.get_element_css(el.selector, viewport_width)
        if css is None:
            continue

        # Estimate element width (consider both width and max-width)
        el_width = _estimate_element_width(css, viewport_width)

        # Estimate font size (handle rem/em → px conversion)
        fs_px = _estimate_font_size_px(css.get("font-size", ""))

        if fs_px <= 0:
            continue

        chars_per_line = el_width / (fs_px * 0.55)

        if chars_per_line > 90:
            issues.append(f"{el.selector} (~{chars_per_line:.0f} chars/line, too wide)")
        elif chars_per_line < 20 and el.text_length > 50:
            issues.append(f"{el.selector} (~{chars_per_line:.0f} chars/line, too narrow)")

    if issues:
        return CheckResult(
            check_name="text_readability",
            passed=False,
            detail=f"{len(issues)} text block(s) have problematic line lengths at {viewport_width}px.",
            evidence=issues[:5],
            severity=0.3,
        )

    return CheckResult(
        check_name="text_readability",
        passed=True,
        detail=f"Text readability is acceptable at {viewport_width}px viewport.",
    )


def check_responsive_tables(
    analysis: PageAnalysis, viewport_width: int,
) -> CheckResult:
    """Check that tables have responsive handling for small screens."""
    table_selectors = analysis.get_selectors_for_tag("table")

    if not table_selectors:
        return CheckResult(
            check_name="responsive_tables",
            passed=True,
            detail="No <table> elements found.",
        )

    if viewport_width > 768:
        return CheckResult(
            check_name="responsive_tables",
            passed=True,
            detail=f"Table responsiveness check mainly applies at <=768px. Current: {viewport_width}px.",
        )

    non_responsive: list[str] = []
    for sel in table_selectors:
        css = analysis.get_element_css(sel, viewport_width)
        if css is None:
            non_responsive.append(sel)
            continue

        overflow = css.get("overflow-x", css.get("overflow", ""))
        display = css.get("display", "")
        max_width = css.get("max-width", "")

        is_handled = (
            overflow in ("auto", "scroll", "hidden")
            or display in ("block", "flex", "grid")
            or max_width == "100%"
        )

        if not is_handled:
            non_responsive.append(f"{sel} (no overflow-x or responsive display)")

    if non_responsive:
        return CheckResult(
            check_name="responsive_tables",
            passed=False,
            detail=f"{len(non_responsive)} of {len(table_selectors)} tables lack responsive handling at {viewport_width}px.",
            evidence=non_responsive[:5],
            severity=0.5,
        )

    return CheckResult(
        check_name="responsive_tables",
        passed=True,
        detail=f"All {len(table_selectors)} tables have responsive handling.",
    )


# --- Dispatcher ---

_CHECK_FUNCTIONS = {
    "viewport_meta": check_viewport_meta,
    "media_queries": check_media_queries,
    "fixed_width_elements": check_fixed_width_elements,
    "responsive_images": check_responsive_images,
    "font_sizing": check_font_sizing,
    "flexible_layouts": check_flexible_layouts,
    "touch_targets": check_touch_targets,
    "horizontal_scroll": check_horizontal_scroll,
    "text_readability": check_text_readability,
    "responsive_tables": check_responsive_tables,
}


def run_check(
    check_name: str, analysis: PageAnalysis, viewport_width: int,
) -> CheckResult:
    """Run a named check. Raises ValueError for unknown check names."""
    fn = _CHECK_FUNCTIONS.get(check_name)
    if fn is None:
        msg = f"Unknown check: {check_name}. Valid: {list(_CHECK_FUNCTIONS.keys())}"
        raise ValueError(msg)
    return fn(analysis, viewport_width)


def run_all_checks(
    analysis: PageAnalysis, viewport_width: int,
) -> dict[str, CheckResult]:
    """Run all 10 checks at a given viewport width."""
    return {
        name: fn(analysis, viewport_width)
        for name, fn in _CHECK_FUNCTIONS.items()
    }
