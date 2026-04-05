"""Ground truth computation and reward scoring (v2).

Computes ground truth by running all checks across standard viewports.
Scores agent reports using F1 + evidence bonus + coverage + efficiency.

v2: identified_issues is list[dict] with evidence fields,
    checks_run removed, evidence bonus added.
"""

from __future__ import annotations

from dalaal_env.analyzer.checks import (
    ALL_CHECK_NAMES,
    STANDARD_VIEWPORTS,
    CheckResult,
    run_all_checks,
    run_check,
)
from dalaal_env.analyzer.parser import PageAnalysis

STANDARD_VIEWPORTS_SET = frozenset(STANDARD_VIEWPORTS)


def compute_ground_truth(
    analysis: PageAnalysis,
) -> dict[str, CheckResult]:
    """Run all 10 checks across all standard viewports.

    An issue is in ground truth if it FAILS at ANY standard viewport.
    Returns {check_name: CheckResult} for all failing checks.
    The CheckResult stored is the worst (highest severity) across viewports.
    """
    worst_failures: dict[str, CheckResult] = {}

    for vw in STANDARD_VIEWPORTS:
        results = run_all_checks(analysis, vw)
        for name, result in results.items():
            if not result.passed:
                existing = worst_failures.get(name)
                if existing is None or result.severity > existing.severity:
                    worst_failures[name] = result

    return worst_failures


def compute_ground_truth_by_viewport(
    analysis: PageAnalysis,
) -> dict[str, list[int]]:
    """For each failing check, return which viewports it fails at.

    Used for evidence bonus validation.
    Returns {check_name: [viewport_widths_where_it_fails]}.
    """
    failing_viewports: dict[str, list[int]] = {}

    for vw in STANDARD_VIEWPORTS:
        results = run_all_checks(analysis, vw)
        for name, result in results.items():
            if not result.passed:
                if name not in failing_viewports:
                    failing_viewports[name] = []
                failing_viewports[name].append(vw)

    return failing_viewports


def _compute_evidence_bonus(
    identified: list[dict],
    ground_truth: dict[str, CheckResult],
    gt_viewports: dict[str, list[int]],
) -> float:
    """Score the quality of evidence cited by the agent.

    For each true-positive issue, check if:
    - cited selectors appear in the GT evidence strings
    - cited viewports are ones where the check actually fails

    Returns a score in [0.0, 0.05].
    """
    gt_set = set(ground_truth.keys())
    tp_scores: list[float] = []

    for item in identified:
        issue_name = item.get("issue", "")
        if issue_name not in gt_set:
            continue

        score = 0.0
        parts = 0

        # Check selector evidence
        cited_selectors = item.get("affected_selectors", [])
        if cited_selectors:
            gt_evidence = ground_truth[issue_name].evidence
            gt_evidence_str = " ".join(gt_evidence).lower()
            matches = sum(
                1 for sel in cited_selectors
                if sel.lower() in gt_evidence_str
            )
            score += matches / len(cited_selectors) if cited_selectors else 0
            parts += 1

        # Check viewport evidence
        cited_viewports = item.get("affected_viewports", [])
        if cited_viewports and issue_name in gt_viewports:
            actual_failing = set(gt_viewports[issue_name])
            matches = sum(1 for vw in cited_viewports if vw in actual_failing)
            score += matches / len(cited_viewports) if cited_viewports else 0
            parts += 1

        if parts > 0:
            tp_scores.append(score / parts)
        else:
            tp_scores.append(0.0)

    if not tp_scores:
        return 0.0

    avg_evidence = sum(tp_scores) / len(tp_scores)
    return 0.05 * avg_evidence


def compute_reward(
    identified: list[dict],
    ground_truth: dict[str, CheckResult],
    gt_viewports: dict[str, list[int]],
    viewports_tested: list[int],
    step_count: int,
    max_steps: int,
) -> tuple[float, dict]:
    """Compute the episode reward from the agent's submitted report.

    Returns (total_reward, breakdown_dict).

    Reward formula:
        R = F1 * 0.85 + evidence_bonus (0.05) + coverage (0.05) + efficiency (0.05)
    """
    # Extract issue names from rich format
    identified_names = {
        item.get("issue", "")
        for item in identified
        if item.get("issue", "") in ALL_CHECK_NAMES
    }
    gt_set = set(ground_truth.keys())

    # F1 computation
    tp = len(identified_names & gt_set)
    fp = len(identified_names - gt_set)
    fn = len(gt_set - identified_names)

    if tp == 0 and fp == 0 and fn == 0:
        precision = 1.0
        recall = 1.0
        f1 = 1.0
    else:
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 1.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

    # Evidence bonus
    evidence_bonus = _compute_evidence_bonus(identified, ground_truth, gt_viewports)

    # Coverage bonus: proportion of standard viewports tested
    viewports_set = frozenset(viewports_tested)
    coverage_hits = len(viewports_set & STANDARD_VIEWPORTS_SET)
    coverage_bonus = 0.05 * (coverage_hits / len(STANDARD_VIEWPORTS_SET))

    # Efficiency bonus
    efficiency_bonus = 0.05 * max(0.0, (max_steps - step_count) / max_steps)

    total = f1 * 0.85 + evidence_bonus + coverage_bonus + efficiency_bonus

    breakdown = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "evidence_bonus": round(evidence_bonus, 4),
        "coverage_bonus": round(coverage_bonus, 4),
        "viewports_tested_count": len(viewports_set),
        "efficiency_bonus": round(efficiency_bonus, 4),
        "steps_used": step_count,
        "total": round(total, 4),
    }

    return total, breakdown
