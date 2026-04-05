"""Ground truth computation and reward scoring.

Computes ground truth by running all checks across standard viewports.
Scores agent reports using F1 + coverage bonus + efficiency bonus.
"""

from __future__ import annotations

from dalaal_env.analyzer.checks import (
    ALL_CHECK_NAMES,
    STANDARD_VIEWPORTS,
    CheckResult,
    run_all_checks,
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


def compute_reward(
    identified: list[str],
    ground_truth: dict[str, CheckResult],
    viewports_tested: list[int],
    checks_run: list[str],
    step_count: int,
    max_steps: int,
) -> tuple[float, dict]:
    """Compute the episode reward from the agent's submitted report.

    Returns (total_reward, breakdown_dict).

    Reward formula:
        R = F1 * 0.85 + coverage_bonus (max 0.10) + efficiency_bonus (max 0.10)
    """
    # Deduplicate and validate identified issues
    identified_set = {
        name for name in set(identified)
        if name in ALL_CHECK_NAMES
    }
    gt_set = set(ground_truth.keys())

    # F1 computation
    tp = len(identified_set & gt_set)
    fp = len(identified_set - gt_set)
    fn = len(gt_set - identified_set)

    # Edge case: no issues exist AND agent reports none → perfect score
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

    # Coverage bonus: proportion of standard viewports tested
    viewports_set = frozenset(viewports_tested)
    coverage_hits = len(viewports_set & STANDARD_VIEWPORTS_SET)
    coverage_bonus = 0.10 * (coverage_hits / len(STANDARD_VIEWPORTS_SET))

    # Efficiency bonus: reward for finishing early
    efficiency_bonus = 0.10 * max(0.0, (max_steps - step_count) / max_steps)

    total = f1 * 0.85 + coverage_bonus + efficiency_bonus

    breakdown = {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "coverage_bonus": round(coverage_bonus, 4),
        "viewports_tested_count": len(viewports_set),
        "efficiency_bonus": round(efficiency_bonus, 4),
        "steps_used": step_count,
        "total": round(total, 4),
    }

    return total, breakdown
