#!/usr/bin/env python3
"""When does a multi-agent architecture pay for itself?

Closed-form models of the common architectures, checked against Monte Carlo in the tests.

Correlation model: with probability `rho` a task has a common cause, so every attempt at it gets
the same outcome (a misleading ticket, a wrong premise in the shared prompt, the same blind spot in
the same model). Otherwise each attempt is an independent draw with success probability `p`.
"""

from __future__ import annotations

import argparse
import math


def _check(*probabilities: float) -> None:
    if not all(0 <= x <= 1 for x in probabilities):
        raise ValueError("probabilities must be in [0, 1]")


def vote3(p: float, rho: float) -> float:
    """Three agents, majority wins."""
    _check(p, rho)
    return rho * p + (1 - rho) * (p**3 + 3 * p**2 * (1 - p))


def vote3_gain(p: float, rho: float) -> float:
    """Points a vote adds over one agent: (1-rho) * p(1-p)(2p-1)."""
    _check(p, rho)
    return (1 - rho) * p * (1 - p) * (2 * p - 1)


PEAK_P = 0.5 + math.sqrt(3) / 6  # where p(1-p)(2p-1) is largest
PEAK_GAIN = PEAK_P * (1 - PEAK_P) * (2 * PEAK_P - 1)


def verify_retry(p: float, rho: float, recall: float, false_alarm: float) -> tuple[float, float]:
    """One worker, one verifier, one retry on a flag. Returns (success, expected worker attempts).

    The retry's result is final. A failure caused by a common cause fails again on retry.
    """
    _check(p, rho, recall, false_alarm)
    retry_after_fail = (1 - rho) * p
    retry_after_pass = rho + (1 - rho) * p
    success = (
        p * (1 - false_alarm)
        + p * false_alarm * retry_after_pass
        + (1 - p) * recall * retry_after_fail
    )
    return success, 1 + p * false_alarm + (1 - p) * recall


def pipeline(p_stage: float, stages: int, handoff: float) -> float:
    """Stages in series, where each handoff keeps the work intact with probability `handoff`."""
    _check(p_stage, handoff)
    return p_stage**stages * handoff ** (stages - 1)


def pipeline_breakeven(stages: int, handoff: float) -> float:
    """How much better each specialist must be than the generalist for a pipeline to tie."""
    _check(handoff)
    return handoff ** (-(stages - 1) / stages)


def fanout(p_sub: float, subtasks: int, synthesis: float) -> float:
    """Independent subtasks in clean contexts, each kept intact through synthesis."""
    _check(p_sub, synthesis)
    return (p_sub * synthesis) ** subtasks


def single_over_subtasks(p_sub: float, subtasks: int, context_penalty: float) -> float:
    """One agent doing all the subtasks, at a per-subtask penalty for the crowded context."""
    _check(p_sub, context_penalty)
    return (p_sub * context_penalty) ** subtasks


def breakeven_value(cost_a: float, p_a: float, cost_b: float, p_b: float) -> float:
    """Value of a successful task above which B beats A. Infinite if B is not more reliable."""
    if p_b <= p_a:
        return math.inf
    return (cost_b - cost_a) / (p_b - p_a)


def estimate_rho(both_pass: int, only_one: int, both_fail: int) -> tuple[float, float]:
    """Estimate (p, rho) from running every case twice.

    P(both fail) = rho(1-p) + (1-rho)(1-p)^2, and p comes from the marginal pass rate.
    """
    n = both_pass + only_one + both_fail
    if n == 0:
        raise ValueError("no pairs")
    p = (2 * both_pass + only_one) / (2 * n)
    q = 1 - p
    if p in (0, 1):
        return p, 1.0
    return p, max(0.0, min(1.0, (both_fail / n - q * q) / (q - q * q)))


def table(p: float, rho: float, single_cost: float, multi_ratio: float, verify_cost: float):
    vote = vote3(p, rho)
    verified, attempts = verify_retry(p, rho, recall=0.7, false_alarm=0.1)
    verified_cost = single_cost * (attempts + verify_cost)
    vote_cost = single_cost * multi_ratio
    return [
        ("single agent", p, single_cost, 0.0),
        (
            "verifier + one retry",
            verified,
            verified_cost,
            breakeven_value(single_cost, p, verified_cost, verified),
        ),
        (
            "three agents, majority",
            vote,
            vote_cost,
            breakeven_value(single_cost, p, vote_cost, vote),
        ),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--baseline", type=float, required=True, help="single-agent success rate")
    parser.add_argument("--rho", type=float, default=0.5, help="failure correlation")
    parser.add_argument("--cost", type=float, default=1.0, help="cost of one single-agent run")
    parser.add_argument("--multi-ratio", type=float, default=3.75, help="team cost / single cost")
    parser.add_argument(
        "--verify-cost", type=float, default=0.3, help="verifier cost / single cost"
    )
    args = parser.parse_args()
    print(f"{'architecture':<26}{'success':>9}{'cost':>9}{'worth it above':>16}")
    for name, success, cost, value in table(
        args.baseline, args.rho, args.cost, args.multi_ratio, args.verify_cost
    ):
        worth = (
            "-" if name == "single agent" else ("never" if math.isinf(value) else f"{value:,.2f}")
        )
        print(f"{name:<26}{success:>9.1%}{cost:>9.2f}{worth:>16}")


if __name__ == "__main__":
    main()
