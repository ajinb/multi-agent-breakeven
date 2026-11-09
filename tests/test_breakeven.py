import math
import random

import pytest

import breakeven as be

TRIALS = 200_000


def attempt(rng, p, rho, shared):
    if shared is not None:
        return shared
    return rng.random() < p


def task(rng, p, rho):
    return (rng.random() < p) if rng.random() < rho else None


@pytest.mark.parametrize("p,rho", [(0.6, 0.0), (0.8, 0.5), (0.9, 0.9)])
def test_vote3_matches_monte_carlo(p, rho):
    rng = random.Random(1)
    wins = 0
    for _ in range(TRIALS):
        shared = task(rng, p, rho)
        wins += sum(attempt(rng, p, rho, shared) for _ in range(3)) >= 2
    assert wins / TRIALS == pytest.approx(be.vote3(p, rho), abs=0.005)


@pytest.mark.parametrize("p,rho,recall,fa", [(0.7, 0.0, 0.8, 0.1), (0.8, 0.5, 0.7, 0.1)])
def test_verify_retry_matches_monte_carlo(p, rho, recall, fa):
    rng = random.Random(2)
    wins = runs = 0
    for _ in range(TRIALS):
        shared = task(rng, p, rho)
        first = attempt(rng, p, rho, shared)
        runs += 1
        flagged = rng.random() < (fa if first else recall)
        outcome = first
        if flagged:
            runs += 1
            outcome = attempt(rng, p, rho, shared)
        wins += outcome
    success, attempts = be.verify_retry(p, rho, recall, fa)
    assert wins / TRIALS == pytest.approx(success, abs=0.005)
    assert runs / TRIALS == pytest.approx(attempts, abs=0.005)


def test_vote_gain_formula_agrees_with_vote3():
    for p in (0.55, 0.7, 0.95):
        assert be.vote3(p, 0.3) - p == pytest.approx(be.vote3_gain(p, 0.3))


def test_vote_gain_peak():
    grid = max((be.vote3_gain(i / 10000, 0), i / 10000) for i in range(10001))
    assert grid[1] == pytest.approx(be.PEAK_P, abs=0.001)
    assert grid[0] == pytest.approx(be.PEAK_GAIN, abs=1e-6)


def test_vote_gain_vanishes_at_the_ends_and_with_full_correlation():
    assert be.vote3_gain(0.5, 0) == 0
    assert be.vote3_gain(1.0, 0) == 0
    assert be.vote3_gain(0.8, 1.0) == 0


def test_pipeline_breakeven_ties():
    lift = be.pipeline_breakeven(4, 0.95)
    assert be.pipeline(0.7 * lift, 4, 0.95) == pytest.approx(0.7**4)


def test_breakeven_value():
    assert be.breakeven_value(1, 0.8, 4, 0.9) == pytest.approx(30)
    assert math.isinf(be.breakeven_value(1, 0.9, 4, 0.8))


@pytest.mark.parametrize("p,rho", [(0.7, 0.0), (0.8, 0.5), (0.6, 0.9)])
def test_estimate_rho_recovers_parameters(p, rho):
    rng = random.Random(3)
    counts = [0, 0, 0]
    for _ in range(TRIALS):
        shared = task(rng, p, rho)
        passes = attempt(rng, p, rho, shared) + attempt(rng, p, rho, shared)
        counts[2 - passes] += 1
    p_hat, rho_hat = be.estimate_rho(*counts)
    assert p_hat == pytest.approx(p, abs=0.01)
    assert rho_hat == pytest.approx(rho, abs=0.02)


def test_rejects_bad_probability():
    with pytest.raises(ValueError):
        be.vote3(1.2, 0)
