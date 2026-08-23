"""Tests for budget accounting."""

from __future__ import annotations

from typing import Any

import pytest

from awarebench.harness.budget import BudgetAccountant


def test_totals_start_at_zero() -> None:
    accountant = BudgetAccountant()

    assert accountant.prompt_tokens == 0
    assert accountant.completion_tokens == 0
    assert accountant.total_tokens == 0
    assert accountant.tool_calls == 0
    assert accountant.wall_us_used == 0
    assert accountant.snapshot() == {
        "prompt_tokens": 0,
        "completion_tokens": 0,
        "total_tokens": 0,
        "tool_calls": 0,
        "wall_us_used": 0,
    }


def test_add_tokens_splits_prompt_and_completion() -> None:
    accountant = BudgetAccountant()
    accountant.add_tokens(prompt=120, completion=30)
    accountant.add_tokens(prompt=80, completion=50)

    assert accountant.prompt_tokens == 200
    assert accountant.completion_tokens == 80
    assert accountant.total_tokens == 280


@pytest.mark.parametrize(
    ("prompt", "completion"),
    [(True, 0), (0, True), (1.5, 0), (0, 2.5), (-1, 0), (0, -1)],
)
def test_add_tokens_rejects_bools_floats_and_negatives(prompt: Any, completion: Any) -> None:
    accountant = BudgetAccountant()

    with pytest.raises(ValueError):
        accountant.add_tokens(prompt, completion)

    assert accountant.snapshot()["total_tokens"] == 0


def test_add_tool_call_counts_each_invocation() -> None:
    accountant = BudgetAccountant()
    accountant.add_tool_call()
    accountant.add_tool_call()
    accountant.add_tool_call()

    assert accountant.tool_calls == 3


def test_add_wall_us_sums() -> None:
    accountant = BudgetAccountant()
    accountant.add_wall_us(1500)
    accountant.add_wall_us(500)

    assert accountant.wall_us_used == 2000


@pytest.mark.parametrize("us", [True, 2.5, -5])
def test_add_wall_us_rejects_bools_floats_and_negatives(us: Any) -> None:
    accountant = BudgetAccountant()

    with pytest.raises(ValueError):
        accountant.add_wall_us(us)

    assert accountant.wall_us_used == 0


def test_snapshot_matches_properties() -> None:
    accountant = BudgetAccountant()
    accountant.add_tokens(prompt=300, completion=45)
    accountant.add_tool_call()
    accountant.add_tool_call()
    accountant.add_wall_us(9999)

    snapshot = accountant.snapshot()

    assert snapshot["prompt_tokens"] == accountant.prompt_tokens == 300
    assert snapshot["completion_tokens"] == accountant.completion_tokens == 45
    assert snapshot["total_tokens"] == accountant.total_tokens == 345
    assert snapshot["tool_calls"] == accountant.tool_calls == 2
    assert snapshot["wall_us_used"] == accountant.wall_us_used == 9999
