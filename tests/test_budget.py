"""Tests for budget accounting."""

from __future__ import annotations

import pytest

from awarebench.harness.budget import BudgetAccountant


def test_totals_start_at_zero() -> None:
    accountant = BudgetAccountant()

    assert accountant.tokens_used == 0
    assert accountant.tool_calls == 0
    assert accountant.wall_us_used == 0
    assert accountant.snapshot() == {
        "tokens_used": 0,
        "tool_calls": 0,
        "wall_us_used": 0,
    }


def test_add_tokens_sums() -> None:
    accountant = BudgetAccountant()
    accountant.add_tokens(120)
    accountant.add_tokens(80)
    accountant.add_tokens(0)

    assert accountant.tokens_used == 200


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


def test_negative_inputs_raise_and_leave_totals_unchanged() -> None:
    accountant = BudgetAccountant()
    accountant.add_tokens(10)
    accountant.add_wall_us(20)

    with pytest.raises(ValueError, match="n must be >= 0"):
        accountant.add_tokens(-1)
    with pytest.raises(ValueError, match="us must be >= 0"):
        accountant.add_wall_us(-5)

    assert accountant.tokens_used == 10
    assert accountant.wall_us_used == 20


def test_snapshot_matches_properties() -> None:
    accountant = BudgetAccountant()
    accountant.add_tokens(300)
    accountant.add_tool_call()
    accountant.add_tool_call()
    accountant.add_wall_us(9999)

    snapshot = accountant.snapshot()

    assert snapshot["tokens_used"] == accountant.tokens_used == 300
    assert snapshot["tool_calls"] == accountant.tool_calls == 2
    assert snapshot["wall_us_used"] == accountant.wall_us_used == 9999
