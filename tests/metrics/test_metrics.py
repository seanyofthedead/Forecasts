"""Tests for performance metric utilities."""
from __future__ import annotations

from metrics import metrics


def test_compute_metrics_handles_empty_logs():
    assert metrics.compute_metrics([]) == {}


def test_compute_metrics_calculates_summary_values():
    trade_log = [
        {"exit_price": 2.0, "pnl": 100.0},
        {"exit_price": 1.0, "pnl": -50.0},
        {"exit_price": 1.5, "pnl": 75.0},
    ]
    summary = metrics.compute_metrics(trade_log)
    assert summary["Total Trades"] == 3
    assert summary["Win Rate (%)"] == 66.67
    assert summary["Consecutive Losers"] == 1
    assert summary["Net Profit"] == 125.0


def test_log_trades_to_csv(tmp_path):
    trade_log = [
        {
            "entry_time": "2024-01-02 09:30",
            "exit_time": "2024-01-02 09:35",
            "action": "BUY",
            "shares": 10,
            "entry_price": 100.0,
            "exit_price": 105.0,
            "pnl": 50.0,
            "reason": "test",
        }
    ]
    filename = tmp_path / "trades.csv"
    metrics.log_trades_to_csv(trade_log, filename=str(filename))
    contents = filename.read_text().strip().splitlines()
    assert contents[0].startswith("entry_time")
    assert "105.0" in contents[1]
