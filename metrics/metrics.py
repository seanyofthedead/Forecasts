"""Trade logging and performance metric helpers."""
from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List


def log_trades_to_csv(trade_log: Iterable[Dict], filename: str = "trades_log.csv") -> None:
    """Append completed trades from ``trade_log`` into ``filename``."""
    fieldnames = [
        "entry_time",
        "exit_time",
        "action",
        "shares",
        "entry_price",
        "exit_price",
        "pnl",
        "reason",
    ]
    write_header = not os.path.isfile(filename)
    with open(filename, mode="a", newline="") as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for trade in trade_log:
            if trade.get("exit_time") is None:
                continue
            writer.writerow(trade)


def compute_metrics(trade_log: List[Dict]) -> Dict[str, float]:
    if not trade_log:
        return {}

    completed = [trade for trade in trade_log if trade.get("exit_price") is not None]
    if not completed:
        return {}

    wins = [t for t in completed if (t.get("pnl") or 0) > 0]
    losses = [t for t in completed if (t.get("pnl") or 0) < 0]

    num_trades = len(completed)
    num_wins = len(wins)
    num_losses = len(losses)

    total_profit = sum(t.get("pnl", 0) for t in wins)
    total_loss = sum(t.get("pnl", 0) for t in losses)

    win_rate = (num_wins / num_trades * 100.0) if num_trades else 0.0
    average_win = (total_profit / num_wins) if num_wins else 0.0
    average_loss = abs(total_loss / num_losses) if num_losses else 0.0
    profit_factor = abs(total_profit) / abs(total_loss) if total_loss else float("inf")

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    for trade in completed:
        pnl = trade.get("pnl") or 0.0
        equity += pnl
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, peak - equity)
    max_drawdown_pct = (max_drawdown / peak * 100.0) if peak else 0.0

    longest_losing_streak = 0
    current_streak = 0
    for trade in completed:
        if (trade.get("pnl") or 0) < 0:
            current_streak += 1
            longest_losing_streak = max(longest_losing_streak, current_streak)
        else:
            current_streak = 0

    return {
        "Total Trades": num_trades,
        "Win Rate (%)": round(win_rate, 2),
        "Average Win": round(average_win, 2),
        "Average Loss": round(average_loss, 2),
        "Profit Factor": round(profit_factor, 2) if profit_factor != float("inf") else float("inf"),
        "Max Drawdown (%)": round(max_drawdown_pct, 2),
        "Consecutive Losers": longest_losing_streak,
        "Net Profit": round(total_profit + total_loss, 2),
    }
