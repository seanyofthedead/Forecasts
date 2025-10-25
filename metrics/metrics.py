"""Trading performance metrics and logging utilities."""
from __future__ import annotations

import csv
import os
from typing import Dict, Iterable, List


def log_trades_to_csv(trade_log: Iterable[Dict[str, object]], filename: str = "trades_log.csv") -> None:
    fieldnames = ["entry_time", "exit_time", "action", "shares", "entry_price", "exit_price", "pnl", "reason"]
    write_header = not os.path.isfile(filename)
    with open(filename, "a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for trade in trade_log:
            if trade.get("exit_time") is not None:
                writer.writerow(trade)


def compute_metrics(trade_log: List[Dict[str, object]]) -> Dict[str, float]:
    if not trade_log:
        return {}
    completed = [trade for trade in trade_log if trade.get("exit_price") is not None]
    if not completed:
        return {}
    wins = [trade for trade in completed if trade.get("pnl", 0) > 0]
    losses = [trade for trade in completed if trade.get("pnl", 0) < 0]

    total_trades = len(completed)
    win_rate = (len(wins) / total_trades) * 100 if total_trades else 0.0
    avg_win = sum(t["pnl"] for t in wins) / len(wins) if wins else 0.0
    avg_loss = sum(t["pnl"] for t in losses) / len(losses) if losses else 0.0
    total_profit = sum(t["pnl"] for t in wins)
    total_loss = sum(t["pnl"] for t in losses)
    profit_factor = abs(total_profit) / abs(total_loss) if total_loss != 0 else float("inf")

    equity = 0.0
    peak = 0.0
    max_drawdown = 0.0
    consecutive_losses = 0
    max_consecutive_losses = 0
    for trade in completed:
        pnl = trade.get("pnl", 0.0) or 0.0
        equity += pnl
        if equity > peak:
            peak = equity
        drawdown = peak - equity
        if drawdown > max_drawdown:
            max_drawdown = drawdown
        if pnl < 0:
            consecutive_losses += 1
            max_consecutive_losses = max(max_consecutive_losses, consecutive_losses)
        else:
            consecutive_losses = 0

    max_drawdown_pct = (max_drawdown / peak * 100) if peak else 0.0

    return {
        "Total Trades": total_trades,
        "Win Rate (%)": round(win_rate, 2),
        "Average Win": round(avg_win, 2),
        "Average Loss": round(abs(avg_loss), 2),
        "Profit Factor": round(profit_factor, 2),
        "Max Drawdown (%)": round(max_drawdown_pct, 2),
        "Consecutive Losers": max_consecutive_losses,
        "Net Profit": round(total_profit + total_loss, 2),
    }
