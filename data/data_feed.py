"""Data ingestion utilities for the trading simulation framework."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Generator, Iterable, Optional

import os
import time

try:  # Optional dependency used for historical downloads.
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover - the scaffold should still work without yfinance
    yf = None

import requests

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")


def fetch_intraday_yahoo(symbol: str, interval: str = "1m", period: str = "1d"):
    """Return an OHLCV DataFrame from Yahoo Finance using ``yfinance``.

    The function requires ``yfinance`` to be installed.  The caller is responsible
    for handling the pandas DataFrame that is returned.  If ``yfinance`` is not
    available, a ``RuntimeError`` is raised so that the CLI can report a helpful
    message to the user.
    """
    if yf is None:
        raise RuntimeError("yfinance is not installed. Install it to enable Yahoo Finance downloads.")
    return yf.download(tickers=symbol, interval=interval, period=period)


def fetch_intraday_alphavantage(symbol: str, interval: str = "1min", output_size: str = "compact") -> Dict[str, Dict[str, str]]:
    """Fetch intraday data from the Alpha Vantage REST API.

    Parameters
    ----------
    symbol:
        Stock ticker to request.
    interval:
        Bar interval supported by Alpha Vantage (``1min``, ``5min``, etc.).
    output_size:
        ``compact`` or ``full`` as defined by the API.
    """
    if not ALPHA_VANTAGE_API_KEY:
        raise RuntimeError("Alpha Vantage API key not set. Export ALPHAVANTAGE_API_KEY before calling this function.")
    url = (
        "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY"
        f"&symbol={symbol}&interval={interval}&outputsize={output_size}"
        f"&apikey={ALPHA_VANTAGE_API_KEY}&datatype=json"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    payload: Dict[str, Dict[str, str]] = response.json()
    return payload


@dataclass
class Tick:
    """A single OHLCV bar used by the simulation engine."""

    datetime: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    metadata: Optional[Dict[str, float]] = None

    def as_dict(self) -> Dict[str, float]:
        base = {
            "datetime": self.datetime,
            "open": self.open,
            "high": self.high,
            "low": self.low,
            "close": self.close,
            "volume": self.volume,
        }
        if self.metadata:
            base.update(self.metadata)
        return base


class DataStreamer:
    """Yield market data either from historical files or live downloads.

    The streamer is intentionally simple – it is suitable for simulated live
    trading, and can be extended later to incorporate websockets or other
    real-time APIs.  When ``historical_path`` is provided the file is read once
    and yielded row-by-row.  Otherwise the class will periodically poll Yahoo
    Finance for the most recent data snapshot.
    """

    def __init__(
        self,
        symbol: str,
        source: str = "yahoo",
        historical_path: Optional[Path] = None,
        interval_seconds: int = 60,
    ) -> None:
        self.symbol = symbol
        self.source = source
        self.historical_path = Path(historical_path) if historical_path else None
        self.interval_seconds = interval_seconds

    def _stream_from_history(self) -> Iterable[Tick]:
        import csv

        if not self.historical_path or not self.historical_path.exists():
            raise FileNotFoundError(f"Historical data file {self.historical_path!s} not found.")
        with self.historical_path.open("r", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                yield Tick(
                    datetime=datetime.fromisoformat(row["datetime"]),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    volume=float(row["volume"]),
                )

    def stream(self, continuous: bool = False) -> Generator[Tick, None, None]:
        """Generate :class:`Tick` objects.

        Parameters
        ----------
        continuous:
            When ``True`` the function keeps polling Yahoo Finance.  When ``False``
            it yields the latest snapshot a single time which is useful for unit
            tests or scripted demonstrations.
        """
        if self.historical_path:
            for tick in self._stream_from_history():
                yield tick
            return

        if self.source != "yahoo":
            raise NotImplementedError("Only Yahoo Finance streaming is supported in the scaffold.")

        while True:
            df = fetch_intraday_yahoo(self.symbol, interval="1m", period="1d")
            for index, row in df.iterrows():
                yield Tick(
                    datetime=index.to_pydatetime(),
                    open=float(row["Open"]),
                    high=float(row["High"]),
                    low=float(row["Low"]),
                    close=float(row["Close"]),
                    volume=float(row["Volume"]),
                )
            if not continuous:
                break
            time.sleep(self.interval_seconds)
