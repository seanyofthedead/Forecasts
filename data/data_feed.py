"""Data ingestion utilities for the live trading simulator."""
from __future__ import annotations

import os
import time
from typing import Dict, Generator, Iterable, Optional

import requests

try:  # pragma: no cover - optional dependency
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

ALPHA_VANTAGE_API_KEY = os.getenv("ALPHAVANTAGE_API_KEY", "")


def fetch_intraday_yahoo(symbol: str, interval: str = "1m", period: str = "1d"):
    """Fetch intraday data for ``symbol`` using :mod:`yfinance`.

    Parameters
    ----------
    symbol:
        The ticker symbol (e.g. ``"AAPL"``).
    interval:
        Candle interval passed to yfinance (defaults to one minute).
    period:
        The period of data to fetch (defaults to the latest trading day).

    Returns
    -------
    pandas.DataFrame
        A dataframe indexed by datetime with the usual OHLCV columns.

    Notes
    -----
    This helper requires :mod:`yfinance`.  When the dependency is missing an
    informative :class:`RuntimeError` is raised to keep the scaffold importable
    without optional packages.
    """

    if yf is None:
        raise RuntimeError(
            "yfinance is required to fetch Yahoo Finance data. Install it via "
            "`pip install yfinance`."
        )

    return yf.download(tickers=symbol, interval=interval, period=period)


def fetch_intraday_alphavantage(
    symbol: str, interval: str = "1min", output_size: str = "compact"
) -> Dict:
    """Fetch intraday OHLCV candles from the Alpha Vantage REST API."""
    if not ALPHA_VANTAGE_API_KEY:
        raise RuntimeError("Alpha Vantage API key not set. Export ALPHAVANTAGE_API_KEY.")

    url = (
        "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY"
        f"&symbol={symbol}&interval={interval}&outputsize={output_size}"
        f"&apikey={ALPHA_VANTAGE_API_KEY}&datatype=json"
    )
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    return response.json()


class DataStreamer:
    """Stream real-time or simulated real-time data for a ticker symbol."""

    def __init__(self, symbol: str, source: str = "yahoo", *, poll_interval: float = 1.0):
        self.symbol = symbol
        self.source = source
        self.poll_interval = poll_interval
        self._buffer: Optional[Iterable[Dict]] = None

    def _iter_yahoo(self) -> Iterable[Dict]:
        dataframe = fetch_intraday_yahoo(self.symbol, interval="1m", period="1d")
        for timestamp, row in dataframe.iterrows():
            yield {
                "datetime": timestamp.to_pydatetime(),
                "open": float(row["Open"]),
                "high": float(row["High"]),
                "low": float(row["Low"]),
                "close": float(row["Close"]),
                "volume": float(row["Volume"]),
            }

    def _iter_alphavantage(self) -> Iterable[Dict]:
        payload = fetch_intraday_alphavantage(self.symbol)
        time_series = payload.get("Time Series (1min)", {})
        for timestamp, values in sorted(time_series.items()):
            yield {
                "datetime": timestamp,
                "open": float(values["1. open"]),
                "high": float(values["2. high"]),
                "low": float(values["3. low"]),
                "close": float(values["4. close"]),
                "volume": float(values["5. volume"]),
            }

    def stream(self, live: bool = False) -> Generator[Dict, None, None]:
        """Yield ticks one at a time for the configured data source."""
        if self.source == "yahoo":
            iterator = self._iter_yahoo()
        else:
            iterator = self._iter_alphavantage()

        for tick in iterator:
            yield tick
            if live:
                time.sleep(self.poll_interval)
