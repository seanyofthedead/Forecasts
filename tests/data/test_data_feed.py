"""Tests for data ingestion utilities."""
from __future__ import annotations

from typing import List

import pandas as pd
import pytest

from data import data_feed


class DummyResponse:
    def __init__(self, payload):
        self._payload = payload

    def raise_for_status(self):  # pragma: no cover - nothing to do for dummy
        return None

    def json(self):  # pragma: no cover - simple passthrough
        return self._payload


def test_fetch_intraday_alphavantage_requires_api_key(monkeypatch):
    monkeypatch.setattr(data_feed, "ALPHA_VANTAGE_API_KEY", "")
    with pytest.raises(RuntimeError):
        data_feed.fetch_intraday_alphavantage("TEST")


def test_fetch_intraday_alphavantage_parses_payload(monkeypatch):
    monkeypatch.setattr(data_feed, "ALPHA_VANTAGE_API_KEY", "demo")
    payload = {
        "Time Series (1min)": {
            "2024-01-02 09:30:00": {
                "1. open": "100.0",
                "2. high": "101.0",
                "3. low": "99.5",
                "4. close": "100.5",
                "5. volume": "1000",
            }
        }
    }
    monkeypatch.setattr(data_feed.requests, "get", lambda *a, **k: DummyResponse(payload))
    result = data_feed.fetch_intraday_alphavantage("TEST")
    assert result == payload


def test_data_streamer_iterates_yahoo(monkeypatch):
    timestamps = pd.date_range("2024-01-02 09:30", periods=2, freq="1min")
    dataframe = pd.DataFrame(
        {
            "Open": [100.0, 101.0],
            "High": [101.0, 102.0],
            "Low": [99.0, 100.0],
            "Close": [100.5, 101.5],
            "Volume": [1_000, 1_200],
        },
        index=timestamps,
    )
    monkeypatch.setattr(data_feed, "fetch_intraday_yahoo", lambda *a, **k: dataframe)
    streamer = data_feed.DataStreamer("TEST", source="yahoo")
    ticks = list(streamer.stream())
    assert len(ticks) == 2
    assert ticks[0]["datetime"] == timestamps[0].to_pydatetime()
    assert ticks[1]["close"] == pytest.approx(101.5)


def test_data_streamer_iterates_alphavantage(monkeypatch):
    payload = {
        "Time Series (1min)": {
            "2024-01-02 09:30:00": {
                "1. open": "100.0",
                "2. high": "101.0",
                "3. low": "99.5",
                "4. close": "100.5",
                "5. volume": "1000",
            },
            "2024-01-02 09:31:00": {
                "1. open": "100.5",
                "2. high": "101.5",
                "3. low": "100.0",
                "4. close": "101.0",
                "5. volume": "900",
            },
        }
    }
    monkeypatch.setattr(data_feed, "fetch_intraday_alphavantage", lambda *a, **k: payload)
    streamer = data_feed.DataStreamer("TEST", source="alphavantage")
    ticks: List[dict] = list(streamer.stream())
    assert [tick["close"] for tick in ticks] == [100.5, 101.0]
