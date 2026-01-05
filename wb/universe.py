# wb/universe.py
from __future__ import annotations

import io
from typing import List

import pandas as pd
import requests
import streamlit as st

UNIVERSE_MODULE_VERSION = "2026-01-05_v3_no_read_html_url"

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# S&P 500 constituents (CSV) — avoids Wikipedia scraping entirely
SP500_DATAHUB = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
SP500_GITHUB = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"

# Nasdaq-100 constituents (CSV) — maintained in yfinance repo
NASDAQ100_GITHUB = "https://raw.githubusercontent.com/ranaroussi/yfinance/master/yfinance/data/nasdaq100.csv"


def _normalize_ticker(t: str) -> str:
    t = str(t).strip().upper()
    # BRK.B -> BRK-B, etc.
    t = t.replace(".", "-")
    return t


def _get_text(url: str, timeout: int = 25) -> str:
    r = requests.get(url, headers=_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text


def _read_csv_from_url(url: str) -> pd.DataFrame:
    txt = _get_text(url)
    return pd.read_csv(io.StringIO(txt))


def _extract_tickers(df: pd.DataFrame) -> List[str]:
    # Try common column names
    for col in ["Symbol", "symbol", "Ticker", "ticker"]:
        if col in df.columns:
            vals = df[col].astype(str).map(_normalize_ticker).tolist()
            return _dedupe(vals)

    # Single-column CSV fallback
    if df.shape[1] == 1:
        vals = df.iloc[:, 0].astype(str).map(_normalize_ticker).tolist()
        return _dedupe(vals)

    raise RuntimeError(f"Unable to find a ticker column in CSV columns={list(df.columns)}")


def _dedupe(vals: List[str]) -> List[str]:
    out, seen = [], set()
    for v in vals:
        v = _normalize_ticker(v)
        if not v or v == "NAN":
            continue
        if v not in seen:
            seen.add(v)
            out.append(v)
    return out


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def get_universe(which: str) -> List[str]:
    which = (which or "").strip().lower()

    if which == "sp500":
        # 1) datahub
        try:
            df = _read_csv_from_url(SP500_DATAHUB)
            t = _extract_tickers(df)
            if t:
                return t
        except Exception:
            pass

        # 2) github datasets
        df = _read_csv_from_url(SP500_GITHUB)
        t = _extract_tickers(df)
        if t:
            return t

        raise RuntimeError("Unable to fetch S&P 500 constituents from CSV sources.")

    if which == "nasdaq100":
        df = _read_csv_from_url(NASDAQ100_GITHUB)
        t = _extract_tickers(df)
        if t:
            return t
        raise RuntimeError("Unable to fetch Nasdaq-100 constituents from CSV source.")

    raise ValueError(f"Unknown universe: {which}")
