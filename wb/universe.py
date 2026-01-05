# wb/universe.py
from __future__ import annotations

import io
import re
from typing import List, Optional

import pandas as pd
import requests
import streamlit as st


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

# Prefer CSV endpoints (less likely to block) ✅
DATAHUB_SP500_CSV = "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv"
GITHUB_SP500_CSV = "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/main/data/constituents.csv"

# Nasdaq-100: reliable CSV is harder; we’ll use Wikipedia HTML via requests->read_html(html_str),
# and if blocked, we fall back to a maintained GitHub list.
GITHUB_NDX_CSV = "https://raw.githubusercontent.com/ranaroussi/yfinance/master/yfinance/data/nasdaq100.csv"
WIKI_NDX = "https://en.wikipedia.org/wiki/Nasdaq-100"


def _get_text(url: str, timeout: int = 25) -> str:
    r = requests.get(url, headers=_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text


def _normalize_ticker(t: str) -> str:
    t = str(t).strip().upper()
    # Wikipedia often uses BRK.B; yfinance uses BRK-B
    t = t.replace(".", "-")
    return t


def _dedupe_preserve_order(items: List[str]) -> List[str]:
    out, seen = [], set()
    for x in items:
        x = _normalize_ticker(x)
        if x and x != "NAN" and x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _try_sp500_csv(url: str) -> List[str]:
    txt = _get_text(url)
    df = pd.read_csv(io.StringIO(txt))
    # datahub + datasets format: Symbol column exists
    for col in df.columns:
        if str(col).strip().lower() == "symbol":
            return _dedupe_preserve_order(df[col].astype(str).tolist())
    return []


def _try_ndx_github_csv(url: str) -> List[str]:
    """
    yfinance repo includes a nasdaq100.csv (tickers only). If this ever moves, we’ll just return [].
    """
    txt = _get_text(url)
    df = pd.read_csv(io.StringIO(txt))
    # expected column names vary; handle common ones
    for candidate in ["Symbol", "symbol", "Ticker", "ticker"]:
        if candidate in df.columns:
            return _dedupe_preserve_order(df[candidate].astype(str).tolist())
    # if single column csv
    if df.shape[1] == 1:
        return _dedupe_preserve_order(df.iloc[:, 0].astype(str).tolist())
    return []


def _try_wikipedia_html_tickers(url: str) -> List[str]:
    """
    Fetch HTML with requests, then parse tables from the HTML string (NOT the URL).
    """
    html = _get_text(url)
    tables = pd.read_html(html)
    if not tables:
        return []

    # Find a table with 'Ticker' or 'Symbol'
    best = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns.astype(str)]
        if any("ticker" in c or "symbol" in c for c in cols):
            best = t
            break
    if best is None:
        best = tables[0]

    ticker_col = None
    for c in best.columns:
        lc = str(c).lower()
        if "ticker" in lc or "symbol" in lc:
            ticker_col = c
            break

    if ticker_col is not None:
        return _dedupe_preserve_order(best[ticker_col].astype(str).tolist())

    # Regex fallback if no ticker column
    joined = best.astype(str).agg(" ".join, axis=1).tolist()
    found = []
    for row in joined:
        found.extend(re.findall(r"\b[A-Z]{1,5}(?:[.\-][A-Z])?\b", row))
    return _dedupe_preserve_order(found)


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def get_universe(which: str) -> List[str]:
    which = (which or "").strip().lower()

    if which == "sp500":
        # 1) datahub CSV
        try:
            tickers = _try_sp500_csv(DATAHUB_SP500_CSV)
            if tickers:
                return tickers
        except Exception:
            pass

        # 2) github datasets CSV
        try:
            tickers = _try_sp500_csv(GITHUB_SP500_CSV)
            if tickers:
                return tickers
        except Exception:
            pass

        raise RuntimeError("Unable to fetch S&P 500 tickers from CSV sources (blocked).")

    if which == "nasdaq100":
        # 1) github nasdaq100.csv (fast + stable-ish)
        try:
            tickers = _try_ndx_github_csv(GITHUB_NDX_CSV)
            if tickers:
                return tickers
        except Exception:
            pass

        # 2) wikipedia (requests -> parse html string)
        try:
            tickers = _try_wikipedia_html_tickers(WIKI_NDX)
            if tickers:
                return tickers
        except Exception:
            pass

        raise RuntimeError("Unable to fetch Nasdaq-100 tickers (GitHub + Wikipedia blocked).")

    raise ValueError(f"Unknown universe: {which}")
