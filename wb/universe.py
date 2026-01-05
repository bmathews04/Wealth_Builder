# wb/universe.py
from __future__ import annotations

import re
from typing import List

import pandas as pd
import requests
import streamlit as st


WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_NDX = "https://en.wikipedia.org/wiki/Nasdaq-100"

SLICKCHARTS_SP500 = "https://www.slickcharts.com/sp500"
NASDAQTRADER_LISTED = "https://api.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"


_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
}


def _get_html(url: str, timeout: int = 20) -> str:
    r = requests.get(url, headers=_HEADERS, timeout=timeout)
    r.raise_for_status()
    return r.text


def _normalize_ticker(t: str) -> str:
    t = str(t).strip().upper()
    # Wikipedia uses BRK.B; yfinance typically uses BRK-B
    t = t.replace(".", "-")
    return t


def _tables_from_url(url: str) -> List[pd.DataFrame]:
    html = _get_html(url)
    return pd.read_html(html)


def _try_sp500_wikipedia() -> List[str]:
    tables = _tables_from_url(WIKI_SP500)
    if not tables:
        return []
    df = tables[0]
    if "Symbol" not in df.columns:
        return []
    tickers = df["Symbol"].astype(str).map(_normalize_ticker).tolist()
    return [t for t in tickers if t and t != "NAN"]


def _try_ndx_wikipedia() -> List[str]:
    tables = _tables_from_url(WIKI_NDX)
    if not tables:
        return []

    # Find a table that likely contains tickers
    best = None
    for t in tables:
        cols = [c.lower() for c in t.columns.astype(str)]
        if any("ticker" in c or "symbol" in c for c in cols):
            best = t
            break

    if best is None:
        # Sometimes the first table contains companies with tickers embedded
        best = tables[0]

    # locate ticker column
    ticker_col = None
    for c in best.columns:
        lc = str(c).lower()
        if "ticker" in lc or "symbol" in lc:
            ticker_col = c
            break

    if ticker_col is None:
        # try parse tickers from any column by regex (fallback)
        joined = best.astype(str).agg(" ".join, axis=1).tolist()
        # crude: match 1-5 uppercase letters, optional .X or -X
        found = []
        for row in joined:
            m = re.findall(r"\b[A-Z]{1,5}(?:[.\-][A-Z])?\b", row)
            found.extend(m)
        tickers = [_normalize_ticker(x) for x in found]
        # de-dupe while preserving order
        out = []
        seen = set()
        for t in tickers:
            if t not in seen:
                seen.add(t)
                out.append(t)
        return out

    tickers = best[ticker_col].astype(str).map(_normalize_ticker).tolist()
    # De-dupe while preserving order
    out = []
    seen = set()
    for t in tickers:
        if t and t != "NAN" and t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _fallback_sp500_slickcharts() -> List[str]:
    tables = _tables_from_url(SLICKCHARTS_SP500)
    if not tables:
        return []
    df = tables[0]
    # slickcharts uses "Symbol" column
    sym_col = None
    for c in df.columns:
        if str(c).strip().lower() == "symbol":
            sym_col = c
            break
    if sym_col is None:
        return []
    tickers = df[sym_col].astype(str).map(_normalize_ticker).tolist()
    return [t for t in tickers if t and t != "NAN"]


def _fallback_nasdaqtrader_list() -> List[str]:
    """
    Nasdaqtrader provides ALL nasdaq listed symbols (not specifically NDX).
    We'll return a large liquid universe as last resort rather than erroring.
    """
    txt = _get_html(NASDAQTRADER_LISTED)
    lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
    # File is pipe-delimited with header; last line is footer.
    # Example header: Symbol|Security Name|Market Category|...
    if not lines or "|" not in lines[0]:
        return []

    rows = [ln.split("|") for ln in lines[1:] if "|" in ln]
    # drop footer row(s)
    rows = [r for r in rows if r and r[0] and not r[0].startswith("File Creation Time")]

    syms = []
    for r in rows:
        sym = _normalize_ticker(r[0])
        # Skip test issues / weirdness
        if sym and sym.isascii():
            syms.append(sym)

    # De-dupe preserve order
    out = []
    seen = set()
    for t in syms:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def get_universe(which: str) -> List[str]:
    which = (which or "").strip().lower()

    if which == "sp500":
        # Try Wikipedia with headers first
        try:
            tickers = _try_sp500_wikipedia()
            if tickers:
                return tickers
        except Exception:
            pass

        # Fallback
        try:
            tickers = _fallback_sp500_slickcharts()
            if tickers:
                return tickers
        except Exception:
            pass

        raise RuntimeError("Unable to fetch S&P 500 tickers (Wikipedia and fallback blocked).")

    if which == "nasdaq100":
        # Try Wikipedia first
        try:
            tickers = _try_ndx_wikipedia()
            if tickers:
                return tickers
        except Exception:
            pass

        # Fallback to Nasdaqtrader (not exact NDX, but avoids hard failure)
        try:
            tickers = _fallback_nasdaqtrader_list()
            if tickers:
                return tickers
        except Exception:
            pass

        raise RuntimeError("Unable to fetch Nasdaq-100 tickers (Wikipedia and fallback blocked).")

    raise ValueError(f"Unknown universe: {which}")
