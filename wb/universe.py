# wb/universe.py
from __future__ import annotations

from typing import List
import re
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_NDX = "https://en.wikipedia.org/wiki/Nasdaq-100"

# StockAnalysis tends to have a simple holdings table we can scrape with headers
STOCKANALYSIS_ETF_HOLDINGS = "https://stockanalysis.com/etf/{ticker}/holdings/"

# A browser-like User-Agent avoids lots of 403s in hosted environments
UA_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}


def _normalize_tickers(series: pd.Series) -> List[str]:
    """Uppercase, strip, and normalize common ticker formatting."""
    tickers = (
        series.astype(str)
        .str.upper()
        .str.strip()
        # Yahoo Finance uses '-' for class shares; Wikipedia often uses '.'
        .str.replace(".", "-", regex=False)
    )
    tickers = [t for t in tickers.tolist() if t and t != "NAN"]
    # de-dup while preserving order
    seen = set()
    out = []
    for t in tickers:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


def _read_html_with_headers(url: str) -> List[pd.DataFrame]:
    """Fetch HTML with headers (avoids 403) then parse via pandas."""
    r = requests.get(url, headers=UA_HEADERS, timeout=25)
    r.raise_for_status()
    # pandas can parse from raw html text
    return pd.read_html(r.text)


def _get_sp500_from_wiki() -> List[str]:
    tables = _read_html_with_headers(WIKI_SP500)
    df = tables[0]
    if "Symbol" not in df.columns:
        raise ValueError("Unexpected Wikipedia S&P 500 table format.")
    return _normalize_tickers(df["Symbol"])


def _get_nasdaq100_from_wiki() -> List[str]:
    tables = _read_html_with_headers(WIKI_NDX)
    # Wikipedia table layout can vary; find a table with a Ticker/Symbol column
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any("ticker" in c for c in cols) or any("symbol" in c for c in cols):
            # choose the first matching column
            for c in t.columns:
                cl = str(c).lower()
                if "ticker" in cl or "symbol" in cl:
                    return _normalize_tickers(t[c])
    raise ValueError("Could not find Nasdaq-100 constituent table on Wikipedia.")


def _get_etf_holdings_stockanalysis(etf: str) -> List[str]:
    url = STOCKANALYSIS_ETF_HOLDINGS.format(ticker=etf.lower())
    tables = _read_html_with_headers(url)

    # Find a table that looks like holdings (has "Symbol" or "Ticker")
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if "symbol" in cols:
            return _normalize_tickers(t["Symbol"])
        if "ticker" in cols:
            return _normalize_tickers(t["Ticker"])

    raise ValueError(f"Could not parse holdings table for ETF {etf} from StockAnalysis.")


def _get_etf_holdings_yfinance_fallback(etf: str) -> List[str]:
    """
    Last-resort fallback: yfinance fund data is sometimes only top holdings.
    Still better than crashing.
    """
    tk = yf.Ticker(etf)
    # Newer yfinance versions expose funds_data; some environments don’t.
    fd = getattr(tk, "funds_data", None)
    if fd is None:
        raise ValueError("yfinance funds_data not available in this environment.")

    top = getattr(fd, "top_holdings", None)
    if top is None or not isinstance(top, pd.DataFrame) or top.empty:
        raise ValueError("yfinance funds_data.top_holdings not available/empty.")

    # Column name varies; try common ones
    for col in ["Symbol", "symbol", "Holding", "holding", "Ticker", "ticker"]:
        if col in top.columns:
            return _normalize_tickers(top[col])

    # Sometimes it’s index-like
    if top.index is not None and len(top.index) > 0:
        idx = pd.Series(top.index.astype(str))
        # keep only things that resemble tickers
        idx = idx[idx.str.match(r"^[A-Z0-9\.-]{1,10}$", na=False)]
        if len(idx) > 0:
            return _normalize_tickers(idx)

    raise ValueError("Could not infer tickers from yfinance top_holdings.")


@st.cache_data(show_spinner=False, ttl=60 * 60 * 12)
def get_universe(which: str) -> List[str]:
    """
    Supported:
      - "sp500"         (Wikipedia, with UA headers)
      - "nasdaq100"     (Wikipedia, with UA headers)
      - "etf:SPY"       (ETF holdings; tries StockAnalysis; falls back)
      - "etf:QQQ"
    """
    which = (which or "").strip().lower()

    if which == "sp500":
        return _get_sp500_from_wiki()

    if which == "nasdaq100":
        return _get_nasdaq100_from_wiki()

    if which.startswith("etf:"):
        etf = which.split(":", 1)[1].strip().upper()
        if not etf:
            raise ValueError("ETF universe requires a ticker, e.g. etf:SPY")

        # Primary: StockAnalysis holdings
        try:
            return _get_etf_holdings_stockanalysis(etf)
        except Exception:
            # Fallback: if someone passes SPY/QQQ and StockAnalysis fails,
            # try Wikipedia proxies as a backup where applicable
            if etf in {"SPY", "VOO", "IVV"}:
                try:
                    return _get_sp500_from_wiki()
                except Exception:
                    pass
            if etf in {"QQQ", "QQQM"}:
                try:
                    return _get_nasdaq100_from_wiki()
                except Exception:
                    pass

            # Last fallback: yfinance (may be top holdings only)
            return _get_etf_holdings_yfinance_fallback(etf)

    raise ValueError(f"Unknown universe '{which}'. Try 'sp500', 'nasdaq100', or 'etf:SPY'.")
