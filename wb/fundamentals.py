from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

import pandas as pd
import streamlit as st
import yfinance as yf

from wb.providers import FundamentalsBundle

@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_one(ticker: str) -> FundamentalsBundle:
    t = yf.Ticker(ticker)

    income_a = t.financials if isinstance(t.financials, pd.DataFrame) else pd.DataFrame()
    balance_a = t.balance_sheet if isinstance(t.balance_sheet, pd.DataFrame) else pd.DataFrame()
    cashflow_a = t.cashflow if isinstance(t.cashflow, pd.DataFrame) else pd.DataFrame()

    income_q = getattr(t, "quarterly_financials", pd.DataFrame())
    balance_q = getattr(t, "quarterly_balance_sheet", pd.DataFrame())
    cashflow_q = getattr(t, "quarterly_cashflow", pd.DataFrame())

    if not isinstance(income_q, pd.DataFrame): income_q = pd.DataFrame()
    if not isinstance(balance_q, pd.DataFrame): balance_q = pd.DataFrame()
    if not isinstance(cashflow_q, pd.DataFrame): cashflow_q = pd.DataFrame()

    # Best-effort metadata (may be missing)
    sector = None
    industry = None
    try:
        info = getattr(t, "info", None)
        if isinstance(info, dict):
            sector = info.get("sector")
            industry = info.get("industry")
    except Exception:
        pass

    return FundamentalsBundle(
        income_a=income_a,
        balance_a=balance_a,
        cashflow_a=cashflow_a,
        income_q=income_q,
        balance_q=balance_q,
        cashflow_q=cashflow_q,
        meta={"sector": sector, "industry": industry},
    )

def fetch_fundamentals_parallel(tickers, max_workers: int = 16) -> Dict[str, dict]:
    """
    Returns a dict keyed by ticker containing:
      income_a, balance_a, cashflow_a, income_q, balance_q, cashflow_q, meta
    """
    out = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_one, t): t for t in tickers}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                b = fut.result()
                out[t] = {
                    "income_a": b.income_a,
                    "balance_a": b.balance_a,
                    "cashflow_a": b.cashflow_a,
                    "income_q": b.income_q,
                    "balance_q": b.balance_q,
                    "cashflow_q": b.cashflow_q,
                    "meta": b.meta,
                }
            except Exception:
                out[t] = {
                    "income_a": pd.DataFrame(),
                    "balance_a": pd.DataFrame(),
                    "cashflow_a": pd.DataFrame(),
                    "income_q": pd.DataFrame(),
                    "balance_q": pd.DataFrame(),
                    "cashflow_q": pd.DataFrame(),
                    "meta": {"sector": None, "industry": None},
                }
    return out
