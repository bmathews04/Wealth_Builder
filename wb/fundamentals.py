from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict

import pandas as pd
import streamlit as st
import yfinance as yf

@st.cache_data(show_spinner=False, ttl=60 * 60)
def fetch_one(ticker: str) -> Dict[str, pd.DataFrame]:
    t = yf.Ticker(ticker)

    income = t.financials if isinstance(t.financials, pd.DataFrame) else pd.DataFrame()
    balance = t.balance_sheet if isinstance(t.balance_sheet, pd.DataFrame) else pd.DataFrame()
    cashflow = t.cashflow if isinstance(t.cashflow, pd.DataFrame) else pd.DataFrame()

    # Quarterly (helps trend scoring). Many tickers have it; some don’t.
    income_q = getattr(t, "quarterly_financials", pd.DataFrame())
    balance_q = getattr(t, "quarterly_balance_sheet", pd.DataFrame())
    cashflow_q = getattr(t, "quarterly_cashflow", pd.DataFrame())

    return {
        "income_a": income,
        "balance_a": balance,
        "cashflow_a": cashflow,
        "income_q": income_q if isinstance(income_q, pd.DataFrame) else pd.DataFrame(),
        "balance_q": balance_q if isinstance(balance_q, pd.DataFrame) else pd.DataFrame(),
        "cashflow_q": cashflow_q if isinstance(cashflow_q, pd.DataFrame) else pd.DataFrame(),
    }

def fetch_fundamentals_parallel(tickers, max_workers: int = 16) -> Dict[str, Dict[str, pd.DataFrame]]:
    out = {}
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = {ex.submit(fetch_one, t): t for t in tickers}
        for fut in as_completed(futs):
            t = futs[fut]
            try:
                out[t] = fut.result()
            except Exception:
                out[t] = {
                    "income_a": pd.DataFrame(),
                    "balance_a": pd.DataFrame(),
                    "cashflow_a": pd.DataFrame(),
                    "income_q": pd.DataFrame(),
                    "balance_q": pd.DataFrame(),
                    "cashflow_q": pd.DataFrame(),
                }
    return out
