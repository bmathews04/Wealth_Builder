from __future__ import annotations
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Optional
import random
import time
import pandas as pd
import streamlit as st

from wb.providers import FundamentalsBundle
from wb.providers_yahoo import YahooProvider
from wb.providers_fmp import get_fmp_provider_from_secrets


def _retry_get(provider, ticker: str, retries: int, base_sleep: float) -> FundamentalsBundle:
    last_exc = None
    for i in range(retries + 1):
        try:
            return provider.get(ticker)
        except Exception as e:
            last_exc = e
            # exponential backoff + jitter
            sleep = base_sleep * (2 ** i) + random.random() * 0.25
            time.sleep(min(sleep, 6.0))
    # Return empty bundle on failure
    return FundamentalsBundle(
        income_a=pd.DataFrame(),
        balance_a=pd.DataFrame(),
        cashflow_a=pd.DataFrame(),
        income_q=pd.DataFrame(),
        balance_q=pd.DataFrame(),
        cashflow_q=pd.DataFrame(),
        meta={"sector": None, "industry": None},
    )


def _select_provider() -> object:
    # Priority: st.secrets["DATA_PROVIDER"] if set; default yahoo
    provider_name = "yahoo"
    try:
        provider_name = st.secrets.get("DATA_PROVIDER", "yahoo")
    except Exception:
        provider_name = "yahoo"

    provider_name = str(provider_name).lower().strip()

    if provider_name == "fmp":
        fmp = get_fmp_provider_from_secrets()
        if fmp is not None:
            return fmp
        # fallback
        return YahooProvider()

    return YahooProvider()


def fetch_fundamentals_parallel(tickers, max_workers: int = 16) -> Dict[str, dict]:
    """
    Provider-backed fundamentals fetch with retries + throttling safety.
    Returns dict per ticker compatible with existing metrics code.
    """
    provider = _select_provider()

    throttle_safe = False
    try:
        throttle_safe = bool(st.secrets.get("THROTTLE_SAFE_MODE", False))
    except Exception:
        throttle_safe = False

    # In throttle-safe mode, reduce workers and increase retries
    workers = min(max_workers, 8) if throttle_safe else max_workers
    retries = 4 if throttle_safe else 2
    base_sleep = 0.4 if throttle_safe else 0.2

    out: Dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_retry_get, provider, t, retries, base_sleep): t for t in tickers}
        for fut in as_completed(futs):
            t = futs[fut]
            b = fut.result()
            out[t] = {
                "income_a": b.income_a,
                "balance_a": b.balance_a,
                "cashflow_a": b.cashflow_a,
                "income_q": b.income_q,
                "balance_q": b.balance_q,
                "cashflow_q": b.cashflow_q,
                "meta": b.meta,
                "provider": getattr(provider, "name", "unknown"),
            }
    return out
