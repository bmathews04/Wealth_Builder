from __future__ import annotations
from typing import Dict, Optional, List
import pandas as pd
import requests
import streamlit as st

from wb.providers import FundamentalsBundle, FundamentalsProvider


class FMPProvider(FundamentalsProvider):
    name = "fmp"

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base = "https://financialmodelingprep.com"

    def _get_json(self, path: str, params: Dict) -> List[Dict]:
        url = f"{self.base}{path}"
        r = requests.get(url, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
        return data if isinstance(data, list) else []

    def _df_from_list(self, rows: List[Dict]) -> pd.DataFrame:
        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        # Use 'date' as columns (statement format similar to yfinance: rows as line items)
        if "date" in df.columns:
            df = df.set_index("date").T
        return df

    def get(self, ticker: str) -> FundamentalsBundle:
        # Annual statements
        income = self._df_from_list(self._get_json(
            f"/api/v3/income-statement/{ticker}",
            {"period": "annual", "limit": 5, "apikey": self.api_key},
        ))
        balance = self._df_from_list(self._get_json(
            f"/api/v3/balance-sheet-statement/{ticker}",
            {"period": "annual", "limit": 5, "apikey": self.api_key},
        ))
        cashflow = self._df_from_list(self._get_json(
            f"/api/v3/cash-flow-statement/{ticker}",
            {"period": "annual", "limit": 5, "apikey": self.api_key},
        ))

        # Metadata: profile endpoint (best-effort)
        sector = None
        industry = None
        try:
            prof = self._get_json(f"/api/v3/profile/{ticker}", {"apikey": self.api_key})
            if prof and isinstance(prof[0], dict):
                sector = prof[0].get("sector")
                industry = prof[0].get("industry")
        except Exception:
            pass

        # Quarterly left empty for now (we can add in next round)
        return FundamentalsBundle(
            income_a=income,
            balance_a=balance,
            cashflow_a=cashflow,
            income_q=pd.DataFrame(),
            balance_q=pd.DataFrame(),
            cashflow_q=pd.DataFrame(),
            meta={"sector": sector, "industry": industry},
        )


def get_fmp_provider_from_secrets() -> Optional[FMPProvider]:
    api_key = None
    try:
        api_key = st.secrets.get("FMP_API_KEY", None)
    except Exception:
        api_key = None
    if not api_key:
        return None
    return FMPProvider(api_key=api_key)
