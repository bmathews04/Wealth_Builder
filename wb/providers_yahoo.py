from __future__ import annotations
from typing import Optional, Dict
import pandas as pd
import yfinance as yf

from wb.providers import FundamentalsBundle, FundamentalsProvider


class YahooProvider(FundamentalsProvider):
    name = "yahoo"

    def get(self, ticker: str) -> FundamentalsBundle:
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
