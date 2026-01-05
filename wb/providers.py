from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional

import pandas as pd

@dataclass
class FundamentalsBundle:
    income_a: pd.DataFrame
    balance_a: pd.DataFrame
    cashflow_a: pd.DataFrame
    income_q: pd.DataFrame
    balance_q: pd.DataFrame
    cashflow_q: pd.DataFrame
    meta: Dict[str, Optional[str]]  # sector, industry

class FundamentalsProvider:
    def get(self, ticker: str) -> FundamentalsBundle:
        raise NotImplementedError
