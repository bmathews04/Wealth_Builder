from typing import Dict, Optional, List
import numpy as np
import pandas as pd

def _last(series: pd.Series) -> Optional[float]:
    s = series.dropna()
    if s.empty:
        return None
    return float(s.iloc[0])

def _cagr(start: float, end: float, years: float) -> Optional[float]:
    if start is None or end is None or start <= 0 or years <= 0:
        return None
    return (end / start) ** (1 / years) - 1

def _get_row(df: pd.DataFrame, keys: List[str]) -> Optional[pd.Series]:
    for k in keys:
        if k in df.index:
            return df.loc[k]
    return None

def _price_metrics(prices: pd.DataFrame, ticker: str) -> Dict[str, Optional[float]]:
    g = prices[prices["ticker"] == ticker].dropna(subset=["close"])
    if g.empty:
        return {
            "above_ma200": None,
            "above_ma40w": None,
            "rs_vs_spy_12m": None,
            "max_drawdown_2y": None,
        }

    last = g.iloc[-1]
    above_ma200 = None if pd.isna(last.get("ma200")) else bool(last["close"] > last["ma200"])
    above_ma40w = None if pd.isna(last.get("ma40w")) else bool(last["close"] > last["ma40w"])

    # Relative strength vs SPY over 12 months: (stock_12m - spy_12m)
    stock_12m = last.get("ret_252d")
    rs = None
    spy = prices[prices["ticker"] == "SPY"].dropna(subset=["close"])
    if spy.shape[0] > 260 and g.shape[0] > 260:
        spy_last = spy.iloc[-1]
        spy_12m = spy_last.get("ret_252d")
        if pd.notna(stock_12m) and pd.notna(spy_12m):
            rs = float(stock_12m - spy_12m)

    # Max drawdown (2y): use last 504 trading days
    g2 = g.tail(504)
    if g2.empty:
        mdd = None
    else:
        cummax = g2["close"].cummax()
        dd = (g2["close"] / cummax) - 1.0
        mdd = float(dd.min()) if not dd.empty else None
        if mdd is not None:
            mdd = abs(mdd)

    return {
        "above_ma200": above_ma200,
        "above_ma40w": above_ma40w,
        "rs_vs_spy_12m": rs,
        "max_drawdown_2y": mdd,
    }

def _fund_metrics(f: Dict[str, pd.DataFrame]) -> Dict[str, Optional[float]]:
    inc = f.get("income_a", pd.DataFrame())
    bal = f.get("balance_a", pd.DataFrame())
    cf = f.get("cashflow_a", pd.DataFrame())

    # Revenue CAGR (3y if possible)
    rev_cagr_3y = None
    rev = _get_row(inc, ["Total Revenue", "TotalRevenue", "Revenue"])
    if rev is not None:
        r = rev.dropna()
        if len(r) >= 4:
            end = float(r.iloc[0])
            start = float(r.iloc[3])
            rev_cagr_3y = _cagr(start, end, 3)

    # Gross margin and op margin (latest)
    gross_margin = None
    op_margin = None

    gp = _get_row(inc, ["Gross Profit", "GrossProfit"])
    op = _get_row(inc, ["Operating Income", "OperatingIncome"])
    if rev is not None and gp is not None:
        tr = _last(rev)
        gp_last = _last(gp)
        if tr and gp_last is not None and tr != 0:
            gross_margin = gp_last / tr

    if rev is not None and op is not None:
        tr = _last(rev)
        op_last = _last(op)
        if tr and op_last is not None and tr != 0:
            op_margin = op_last / tr

    # FCF (approx): OCF + CapEx
    ocf = _get_row(cf, ["Total Cash From Operating Activities", "Operating Cash Flow", "OperatingCashFlow"])
    capex = _get_row(cf, ["Capital Expenditures", "CapitalExpenditures"])
    fcf_ttm = None
    if ocf is not None and capex is not None:
        ocf_last = _last(ocf)
        capex_last = _last(capex)
        if ocf_last is not None and capex_last is not None:
            fcf_ttm = ocf_last + capex_last

    fcf_margin = None
    tr = _last(rev) if rev is not None else None
    if fcf_ttm is not None and tr is not None and tr != 0:
        fcf_margin = fcf_ttm / tr

    # Debt/Equity
    debt = _get_row(bal, ["Total Debt", "TotalDebt", "Long Term Debt", "LongTermDebt"])
    equity = _get_row(bal, ["Total Stockholder Equity", "Stockholders Equity", "Total Equity Gross Minority Interest"])
    debt_to_equity = None
    if debt is not None and equity is not None:
        d = _last(debt)
        e = _last(equity)
        if d is not None and e is not None and e != 0:
            debt_to_equity = d / e

    # SBC % revenue (approx, annual)
    sbc = _get_row(cf, ["Stock Based Compensation", "StockBasedCompensation"])
    sbc_pct_rev = None
    if sbc is not None and tr is not None and tr != 0:
        sbc_last = _last(sbc)
        if sbc_last is not None:
            sbc_pct_rev = sbc_last / tr

    # Shares change proxy (3y) – often missing; try a few rows
    shares = _get_row(bal, ["Common Stock Shares Outstanding", "Ordinary Shares Number", "Share Issued"])
    shares_change_3y = None
    if shares is not None:
        s = shares.dropna()
        if len(s) >= 4:
            end = float(s.iloc[0])
            start = float(s.iloc[3])
            if start != 0:
                shares_change_3y = (end / start) - 1

    # ROIC-ish proxy: EBIT / invested capital proxy
    ebit = _get_row(inc, ["EBIT", "Ebit"])
    cash = _get_row(bal, ["Cash And Cash Equivalents", "CashAndCashEquivalents", "Cash"])
    roic_proxy = None
    if ebit is not None and equity is not None:
        ebit_last = _last(ebit)
        d = _last(debt) if debt is not None else 0.0
        e = _last(equity)
        c = _last(cash) if cash is not None else 0.0
        if ebit_last is not None and e is not None:
            invested_cap = (d + e - c)
            if invested_cap and invested_cap > 0:
                roic_proxy = ebit_last / invested_cap

    return {
        "rev_cagr_3y": rev_cagr_3y,
        "gross_margin": gross_margin,
        "op_margin": op_margin,
        "fcf_ttm": fcf_ttm,
        "fcf_margin": fcf_margin,
        "debt_to_equity": debt_to_equity,
        "sbc_pct_rev": sbc_pct_rev,
        "shares_change_3y": shares_change_3y,
        "roic_proxy": roic_proxy,
    }

def build_metrics_table(tickers, prices: pd.DataFrame, fundamentals: Dict[str, Dict[str, pd.DataFrame]]) -> pd.DataFrame:
    # Ensure SPY in prices for RS calc
    want = list(dict.fromkeys(tickers + ["SPY"]))
    # Prices already fetched in app; if SPY missing, RS metric will be None.

    rows = []
    for t in tickers:
        pm = _price_metrics(prices, t)
        fm = _fund_metrics(fundamentals.get(t, {}))
        rows.append({"ticker": t, **pm, **fm})

    return pd.DataFrame(rows)
