from typing import Dict, Optional, List, Tuple
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

def _annual_series_to_ts(row: pd.Series) -> pd.Series:
    """
    yfinance statement columns are dates; convert to sorted time series.
    We coerce col names to datetimes if possible.
    """
    s = row.dropna()
    try:
        s.index = pd.to_datetime(s.index)
        s = s.sort_index()
    except Exception:
        # fallback: keep as-is but reverse if looks newest-first
        pass
    return s

def _price_metrics(prices: pd.DataFrame, ticker: str) -> Dict[str, Optional[float]]:
    g = prices[prices["ticker"] == ticker].dropna(subset=["close"])
    if g.empty:
        return {"above_ma200": None, "above_ma40w": None, "rs_vs_spy_12m": None, "max_drawdown_2y": None}

    last = g.iloc[-1]
    above_ma200 = None if pd.isna(last.get("ma200")) else bool(last["close"] > last["ma200"])
    above_ma40w = None if pd.isna(last.get("ma40w")) else bool(last["close"] > last["ma40w"])

    # RS vs SPY (12m): stock_12m - spy_12m
    rs = None
    stock_12m = last.get("ret_252d")
    spy = prices[prices["ticker"] == "SPY"].dropna(subset=["close"])
    if spy.shape[0] > 260 and g.shape[0] > 260:
        spy_last = spy.iloc[-1]
        spy_12m = spy_last.get("ret_252d")
        if pd.notna(stock_12m) and pd.notna(spy_12m):
            rs = float(stock_12m - spy_12m)

    # Max drawdown over ~2y
    g2 = g.tail(504)
    if g2.empty:
        mdd = None
    else:
        peak = g2["close"].cummax()
        dd = (g2["close"] / peak) - 1.0
        mdd = float(dd.min()) if not dd.empty else None
        if mdd is not None:
            mdd = abs(mdd)

    return {
        "above_ma200": above_ma200,
        "above_ma40w": above_ma40w,
        "rs_vs_spy_12m": rs,
        "max_drawdown_2y": mdd,
    }

def _calc_fcf_series(cf: pd.DataFrame) -> Optional[pd.Series]:
    ocf = _get_row(cf, ["Total Cash From Operating Activities", "Operating Cash Flow", "OperatingCashFlow"])
    capex = _get_row(cf, ["Capital Expenditures", "CapitalExpenditures"])
    if ocf is None or capex is None:
        return None
    ocf_ts = _annual_series_to_ts(ocf)
    capex_ts = _annual_series_to_ts(capex)
    joined = pd.concat([ocf_ts.rename("ocf"), capex_ts.rename("capex")], axis=1).dropna()
    if joined.empty:
        return None
    return joined["ocf"] + joined["capex"]

def _fund_metrics(f: Dict[str, pd.DataFrame], gm_floor: float = 0.30) -> Dict[str, Optional[float]]:
    inc = f.get("income_a", pd.DataFrame())
    bal = f.get("balance_a", pd.DataFrame())
    cf = f.get("cashflow_a", pd.DataFrame())
    meta = f.get("meta", {}) if isinstance(f.get("meta", {}), dict) else {}

    sector = meta.get("sector")
    industry = meta.get("industry")

    # Revenue CAGR (3y)
    rev_cagr_3y = None
    rev_row = _get_row(inc, ["Total Revenue", "TotalRevenue", "Revenue"])
    if rev_row is not None:
        rev_ts = _annual_series_to_ts(rev_row)
        if len(rev_ts) >= 4:
            start = float(rev_ts.iloc[-4])
            end = float(rev_ts.iloc[-1])
            rev_cagr_3y = _cagr(start, end, 3)

    # Margins latest annual
    gross_margin = None
    op_margin = None
    gp_row = _get_row(inc, ["Gross Profit", "GrossProfit"])
    op_row = _get_row(inc, ["Operating Income", "OperatingIncome"])

    if rev_row is not None and gp_row is not None:
        tr = _last(rev_row)
        gp = _last(gp_row)
        if tr and gp is not None and tr != 0:
            gross_margin = gp / tr

    if rev_row is not None and op_row is not None:
        tr = _last(rev_row)
        op = _last(op_row)
        if tr and op is not None and tr != 0:
            op_margin = op / tr

    # FCF ttm-ish (actually latest annual in yfinance)
    fcf_ttm = None
    fcf_series = _calc_fcf_series(cf)
    if fcf_series is not None and len(fcf_series) >= 1:
        fcf_ttm = float(fcf_series.iloc[-1])

    fcf_margin = None
    tr_latest = _last(rev_row) if rev_row is not None else None
    if fcf_ttm is not None and tr_latest is not None and tr_latest != 0:
        fcf_margin = fcf_ttm / tr_latest

    # Debt/Equity
    debt = _get_row(bal, ["Total Debt", "TotalDebt", "Long Term Debt", "LongTermDebt"])
    equity = _get_row(bal, ["Total Stockholder Equity", "Stockholders Equity", "Total Equity Gross Minority Interest"])
    debt_to_equity = None
    if debt is not None and equity is not None:
        d = _last(debt)
        e = _last(equity)
        if d is not None and e is not None and e != 0:
            debt_to_equity = d / e

    # SBC % revenue (approx) from cash flow
    sbc = _get_row(cf, ["Stock Based Compensation", "StockBasedCompensation"])
    sbc_pct_rev = None
    if sbc is not None and tr_latest is not None and tr_latest != 0:
        sbc_last = _last(sbc)
        if sbc_last is not None:
            sbc_pct_rev = sbc_last / tr_latest

    # Shares change proxy (3y)
    shares = _get_row(bal, ["Common Stock Shares Outstanding", "Ordinary Shares Number", "Share Issued"])
    shares_change_3y = None
    if shares is not None:
        s_ts = _annual_series_to_ts(shares)
        if len(s_ts) >= 4:
            start = float(s_ts.iloc[-4])
            end = float(s_ts.iloc[-1])
            if start != 0:
                shares_change_3y = (end / start) - 1

    # ROIC-ish proxy: EBIT / (Debt + Equity - Cash)
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

    # -------- Consistency metrics (last 5 annual points, where available) --------
    fcf_pos_years_5 = None
    if fcf_series is not None:
        tail = fcf_series.tail(5).dropna()
        if len(tail) > 0:
            fcf_pos_years_5 = int((tail > 0).sum())

    gm_floor_years_5 = None
    if rev_row is not None and gp_row is not None:
        rev_ts = _annual_series_to_ts(rev_row)
        gp_ts = _annual_series_to_ts(gp_row)
        joined = pd.concat([rev_ts.rename("rev"), gp_ts.rename("gp")], axis=1).dropna().tail(5)
        if not joined.empty:
            gm = joined["gp"] / joined["rev"]
            gm_floor_years_5 = int((gm >= gm_floor).sum())

    rev_up_years_5 = None
    if rev_row is not None:
        rev_ts = _annual_series_to_ts(rev_row).dropna().tail(6)  # 6 points => 5 YoY comps
        if len(rev_ts) >= 2:
            yoy = rev_ts.pct_change().dropna().tail(5)
            if len(yoy) > 0:
                rev_up_years_5 = int((yoy > 0).sum())

    # Data quality: count how many key fields exist
    quality_fields = {
        "rev": rev_row is not None,
        "gp": gp_row is not None,
        "op": op_row is not None,
        "cf_ocf_capex": fcf_series is not None,
        "debt_equity": (debt is not None and equity is not None),
        "sbc": sbc is not None,
        "shares": shares is not None,
        "meta": bool(sector or industry),
    }
    data_quality_score = int(sum(1 for v in quality_fields.values() if v))

    return {
        "sector": sector,
        "industry": industry,
        "rev_cagr_3y": rev_cagr_3y,
        "gross_margin": gross_margin,
        "op_margin": op_margin,
        "fcf_ttm": fcf_ttm,
        "fcf_margin": fcf_margin,
        "debt_to_equity": debt_to_equity,
        "sbc_pct_rev": sbc_pct_rev,
        "shares_change_3y": shares_change_3y,
        "roic_proxy": roic_proxy,
        "fcf_pos_years_5": fcf_pos_years_5,
        "gm_floor_years_5": gm_floor_years_5,
        "rev_up_years_5": rev_up_years_5,
        "data_quality_score": data_quality_score,
    }

def build_metrics_table(tickers, prices: pd.DataFrame, fundamentals: Dict[str, Dict[str, pd.DataFrame]], gm_floor: float = 0.30) -> pd.DataFrame:
    rows = []
    for t in tickers:
        pm = _price_metrics(prices, t)
        fm = _fund_metrics(fundamentals.get(t, {}), gm_floor=gm_floor)
        rows.append({"ticker": t, **pm, **fm})
    return pd.DataFrame(rows)
