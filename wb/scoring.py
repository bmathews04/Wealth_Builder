import numpy as np
import pandas as pd

def _clip01(x: float) -> float:
    return float(np.clip(x, 0.0, 1.0))

def score_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    0–100 score designed for long-term compounding:
      - Trend health (28)
      - Growth (18)
      - Margins (14)
      - Cash flow (18)
      - Balance sheet + quality (12)
      - Consistency (10)
    """
    out = df.copy()
    score = np.zeros(len(out), dtype=float)

    # Trend (28)
    score += out["above_ma200"].fillna(False).astype(int) * 14
    score += out["above_ma40w"].fillna(False).astype(int) * 9

    rs = out["rs_vs_spy_12m"].fillna(0.0)
    # map -20%..+40% => 0..1
    score += rs.apply(lambda x: _clip01((x + 0.20) / 0.60)) * 5

    # Growth (18)
    rc = out["rev_cagr_3y"].fillna(0.0)
    score += rc.apply(lambda x: _clip01(x / 0.30)) * 18

    # Margins (14)
    gm = out["gross_margin"].fillna(0.0)
    score += gm.apply(lambda x: _clip01((x - 0.20) / 0.40)) * 9  # 20%..60%
    om = out["op_margin"].fillna(-0.05)
    score += om.apply(lambda x: _clip01((x + 0.05) / 0.25)) * 5  # -5%..20%

    # Cash flow (18)
    fcf = out["fcf_ttm"].fillna(-1.0)
    score += (fcf > 0).astype(int) * 10
    fcfm = out["fcf_margin"].fillna(-0.05)
    score += fcfm.apply(lambda x: _clip01((x + 0.05) / 0.25)) * 8  # -5%..20%

    # Balance sheet + quality (12)
    dte = out["debt_to_equity"].fillna(10.0)
    score += dte.apply(lambda x: 8 * _clip01(1 - (x / 1.0)))  # 0..8
    sbc = out["sbc_pct_rev"].fillna(0.20)
    score += sbc.apply(lambda x: 4 * _clip01(1 - (x / 0.20)))  # 0..4 (<=20% better)

    # Consistency (10)
    # FCF positive years (out of 5): 0..5 => 0..5 points
    fcfy = out["fcf_pos_years_5"].fillna(0).astype(float)
    score += fcfy.apply(lambda x: _clip01(x / 5.0)) * 5

    # Revenue up years (out of 5): 0..5 => 0..3 points
    revy = out["rev_up_years_5"].fillna(0).astype(float)
    score += revy.apply(lambda x: _clip01(x / 5.0)) * 3

    # GM floor years: 0..5 => 0..2 points
    gmy = out["gm_floor_years_5"].fillna(0).astype(float)
    score += gmy.apply(lambda x: _clip01(x / 5.0)) * 2

    out["score"] = np.round(score, 1)
    return out

def add_data_quality_badges(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds a simple data quality badge based on presence of key fields.
    """
    out = df.copy()
    # data_quality_score max ~8 in metrics.py
    s = out.get("data_quality_score", pd.Series([0]*len(out))).fillna(0)

    def badge(x: float) -> str:
        if x >= 7:
            return "🟢 High"
        if x >= 5:
            return "🟡 Medium"
        return "🔴 Low"

    out["data_quality"] = s.apply(badge)
    return out
