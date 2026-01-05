import numpy as np
import pandas as pd

def score_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    0–100 score tuned for long-term compounding:
    - Trend health (30)
    - Growth (20)
    - Margins (15)
    - Cash flow (20)
    - Balance sheet + quality (15)
    """
    out = df.copy()

    def clip01(x): return float(np.clip(x, 0, 1))

    score = np.zeros(len(out), dtype=float)

    # Trend (30)
    score += out["above_ma200"].fillna(False).astype(int) * 15
    score += out["above_ma40w"].fillna(False).astype(int) * 10
    # RS vs SPY (12m): map -20%..+40% to 0..1
    rs = out["rs_vs_spy_12m"].fillna(0.0)
    score += rs.apply(lambda x: clip01((x + 0.20) / 0.60)) * 5

    # Growth (20): rev cagr 0..30% => 0..1
    rc = out["rev_cagr_3y"].fillna(0.0)
    score += rc.apply(lambda x: clip01(x / 0.30)) * 20

    # Margins (15): GM 20..60 => 0..1 (10 pts), OM -5..20 => 0..1 (5 pts)
    gm = out["gross_margin"].fillna(0.0)
    score += gm.apply(lambda x: clip01((x - 0.20) / 0.40)) * 10
    om = out["op_margin"].fillna(-0.05)
    score += om.apply(lambda x: clip01((x + 0.05) / 0.25)) * 5

    # Cash flow (20): FCF positive = 12 pts; FCF margin -5..20 => 0..1 (8 pts)
    fcf = out["fcf_ttm"].fillna(-1.0)
    score += (fcf > 0).astype(int) * 12
    fcfm = out["fcf_margin"].fillna(-0.05)
    score += fcfm.apply(lambda x: clip01((x + 0.05) / 0.25)) * 8

    # Balance sheet + quality (15)
    dte = out["debt_to_equity"].fillna(10.0)
    # debt/equity <0.3 => 1, 0.3..1.0 scale down
    score += dte.apply(lambda x: 10 * clip01(1 - (x / 1.0)))  # 0..10
    sbc = out["sbc_pct_rev"].fillna(0.20)
    score += sbc.apply(lambda x: 5 * clip01(1 - (x / 0.20)))  # 0..5 (<=20% gets higher)

    out["score"] = np.round(score, 1)
    return out
