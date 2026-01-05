# wb/causal_ranker.py
from __future__ import annotations
from typing import Callable, List, Optional

import pandas as pd


def ranker_from_snapshots(
    snapshots: pd.DataFrame,
    universe: Optional[list[str]] = None,
    score_col: str = "score",
) -> Callable[[pd.Timestamp], List[str]]:
    """
    Returns a function ranker(asof)->ranked tickers using the latest snapshot <= asof.
    """
    if snapshots is None or snapshots.empty:
        def empty_ranker(_asof: pd.Timestamp) -> list[str]:
            return []
        return empty_ranker

    snaps = snapshots.copy()
    snaps["asof"] = pd.to_datetime(snaps["asof"]).dt.normalize()
    snaps["ticker"] = snaps["ticker"].astype(str).str.upper().str.strip()

    universe_set = None
    if universe:
        universe_set = set([u.upper().strip() for u in universe])

    def ranker(asof: pd.Timestamp) -> list[str]:
        d = pd.to_datetime(asof).normalize()
        eligible = snaps[snaps["asof"] <= d]
        if eligible.empty:
            return []

        latest_date = eligible["asof"].max()
        latest = eligible[eligible["asof"] == latest_date].copy()

        if universe_set is not None:
            latest = latest[latest["ticker"].isin(universe_set)]

        if score_col not in latest.columns:
            return []

        latest = latest[latest[score_col].notna()].sort_values(score_col, ascending=False)
        return latest["ticker"].astype(str).tolist()

    return ranker
