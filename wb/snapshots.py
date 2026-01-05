# wb/snapshots.py
from __future__ import annotations

import gzip
import io
from dataclasses import dataclass
from typing import Optional

import pandas as pd


SNAPSHOT_COLUMNS = [
    "asof",
    "ticker",
    "score",
    "data_quality",
    "sector",
    "industry",
]


def make_snapshot(scored_df: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """
    Create a minimal snapshot from your scored long-term table.
    """
    df = scored_df.copy()

    # Ensure required columns exist
    for c in ["ticker", "score"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")

    for c in ["data_quality", "sector", "industry"]:
        if c not in df.columns:
            df[c] = None

    snap = df[["ticker", "score", "data_quality", "sector", "industry"]].copy()
    snap["asof"] = pd.to_datetime(asof).normalize()
    snap = snap[SNAPSHOT_COLUMNS]
    snap["ticker"] = snap["ticker"].astype(str).str.upper().str.strip()
    return snap


def snapshots_to_bytes(snapshots: pd.DataFrame) -> bytes:
    """
    Serialize snapshots DataFrame to gzipped CSV bytes.
    """
    buf = io.BytesIO()
    with gzip.GzipFile(fileobj=buf, mode="wb") as gz:
        gz.write(snapshots.to_csv(index=False).encode("utf-8"))
    return buf.getvalue()


def snapshots_from_bytes(data: bytes) -> pd.DataFrame:
    """
    Read gzipped CSV bytes into DataFrame.
    """
    with gzip.GzipFile(fileobj=io.BytesIO(data), mode="rb") as gz:
        raw = gz.read()
    df = pd.read_csv(io.BytesIO(raw))
    if "asof" in df.columns:
        df["asof"] = pd.to_datetime(df["asof"])
    if "ticker" in df.columns:
        df["ticker"] = df["ticker"].astype(str).str.upper().str.strip()
    return df


def merge_snapshots(existing: Optional[pd.DataFrame], new_snap: pd.DataFrame) -> pd.DataFrame:
    """
    Append and de-duplicate (asof,ticker).
    """
    if existing is None or existing.empty:
        out = new_snap.copy()
    else:
        out = pd.concat([existing, new_snap], ignore_index=True)

    out["asof"] = pd.to_datetime(out["asof"]).dt.normalize()
    out["ticker"] = out["ticker"].astype(str).str.upper().str.strip()

    out = out.drop_duplicates(subset=["asof", "ticker"], keep="last")
    out = out.sort_values(["asof", "score"], ascending=[True, False]).reset_index(drop=True)
    return out
