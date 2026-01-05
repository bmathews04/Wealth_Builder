import pandas as pd
import streamlit as st
import yfinance as yf

@st.cache_data(show_spinner=False, ttl=60 * 60)
def batch_fetch_prices(tickers, years: int = 5) -> pd.DataFrame:
    """
    Batch fetch daily adjusted OHLCV for many tickers.
    Returns long-form DataFrame:
      date, ticker, open, high, low, close, volume,
      ma200, ma40w, ret_252d, ret_126d
    """
    if not tickers:
        return pd.DataFrame()

    df = yf.download(
        tickers=tickers,
        period=f"{years}y",
        interval="1d",
        auto_adjust=True,
        group_by="ticker",
        threads=True,
        progress=False,
    )

    frames = []
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            if (t, "Close") not in df.columns:
                continue
            tmp = df[t][["Open", "High", "Low", "Close", "Volume"]].copy()
            tmp.columns = ["open", "high", "low", "close", "volume"]
            tmp["ticker"] = t
            frames.append(tmp)
        if not frames:
            return pd.DataFrame()
        out = pd.concat(frames).reset_index().rename(columns={"Date": "date"})
    else:
        # single ticker
        out = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        out.columns = ["open", "high", "low", "close", "volume"]
        out = out.reset_index().rename(columns={"Date": "date"})
        out["ticker"] = tickers[0]

    out = out.sort_values(["ticker", "date"]).reset_index(drop=True)

    def add_ind(g: pd.DataFrame) -> pd.DataFrame:
        g = g.copy()
        g["ma200"] = g["close"].rolling(200).mean()

        # Weekly 40W MA computed from weekly closes
        wk = g.set_index("date")["close"].resample("W-FRI").last()
        ma40w = wk.rolling(40).mean()
        g = g.set_index("date")
        g["ma40w"] = ma40w.reindex(g.index, method="ffill")
        g = g.reset_index()

        g["ret_252d"] = g["close"].pct_change(252)
        g["ret_126d"] = g["close"].pct_change(126)
        return g

    out = out.groupby("ticker", group_keys=False).apply(add_ind)
    return out
