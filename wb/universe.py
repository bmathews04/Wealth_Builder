import pandas as pd
import streamlit as st

WIKI_SP500 = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
WIKI_NDX = "https://en.wikipedia.org/wiki/Nasdaq-100"

@st.cache_data(show_spinner=False, ttl=60 * 60 * 24)
def get_universe(which: str):
    if which == "sp500":
        tables = pd.read_html(WIKI_SP500)
        df = tables[0]
        tickers = df["Symbol"].astype(str).str.upper().str.replace(".", "-", regex=False).tolist()
        return tickers

    if which == "nasdaq100":
        tables = pd.read_html(WIKI_NDX)
        for t in tables:
            cols = [c.lower() for c in t.columns.astype(str)]
            if "ticker" in cols or "symbol" in cols:
                col = t.columns[cols.index("ticker")] if "ticker" in cols else t.columns[cols.index("symbol")]
                tickers = t[col].astype(str).str.upper().str.replace(".", "-", regex=False).tolist()
                tickers = [x for x in tickers if x and x != "NAN" and x != "—"]
                return tickers

    return []
