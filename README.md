# Wealth Builder Screener

A Streamlit app to screen stocks for long-term compounding using:
- Price action (trend + relative strength + drawdowns)
- Fundamentals (growth, margins, FCF, balance sheet)
- Quality add-ons (SBC burden, dilution proxy, ROIC-ish proxy)
- Ranked scoring + click-to-dive dashboards

## Run locally

```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
streamlit run app.py
