# ═══════════════════════════════════════════════════════════════════════════════
#              QuantAI Elite — World-Class Stock & Commodity Predictor
#   FIXED: Log-return targets · Momentum bias · Reduced decay · AI validator
#          Walk-forward validation · Fundamentals · Top 5 Picks · No-zero bug
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
from huggingface_hub import InferenceClient
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import StandardScaler
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ─── PAGE CONFIG ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="QuantAI Elite",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─── SECRETS ──────────────────────────────────────────────────────────────────
HF_TOKEN        = "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"
SERP_API_KEY    = "6ff2e871beaffd46ffca7bf6c4007814e1fa27937b662e1408e2fd04d9937b0f"
_ai_client      = InferenceClient(token=HF_TOKEN)
_AI_MODEL       = "Qwen/Qwen2.5-7B-Instruct"

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background: #070b14; }
.stTabs [data-baseweb="tab-list"] { gap: 8px; }
.stTabs [data-baseweb="tab"] {
    font-size: 14px; font-weight: 600; padding: 8px 18px;
    border-radius: 8px 8px 0 0; background: #111827;
    color: #6b7280; border: none;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    color: white !important;
}
.stMetric { background: #0f1623; border-radius: 12px; padding: 14px 16px; border: 1px solid #1f2937; }
.stMetric label { color: #6b7280 !important; font-size: 12px !important; }
.stMetric [data-testid="metric-container"] > div:last-child { font-size: 13px !important; }
.stSidebar { background: #080c18 !important; }
[data-testid="stSidebarContent"] { padding: 0 !important; }
.stSelectbox > div > div { background: #111827 !important; border-color: #1f2937 !important; color: #e2e8f0 !important; }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    border: none !important; font-weight: 700 !important;
    letter-spacing: 0.5px; transition: all .25s ease;
    box-shadow: 0 4px 20px rgba(99,102,241,.4);
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 30px rgba(99,102,241,.7);
    transform: translateY(-1px);
}
.quant-fab {
    position: fixed; bottom: 26px; right: 26px;
    width: 64px; height: 64px; border-radius: 50%;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 30px; cursor: pointer;
    box-shadow: 0 0 0 0 rgba(99,102,241,.75);
    animation: fab-ring 2.4s infinite;
    z-index: 999999; border: 2px solid rgba(167,139,250,.4);
}
.quant-fab-label {
    position: fixed; bottom: 98px; right: 18px;
    background: #1e1b4b; color: #c7d2fe;
    border: 1px solid #4338ca; border-radius: 10px;
    padding: 6px 12px; font-size: 12px; font-weight: 600;
    z-index: 999998; white-space: nowrap;
    opacity: 0; animation: label-appear 0.6s 4s forwards;
    pointer-events: none;
}
@keyframes fab-ring {
    0%   { box-shadow: 0 0 0 0   rgba(99,102,241,.75); }
    70%  { box-shadow: 0 0 0 20px rgba(99,102,241,0);  }
    100% { box-shadow: 0 0 0 0   rgba(99,102,241,0);  }
}
@keyframes label-appear { to { opacity: 1; } }
.model-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #1e1b4b; color: #a5b4fc;
    border: 1px solid #4338ca; border-radius: 20px;
    padding: 4px 12px; font-size: 11px; font-weight: 700;
    letter-spacing: .5px;
}
.sanity-ok  { background:#14532d; color:#86efac; border-radius:8px; padding:8px 14px; font-size:12px; font-weight:600; }
.sanity-warn{ background:#451a03; color:#fed7aa; border-radius:8px; padding:8px 14px; font-size:12px; font-weight:600; }
.ai-ok      { background:#0c2a1a; color:#4ade80; border-radius:8px; padding:10px 14px; font-size:12px; border:1px solid #166534; margin-top:6px; }
.ai-warn    { background:#2a1a0c; color:#fbbf24; border-radius:8px; padding:10px 14px; font-size:12px; border:1px solid #92400e; margin-top:6px; }
.nse-badge  { background:#14532d; color:#86efac; border-radius:6px; padding:2px 8px; font-size:11px; font-weight:700; }
.bse-badge  { background:#1e1b4b; color:#a5b4fc; border-radius:6px; padding:2px 8px; font-size:11px; font-weight:700; }
.us-badge   { background:#1c1917; color:#fbbf24; border-radius:6px; padding:2px 8px; font-size:11px; font-weight:700; }
.quant-header { text-align:center; padding: 10px 0 4px; }
.quant-header h1 {
    font-size: 2.2rem; font-weight: 900;
    background: linear-gradient(135deg,#6366f1,#a78bfa,#38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -1px; margin-bottom: 2px;
}
.quant-header p { color: #6b7280; font-size: 13px; margin: 0; }
.ticker-search-box {
    background: #111827; border: 1px dashed #4338ca;
    border-radius: 10px; padding: 10px 12px; margin-top: 6px;
    font-size: 12px; color: #a5b4fc;
}
.top5-card {
    background: linear-gradient(135deg, #0f1623, #111827);
    border: 1px solid #1f2937; border-radius: 12px;
    padding: 14px 16px; margin-bottom: 10px;
}
.top5-rank { font-size: 24px; font-weight: 900; color: #a78bfa; }
.top5-name { font-size: 15px; font-weight: 700; color: #e2e8f0; }
.top5-ticker { font-size: 11px; color: #6b7280; }
.fund-row { background:#0f1623; border-radius:10px; padding:8px 12px; border:1px solid #1f2937; margin-top:6px; }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="quant-header">
  <h1>📈 QuantAI Elite</h1>
  <p>Log-Return Ensemble · AI Validator · Fundamentals · Top 5 ROI Picks · US & India NSE/BSE · Metals · Jewelry · Custom Tickers</p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
#  STOCK LISTS
# ═══════════════════════════════════════════════════════════════════════════════

us_large_cap = {
    "Apple": "AAPL", "Microsoft": "MSFT", "NVIDIA": "NVDA",
    "Amazon": "AMZN", "Alphabet (Google)": "GOOGL", "Meta": "META",
    "Tesla": "TSLA", "Berkshire Hathaway": "BRK-B", "JPMorgan Chase": "JPM",
    "Visa": "V", "Johnson & Johnson": "JNJ", "Walmart": "WMT",
    "Exxon Mobil": "XOM", "UnitedHealth": "UNH", "Procter & Gamble": "PG",
    "Mastercard": "MA", "Broadcom": "AVGO", "Eli Lilly": "LLY",
    "Chevron": "CVX", "Home Depot": "HD",
}
us_mid_cap = {
    "Lyft": "LYFT", "Robinhood": "HOOD", "Chewy": "CHWY",
    "Duolingo": "DUOL", "Zillow": "Z", "Wingstop": "WING",
    "Five Below": "FIVE", "Crocs": "CROX", "Hasbro": "HAS",
    "Caesars Entertainment": "CZR", "Vimeo": "VMEO", "Sweetgreen": "SG",
}
us_small_cap = {
    "Genie Energy": "GNE", "Coda Octopus": "CODA", "Turtle Beach": "HEAR",
    "Intellicheck": "IDN", "Frequency Electronics": "FEIM", "Transcat": "TRNS",
    "ProQR Therapeutics": "PRQR", "Can-Fite BioPharma": "CANF",
}

india_large_base = {
    "Reliance Industries": "RELIANCE", "Tata Consultancy Services": "TCS",
    "Infosys": "INFY", "HDFC Bank": "HDFCBANK", "ICICI Bank": "ICICIBANK",
    "Bharti Airtel": "BHARTIARTL", "ITC Limited": "ITC",
    "Kotak Mahindra Bank": "KOTAKBANK", "Larsen & Toubro": "LT",
    "Hindustan Unilever": "HINDUNILVR", "Bajaj Finance": "BAJFINANCE",
    "Wipro": "WIPRO", "Axis Bank": "AXISBANK", "Maruti Suzuki": "MARUTI",
    "State Bank of India": "SBIN", "ONGC": "ONGC",
    "Asian Paints": "ASIANPAINT", "HCL Technologies": "HCLTECH",
    "Sun Pharmaceutical": "SUNPHARMA", "Titan Company": "TITAN",
    "UltraTech Cement": "ULTRACEMCO", "Power Grid Corp": "POWERGRID",
    "NTPC": "NTPC", "Tata Motors": "TATAMOTORS", "JSW Steel": "JSWSTEEL",
    "Tata Steel": "TATASTEEL", "Mahindra & Mahindra": "M&M",
    "Bajaj Auto": "BAJAJ-AUTO", "Tech Mahindra": "TECHM",
    "Adani Enterprises": "ADANIENT", "Adani Ports": "ADANIPORTS",
    "Cipla": "CIPLA", "Dr Reddy's Labs": "DRREDDY",
    "Eicher Motors": "EICHERMOT", "Nestle India": "NESTLEIND",
}
india_mid_base = {
    "Zomato": "ZOMATO", "Paytm": "PAYTM", "Nykaa": "NYKAA",
    "IRCTC": "IRCTC", "Muthoot Finance": "MUTHOOTFIN",
    "Delhivery": "DELHIVERY", "PNB": "PNB", "Bank of Baroda": "BANKBARODA",
    "Indian Hotels (Taj)": "INDHOTEL", "Apollo Hospitals": "APOLLOHOSP",
    "Tata Power": "TATAPOWER", "Godrej Properties": "GODREJPROP",
    "PI Industries": "PIIND", "Coforge": "COFORGE",
    "Dixon Technologies": "DIXON", "Polycab India": "POLYCAB",
    "Persistent Systems": "PERSISTENT", "Mphasis": "MPHASIS",
    "Hindustan Aeronautics (HAL)": "HAL", "Bharat Electronics": "BEL",
    "Interglobe Aviation (IndiGo)": "INDIGO",
    "SBI Life Insurance": "SBILIFE", "HDFC Life Insurance": "HDFCLIFE",
}
india_small_base = {
    "Kalyan Jewellers": "KALYANKJIL", "Senco Gold": "SENCO",
    "PC Jeweller": "PCJEWELLER", "Thangamayil Jewellery": "THANGAMAYL",
    "Easy Trip Planners": "EASEMYTRIP", "Kaynes Technology": "KAYNES",
    "Happiest Minds": "HAPPSTMNDS", "Medplus Health": "MEDPLUS",
    "Campus Activewear": "CAMPUS", "Vedant Fashions (Manyavar)": "MANYAVAR",
    "Bikaji Foods": "BIKAJI", "RBL Bank": "RBLBANK",
    "Ujjivan Small Finance": "UJJIVANSFB", "CMS Info Systems": "CMSINFO",
    "Sapphire Foods": "SAPPHIRE",
}

def make_india_map(base: dict, exchange: str) -> dict:
    suffix = ".NS" if exchange == "NSE" else ".BO"
    return {name: ticker + suffix for name, ticker in base.items()}

commodities_futures = {
    "🥇 Gold Futures": "GC=F", "🥈 Silver Futures": "SI=F",
    "⬜ Platinum": "PL=F", "🔵 Palladium": "PA=F",
    "🛢️ Crude Oil (WTI)": "CL=F", "🔥 Natural Gas": "NG=F",
    "🔶 Copper": "HG=F", "🌽 Corn": "ZC=F",
    "🌾 Wheat": "ZW=F", "🫘 Soybeans": "ZS=F",
}
commodities_etf = {
    "Gold ETF (GLD)": "GLD", "Silver ETF (SLV)": "SLV",
    "Gold Miners (GDX)": "GDX", "Junior Gold Miners (GDXJ)": "GDXJ",
    "Silver Miners (SIL)": "SIL", "Commodities Index (PDBC)": "PDBC",
    "Gold ETF India (GOLDBEES)": "GOLDBEES.NS",
}
jewelry_stocks_map = {
    "Titan Company (NSE)": "TITAN.NS", "Kalyan Jewellers (NSE)": "KALYANKJIL.NS",
    "Senco Gold (NSE)": "SENCO.NS", "PC Jeweller (NSE)": "PCJEWELLER.NS",
    "Thangamayil Jewellery (NSE)": "THANGAMAYL.NS",
    "Tribhovandas Bhimji Zaveri (NSE)": "TBZ.NS",
    "Signet Jewelers (NYSE)": "SIG", "Tapestry (Coach/Kate Spade)": "TPR",
    "Movado Group": "MOV",
}

# ═══════════════════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def currency_symbol(ticker: str) -> str:
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return "₹"
    elif ticker.endswith(".CO"):
        return "kr"
    elif ticker.endswith(".PA") or ticker.endswith(".SW"):
        return "€"
    elif ticker.endswith(".HK"):
        return "HK$"
    return "$"

@st.cache_data(ttl=300, show_spinner=False)
def load_stock(ticker: str, period: str) -> pd.DataFrame:
    try:
        df = yf.Ticker(ticker).history(period=period)
        df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
        return df
    except Exception:
        return pd.DataFrame()

@st.cache_data(ttl=3600, show_spinner=False)
def get_fundamentals(ticker: str) -> dict:
    """Fetch key fundamental metrics from yfinance."""
    try:
        info = yf.Ticker(ticker).info
        return {
            "pe_ratio":        info.get("trailingPE"),
            "fwd_pe":          info.get("forwardPE"),
            "revenue_growth":  info.get("revenueGrowth"),
            "earnings_growth": info.get("earningsGrowth"),
            "debt_equity":     info.get("debtToEquity"),
            "profit_margin":   info.get("profitMargins"),
            "operating_margin":info.get("operatingMargins"),
            "gross_margin":    info.get("grossMargins"),
            "market_cap":      info.get("marketCap"),
            "sector":          info.get("sector", "N/A"),
            "peg_ratio":       info.get("pegRatio"),
            "return_on_equity":info.get("returnOnEquity"),
            "free_cashflow":   info.get("freeCashflow"),
        }
    except Exception:
        return {}

def resolve_custom_ticker(raw: str) -> tuple:
    raw = raw.strip()
    if not raw:
        return "", ""
    try:
        t = yf.Ticker(raw.upper())
        info = t.info
        if info.get("regularMarketPrice") or info.get("currentPrice"):
            name = info.get("shortName") or info.get("longName") or raw.upper()
            return raw.upper(), name
    except Exception:
        pass
    try:
        results = yf.Search(raw, max_results=5).quotes
        if results:
            best   = results[0]
            symbol = best.get("symbol", raw.upper())
            name   = best.get("shortname") or best.get("longname") or symbol
            return symbol, name
    except Exception:
        pass
    return raw.upper(), raw.upper()

def fetch_real_time(query: str) -> str:
    try:
        url = (f"https://serpapi.com/search.json"
               f"?q={requests.utils.quote(query)}&api_key={SERP_API_KEY}")
        r = requests.get(url, timeout=6).json()
        if "answer_box" in r and "snippet" in r["answer_box"]:
            return r["answer_box"]["snippet"]
        snips = [x.get("snippet","") for x in r.get("organic_results",[])[:4] if x.get("snippet")]
        return " | ".join(snips) or "No real-time data available."
    except Exception as e:
        return f"Search error: {e}"

@st.cache_data(ttl=1800, show_spinner=False)
def get_news_sentiment(query: str) -> float:
    try:
        url = (f"https://serpapi.com/search.json"
               f"?q={requests.utils.quote(query + ' stock market news')}"
               f"&tbm=nws&num=5&api_key={SERP_API_KEY}")
        r = requests.get(url, timeout=6).json()
        headlines = []
        for src in ("news_results", "organic_results"):
            for item in r.get(src, [])[:5]:
                text = item.get("title") or item.get("snippet","")
                if text:
                    headlines.append(text)
            if headlines:
                break
        if not headlines:
            return 0.0
        combined = " ".join(headlines)[:512]
        ic = InferenceClient(token=HF_TOKEN)
        result = ic.text_classification(combined, model="ProsusAI/finbert")
        scores = {item.label.lower(): item.score for item in result}
        return float(np.clip(scores.get("positive",0.33) - scores.get("negative",0.33), -1, 1))
    except Exception:
        return 0.0

def ai_validate_prediction(
    company: str, ticker: str,
    current_price: float, pred_price: float, pred_pct: float,
    days: int, hist_vol: float, sentiment: float,
    fundamentals: dict, sym: str,
) -> str:
    """
    Use Qwen to double-check the model's prediction for reasonableness.
    Returns a validation string with VERDICT: ACCEPTED or NEEDS REVIEW.
    """
    try:
        fund_lines = []
        if fundamentals:
            pe  = fundamentals.get("pe_ratio")
            rg  = fundamentals.get("revenue_growth")
            eg  = fundamentals.get("earnings_growth")
            de  = fundamentals.get("debt_equity")
            pm  = fundamentals.get("profit_margin")
            roe = fundamentals.get("return_on_equity")
            if pe  is not None: fund_lines.append(f"  P/E Ratio: {pe:.1f}")
            if rg  is not None: fund_lines.append(f"  Revenue Growth: {rg*100:.1f}%")
            if eg  is not None: fund_lines.append(f"  Earnings Growth: {eg*100:.1f}%")
            if de  is not None: fund_lines.append(f"  Debt/Equity: {de:.2f}")
            if pm  is not None: fund_lines.append(f"  Profit Margin: {pm*100:.1f}%")
            if roe is not None: fund_lines.append(f"  Return on Equity: {roe*100:.1f}%")
        fund_text = "\n".join(fund_lines) if fund_lines else "  Not available (likely commodity/ETF)"

        two_sigma = hist_vol * 1.96 * (days ** 0.5) * 100.0

        prompt = f"""You are a senior quantitative analyst reviewing an AI price prediction.

STOCK: {company} ({ticker})
Current Price:       {sym}{current_price:,.2f}
Predicted ({days}d): {sym}{pred_price:,.2f}  ({pred_pct:+.2f}%)
Historical Daily Vol: {hist_vol*100:.2f}%/day
2σ Range ({days}d):  ±{two_sigma:.1f}%
News Sentiment:      {sentiment:+.2f} (range −1 to +1)

Fundamentals:
{fund_text}

Your task:
1. Is {pred_pct:+.2f}% over {days} days statistically plausible given {hist_vol*100:.2f}%/day vol?
2. Does the direction align with sentiment ({sentiment:+.2f}) and fundamentals?
3. Any red flags (extreme P/E, high debt, negative growth that contradict a bullish call)?
4. Final verdict.

Be concise (≤120 words). Finish with exactly one of:
VERDICT: ACCEPTED — [one-line reason]
VERDICT: NEEDS REVIEW — [one-line reason]"""

        res = _ai_client.chat.completions.create(
            model=_AI_MODEL,
            messages=[
                {"role": "system", "content": "You are a quantitative financial analyst. Be precise, data-driven, and brief."},
                {"role": "user",   "content": prompt},
            ],
            max_tokens=300,
            temperature=0.25,
        )
        return res.choices[0].message.content
    except Exception as e:
        return f"AI validation unavailable: {e}"

# ═══════════════════════════════════════════════════════════════════════════════
#  TOP 5 STOCKS SCORER
# ═══════════════════════════════════════════════════════════════════════════════

def _score_stock(name: str, ticker: str) -> dict | None:
    """Score a single stock for ROI potential. Returns None if data unavailable."""
    try:
        df = load_stock(ticker, "3mo")
        if df.empty or len(df) < 30:
            return None

        close    = df["Close"]
        sym      = currency_symbol(ticker)
        log_rets = np.log(close / close.shift(1)).dropna()
        hist_vol = float(log_rets.std()) if len(log_rets) > 0 else 0.02

        # ── Technical signals ─────────────────────────────────────────────
        mom_30 = float((close.iloc[-1] / close.iloc[0] - 1) * 100)
        mom_10 = float((close.iloc[-1] / close.iloc[-10] - 1) * 100) if len(close) >= 10 else 0.0

        # RSI
        delta = close.diff()
        gain  = delta.clip(lower=0).rolling(14).mean()
        loss  = (-delta.clip(upper=0)).rolling(14).mean()
        rs    = gain / (loss.replace(0, np.nan))
        rsi   = float((100 - 100 / (1 + rs)).iloc[-1]) if not rs.empty else 50.0
        if np.isnan(rsi):
            rsi = 50.0

        # RSI in sweet spot (not overbought, not oversold crash)
        rsi_score = 1.0 if 40 <= rsi <= 62 else 0.6 if 30 <= rsi <= 70 else 0.2

        # Volume trend (recent vs 30-day avg)
        vol_ratio = float(df["Volume"].iloc[-5:].mean() / (df["Volume"].mean() + 1))
        vol_score = min(vol_ratio / 1.5, 1.0)

        # Price above 50-day MA (trend confirmation)
        ma50        = close.rolling(50).mean()
        trend_score = 1.0 if (len(ma50.dropna()) > 0 and close.iloc[-1] > ma50.dropna().iloc[-1]) else 0.0

        # Sharpe proxy (10-day)
        ret_10   = log_rets.tail(10)
        sharpe   = float(ret_10.mean() / (ret_10.std() + 1e-8)) if len(ret_10) >= 5 else 0.0
        sharpe_n = min(max((sharpe + 3) / 6, 0), 1)   # normalise to [0,1]

        tech_score = (
            min(max(mom_30 / 15 + 0.5, 0), 1) * 0.25 +
            rsi_score                          * 0.25 +
            vol_score                          * 0.15 +
            trend_score                        * 0.20 +
            sharpe_n                           * 0.15
        )

        # ── Fundamental signals ───────────────────────────────────────────
        fund       = get_fundamentals(ticker)
        fund_score = 0.5   # neutral default (for commodities / ETFs with no fundamentals)
        if fund:
            pe   = fund.get("pe_ratio")
            rg   = fund.get("revenue_growth") or 0.0
            eg   = fund.get("earnings_growth") or 0.0
            pm   = fund.get("profit_margin") or 0.0
            de   = fund.get("debt_equity") or 0.0
            peg  = fund.get("peg_ratio")

            f = 0.0
            count = 0

            if pe is not None and not np.isnan(pe):
                f += 1.0 if 8 < pe < 30 else 0.5 if pe <= 40 else 0.2
                count += 1
            if rg:
                f += min(max(rg / 0.30, 0), 1)
                count += 1
            if eg:
                f += min(max(eg / 0.25, 0), 1)
                count += 1
            if pm:
                f += min(max(pm / 0.20, 0), 1)
                count += 1
            if de:
                f += 1.0 if de < 50 else 0.6 if de < 150 else 0.2
                count += 1
            if peg is not None and not np.isnan(peg):
                f += 1.0 if 0 < peg < 1.5 else 0.6 if peg < 3 else 0.2
                count += 1

            if count > 0:
                fund_score = f / count

        # ── Composite ─────────────────────────────────────────────────────
        composite = tech_score * 0.55 + fund_score * 0.45

        return {
            "name":       name,
            "ticker":     ticker,
            "sym":        sym,
            "price":      f"{sym}{close.iloc[-1]:,.2f}",
            "mom_30":     mom_30,
            "mom_10":     mom_10,
            "rsi":        rsi,
            "vol_ratio":  vol_ratio,
            "hist_vol":   hist_vol * 100,
            "tech_score": tech_score * 100,
            "fund_score": fund_score * 100,
            "composite":  composite * 100,
            "fund":       fund,
        }
    except Exception:
        return None


@st.cache_data(ttl=600, show_spinner=False)
def get_top_5_stocks(market_label: str, stocks_json: str) -> list:
    """
    Score up to 20 stocks and return the top 5 by composite ROI score.
    stocks_json is a JSON string of the dict to allow cache-key hashing.
    """
    import json
    stocks_map = json.loads(stocks_json)
    tickers    = list(stocks_map.items())[:20]
    results    = []
    for name, tkr in tickers:
        scored = _score_stock(name, tkr)
        if scored:
            results.append(scored)
    results.sort(key=lambda x: x["composite"], reverse=True)
    return results[:5]

# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING — ONLY NORMALISED/RATIO FEATURES (no absolute prices)
# ═══════════════════════════════════════════════════════════════════════════════

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

_OHLCV = {"Open","High","Low","Close","Volume","Dividends","Stock Splits","target"}

def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in _OHLCV]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df[["Open","High","Low","Close","Volume"]].copy().astype(float)
    c = d["Close"]; o = d["Open"]; h = d["High"]; l = d["Low"]; v = d["Volume"]

    # ── Log-returns ───────────────────────────────────────────────────────
    log_ret = np.log(c / c.shift(1).replace(0, np.nan))
    d["log_ret"] = log_ret
    d["ret_1"]   = c.pct_change()
    for lag in (2, 3, 5, 10, 15, 20):
        d[f"ret_{lag}"] = c.pct_change(lag)

    # ── Price / MA RATIOS ─────────────────────────────────────────────────
    for w in (5, 10, 20, 50, 100, 200):
        ma = c.rolling(w).mean()
        d[f"c_ma{w}"] = c / (ma + 1e-10) - 1.0
    for span in (9, 21, 50):
        ema = c.ewm(span=span).mean()
        d[f"c_ema{span}"] = c / (ema + 1e-10) - 1.0

    # ── MACD (normalised) ─────────────────────────────────────────────────
    ema12  = c.ewm(span=12).mean()
    ema26  = c.ewm(span=26).mean()
    macd   = ema12 - ema26
    macd_s = macd.ewm(span=9).mean()
    d["macd_r"]      = macd            / (c + 1e-10)
    d["macd_sig_r"]  = macd_s          / (c + 1e-10)
    d["macd_hist_r"] = (macd - macd_s) / (c + 1e-10)

    # ── Bollinger Bands ───────────────────────────────────────────────────
    bb_m = c.rolling(20).mean()
    bb_s = c.rolling(20).std()
    d["bb_width"] = (2 * bb_s) / (bb_m + 1e-10)
    d["bb_pos"]   = np.clip((c - (bb_m - 2*bb_s)) / (4*bb_s + 1e-6), -3, 3)

    # ── RSI family ────────────────────────────────────────────────────────
    d["rsi_7"]     = calc_rsi(c, 7)  / 100.0
    d["rsi_14"]    = calc_rsi(c, 14) / 100.0
    d["rsi_21"]    = calc_rsi(c, 21) / 100.0
    d["rsi_7_ch"]  = d["rsi_7"].diff()
    d["rsi_14_ch"] = d["rsi_14"].diff()

    # ── Stochastic %K / %D ────────────────────────────────────────────────
    lo14 = c.rolling(14).min(); hi14 = c.rolling(14).max()
    stk  = (c - lo14) / (hi14 - lo14 + 1e-10)
    d["stoch_k"]  = stk
    d["stoch_d"]  = stk.rolling(3).mean()
    d["stoch_kd"] = stk - stk.rolling(3).mean()

    # ── Williams %R ───────────────────────────────────────────────────────
    hi14w = h.rolling(14).max(); lo14w = l.rolling(14).min()
    d["williams_r"] = (hi14w - c) / (hi14w - lo14w + 1e-10)

    # ── CCI (Commodity Channel Index, normalised) ─────────────────────────
    tp   = (h + l + c) / 3
    tp_m = tp.rolling(20).mean()
    tp_d = tp.rolling(20).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    d["cci"] = np.clip((tp - tp_m) / (0.015 * tp_d + 1e-10) / 200.0, -3, 3)

    # ── ADX (Average Directional Index, normalised) ───────────────────────
    tr      = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    dm_plus = (h - h.shift()).clip(lower=0)
    dm_minus= (l.shift() - l).clip(lower=0)
    dm_plus[dm_plus < dm_minus]  = 0
    dm_minus[dm_minus < dm_plus] = 0
    atr14_adx = tr.rolling(14).mean()
    di_plus   = 100 * dm_plus.rolling(14).mean()  / (atr14_adx + 1e-10)
    di_minus  = 100 * dm_minus.rolling(14).mean() / (atr14_adx + 1e-10)
    dx        = (di_plus - di_minus).abs() / (di_plus + di_minus + 1e-10) * 100
    d["adx"]     = dx.rolling(14).mean() / 100.0
    d["di_diff"] = (di_plus - di_minus) / 100.0

    # ── Rate of Change (ROC) ──────────────────────────────────────────────
    for roc_n in (5, 10, 20):
        d[f"roc_{roc_n}"] = (c - c.shift(roc_n)) / (c.shift(roc_n) + 1e-10)

    # ── ATR (normalised) ──────────────────────────────────────────────────
    hl = h - l
    hc = (h - c.shift()).abs()
    lc = (l - c.shift()).abs()
    tr2   = pd.concat([hl, hc, lc], axis=1).max(axis=1)
    atr14 = tr2.rolling(14).mean()
    d["atr_r"]    = atr14 / (c + 1e-10)
    d["atr_r_ch"] = d["atr_r"].diff()

    # ── Volume (normalised) ───────────────────────────────────────────────
    vol_ma20      = v.rolling(20).mean()
    d["vol_r"]    = v / (vol_ma20 + 1)
    d["vol_r5"]   = v / (v.rolling(5).mean() + 1)
    d["vol_std_r"]= v.rolling(20).std() / (vol_ma20 + 1)
    d["vol_ret"]  = v.pct_change()
    sign          = np.sign(c.diff().fillna(0))
    obv           = (sign * v).cumsum()
    obv_ma        = obv.rolling(10).mean()
    d["obv_slope_r"] = (obv - obv.shift(5)) / (obv.abs().rolling(5).mean() + 1e-10)
    d["obv_vs_ma"]   = (obv - obv_ma) / (obv_ma.abs() + 1e-10)

    # ── Candle structure ──────────────────────────────────────────────────
    d["hl_r"]         = (h - l) / (c + 1e-10)
    d["co_r"]         = (c - o) / (o + 1e-10)
    d["gap"]          = (o - c.shift()) / (c.shift() + 1e-10)
    d["upper_shadow"] = (h - pd.concat([c, o], axis=1).max(axis=1)) / (c + 1e-10)
    d["lower_shadow"] = (pd.concat([c, o], axis=1).min(axis=1) - l) / (c + 1e-10)

    # ── Volatility regime ─────────────────────────────────────────────────
    d["vol5"]      = log_ret.rolling(5).std()
    d["vol10"]     = log_ret.rolling(10).std()
    d["vol20"]     = log_ret.rolling(20).std()
    d["vol60"]     = log_ret.rolling(60).std()
    d["vol_ratio"] = d["vol5"] / (d["vol20"] + 1e-10)

    # ── Z-score of price vs rolling mean ─────────────────────────────────
    for w in (10, 20, 50):
        rm = c.rolling(w).mean()
        rs = c.rolling(w).std()
        d[f"zscore_{w}"] = np.clip((c - rm) / (rs + 1e-6), -5, 5)

    # ── Autocorrelation of returns ────────────────────────────────────────
    d["ret_autocorr_5"] = log_ret.rolling(5).apply(
        lambda x: np.nan_to_num(x.autocorr(lag=1), nan=0.0), raw=False)
    d["ret_autocorr_10"] = log_ret.rolling(10).apply(
        lambda x: np.nan_to_num(x.autocorr(lag=1), nan=0.0), raw=False)

    # ── Momentum persistence features ────────────────────────────────────
    d["mom_5_sign"]  = np.sign(c.pct_change(5))
    d["mom_10_sign"] = np.sign(c.pct_change(10))
    d["mom_streak"]  = log_ret.apply(lambda x: 1.0 if x > 0 else -1.0
                                     ).rolling(5).sum() / 5.0

    # ── Calendar dummies ──────────────────────────────────────────────────
    idx = d.index
    if isinstance(idx, pd.DatetimeIndex):
        d["dow"]       = idx.dayofweek / 4.0
        d["dom"]       = idx.day / 31.0
        d["month"]     = idx.month / 12.0
        d["is_mon"]    = (idx.dayofweek == 0).astype(float)
        d["is_fri"]    = (idx.dayofweek == 4).astype(float)
        d["month_sin"] = np.sin(2 * np.pi * idx.month / 12)
        d["month_cos"] = np.cos(2 * np.pi * idx.month / 12)

    return d.drop(columns=["Open","High","Low","Volume"], errors="ignore").dropna()


def clean_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.clip(lower=-10, upper=10)
    df = df.dropna()
    return df

# ═══════════════════════════════════════════════════════════════════════════════
#  ENSEMBLE TRAINING — Log-return targets, reduced regularisation
# ═══════════════════════════════════════════════════════════════════════════════

def train_ensemble(feat_df: pd.DataFrame, feature_cols: list, hist_vol: float):
    df     = feat_df.copy()
    close  = df["Close"]
    log_ret_t1 = np.log(close.shift(-1) / close)
    df["target"] = log_ret_t1
    df = df.dropna(subset=["target"] + feature_cols)

    if len(df) < 60:
        return None

    X = df[feature_cols].values.astype(np.float32)
    y = df["target"].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)
    X = np.clip(X, -10, 10)

    scaler_X = StandardScaler()
    X_sc     = scaler_X.fit_transform(X)

    split  = max(int(len(X) * 0.80), 50)
    X_tr, X_val = X_sc[:split], X_sc[split:]
    y_tr, y_val = y[:split],    y[split:]

    # Cap at 5× historical vol (slightly looser to preserve signal)
    cap   = max(5.0 * hist_vol, 0.04)
    y_tr  = np.clip(y_tr,  -cap, cap)
    y_val = np.clip(y_val, -cap, cap)

    # ── XGBoost — reduced regularisation vs original ──────────────────────
    # reg_lambda: 3.0→1.0, reg_alpha: 0.3→0.05, min_child_weight: 5→3
    # These changes let the model capture real log-return signals instead
    # of being penalised into predicting zero every time.
    xgb_m = XGBRegressor(
        n_estimators=400, max_depth=5, learning_rate=0.025,
        subsample=0.80, colsample_bytree=0.70,
        min_child_weight=3,
        reg_alpha=0.05, reg_lambda=1.0, gamma=0.05,
        random_state=42, verbosity=0, n_jobs=-1,
    )
    try:
        xgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                   verbose=False, early_stopping_rounds=30)
    except TypeError:
        xgb_m.fit(X_tr, y_tr)

    # ── GradientBoosting — slightly looser ────────────────────────────────
    gbr_m = GradientBoostingRegressor(
        n_estimators=250, max_depth=4, learning_rate=0.03,
        subsample=0.80, min_samples_leaf=4,
        max_features="sqrt", random_state=42,
    )
    gbr_m.fit(X_tr, y_tr)

    # ── RandomForest — moderate depth ────────────────────────────────────
    rf_m = RandomForestRegressor(
        n_estimators=250, max_depth=8, min_samples_leaf=4,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    rf_m.fit(X_tr, y_tr)

    def rmse(model):
        p = model.predict(X_val)
        return float(np.sqrt(np.mean((p - y_val)**2))) + 1e-10

    rmses     = {"xgb": rmse(xgb_m), "gbr": rmse(gbr_m), "rf": rmse(rf_m)}
    inv_total = sum(1/v for v in rmses.values())
    w         = {k: (1/v)/inv_total for k, v in rmses.items()}

    val_ret_pred  = (w["xgb"] * xgb_m.predict(X_val)
                     + w["gbr"] * gbr_m.predict(X_val)
                     + w["rf"]  * rf_m.predict(X_val))
    val_close_start = close.iloc[split:split+len(y_val)].values
    val_price_pred  = val_close_start * np.exp(val_ret_pred)
    val_price_real  = val_close_start * np.exp(y_val)

    val_mape = float(np.mean(
        np.abs(val_price_pred - val_price_real) / (np.abs(val_price_real) + 1e-10)
    ))
    dir_acc  = float(np.mean(np.sign(val_ret_pred) == np.sign(y_val)))

    return {
        "xgb": xgb_m, "gbr": gbr_m, "rf": rf_m, "w": w,
        "scaler_X": scaler_X, "feature_cols": feature_cols,
        "val_mape": val_mape, "dir_acc": dir_acc,
        "val_ret_pred": val_ret_pred, "y_val": y_val,
        "val_price_pred": val_price_pred, "val_price_real": val_price_real,
        "rmses": rmses, "split": split,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SANITY-CHECKED FORECASTER
#
#  KEY FIX (zero-prediction bug):
#  1. Decay changed from 0.92^step (kills signal by day 7) to 0.985^step
#  2. Momentum persistence: blend model signal with recent historical momentum
#  3. Per-step cap slightly increased (3× vol) so realistic moves survive
#  4. Sentiment nudge doubled for visibility
# ═══════════════════════════════════════════════════════════════════════════════

def predict_future(
    df_orig: pd.DataFrame,
    days: int,
    m: dict,
    hist_vol: float,
    sentiment: float = 0.0,
    skip_weekends: bool = True,
):
    df   = df_orig[["Open","High","Low","Close","Volume"]].copy().astype(float)
    feat = m["feature_cols"]
    sx   = m["scaler_X"]
    w    = m["w"]

    last_close = float(df["Close"].iloc[-1])

    # Historical momentum (10-day avg log return) used as persistence anchor.
    # This ensures the model isn't pulled to zero when the stock is trending.
    log_rets_hist = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    hist_mom_10   = float(log_rets_hist.tail(10).mean()) if len(log_rets_hist) >= 10 else 0.0
    hist_mom_5    = float(log_rets_hist.tail(5).mean())  if len(log_rets_hist) >= 5  else 0.0

    # Cap at 3× hist_vol per step — realistic, prevents blow-up
    per_step_cap = float(np.clip(3.0 * hist_vol, 0.01, 0.05))

    # Sentiment nudge in log-return units (scaled up for visibility)
    sent_adj_per_step = float(np.clip(sentiment * 0.0004, -0.001, 0.001))

    log_ret_preds = []
    price_preds   = []
    price_lo      = []
    price_hi      = []
    cur_price     = last_close

    for step in range(days):
        feat_df = engineer_features(df.tail(250))

        if feat_df.empty or len(feat_df) < 10:
            # Fallback: use historical momentum
            pred_lr = hist_mom_5 * 0.6
        else:
            for col in feat:
                if col not in feat_df.columns:
                    feat_df[col] = 0.0

            X_last = feat_df[feat].iloc[-1:].values.astype(np.float32)
            X_last = np.nan_to_num(X_last, nan=0.0, posinf=0.0, neginf=0.0)

            try:
                X_sc = sx.transform(X_last)
            except Exception:
                pred_lr = hist_mom_5 * 0.6
                log_ret_preds.append(pred_lr)
                cur_price = cur_price * np.exp(pred_lr)
                sigma_steps = hist_vol * np.sqrt(step + 1)
                price_preds.append(float(cur_price))
                price_lo.append(float(cur_price * np.exp(-1.96 * sigma_steps)))
                price_hi.append(float(cur_price * np.exp( 1.96 * sigma_steps)))
                last_idx = df.index[-1]
                next_d   = pd.Timestamp(last_idx) + timedelta(days=1)
                if skip_weekends:
                    while next_d.weekday() >= 5:
                        next_d += timedelta(days=1)
                avg_range = float((df["High"] - df["Low"]).tail(20).mean())
                new_row   = pd.DataFrame({
                    "Open": [cur_price], "High": [cur_price + avg_range*0.5],
                    "Low":  [max(cur_price - avg_range*0.5, 0.01)],
                    "Close": [cur_price], "Volume": [float(df["Volume"].median())],
                }, index=[next_d])
                df = pd.concat([df, new_row])
                continue

            # Weighted ensemble in log-return space
            pred_lr = float(
                w["xgb"] * m["xgb"].predict(X_sc)[0]
                + w["gbr"] * m["gbr"].predict(X_sc)[0]
                + w["rf"]  * m["rf"].predict(X_sc)[0]
            )

            # ── MOMENTUM PERSISTENCE (the key fix) ───────────────────────
            # Blend model prediction with recent historical momentum.
            # blend_w decays over the horizon so early steps lean more on
            # momentum (fast signal) and late steps lean on the model.
            blend_w = max(0.0, 1.0 - step * 0.07)   # 1.0 → 0.0 over ~14 steps
            pred_lr = pred_lr * (1 - blend_w * 0.30) + hist_mom_10 * blend_w * 0.30

            # Sentiment nudge
            pred_lr += sent_adj_per_step

            # ── MILD DECAY (was 0.92^step — far too aggressive) ──────────
            # 0.985^step: at day 7 → 0.90, at day 14 → 0.81, at day 30 → 0.64
            # This preserves directional signal while dampening multi-day drift.
            decay    = 0.985 ** step
            pred_lr *= decay

            # Hard cap per step
            pred_lr = float(np.clip(pred_lr, -per_step_cap, per_step_cap))

        log_ret_preds.append(pred_lr)
        cur_price = cur_price * np.exp(log_ret_preds[-1])

        sigma_steps = hist_vol * np.sqrt(step + 1)
        price_preds.append(float(cur_price))
        price_lo.append(float(cur_price * np.exp(-1.96 * sigma_steps)))
        price_hi.append(float(cur_price * np.exp( 1.96 * sigma_steps)))

        last_idx  = df.index[-1]
        next_d    = pd.Timestamp(last_idx) + timedelta(days=1)
        if skip_weekends:
            while next_d.weekday() >= 5:
                next_d += timedelta(days=1)
        avg_range = float((df["High"] - df["Low"]).tail(20).mean())
        new_row   = pd.DataFrame({
            "Open":   [cur_price],
            "High":   [cur_price + avg_range * 0.5],
            "Low":    [max(cur_price - avg_range * 0.5, 0.01)],
            "Close":  [cur_price],
            "Volume": [float(df["Volume"].median())],
        }, index=[next_d])
        df = pd.concat([df, new_row])

    # ── SANITY CHECK ──────────────────────────────────────────────────────
    total_pct         = (price_preds[-1] - last_close) / last_close * 100.0
    max_realistic_pct = 2.0 * hist_vol * np.sqrt(days) * 100.0
    sanity_ok         = abs(total_pct) <= max_realistic_pct * 1.5
    sanity_msg        = (
        f"✅ Prediction sanity OK — {total_pct:+.1f}% over {days} days "
        f"(2σ historical range: ±{max_realistic_pct:.1f}%)"
        if sanity_ok else
        f"⚠️ Prediction clipped — raw model wanted {total_pct:+.1f}%, "
        f"capped to ±{max_realistic_pct:.1f}% (2σ × {days}d volatility)"
    )

    if not sanity_ok:
        clip_factor = (max_realistic_pct / abs(total_pct)) if abs(total_pct) > 1e-6 else 1.0
        price_preds = [last_close + (p - last_close) * clip_factor for p in price_preds]
        price_lo    = [p * np.exp(-1.96 * hist_vol * np.sqrt(i+1)) for i, p in enumerate(price_preds)]
        price_hi    = [p * np.exp( 1.96 * hist_vol * np.sqrt(i+1)) for i, p in enumerate(price_preds)]
        sanity_ok   = True

    return price_preds, price_lo, price_hi, sanity_ok, sanity_msg


def future_trading_dates(last_date, days: int, skip_weekends: bool = True) -> list:
    dates = []
    d     = pd.Timestamp(last_date)
    while len(dates) < days:
        d += timedelta(days=1)
        if skip_weekends and d.weekday() >= 5:
            continue
        dates.append(d)
    return dates

# ═══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

_DARK   = {"paper_bgcolor": "#070b14", "plot_bgcolor": "#0f1623"}
_MARGIN = dict(l=0, r=0, t=40, b=0)

def build_price_chart(df, preds, future_dates, lo_ci, hi_ci,
                       company, ticker, chart_type="Candlestick"):
    fig = go.Figure()
    if chart_type == "Candlestick":
        fig.add_trace(go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"], name="Price",
            increasing_line_color="#22c55e", decreasing_line_color="#ef4444",
            increasing_fillcolor="#166534", decreasing_fillcolor="#991b1b",
        ))
    else:
        fig.add_trace(go.Scatter(
            x=df.index, y=df["Close"], mode="lines", name="Close",
            line=dict(color="#60a5fa", width=2),
        ))
    bb_m = df["Close"].rolling(20).mean()
    bb_s = df["Close"].rolling(20).std()
    fig.add_trace(go.Scatter(x=df.index, y=bb_m+2*bb_s,
        line=dict(color="rgba(99,102,241,.3)", width=.8),
        name="BB Upper", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=df.index, y=bb_m-2*bb_s,
        line=dict(color="rgba(99,102,241,.3)", width=.8),
        fill="tonexty", fillcolor="rgba(99,102,241,.05)",
        name="BB Lower", hoverinfo="skip"))
    for ww, col in ((20,"#facc15"),(50,"#fb923c")):
        if len(df) >= ww:
            fig.add_trace(go.Scatter(x=df.index, y=df["Close"].rolling(ww).mean(),
                mode="lines", name=f"MA {ww}",
                line=dict(color=col, width=1, dash="dot"), opacity=0.9))
    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=hi_ci + lo_ci[::-1],
        fill="toself", fillcolor="rgba(167,139,250,.12)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI", hoverinfo="skip"))
    fig.add_trace(go.Scatter(
        x=future_dates, y=preds, mode="lines+markers",
        name="Forecast", line=dict(color="#a78bfa", dash="dash", width=2.5),
        marker=dict(size=7, color="#7c3aed", line=dict(color="#a78bfa", width=1.5))))
    fig.update_layout(
        title=dict(text=f"{company}  <span style='font-size:13px;color:#6b7280'>({ticker})</span>",
                   font=dict(size=16, color="#e2e8f0")),
        xaxis_title="Date", yaxis_title="Price",
        xaxis_rangeslider_visible=False, height=440,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, font=dict(size=11)),
        template="plotly_dark", margin=_MARGIN, **_DARK,
    )
    return fig

def build_rsi_chart(df):
    rsi = calc_rsi(df["Close"])
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,.07)", line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(34,197,94,.07)",  line_width=0)
    fig.add_trace(go.Scatter(x=df.index, y=rsi, mode="lines", name="RSI (14)",
                              line=dict(color="#c084fc", width=1.5)))
    fig.add_hline(y=70, line_color="#ef4444", line_dash="dash",
                  annotation_text="70", annotation_position="right")
    fig.add_hline(y=30, line_color="#22c55e", line_dash="dash",
                  annotation_text="30", annotation_position="right")
    fig.update_layout(title="RSI (14)", height=200, yaxis=dict(range=[0,100]),
                       template="plotly_dark", margin=_MARGIN, **_DARK)
    return fig

def build_macd_chart(df):
    ema12  = df["Close"].ewm(span=12).mean()
    ema26  = df["Close"].ewm(span=26).mean()
    macd   = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist   = macd - signal
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in hist]
    fig    = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=hist, name="Histogram",
                          marker_color=colors, opacity=0.65))
    fig.add_trace(go.Scatter(x=df.index, y=macd,   mode="lines", name="MACD",
                              line=dict(color="#60a5fa", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=signal, mode="lines", name="Signal",
                              line=dict(color="#fb923c", width=1.5)))
    fig.update_layout(title="MACD", height=210,
                       template="plotly_dark", margin=_MARGIN, **_DARK)
    return fig

def build_volume_chart(df):
    colors  = ["#22c55e" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#ef4444"
                for i in range(len(df))]
    vol_ma  = df["Volume"].rolling(20).mean()
    fig     = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                          marker_color=colors, opacity=0.7))
    fig.add_trace(go.Scatter(x=df.index, y=vol_ma, mode="lines", name="Vol MA 20",
                              line=dict(color="#facc15", width=1.2, dash="dot")))
    fig.update_layout(title="Volume", height=180,
                       template="plotly_dark", margin=_MARGIN, **_DARK)
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR — AI AGENT CHAT
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding:20px 0 10px;">
      <div style="font-size:52px; line-height:1;">🤖</div>
      <h2 style="color:#a78bfa; margin:8px 0 2px; font-size:19px; font-weight:800;">
        QuantAI Agent
      </h2>
      <p style="color:#6b7280; font-size:11px; margin:0;">Real-time market intelligence</p>
    </div>
    <hr style="border-color:#1f2937; margin:10px 0;">
    """, unsafe_allow_html=True)

    if "chat" not in st.session_state:
        st.session_state.chat = []

    for msg in st.session_state.chat:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])

    prompt = st.chat_input("Ask about stocks, trends, earnings…")

    if prompt:
        st.session_state.chat.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.spinner("Fetching market data…"):
            rt_data = fetch_real_time(prompt + " stock finance 2025")
        sys_msg = (
            "You are QuantAI, an elite quantitative financial AI analyst. "
            "You have access to real-time web data below. "
            "Be precise, concise, and data-driven. "
            "Always end with: '⚠️ Not financial advice.'\n\n"
            f"Real-Time Data:\n{rt_data}"
        )
        messages = [
            {"role": "system", "content": sys_msg},
            *[{"role": m["role"], "content": m["content"]}
              for m in st.session_state.chat[-10:]],
        ]
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                res   = _ai_client.chat.completions.create(
                    model=_AI_MODEL, messages=messages,
                    max_tokens=700, temperature=0.6,
                )
                reply = res.choices[0].message.content
            st.write(reply)
        st.session_state.chat.append({"role": "assistant", "content": reply})

    if st.session_state.chat:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.chat = []
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
#  FUNDAMENTALS DISPLAY HELPER
# ═══════════════════════════════════════════════════════════════════════════════

def display_fundamentals(fund: dict) -> None:
    """Render fundamental metrics row. Handles missing values gracefully."""
    if not fund:
        return

    def fmt(val, pct=False, suffix=""):
        if val is None or (isinstance(val, float) and np.isnan(val)):
            return "N/A"
        if pct:
            return f"{val*100:.1f}%"
        return f"{val:.2f}{suffix}"

    pe  = fund.get("pe_ratio")
    fpe = fund.get("fwd_pe")
    rg  = fund.get("revenue_growth")
    eg  = fund.get("earnings_growth")
    de  = fund.get("debt_equity")
    pm  = fund.get("profit_margin")
    om  = fund.get("operating_margin")
    gm  = fund.get("gross_margin")
    roe = fund.get("return_on_equity")
    peg = fund.get("peg_ratio")

    st.markdown("##### 📊 Fundamental Metrics")
    f1, f2, f3, f4, f5 = st.columns(5)
    f1.metric("P/E (TTM)",        fmt(pe),   delta=f"Fwd {fmt(fpe)}" if fpe else None)
    f2.metric("Revenue Growth",   fmt(rg, pct=True))
    f3.metric("Earnings Growth",  fmt(eg, pct=True))
    f4.metric("Debt / Equity",    fmt(de))
    f5.metric("Profit Margin",    fmt(pm, pct=True))

    f6, f7, f8, f9, f10 = st.columns(5)
    f6.metric("Operating Margin", fmt(om, pct=True))
    f7.metric("Gross Margin",     fmt(gm, pct=True))
    f8.metric("ROE",              fmt(roe, pct=True))
    f9.metric("PEG Ratio",        fmt(peg))
    mc = fund.get("market_cap")
    mc_str = f"${mc/1e9:.1f}B" if mc and mc >= 1e9 else (f"${mc/1e6:.1f}M" if mc else "N/A")
    f10.metric("Market Cap",      mc_str)
    st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED PREDICTION UI
# ═══════════════════════════════════════════════════════════════════════════════

def run_prediction_ui(ticker, company, period, days, chart_type,
                       use_sentiment, skip_weekends=True, price_label="Price"):
    sym = currency_symbol(ticker)

    with st.spinner(f"📡 Fetching {company} ({ticker})…"):
        df = load_stock(ticker, period)

    if df.empty:
        st.error(f"⚠️ No data for **{ticker}**. Check the ticker or try a longer period.")
        return
    if len(df) < 60:
        st.warning(f"Only {len(df)} rows — try a longer period (1y+) for better accuracy.")
        return

    log_rets = np.log(df["Close"] / df["Close"].shift(1)).dropna()
    hist_vol = float(log_rets.std())

    # Fundamentals (fetched in background, shown before chart)
    with st.spinner("📊 Loading fundamental data…"):
        fund = get_fundamentals(ticker)

    # Sentiment
    sentiment = 0.0
    if use_sentiment:
        clean = company
        for ch in ["🥇","🥈","⬜","🔵","🛢️","🔥","🔶","🌽","🌾","🫘"]:
            clean = clean.replace(ch,"")
        clean = clean.strip()
        with st.spinner("🧠 Running FinBERT sentiment…"):
            sentiment = get_news_sentiment(clean)

    # Feature engineering
    with st.spinner("⚙️ Building normalised indicators…"):
        feat_df = engineer_features(df.copy())
        feat_df = clean_features(feat_df)

    if len(feat_df) < 60:
        st.warning("⚠️ Insufficient history after feature generation. Try **1y** or **2y**.")
        return

    feature_cols = get_feature_cols(feat_df)

    # Train ensemble
    with st.spinner("🤖 Training XGBoost + GBR + RandomForest on log-returns…"):
        mdl = train_ensemble(feat_df, feature_cols, hist_vol)

    if not mdl:
        st.error("Model training failed — need at least 60 clean rows.")
        return

    # Forecast
    with st.spinner(f"🔮 Forecasting {days} trading days…"):
        preds, lo_ci, hi_ci, sanity_ok, sanity_msg = predict_future(
            df.copy(), days, mdl, hist_vol, sentiment, skip_weekends
        )

    future_dates = future_trading_dates(df.index[-1], days, skip_weekends)

    # ── Metrics ──────────────────────────────────────────────────────────────
    current   = float(df["Close"].iloc[-1])
    prev      = float(df["Close"].iloc[-2]) if len(df) > 1 else current
    pred_last = preds[-1]
    day_pct   = (current - prev)      / prev      * 100
    pred_pct  = (pred_last - current) / current   * 100
    wk_idx    = max(-7, -len(df))
    wk_pct    = (current - float(df["Close"].iloc[wk_idx])) / float(df["Close"].iloc[wk_idx]) * 100
    accuracy  = max(0.0, (1 - mdl["val_mape"]) * 100)
    dir_acc   = mdl["dir_acc"] * 100
    sent_lbl  = ("🟢 Bullish" if sentiment > 0.1
                 else "🔴 Bearish" if sentiment < -0.1
                 else "⚪ Neutral")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(price_label,          f"{sym}{current:,.2f}",   f"{day_pct:+.2f}%")
    c2.metric(f"+{days}d Forecast", f"{sym}{pred_last:,.2f}", f"{pred_pct:+.2f}%")
    c3.metric("7-Day Δ",            f"{wk_pct:+.2f}%")
    c4.metric("Price Accuracy",     f"{accuracy:.1f}%",        f"Dir {dir_acc:.0f}%")
    c5.metric("Sentiment",          sent_lbl,                  f"{sentiment*100:+.1f}%")

    st.markdown(
        f'<div class="model-badge">⚡ XGBoost&nbsp;{mdl["w"]["xgb"]*100:.0f}% · '
        f'GBR&nbsp;{mdl["w"]["gbr"]*100:.0f}% · '
        f'RF&nbsp;{mdl["w"]["rf"]*100:.0f}% · '
        f'hist_vol&nbsp;{hist_vol*100:.2f}%/day</div>',
        unsafe_allow_html=True,
    )

    # Sanity banner
    css_cls = "sanity-ok" if sanity_ok else "sanity-warn"
    st.markdown(f'<div class="{css_cls}">{sanity_msg}</div>', unsafe_allow_html=True)

    # ── Fundamental metrics ───────────────────────────────────────────────────
    st.markdown("")
    display_fundamentals(fund)

    # ── Price chart ───────────────────────────────────────────────────────────
    st.plotly_chart(
        build_price_chart(df, preds, future_dates, lo_ci, hi_ci,
                           company, ticker, chart_type),
        use_container_width=True,
    )

    # Indicators
    ic1, ic2 = st.columns(2)
    with ic1:
        st.plotly_chart(build_rsi_chart(df), use_container_width=True)
    with ic2:
        st.plotly_chart(build_macd_chart(df), use_container_width=True)
    st.plotly_chart(build_volume_chart(df), use_container_width=True)

    # ── AI Validator ──────────────────────────────────────────────────────────
    with st.expander("🤖 AI Prediction Validator (Qwen Double-Check)", expanded=True):
        with st.spinner("🧠 AI is reviewing the prediction…"):
            verdict_text = ai_validate_prediction(
                company, ticker, current, pred_last, pred_pct,
                days, hist_vol, sentiment, fund, sym,
            )
        # Determine CSS class from verdict
        verdict_cls = "ai-ok" if "VERDICT: ACCEPTED" in verdict_text else "ai-warn"
        st.markdown(
            f'<div class="{verdict_cls}">🤖 <strong>AI Validator:</strong><br>'
            f'{verdict_text.replace(chr(10), "<br>")}</div>',
            unsafe_allow_html=True,
        )

    # ── Validation expander ───────────────────────────────────────────────────
    with st.expander("🧪 Model Validation (Actual vs Predicted)"):
        n_val   = len(mdl["val_price_pred"])
        val_idx = list(df.index[mdl["split"]:mdl["split"]+n_val])
        if len(val_idx) < n_val:
            val_idx += [val_idx[-1] + timedelta(days=i+1)
                        for i in range(n_val - len(val_idx))]
        vfig = go.Figure()
        vfig.add_trace(go.Scatter(x=val_idx, y=mdl["val_price_real"],
                                   mode="lines", name="Actual",
                                   line=dict(color="#60a5fa", width=2)))
        vfig.add_trace(go.Scatter(x=val_idx, y=mdl["val_price_pred"],
                                   mode="lines", name="Predicted",
                                   line=dict(color="#a78bfa", width=2, dash="dash")))
        vfig.update_layout(title="Hold-Out Validation — Actual vs Predicted",
                            template="plotly_dark", height=260,
                            margin=_MARGIN, **_DARK)
        st.plotly_chart(vfig, use_container_width=True)
        rc1, rc2, rc3, rc4 = st.columns(4)
        rc1.metric("XGBoost RMSE", f"{mdl['rmses']['xgb']*100:.4f}%",
                   f"W={mdl['w']['xgb']*100:.0f}%")
        rc2.metric("GBR RMSE",     f"{mdl['rmses']['gbr']*100:.4f}%",
                   f"W={mdl['w']['gbr']*100:.0f}%")
        rc3.metric("RF RMSE",      f"{mdl['rmses']['rf']*100:.4f}%",
                   f"W={mdl['w']['rf']*100:.0f}%")
        rc4.metric("Directional Acc", f"{dir_acc:.1f}%",
                   f"MAPE {mdl['val_mape']*100:.2f}%")

    # ── Forecast table ────────────────────────────────────────────────────────
    with st.expander("📋 Full Forecast Table"):
        fc_df = pd.DataFrame({
            "Date":            [d.strftime("%Y-%m-%d") for d in future_dates],
            "Predicted":       [f"{sym}{p:,.2f}" for p in preds],
            "Lower CI (95%)":  [f"{sym}{l:,.2f}" for l in lo_ci],
            "Upper CI (95%)":  [f"{sym}{h:,.2f}" for h in hi_ci],
            "Δ% vs Today":     [f"{((p-current)/current)*100:+.2f}%" for p in preds],
            "Daily Δ%":        [
                f"{((preds[i]-preds[i-1])/preds[i-1]*100):+.2f}%" if i > 0
                else f"{((preds[0]-current)/current*100):+.2f}%"
                for i in range(len(preds))
            ],
        })
        st.dataframe(fc_df, use_container_width=True, hide_index=True)

    with st.expander("📈 Last 10 Trading Days"):
        st.dataframe(
            df[["Open","High","Low","Close","Volume"]].tail(10).round(3),
            use_container_width=True,
        )

# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Stock Predictor",
    "📊 Market Overview",
    "💎 Elements & Jewelry",
    "🏆 Top 5 ROI Picks",
])

# ─── TAB 1: STOCK PREDICTOR ───────────────────────────────────────────────────
with tab1:
    st.markdown(
        "<h4 style='text-align:center; color:#e2e8f0;'>Elite Stock Price Predictor</h4>",
        unsafe_allow_html=True,
    )

    ctrl_col, chart_col = st.columns([1, 2.6])

    with ctrl_col:
        st.markdown("#### ⚙️ Configuration")

        market = st.selectbox(
            "🌍 Market",
            ["🇺🇸 US Markets", "🇮🇳 India NSE", "🇮🇳 India BSE"],
        )

        if "US" in market:
            cap_cat    = st.selectbox("📂 Cap Category",
                                       ["🏛️ Large Cap","🏢 Mid Cap","🏬 Small Cap"])
            stocks_map = (us_large_cap if "Large" in cap_cat
                          else us_mid_cap if "Mid" in cap_cat else us_small_cap)
            exchange_badge = '<span class="us-badge">NYSE/NASDAQ</span>'
            skip_wk = True
        else:
            exch    = "NSE" if "NSE" in market else "BSE"
            cap_cat = st.selectbox("📂 Cap Category",
                                    ["🏛️ Large Cap","🏢 Mid Cap","🏬 Small Cap"])
            base    = (india_large_base if "Large" in cap_cat
                       else india_mid_base if "Mid" in cap_cat else india_small_base)
            stocks_map   = make_india_map(base, exch)
            badge_cls    = "nse-badge" if exch == "NSE" else "bse-badge"
            exchange_badge = f'<span class="{badge_cls}">{exch}</span>'
            skip_wk = True

        use_custom = st.toggle("🔍 Enter custom ticker / company name", value=False)

        if use_custom:
            st.markdown(
                '<div class="ticker-search-box">'
                '💡 Type any ticker (e.g. <b>GOOG</b>, <b>NVDA</b>, <b>TCS.NS</b>) '
                'or company name (e.g. <b>Infosys</b>). '
                'The app will auto-resolve it.</div>',
                unsafe_allow_html=True,
            )
            custom_input = st.text_input(
                "Ticker / Company Name",
                placeholder="e.g. NVDA, TCS.NS, Zomato, Gold Futures",
            )
            if custom_input:
                with st.spinner(f"🔎 Resolving '{custom_input}'…"):
                    ticker, company = resolve_custom_ticker(custom_input)
                if ticker:
                    st.caption(f"✅ Resolved → **{ticker}** ({company})")
                else:
                    ticker, company = "", custom_input
                    st.warning("Could not auto-resolve. Check the ticker spelling.")
            else:
                ticker, company = "", ""
                st.info("Type a ticker or company name above.")
            exchange_badge = '<span class="us-badge">Custom</span>'
        else:
            company = st.selectbox("🏷️ Company", list(stocks_map.keys()))
            ticker  = stocks_map[company]

        if ticker:
            st.caption(f"Ticker: **{ticker}**")
            st.markdown(exchange_badge, unsafe_allow_html=True)

        period     = st.selectbox("📅 Period", ["3mo","6mo","1y","2y","5y"], index=2)
        days       = st.slider("🔮 Forecast Days", 1, 30, 7)
        chart_type = st.radio("📉 Chart Style", ["Candlestick","Line"], horizontal=True)
        use_sent   = st.toggle("🧠 Sentiment (FinBERT)", value=True)

        run_btn = st.button("🚀 Run Prediction", use_container_width=True,
                             type="primary", disabled=not bool(ticker))

    with chart_col:
        if run_btn and ticker:
            run_prediction_ui(
                ticker, company, period, days,
                chart_type, use_sent, skip_wk,
            )
        elif not ticker:
            st.info("👈 Enter or select a ticker to get started.")
        else:
            st.markdown("""
            <div style="height:400px; display:flex; flex-direction:column;
                        align-items:center; justify-content:center;
                        color:#374151; font-size:14px; gap:8px;">
              <div style="font-size:56px;">📈</div>
              <p>Configure settings → click <strong>Run Prediction</strong></p>
              <p style="font-size:12px; color:#1f2937;">
                Log-return model · Momentum persistence · AI validator · Fundamentals
              </p>
            </div>
            """, unsafe_allow_html=True)

# ─── TAB 2: MARKET OVERVIEW ───────────────────────────────────────────────────
with tab2:
    st.markdown(
        "<h4 style='text-align:center; color:#e2e8f0;'>📊 Market Overview</h4>",
        unsafe_allow_html=True,
    )

    ov_market = st.selectbox(
        "🌍 Market",
        ["US Large Cap","US Mid Cap","India NSE Large","India NSE Mid",
         "India BSE Large","India BSE Mid"],
        key="ov_market",
    )
    ov_map_lookup = {
        "US Large Cap":    us_large_cap,
        "US Mid Cap":      us_mid_cap,
        "India NSE Large": make_india_map(india_large_base, "NSE"),
        "India NSE Mid":   make_india_map(india_mid_base,   "NSE"),
        "India BSE Large": make_india_map(india_large_base, "BSE"),
        "India BSE Mid":   make_india_map(india_mid_base,   "BSE"),
    }
    ov_map    = ov_map_lookup[ov_market]
    ov_picked = st.multiselect(
        "Select companies",
        list(ov_map.keys()),
        default=list(ov_map.keys())[:6],
    )

    if st.button("📥 Load Market Data", use_container_width=True, key="ov_btn"):
        rows = []
        prog = st.progress(0)
        for i, name in enumerate(ov_picked):
            tkr = ov_map[name]
            sym = currency_symbol(tkr)
            try:
                d = load_stock(tkr, "5d")
                if not d.empty and len(d) >= 2:
                    cur  = float(d["Close"].iloc[-1])
                    prv  = float(d["Close"].iloc[-2])
                    chg  = (cur - prv) / prv * 100
                    vol  = int(d["Volume"].iloc[-1])
                    hi5  = float(d["Close"].max())
                    lo5  = float(d["Close"].min())
                    rows.append({
                        "Company": name, "Ticker": tkr,
                        "Price":   f"{sym}{cur:,.2f}",
                        "Day Δ%":  round(chg, 2),
                        "5d High": f"{sym}{hi5:,.2f}",
                        "5d Low":  f"{sym}{lo5:,.2f}",
                        "Volume":  f"{vol:,}",
                    })
            except Exception:
                pass
            prog.progress((i+1) / max(len(ov_picked), 1))
        prog.empty()

        if rows:
            ov_df = pd.DataFrame(rows)
            st.dataframe(ov_df, use_container_width=True, hide_index=True)
            bar = px.bar(ov_df, x="Company", y="Day Δ%", color="Day Δ%",
                          color_continuous_scale=["#ef4444","#374151","#22c55e"],
                          template="plotly_dark",
                          title=f"{ov_market} — Daily % Change")
            bar.update_layout(margin=_MARGIN, **_DARK, height=380)
            st.plotly_chart(bar, use_container_width=True)
        else:
            st.warning("No data returned. Try again shortly.")

# ─── TAB 3: ELEMENTS & JEWELRY ────────────────────────────────────────────────
with tab3:
    st.markdown(
        "<h4 style='text-align:center; color:#e2e8f0;'>💎 Elements, Commodities & Jewelry Predictor</h4>",
        unsafe_allow_html=True,
    )

    el_type = st.radio(
        "Category",
        ["🥇 Precious Metals (Futures)", "📈 Commodity ETFs", "💍 Jewelry Stocks"],
        horizontal=True,
    )
    el_map = (
        commodities_futures if "Futures" in el_type
        else commodities_etf if "ETF" in el_type
        else jewelry_stocks_map
    )
    price_label = (
        "Futures Price" if "Futures" in el_type
        else "ETF Price" if "ETF" in el_type
        else "Stock Price"
    )

    el_ctrl, el_chart = st.columns([1, 2.6])

    with el_ctrl:
        el_company = st.selectbox("Select Asset", list(el_map.keys()))
        el_ticker  = el_map[el_company]
        st.caption(f"Ticker: **{el_ticker}**")

        el_use_custom = st.toggle("🔍 Custom ticker", value=False, key="el_custom_tog")
        if el_use_custom:
            el_raw = st.text_input("Custom ticker",
                                    placeholder="GC=F, SLV, TITAN.NS…", key="el_custom_in")
            if el_raw:
                with st.spinner("Resolving…"):
                    el_ticker, el_company = resolve_custom_ticker(el_raw)
                st.caption(f"✅ → **{el_ticker}** ({el_company})")

        el_period     = st.selectbox("Period", ["3mo","6mo","1y","2y","5y"],
                                     index=2, key="el_period")
        el_days       = st.slider("Forecast Days", 1, 30, 7, key="el_days")
        el_chart_type = st.radio("Chart Style", ["Candlestick","Line"],
                                  horizontal=True, key="el_ct")
        el_sent       = st.toggle("🧠 Sentiment Analysis", value=True, key="el_sent")

        el_run = st.button("🚀 Run Prediction", use_container_width=True,
                            type="primary", key="el_run")

    with el_chart:
        if el_run:
            run_prediction_ui(
                el_ticker, el_company, el_period, el_days,
                el_chart_type, el_sent, True,
                price_label=price_label,
            )
        else:
            st.markdown("""
            <div style="height:400px; display:flex; flex-direction:column;
                        align-items:center; justify-content:center;
                        color:#374151; font-size:14px; gap:8px;">
              <div style="font-size:56px;">💎</div>
              <p>Select an asset → click <strong>Run Prediction</strong></p>
              <p style="font-size:12px; color:#1f2937;">
                Gold · Silver · Platinum · Jewelry stocks · Custom tickers
              </p>
            </div>
            """, unsafe_allow_html=True)

# ─── TAB 4: TOP 5 ROI PICKS ───────────────────────────────────────────────────
with tab4:
    st.markdown(
        "<h4 style='text-align:center; color:#e2e8f0;'>🏆 Top 5 Stocks for Best ROI Right Now</h4>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='text-align:center; color:#6b7280; font-size:13px;'>"
        "Composite score = 55% Technical (momentum · RSI · volume · trend · Sharpe) "
        "+ 45% Fundamental (P/E · revenue growth · earnings growth · debt/equity · margins · PEG)"
        "</p>",
        unsafe_allow_html=True,
    )

    t4_market = st.selectbox(
        "🌍 Analyse Market",
        ["US Large Cap", "US Mid Cap", "India NSE Large", "India NSE Mid",
         "India BSE Large", "India BSE Mid"],
        key="t4_market",
    )
    t4_map_lookup = {
        "US Large Cap":    us_large_cap,
        "US Mid Cap":      us_mid_cap,
        "India NSE Large": make_india_map(india_large_base, "NSE"),
        "India NSE Mid":   make_india_map(india_mid_base,   "NSE"),
        "India BSE Large": make_india_map(india_large_base, "BSE"),
        "India BSE Mid":   make_india_map(india_mid_base,   "BSE"),
    }
    t4_map = t4_map_lookup[t4_market]

    if st.button("🔍 Find Top 5 ROI Picks", use_container_width=True, type="primary", key="t4_btn"):
        import json
        with st.spinner(f"🔬 Scoring {min(len(t4_map), 20)} stocks across technical + fundamental dimensions…"):
            top5 = get_top_5_stocks(t4_market, json.dumps(t4_map))

        if not top5:
            st.warning("Could not score stocks. Check your connection and try again.")
        else:
            medals = ["🥇","🥈","🥉","4️⃣","5️⃣"]
            st.markdown("")

            for rank, stock in enumerate(top5):
                sym_  = stock["sym"]
                fund_ = stock.get("fund", {})

                # Helper to format fund value
                def fmtf(val, pct=False):
                    if val is None or (isinstance(val, float) and np.isnan(val)):
                        return "N/A"
                    return f"{val*100:.1f}%" if pct else f"{val:.2f}"

                pe_v  = fund_.get("pe_ratio")
                rg_v  = fund_.get("revenue_growth")
                eg_v  = fund_.get("earnings_growth")
                de_v  = fund_.get("debt_equity")
                pm_v  = fund_.get("profit_margin")

                with st.container():
                    st.markdown(f"""
                    <div class="top5-card">
                      <div style="display:flex; align-items:center; gap:12px; margin-bottom:8px;">
                        <span class="top5-rank">{medals[rank]}</span>
                        <div>
                          <div class="top5-name">{stock['name']}</div>
                          <div class="top5-ticker">{stock['ticker']} &nbsp;|&nbsp; {stock['price']}</div>
                        </div>
                        <div style="margin-left:auto; text-align:right;">
                          <div style="font-size:22px; font-weight:900;
                               color:{'#4ade80' if stock['composite']>=55 else '#fbbf24' if stock['composite']>=45 else '#f87171'};">
                            {stock['composite']:.0f}/100
                          </div>
                          <div style="font-size:11px; color:#6b7280;">Composite Score</div>
                        </div>
                      </div>
                    </div>
                    """, unsafe_allow_html=True)

                    col_a, col_b, col_c, col_d, col_e, col_f, col_g = st.columns(7)
                    col_a.metric("30d Mom",     f"{stock['mom_30']:+.1f}%")
                    col_b.metric("10d Mom",     f"{stock['mom_10']:+.1f}%")
                    col_c.metric("RSI",         f"{stock['rsi']:.0f}")
                    col_d.metric("Tech Score",  f"{stock['tech_score']:.0f}/100")
                    col_e.metric("Fund Score",  f"{stock['fund_score']:.0f}/100")
                    col_f.metric("Daily Vol",   f"{stock['hist_vol']:.2f}%")
                    col_g.metric("Vol Ratio",   f"{stock['vol_ratio']:.2f}x")

                    # Fundamental mini-row
                    fa, fb, fc, fd, fe = st.columns(5)
                    fa.metric("P/E Ratio",      fmtf(pe_v))
                    fb.metric("Revenue Growth", fmtf(rg_v, pct=True))
                    fc.metric("Earnings Growth",fmtf(eg_v, pct=True))
                    fd.metric("Debt/Equity",    fmtf(de_v))
                    fe.metric("Profit Margin",  fmtf(pm_v, pct=True))

                    st.divider()

            # ── Composite bar chart ───────────────────────────────────────────
            chart_data = pd.DataFrame({
                "Stock":          [f"{medals[i]} {s['name']}" for i, s in enumerate(top5)],
                "Composite Score":[s["composite"] for s in top5],
                "Tech Score":     [s["tech_score"] for s in top5],
                "Fund Score":     [s["fund_score"] for s in top5],
            })
            fig_top5 = go.Figure()
            fig_top5.add_trace(go.Bar(
                x=chart_data["Stock"], y=chart_data["Tech Score"],
                name="Technical", marker_color="#6366f1", opacity=0.85,
            ))
            fig_top5.add_trace(go.Bar(
                x=chart_data["Stock"], y=chart_data["Fund Score"],
                name="Fundamental", marker_color="#22c55e", opacity=0.85,
            ))
            fig_top5.update_layout(
                barmode="group",
                title="Top 5 — Technical vs Fundamental Score Breakdown",
                yaxis=dict(range=[0, 100], title="Score"),
                template="plotly_dark", height=360,
                margin=_MARGIN, **_DARK,
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_top5, use_container_width=True)

            st.markdown(
                "<p style='color:#374151; font-size:11px; text-align:center;'>"
                "⚠️ Scores are quantitative signals only. Not financial advice. "
                "Past performance does not guarantee future results.</p>",
                unsafe_allow_html=True,
            )
    else:
        st.markdown("""
        <div style="height:300px; display:flex; flex-direction:column;
                    align-items:center; justify-content:center;
                    color:#374151; font-size:14px; gap:8px;">
          <div style="font-size:56px;">🏆</div>
          <p>Select a market → click <strong>Find Top 5 ROI Picks</strong></p>
          <p style="font-size:12px; color:#1f2937;">
            Scores up to 20 stocks · Technical + Fundamental composite · P/E · Revenue · Margins
          </p>
        </div>
        """, unsafe_allow_html=True)

# ─── FLOATING AI FAB ──────────────────────────────────────────────────────────
st.markdown("""
<div class="quant-fab" title="Open sidebar for AI chat →">🤖</div>
<div class="quant-fab-label">← Open sidebar for AI chat</div>
""", unsafe_allow_html=True)

# ─── FOOTER ───────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center; color:#374151; font-size:11px; margin-top:40px; padding:20px;">
  QuantAI Elite · Log-return ensemble · Momentum persistence · AI validator · Fundamentals ·
  Top 5 ROI Picks · US NYSE/NASDAQ · India NSE/BSE · Precious Metals · Jewelry · Custom Tickers<br>
  ⚠️ For informational purposes only. Not financial advice.
</div>
""", unsafe_allow_html=True)
