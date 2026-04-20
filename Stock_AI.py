# ═══════════════════════════════════════════════════════════════════════════════
#              QuantAI Elite — World-Class Stock & Commodity Predictor
#   Ensemble ML: XGBoost + Gradient Boosting + RandomForest + Sentiment AI
#       US Markets • India NSE/BSE • Precious Metals • Jewelry Stocks
# ═══════════════════════════════════════════════════════════════════════════════

import streamlit as st
from huggingface_hub import InferenceClient
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
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
HF_TOKEN     = st.secrets["HF_TOKEN"]
SERP_API_KEY = st.secrets["SERP_API_KEY"]
_ai_client   = InferenceClient(token=HF_TOKEN)
_AI_MODEL    = "Qwen/Qwen2.5-7B-Instruct"

# ─── GLOBAL CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Base ── */
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

/* ── Sidebar ── */
.stSidebar { background: #080c18 !important; }
[data-testid="stSidebarContent"] { padding: 0 !important; }

/* ── Selectbox, buttons ── */
.stSelectbox > div > div { background: #111827 !important; border-color: #1f2937 !important; color: #e2e8f0 !important; }
.stButton > button[kind="primary"] {
    background: linear-gradient(135deg,#6366f1,#8b5cf6) !important;
    border: none !important; font-weight: 700 !important;
    letter-spacing: 0.5px;
    transition: all .25s ease;
    box-shadow: 0 4px 20px rgba(99,102,241,.4);
}
.stButton > button[kind="primary"]:hover {
    box-shadow: 0 6px 30px rgba(99,102,241,.7);
    transform: translateY(-1px);
}

/* ── Floating AI FAB ── */
.quant-fab {
    position: fixed; bottom: 26px; right: 26px;
    width: 64px; height: 64px; border-radius: 50%;
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    display: flex; align-items: center; justify-content: center;
    font-size: 30px; cursor: pointer;
    box-shadow: 0 0 0 0 rgba(99,102,241,.75);
    animation: fab-ring 2.4s infinite;
    z-index: 999999;
    border: 2px solid rgba(167,139,250,.4);
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

/* ── Model badge ── */
.model-badge {
    display: inline-flex; align-items: center; gap: 6px;
    background: #1e1b4b; color: #a5b4fc;
    border: 1px solid #4338ca; border-radius: 20px;
    padding: 4px 12px; font-size: 11px; font-weight: 700;
    letter-spacing: .5px;
}
/* ── Sentiment card ── */
.sent-card {
    background: #0f1623; border-radius: 10px;
    border-left: 4px solid #6366f1;
    padding: 10px 16px; margin: 6px 0;
    font-size: 13px;
}
/* ── Exchange badge ── */
.nse-badge { background:#14532d; color:#86efac; border-radius:6px; padding:2px 8px; font-size:11px; font-weight:700; }
.bse-badge { background:#1e1b4b; color:#a5b4fc; border-radius:6px; padding:2px 8px; font-size:11px; font-weight:700; }
.us-badge  { background:#1c1917; color:#fbbf24; border-radius:6px; padding:2px 8px; font-size:11px; font-weight:700; }

/* ── Header ── */
.quant-header { text-align:center; padding: 10px 0 4px; }
.quant-header h1 {
    font-size: 2.2rem; font-weight: 900;
    background: linear-gradient(135deg,#6366f1,#a78bfa,#38bdf8);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -1px; margin-bottom: 2px;
}
.quant-header p { color: #6b7280; font-size: 13px; margin: 0; }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div class="quant-header">
  <h1>📈 QuantAI Elite</h1>
  <p>Ensemble ML · Sentiment AI · US & India NSE/BSE · Precious Metals · Jewelry</p>
</div>
""", unsafe_allow_html=True)
st.divider()

# ═══════════════════════════════════════════════════════════════════════════════
#  STOCK LISTS
# ═══════════════════════════════════════════════════════════════════════════════

# ── US ────────────────────────────────────────────────────────────────────────
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
    "Caesars Entertainment": "CZR", "Vimeo": "VMEO",
    "Sweetgreen": "SG", "Petco": "WOOF",
}
us_small_cap = {
    "Genie Energy": "GNE", "Coda Octopus": "CODA",
    "Turtle Beach": "HEAR", "Intellicheck": "IDN",
    "Frequency Electronics": "FEIM", "Transcat": "TRNS",
    "ProQR Therapeutics": "PRQR", "Can-Fite BioPharma": "CANF",
}

# ── India (base tickers, suffix added dynamically) ────────────────────────────
india_large_base = {
    "Reliance Industries": "RELIANCE",
    "Tata Consultancy Services": "TCS",
    "Infosys": "INFY",
    "HDFC Bank": "HDFCBANK",
    "ICICI Bank": "ICICIBANK",
    "Bharti Airtel": "BHARTIARTL",
    "ITC Limited": "ITC",
    "Kotak Mahindra Bank": "KOTAKBANK",
    "Larsen & Toubro": "LT",
    "Hindustan Unilever": "HINDUNILVR",
    "Bajaj Finance": "BAJFINANCE",
    "Wipro": "WIPRO",
    "Axis Bank": "AXISBANK",
    "Maruti Suzuki": "MARUTI",
    "State Bank of India": "SBIN",
    "ONGC": "ONGC",
    "Asian Paints": "ASIANPAINT",
    "HCL Technologies": "HCLTECH",
    "Sun Pharmaceutical": "SUNPHARMA",
    "Titan Company": "TITAN",
    "UltraTech Cement": "ULTRACEMCO",
    "Power Grid Corp": "POWERGRID",
    "NTPC": "NTPC",
    "Tata Motors": "TATAMOTORS",
    "JSW Steel": "JSWSTEEL",
    "Tata Steel": "TATASTEEL",
    "Mahindra & Mahindra": "M&M",
    "Bajaj Auto": "BAJAJ-AUTO",
    "Tech Mahindra": "TECHM",
    "Adani Enterprises": "ADANIENT",
    "Adani Ports": "ADANIPORTS",
    "Cipla": "CIPLA",
    "Dr Reddy's Labs": "DRREDDY",
    "Eicher Motors": "EICHERMOT",
    "Nestle India": "NESTLEIND",
}
india_mid_base = {
    "Zomato": "ZOMATO",
    "Paytm": "PAYTM",
    "Nykaa": "NYKAA",
    "IRCTC": "IRCTC",
    "Muthoot Finance": "MUTHOOTFIN",
    "Delhivery": "DELHIVERY",
    "PNB": "PNB",
    "Bank of Baroda": "BANKBARODA",
    "Indian Hotels (Taj)": "INDHOTEL",
    "Apollo Hospitals": "APOLLOHOSP",
    "Tata Power": "TATAPOWER",
    "Godrej Properties": "GODREJPROP",
    "PI Industries": "PIIND",
    "Coforge": "COFORGE",
    "Dixon Technologies": "DIXON",
    "Polycab India": "POLYCAB",
    "Persistent Systems": "PERSISTENT",
    "Mphasis": "MPHASIS",
    "IRFC": "IRFC",
    "Hindustan Aeronautics (HAL)": "HAL",
    "Bharat Electronics": "BEL",
    "Indian Railway Finance": "IRFC",
    "Interglobe Aviation (IndiGo)": "INDIGO",
    "SBI Life Insurance": "SBILIFE",
    "HDFC Life Insurance": "HDFCLIFE",
}
india_small_base = {
    "Kalyan Jewellers": "KALYANKJIL",
    "Senco Gold": "SENCO",
    "PC Jeweller": "PCJEWELLER",
    "Thangamayil Jewellery": "THANGAMAYL",
    "Easy Trip Planners": "EASEMYTRIP",
    "Kaynes Technology": "KAYNES",
    "Syrma SGS": "SYRMA",
    "Happiest Minds": "HAPPSTMNDS",
    "Medplus Health": "MEDPLUS",
    "Global Health (Medanta)": "MEDANTA",
    "Campus Activewear": "CAMPUS",
    "Vedant Fashions (Manyavar)": "MANYAVAR",
    "Bikaji Foods": "BIKAJI",
    "Landmark Cars": "LANDMARK",
    "RBL Bank": "RBLBANK",
    "Ujjivan Small Finance": "UJJIVANSFB",
    "Suryoday Small Finance": "SURYODAY",
    "CMS Info Systems": "CMSINFO",
    "Sapphire Foods": "SAPPHIRE",
    "Restaurant Brands (Burger King)": "RBA",
}

def make_india_map(base: dict, exchange: str) -> dict:
    suffix = ".NS" if exchange == "NSE" else ".BO"
    return {name: ticker + suffix for name, ticker in base.items()}

# ── Commodities & Elements ────────────────────────────────────────────────────
commodities_futures = {
    "🥇 Gold Futures": "GC=F",
    "🥈 Silver Futures": "SI=F",
    "⬜ Platinum": "PL=F",
    "🔵 Palladium": "PA=F",
    "🛢️ Crude Oil (WTI)": "CL=F",
    "🔥 Natural Gas": "NG=F",
    "🔶 Copper": "HG=F",
    "🌽 Corn": "ZC=F",
    "🌾 Wheat": "ZW=F",
    "🫘 Soybeans": "ZS=F",
}
commodities_etf = {
    "Gold ETF (GLD)": "GLD",
    "Silver ETF (SLV)": "SLV",
    "Gold Miners (GDX)": "GDX",
    "Junior Gold Miners (GDXJ)": "GDXJ",
    "Silver Miners (SIL)": "SIL",
    "Commodities Index (PDBC)": "PDBC",
    "Gold ETF India (GOLDBEES)": "GOLDBEES.NS",
    "Silver ETF India (SILVERETF)": "SILVERETF.NS",
    "Nippon Gold ETF India": "GOLDETF.NS",
}
jewelry_stocks_map = {
    # Indian
    "Titan Company (NSE)": "TITAN.NS",
    "Kalyan Jewellers (NSE)": "KALYANKJIL.NS",
    "Senco Gold (NSE)": "SENCO.NS",
    "PC Jeweller (NSE)": "PCJEWELLER.NS",
    "Thangamayil Jewellery (NSE)": "THANGAMAYL.NS",
    "Renaissance Jewellery (NSE)": "RJL.NS",
    "Tribhovandas Bhimji Zaveri": "TBZ.NS",
    # International
    "Signet Jewelers (NYSE)": "SIG",
    "Pandora A/S (Copenhagen)": "PNDORA.CO",
    "Richemont (Cartier parent)": "CFR.SW",
    "LVMH (Tiffany parent)": "MC.PA",
    "Chow Tai Fook (HK)": "1929.HK",
    "Tapestry (Coach/Kate Spade)": "TPR",
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
def get_stock_info(ticker: str) -> dict:
    try:
        return yf.Ticker(ticker).info
    except Exception:
        return {}

def fetch_real_time(query: str) -> str:
    try:
        url = (f"https://serpapi.com/search.json"
               f"?q={requests.utils.quote(query)}&api_key={SERP_API_KEY}")
        r = requests.get(url, timeout=6).json()
        if "answer_box" in r and "snippet" in r["answer_box"]:
            return r["answer_box"]["snippet"]
        snips = [
            x.get("snippet", "")
            for x in r.get("organic_results", [])[:4]
            if x.get("snippet")
        ]
        return " | ".join(snips) or "No real-time data available."
    except Exception as e:
        return f"Search error: {e}"

@st.cache_data(ttl=1800, show_spinner=False)
def get_news_sentiment(query: str) -> float:
    """Fetch news & run FinBERT sentiment via HF Inference API."""
    try:
        url = (f"https://serpapi.com/search.json"
               f"?q={requests.utils.quote(query + ' stock market news')}"
               f"&tbm=nws&num=5&api_key={SERP_API_KEY}")
        r = requests.get(url, timeout=6).json()

        headlines = []
        for src in ("news_results", "organic_results"):
            for item in r.get(src, [])[:5]:
                text = item.get("title") or item.get("snippet", "")
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
        return float(
            np.clip(scores.get("positive", 0.33) - scores.get("negative", 0.33), -1, 1)
        )
    except Exception:
        return 0.0

# ═══════════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING  (40+ indicators)
# ═══════════════════════════════════════════════════════════════════════════════

def calc_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain  = delta.clip(lower=0).rolling(period).mean()
    loss  = (-delta.clip(upper=0)).rolling(period).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

_EXCLUDE_COLS = {
    "Open","High","Low","Close","Volume",
    "Dividends","Stock Splits","target","tr",
}

def get_feature_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in _EXCLUDE_COLS]

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df[["Open", "High", "Low", "Close", "Volume"]].copy().astype(float)
    c = d["Close"]

    # Returns & momentum
    d["ret_1"]   = c.pct_change()
    d["log_ret"] = np.log(c / c.shift(1).replace(0, np.nan))
    for lag in (2, 3, 5, 10, 15, 20):
        d[f"mom_{lag}"] = c.pct_change(lag)

    # Moving averages
    for w in (5, 10, 20, 50, 100):
        ma = c.rolling(w).mean()
        d[f"ma_{w}"] = ma
        d[f"ma_r_{w}"] = c / ma.replace(0, np.nan)

    for span in (9, 21, 50):
        ema = c.ewm(span=span).mean()
        d[f"ema_{span}"] = ema
        d[f"ema_r_{span}"] = c / ema.replace(0, np.nan)

    # MACD
    ema12, ema26 = c.ewm(span=12).mean(), c.ewm(span=26).mean()
    d["macd"] = ema12 - ema26
    d["macd_sig"] = d["macd"].ewm(span=9).mean()
    d["macd_hist"] = d["macd"] - d["macd_sig"]
    d["macd_r"] = d["macd"] / c.replace(0, np.nan)

    # Bollinger Bands
    bb_m = c.rolling(20).mean()
    bb_s = c.rolling(20).std()
    d["bb_upper"] = bb_m + 2 * bb_s
    d["bb_lower"] = bb_m - 2 * bb_s
    bb_width = d["bb_upper"] - d["bb_lower"]
    d["bb_width"] = bb_width / bb_m.replace(0, np.nan)
    d["bb_pos"] = (c - d["bb_lower"]) / bb_width.replace(0, np.nan)

    # RSI
    d["rsi_7"] = calc_rsi(c, 7)
    d["rsi_14"] = calc_rsi(c, 14)
    d["rsi_21"] = calc_rsi(c, 21)

    # Stochastic
    lo14 = c.rolling(14).min()
    hi14 = c.rolling(14).max()
    stoch_den = (hi14 - lo14).replace(0, np.nan)
    d["stoch_k"] = (c - lo14) / stoch_den
    d["stoch_d"] = d["stoch_k"].rolling(3).mean()

    # ATR
    hl = d["High"] - d["Low"]
    hc = (d["High"] - d["Close"].shift()).abs()
    lc = (d["Low"] - d["Close"].shift()).abs()
    d["atr"] = pd.concat([hl, hc, lc], axis=1).max(axis=1).rolling(14).mean()
    d["atr_r"] = d["atr"] / c.replace(0, np.nan)

    # OBV
    sign = np.sign(c.diff().fillna(0))
    d["obv"] = (sign * d["Volume"]).cumsum()
    d["obv_ma"] = d["obv"].rolling(10).mean()
    d["obv_r"] = d["obv"] / d["obv_ma"].abs().replace(0, np.nan)

    # Volume
    vol = d["Volume"]
    d["vol_ma20"] = vol.rolling(20).mean()
    d["vol_r"] = vol / d["vol_ma20"].replace(0, np.nan)
    d["vol_std"] = vol.rolling(20).std()
    d["vol_ret"] = vol.pct_change()

    # Price structure
    d["hl_r"] = (d["High"] - d["Low"]) / c.replace(0, np.nan)
    d["co_r"] = (c - d["Open"]) / d["Open"].replace(0, np.nan)
    d["gap"] = (d["Open"] - d["Close"].shift()) / d["Close"].shift().replace(0, np.nan)

    # Volatility regimes
    d["vol5"] = d["ret_1"].rolling(5).std()
    d["vol20"] = d["ret_1"].rolling(20).std()
    d["vol_regime"] = d["vol5"] / d["vol20"].replace(0, np.nan)

    # Z-score
    roll_m = c.rolling(20).mean()
    roll_s = c.rolling(20).std()
    d["zscore_20"] = (c - roll_m) / roll_s.replace(0, np.nan)

    # Calendar features
    idx = d.index
    if isinstance(idx, pd.DatetimeIndex):
        d["dow"] = idx.dayofweek
        d["dom"] = idx.day
        d["month"] = idx.month
        d["quarter"] = idx.quarter
        d["is_mon"] = (idx.dayofweek == 0).astype(int)
        d["is_fri"] = (idx.dayofweek == 4).astype(int)

    d = d.replace([np.inf, -np.inf], np.nan)
    d = d.dropna()

    return d

# ═══════════════════════════════════════════════════════════════════════════════
#  ENSEMBLE PREDICTION ENGINE
# ═══════════════════════════════════════════════════════════════════════════════

def train_ensemble(feat_df: pd.DataFrame, feature_cols: list):
    df = feat_df.copy()
    df["target"] = df["Close"].shift(-1)

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=feature_cols + ["target"]).copy()

    if len(df) < 45:
        return None

    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["target"].to_numpy(dtype=np.float64)

    finite_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
    df = df.loc[finite_mask].copy()
    X = df[feature_cols].to_numpy(dtype=np.float64)
    y = df["target"].to_numpy(dtype=np.float64)

    X = np.clip(X, -1e12, 1e12)

    scaler_X = StandardScaler()
    scaler_y = MinMaxScaler()
    X_sc = scaler_X.fit_transform(X)
    y_sc = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

    split = max(int(len(X) * 0.8), 35)
    X_tr, X_val = X_sc[:split], X_sc[split:]
    y_tr, y_val = y_sc[:split], y_sc[split:]

    if len(X_val) == 0:
        return None

    # ── XGBoost ─────────────────────────────────────────────────────────────
    xgb_m = XGBRegressor(
        n_estimators=350, max_depth=5, learning_rate=0.04,
        subsample=0.8, colsample_bytree=0.75,
        min_child_weight=3, reg_alpha=0.1, reg_lambda=1.5,
        gamma=0.05, random_state=42, verbosity=0, n_jobs=-1,
    )
    try:
        xgb_m.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False,
            early_stopping_rounds=30,
        )
    except TypeError:
        xgb_m.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)

    # ── GradientBoosting ────────────────────────────────────────────────────
    gbr_m = GradientBoostingRegressor(
        n_estimators=200, max_depth=4, learning_rate=0.06,
        subsample=0.8, min_samples_leaf=4, max_features="sqrt",
        random_state=42,
    )
    gbr_m.fit(X_tr, y_tr)

    # ── RandomForest ────────────────────────────────────────────────────────
    rf_m = RandomForestRegressor(
        n_estimators=150, max_depth=8, min_samples_leaf=3,
        max_features="sqrt", random_state=42, n_jobs=-1,
    )
    rf_m.fit(X_tr, y_tr)

    # ── Inverse-RMSE weighting ───────────────────────────────────────────────
    def rmse_real(model):
        p_sc = model.predict(X_val)
        p    = scaler_y.inverse_transform(p_sc.reshape(-1, 1)).flatten()
        y    = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()
        return float(np.sqrt(np.mean((p - y) ** 2))) + 1e-8

    rmses = {"xgb": rmse_real(xgb_m), "gbr": rmse_real(gbr_m), "rf": rmse_real(rf_m)}
    inv_total = sum(1 / v for v in rmses.values())
    w = {k: (1 / v) / inv_total for k, v in rmses.items()}

    # ── Ensemble val metrics ─────────────────────────────────────────────────
    ens_sc   = (w["xgb"] * xgb_m.predict(X_val)
                + w["gbr"] * gbr_m.predict(X_val)
                + w["rf"]  * rf_m.predict(X_val))
    val_pred = scaler_y.inverse_transform(ens_sc.reshape(-1, 1)).flatten()
    y_real   = scaler_y.inverse_transform(y_val.reshape(-1, 1)).flatten()

    val_mape = float(np.mean(np.abs(val_pred - y_real) / (np.abs(y_real) + 1e-10)))
    res_std  = float(np.std(val_pred - y_real))

    return {
        "xgb": xgb_m, "gbr": gbr_m, "rf": rf_m,
        "w": w,
        "scaler_X": scaler_X, "scaler_y": scaler_y,
        "feature_cols": feature_cols,
        "val_mape": val_mape,
        "residual_std": res_std,
        "val_preds": val_pred,
        "y_val_real": y_real,
        "rmses": rmses,
    }


def predict_future(
    df_orig: pd.DataFrame,
    days: int,
    m: dict,
    sentiment: float = 0.0,
    skip_weekends: bool = True,
):
    """
    Recursive multi-step forecasting.
    At each step the full indicator suite is re-calculated from the
    rolling window so that technical signals stay accurate.
    Returns (predictions, lower_CI, upper_CI).
    """
    df = df_orig[["Open", "High", "Low", "Close", "Volume"]].copy().astype(float)
    feature_cols = m["feature_cols"]
    scaler_X     = m["scaler_X"]
    scaler_y     = m["scaler_y"]
    res_std      = m["residual_std"]

    preds, lo_ci, hi_ci = [], [], []

    for step in range(days):
        feat_df = engineer_features(df.tail(120))
        feat_df = feat_df.replace([np.inf, -np.inf], np.nan).dropna()

        # ── Fill any missing feature columns ────────────────────────────────
        for col in feature_cols:
            if col not in feat_df.columns:
                feat_df[col] = 0.0

        # ── Guard against empty frames ───────────────────────────────────────
        if feat_df.empty or len(feat_df) < 5:
            last = preds[-1] if preds else float(df["Close"].iloc[-1])
            preds.append(last)
            lo_ci.append(last * 0.97)
            hi_ci.append(last * 1.03)
        else:
            X_last = feat_df[feature_cols].iloc[-1:].to_numpy(dtype=np.float64)
            X_last = np.clip(X_last, -1e12, 1e12)
            X_sc = scaler_X.transform(X_last)
            try:
                X_sc = scaler_X.transform(X_last)
            except Exception:
                last = preds[-1] if preds else float(df["Close"].iloc[-1])
                preds.append(last)
                lo_ci.append(last * 0.97)
                hi_ci.append(last * 1.03)
                continue

            # Weighted ensemble prediction
            p_sc = (
                m["w"]["xgb"] * m["xgb"].predict(X_sc)[0]
                + m["w"]["gbr"] * m["gbr"].predict(X_sc)[0]
                + m["w"]["rf"]  * m["rf"].predict(X_sc)[0]
            )

            # Sentiment nudge (capped at ±0.3 % per step)
            sent_adj = float(np.clip(sentiment * 0.003, -0.003, 0.003))
            p_sc_adj = float(p_sc) * (1.0 + sent_adj)

            pred = float(scaler_y.inverse_transform([[p_sc_adj]])[0][0])

            # CI expands as sqrt(step) to model compounding uncertainty
            sigma = res_std * np.sqrt(step + 1)
            lo    = pred - 1.96 * sigma
            hi    = pred + 1.96 * sigma

            preds.append(pred)
            lo_ci.append(lo)
            hi_ci.append(hi)

        # ── Append synthetic next row for rolling re-calculation ─────────────
        last_idx = df.index[-1]
        next_d   = pd.Timestamp(last_idx) + timedelta(days=1)
        if skip_weekends:
            while next_d.weekday() >= 5:
                next_d += timedelta(days=1)

        daily_range = float(df["High"].tail(20).mean() - df["Low"].tail(20).mean())
        pred_close  = preds[-1]
        new_row = pd.DataFrame(
            {
                "Open":   [pred_close * 0.9998],
                "High":   [pred_close + daily_range * 0.5],
                "Low":    [pred_close - daily_range * 0.5],
                "Close":  [pred_close],
                "Volume": [float(df["Volume"].median())],
            },
            index=[next_d],
        )
        df = pd.concat([df, new_row])

    return preds, lo_ci, hi_ci


def future_trading_dates(last_date, days: int, skip_weekends: bool = True) -> list:
    dates = []
    d = pd.Timestamp(last_date)
    while len(dates) < days:
        d += timedelta(days=1)
        if skip_weekends and d.weekday() >= 5:
            continue
        dates.append(d)
    return dates

# ═══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════

_DARK = {"paper_bgcolor": "#070b14", "plot_bgcolor": "#0f1623"}
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

    # ── Bollinger Bands (subtle fill) ────────────────────────────────────────
    bb_m = df["Close"].rolling(20).mean()
    bb_s = df["Close"].rolling(20).std()
    fig.add_trace(go.Scatter(
        x=df.index, y=bb_m + 2*bb_s, line=dict(color="rgba(99,102,241,.3)", width=.8),
        name="BB Upper", hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=df.index, y=bb_m - 2*bb_s, line=dict(color="rgba(99,102,241,.3)", width=.8),
        fill="tonexty", fillcolor="rgba(99,102,241,.05)",
        name="BB Lower", hoverinfo="skip",
    ))

    # ── Moving averages ───────────────────────────────────────────────────────
    for w, col in ((20, "#facc15"), (50, "#fb923c")):
        if len(df) >= w:
            fig.add_trace(go.Scatter(
                x=df.index, y=df["Close"].rolling(w).mean(), mode="lines",
                name=f"MA {w}", line=dict(color=col, width=1, dash="dot"), opacity=0.9,
            ))

    # ── Confidence band (95 %) ────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=future_dates + future_dates[::-1],
        y=hi_ci + lo_ci[::-1],
        fill="toself", fillcolor="rgba(167,139,250,.12)",
        line=dict(color="rgba(0,0,0,0)"), name="95% CI", hoverinfo="skip",
    ))

    # ── Forecast line ─────────────────────────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=future_dates, y=preds, mode="lines+markers",
        name="Forecast", line=dict(color="#a78bfa", dash="dash", width=2.5),
        marker=dict(size=7, color="#7c3aed", line=dict(color="#a78bfa", width=1.5)),
    ))

    fig.update_layout(
        title=dict(text=f"{company}  <span style='font-size:13px;color:#6b7280'>({ticker})</span>",
                   font=dict(size=16, color="#e2e8f0")),
        xaxis_title="Date", yaxis_title="Price",
        xaxis_rangeslider_visible=False, height=430,
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    font=dict(size=11)),
        template="plotly_dark", margin=_MARGIN, **_DARK,
    )
    return fig


def build_rsi_chart(df):
    rsi = calc_rsi(df["Close"])
    fig = go.Figure()
    fig.add_hrect(y0=70, y1=100, fillcolor="rgba(239,68,68,.07)", line_width=0)
    fig.add_hrect(y0=0,  y1=30,  fillcolor="rgba(34,197,94,.07)",  line_width=0)
    fig.add_trace(go.Scatter(
        x=df.index, y=rsi, mode="lines", name="RSI (14)",
        line=dict(color="#c084fc", width=1.5),
    ))
    fig.add_hline(y=70, line_color="#ef4444", line_dash="dash",
                  annotation_text="70", annotation_position="right")
    fig.add_hline(y=30, line_color="#22c55e", line_dash="dash",
                  annotation_text="30", annotation_position="right")
    fig.update_layout(
        title="RSI (14)", height=200, yaxis=dict(range=[0, 100]),
        template="plotly_dark", margin=_MARGIN, **_DARK,
    )
    return fig


def build_macd_chart(df):
    ema12 = df["Close"].ewm(span=12).mean()
    ema26 = df["Close"].ewm(span=26).mean()
    macd  = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist   = macd - signal
    colors = ["#22c55e" if v >= 0 else "#ef4444" for v in hist]

    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=hist, name="Histogram",
                          marker_color=colors, opacity=0.65))
    fig.add_trace(go.Scatter(x=df.index, y=macd,   mode="lines", name="MACD",
                              line=dict(color="#60a5fa", width=1.5)))
    fig.add_trace(go.Scatter(x=df.index, y=signal, mode="lines", name="Signal",
                              line=dict(color="#fb923c", width=1.5)))
    fig.update_layout(
        title="MACD", height=210,
        template="plotly_dark", margin=_MARGIN, **_DARK,
    )
    return fig


def build_volume_chart(df):
    colors = [
        "#22c55e" if df["Close"].iloc[i] >= df["Open"].iloc[i] else "#ef4444"
        for i in range(len(df))
    ]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=df.index, y=df["Volume"], name="Volume",
                          marker_color=colors, opacity=0.7))
    vol_ma = df["Volume"].rolling(20).mean()
    fig.add_trace(go.Scatter(x=df.index, y=vol_ma, mode="lines", name="Vol MA 20",
                              line=dict(color="#facc15", width=1.2, dash="dot")))
    fig.update_layout(
        title="Volume", height=180,
        template="plotly_dark", margin=_MARGIN, **_DARK,
    )
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
      <p style="color:#6b7280; font-size:11px; margin:0;">
        Real-time market intelligence
      </p>
    </div>
    <hr style="border-color:#1f2937; margin:10px 0;">
    """, unsafe_allow_html=True)

    if "chat" not in st.session_state:
        st.session_state.chat = []

    # Render history
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
                res = _ai_client.chat.completions.create(
                    model=_AI_MODEL,
                    messages=messages,
                    max_tokens=700,
                    temperature=0.6,
                )
                reply = res.choices[0].message.content
            st.write(reply)

        st.session_state.chat.append({"role": "assistant", "content": reply})

    if st.session_state.chat:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.chat = []
            st.rerun()

# ═══════════════════════════════════════════════════════════════════════════════
#  SHARED PREDICTION UI  (reused by Tab1 and Tab3)
# ═══════════════════════════════════════════════════════════════════════════════

def run_prediction_ui(ticker, company, period, days, chart_type, use_sentiment,
                       skip_weekends=True, price_label="Price"):
    """Full end-to-end prediction pipeline rendered inside a st.column."""
    sym = currency_symbol(ticker)

    with st.spinner(f"📡 Fetching {company} ({ticker})…"):
        df = load_stock(ticker, period)

    if df.empty:
        st.error(f"⚠️ No data for **{ticker}**. "
                 "Check the ticker or try a different period.")
        return

    # Sentiment
    sentiment = 0.0
    if use_sentiment:
        clean_name = company.replace("🥇","").replace("🥈","").replace("⬜","") \
                             .replace("🔵","").replace("🛢️","").replace("🔥","") \
                             .replace("🔶","").replace("🌽","").replace("🌾","") \
                             .replace("🫘","").strip()
        with st.spinner("🧠 Running FinBERT sentiment analysis…"):
            sentiment = get_news_sentiment(clean_name)

    # Feature engineering
    with st.spinner("⚙️ Engineering 40+ technical indicators…"):
        feat_df = engineer_features(df.copy())

    if len(feat_df) < 45:
        st.warning("⚠️ Need more data for robust ML. Switch to **1y** or **2y** period.")
        return

    feature_cols = get_feature_cols(feat_df)

    # Train ensemble
    with st.spinner("🤖 Training XGBoost + GBR + RandomForest ensemble…"):
        mdl = train_ensemble(feat_df, feature_cols)

    if not mdl:
        st.error("Model training failed — likely insufficient history.")
        return

    # Forecast
    with st.spinner(f"🔮 Forecasting {days} steps ahead…"):
        preds, lo_ci, hi_ci = predict_future(
            df.copy(), days, mdl, sentiment, skip_weekends
        )

    future_dates = future_trading_dates(df.index[-1], days, skip_weekends)

    # ── Metrics ──────────────────────────────────────────────────────────────
    current   = float(df["Close"].iloc[-1])
    pred_last = preds[-1]
    pred_pct  = (pred_last - current) / current * 100
    prev      = float(df["Close"].iloc[-2]) if len(df) > 1 else current
    day_pct   = (current - prev) / prev * 100
    wk        = float(df["Close"].iloc[max(-7, -len(df))])
    wk_pct    = (current - wk) / wk * 100
    accuracy  = max(0.0, (1 - mdl["val_mape"]) * 100)
    sent_lbl  = "🟢 Bullish" if sentiment > 0.1 else "🔴 Bearish" if sentiment < -0.1 else "⚪ Neutral"

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(price_label,            f"{sym}{current:,.2f}",   f"{day_pct:+.2f}%")
    c2.metric(f"+{days}d Forecast",   f"{sym}{pred_last:,.2f}", f"{pred_pct:+.2f}%")
    c3.metric("7-Day Δ",              f"{wk_pct:+.2f}%")
    c4.metric("Model Accuracy",       f"{accuracy:.1f}%")
    c5.metric("Sentiment",            sent_lbl,                 f"{sentiment*100:+.1f}%")

    st.markdown(
        f'<div class="model-badge">⚡ XGBoost&nbsp;{mdl["w"]["xgb"]*100:.0f}% · '
        f'GBR&nbsp;{mdl["w"]["gbr"]*100:.0f}% · '
        f'RF&nbsp;{mdl["w"]["rf"]*100:.0f}%</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    # ── Price chart ──────────────────────────────────────────────────────────
    st.plotly_chart(
        build_price_chart(df, preds, future_dates, lo_ci, hi_ci,
                           company, ticker, chart_type),
        use_container_width=True,
    )

    # ── Indicators ───────────────────────────────────────────────────────────
    ic1, ic2 = st.columns(2)
    with ic1:
        st.plotly_chart(build_rsi_chart(df), use_container_width=True)
    with ic2:
        st.plotly_chart(build_macd_chart(df), use_container_width=True)

    st.plotly_chart(build_volume_chart(df), use_container_width=True)

    # ── Model validation chart ───────────────────────────────────────────────
    with st.expander("🧪 Model Validation & Ensemble Weights"):
        n_val    = len(mdl["val_preds"])
        val_idx  = df.index[-n_val:] if n_val <= len(df) else df.index
        vfig = go.Figure()
        vfig.add_trace(go.Scatter(x=val_idx, y=mdl["y_val_real"],
                                   mode="lines", name="Actual",
                                   line=dict(color="#60a5fa", width=2)))
        vfig.add_trace(go.Scatter(x=val_idx, y=mdl["val_preds"],
                                   mode="lines", name="Ensemble Predicted",
                                   line=dict(color="#a78bfa", width=2, dash="dash")))
        vfig.update_layout(title="Validation — Actual vs Predicted",
                            template="plotly_dark", height=260,
                            margin=_MARGIN, **_DARK)
        st.plotly_chart(vfig, use_container_width=True)

        rc1, rc2, rc3 = st.columns(3)
        rc1.metric("XGBoost RMSE", f"{mdl['rmses']['xgb']:.4f}",
                   f"Weight {mdl['w']['xgb']*100:.1f}%")
        rc2.metric("GBR RMSE",     f"{mdl['rmses']['gbr']:.4f}",
                   f"Weight {mdl['w']['gbr']*100:.1f}%")
        rc3.metric("RF RMSE",      f"{mdl['rmses']['rf']:.4f}",
                   f"Weight {mdl['w']['rf']*100:.1f}%")

    # ── Forecast table ───────────────────────────────────────────────────────
    with st.expander("📋 Full Forecast Table"):
        fc_df = pd.DataFrame({
            "Date":              [d.strftime("%Y-%m-%d") for d in future_dates],
            f"Predicted ({sym})": [f"{sym}{p:,.2f}" for p in preds],
            f"Lower CI ({sym})":  [f"{sym}{l:,.2f}" for l in lo_ci],
            f"Upper CI ({sym})":  [f"{sym}{h:,.2f}" for h in hi_ci],
            "Δ% vs Today":       [f"{((p-current)/current)*100:+.2f}%" for p in preds],
        })
        st.dataframe(fc_df, use_container_width=True, hide_index=True)

    # ── Last 10 days raw ─────────────────────────────────────────────────────
    with st.expander("📈 Last 10 Trading Days"):
        tail = df[["Open","High","Low","Close","Volume"]].tail(10).round(3)
        st.dataframe(tail, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab1, tab2, tab3 = st.tabs([
    "📈 Stock Predictor",
    "📊 Market Overview",
    "💎 Elements & Jewelry",
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
            cap_cat = st.selectbox(
                "📂 Cap Category",
                ["🏛️ Large Cap", "🏢 Mid Cap", "🏬 Small Cap"],
            )
            stocks_map = (
                us_large_cap if "Large" in cap_cat
                else us_mid_cap if "Mid" in cap_cat
                else us_small_cap
            )
            exchange_badge = '<span class="us-badge">NYSE/NASDAQ</span>'
            skip_wk = True
        else:
            exch = "NSE" if "NSE" in market else "BSE"
            cap_cat = st.selectbox(
                "📂 Cap Category",
                ["🏛️ Large Cap", "🏢 Mid Cap", "🏬 Small Cap"],
            )
            base = (
                india_large_base if "Large" in cap_cat
                else india_mid_base if "Mid" in cap_cat
                else india_small_base
            )
            stocks_map = make_india_map(base, exch)
            badge_cls  = "nse-badge" if exch == "NSE" else "bse-badge"
            exchange_badge = f'<span class="{badge_cls}">{exch}</span>'
            skip_wk = True

        company   = st.selectbox("🏷️ Company", list(stocks_map.keys()))
        ticker    = stocks_map[company]
        st.caption(f"Ticker: **{ticker}**")
        st.markdown(exchange_badge, unsafe_allow_html=True)

        period      = st.selectbox("📅 Period", ["3mo","6mo","1y","2y","5y"], index=2)
        days        = st.slider("🔮 Forecast Days", 1, 30, 7)
        chart_type  = st.radio("📉 Chart Style", ["Candlestick","Line"], horizontal=True)
        use_sent    = st.toggle("🧠 Sentiment (FinBERT)", value=True)

        run_btn = st.button("🚀 Run Prediction", use_container_width=True, type="primary")

    with chart_col:
        if run_btn:
            run_prediction_ui(
                ticker, company, period, days,
                chart_type, use_sent, skip_wk,
            )
        else:
            st.markdown("""
            <div style="height:400px; display:flex; flex-direction:column;
                        align-items:center; justify-content:center;
                        color:#374151; font-size:14px; gap:8px;">
              <div style="font-size:56px;">📈</div>
              <p>Configure settings → click <strong>Run Prediction</strong></p>
              <p style="font-size:12px; color:#1f2937;">
                XGBoost · GBR · RandomForest · FinBERT sentiment
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
        "US Large Cap":     us_large_cap,
        "US Mid Cap":       us_mid_cap,
        "India NSE Large":  make_india_map(india_large_base, "NSE"),
        "India NSE Mid":    make_india_map(india_mid_base,   "NSE"),
        "India BSE Large":  make_india_map(india_large_base, "BSE"),
        "India BSE Mid":    make_india_map(india_mid_base,   "BSE"),
    }
    ov_map   = ov_map_lookup[ov_market]
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
                        "Company":   name,
                        "Ticker":    tkr,
                        "Price":     f"{sym}{cur:,.2f}",
                        "Day Δ%":   round(chg, 2),
                        "5d High":   f"{sym}{hi5:,.2f}",
                        "5d Low":    f"{sym}{lo5:,.2f}",
                        "Volume":    f"{vol:,}",
                    })
            except Exception:
                pass
            prog.progress((i + 1) / max(len(ov_picked), 1))
        prog.empty()

        if rows:
            ov_df = pd.DataFrame(rows)
            st.dataframe(ov_df, use_container_width=True, hide_index=True)

            bar = px.bar(
                ov_df, x="Company", y="Day Δ%", color="Day Δ%",
                color_continuous_scale=["#ef4444","#374151","#22c55e"],
                template="plotly_dark",
                title=f"{ov_market} — Daily % Change",
            )
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

    if "Futures" in el_type:
        el_map       = commodities_futures
        price_label  = "Futures Price"
        skip_wk_el   = True
    elif "ETF" in el_type:
        el_map       = commodities_etf
        price_label  = "ETF Price"
        skip_wk_el   = True
    else:
        el_map       = jewelry_stocks_map
        price_label  = "Stock Price"
        skip_wk_el   = True

    el_ctrl, el_chart = st.columns([1, 2.6])

    with el_ctrl:
        el_company  = st.selectbox("Select Asset", list(el_map.keys()))
        el_ticker   = el_map[el_company]
        st.caption(f"Ticker: **{el_ticker}**")

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
                el_chart_type, el_sent, skip_wk_el,
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
                Gold · Silver · Platinum · Jewelry stocks
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
  QuantAI Elite · Ensemble ML: XGBoost + GBR + RF · FinBERT Sentiment ·
  US NYSE/NASDAQ · India NSE/BSE · Precious Metals · Jewelry Stocks<br>
  ⚠️ For informational purposes only. Not financial advice.
</div>
""", unsafe_allow_html=True)
