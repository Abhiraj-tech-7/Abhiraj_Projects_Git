import streamlit as st
from huggingface_hub import InferenceClient
import requests
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta

st.set_page_config("Stock AI Predictor", layout="wide")
st.markdown("<h2 style='text-align:center;'>📈 Stock Price Predictor + AI Assistant</h2>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center; color:gray;'>Powered by real-time market data & AI</p>", unsafe_allow_html=True)

HF_TOKEN="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"
SERP_API_KEY="6ff2e871beaffd46ffca7bf6c4007814e1fa27937b662e1408e2fd04d9937b0f"
model_name="Qwen/Qwen2.5-7B-Instruct"
client=InferenceClient(model=model_name, token=HF_TOKEN)

large_cap={
    "Apple":"AAPL",
    "Microsoft":"MSFT",
    "NVIDIA":"NVDA",
    "Amazon":"AMZN",
    "Alphabet (Google)":"GOOGL",
    "Meta":"META",
    "Tesla":"TSLA",
    "Berkshire Hathaway":"BRK-B",
    "JPMorgan Chase":"JPM",
    "Visa":"V",
    "Johnson & Johnson":"JNJ",
    "Walmart":"WMT",
    "Exxon Mobil":"XOM",
    "UnitedHealth":"UNH",
    "Procter & Gamble":"PG"
}

mid_cap={
    "Lyft":"LYFT",
    "Robinhood":"HOOD",
    "Chewy":"CHWY",
    "Duolingo":"DUOL",
    "Zillow":"Z",
    "Wingstop":"WING",
    "Five Below":"FIVE",
    "Crocs":"CROX",
    "Hasbro":"HAS",
    "Caesars Entertainment":"CZR",
    "Vimeo":"VMEO",
    "Petco":"WOOF",
    "Sweetgreen":"SG",
    "Clearfield":"CLFD",
    "Envista Holdings":"NVST"
}

small_cap={
    "Genie Energy":"GNE",
    "Coda Octopus":"CODA",
    "Ondas Holdings":"ONDS",
    "Turtle Beach":"HEAR",
    "Intellicheck":"IDN",
    "Frequency Electronics":"FEIM",
    "ProQR Therapeutics":"PRQR",
    "Transcat":"TRNS",
    "Primavera Capital":"PV",
    "Expion360":"XPON",
    "Safe Harbor Financial":"SHFS",
    "Greenland Acquisition":"GRNV",
    "FG Financial":"FGF",
    "BRT Analytics":"BRTA",
    "Can-Fite BioPharma":"CANF"
}

def fetch_real_time(query):
    if not query:
        return "No query provided."
    try:
        url=f"https://serpapi.com/search.json?q={query}&api_key={SERP_API_KEY}"
        r=requests.get(url).json()
        if "answer_box" in r and "snippet" in r["answer_box"]:
            return r["answer_box"]["snippet"]
        elif "organic_results" in r:
            snips=[x["snippet"] for x in r["organic_results"][:4] if "snippet" in x]
            return " | ".join(snips) if snips else "No relevant data found."
        return "No real-time data available."
    except Exception as e:
        return f"Error fetching data: {e}"

def load_stock(ticker, period):
    try:
        stock=yf.Ticker(ticker)
        df=stock.history(period=period)
        return df
    except:
        return pd.DataFrame()

def get_stock_info(ticker):
    try:
        stock=yf.Ticker(ticker)
        info=stock.info
        return info
    except:
        return {}

def run_prediction(df, days):
    df=df.copy().reset_index()
    df["idx"]=np.arange(len(df))
    X=df[["idx"]].values
    y=df["Close"].values
    scaler=MinMaxScaler()
    y_scaled=scaler.fit_transform(y.reshape(-1,1)).flatten()
    reg=LinearRegression()
    reg.fit(X, y_scaled)
    future_idx=np.arange(len(df), len(df)+days).reshape(-1,1)
    preds_scaled=reg.predict(future_idx)
    preds=scaler.inverse_transform(preds_scaled.reshape(-1,1)).flatten()
    if "Date" in df.columns:
        last_date=pd.to_datetime(df["Date"].iloc[-1])
    else:
        last_date=datetime.now()
    future_dates=[last_date+timedelta(days=i+1) for i in range(days)]
    return preds, future_dates

def calc_rsi(series, period=14):
    delta=series.diff()
    gain=delta.clip(lower=0)
    loss=-delta.clip(upper=0)
    avg_gain=gain.rolling(period).mean()
    avg_loss=loss.rolling(period).mean()
    rs=avg_gain/avg_loss
    return 100-(100/(1+rs))

t1,t2,t3=st.tabs(["📈 Stock Predictor","📊 Market Overview","🤖 AI Assistant"])

with t1:
    st.markdown("<h4 style='text-align:center;'>Stock Price & Percentage Predictor</h4>", unsafe_allow_html=True)
    
    ctrl,chart=st.columns([1,2.5])
    
    with ctrl:
        st.markdown("#### ⚙️ Controls")
        cap_cat=st.selectbox("📂 Market Cap", ["🏛️ Large Cap","🏢 Mid Cap","🏬 Small Cap"])
        
        if "Large" in cap_cat:
            stocks_map=large_cap
            cap_label="Large Cap"
        elif "Mid" in cap_cat:
            stocks_map=mid_cap
            cap_label="Mid Cap"
        else:
            stocks_map=small_cap
            cap_label="Small Cap"
        
        company=st.selectbox("🏷️ Company", list(stocks_map.keys()))
        ticker=stocks_map[company]
        st.caption(f"Ticker: **{ticker}** · {cap_label}")
        
        period=st.selectbox("📅 Historical Range", ["1mo","3mo","6mo","1y","2y"], index=2)
        days=st.slider("🔮 Forecast Days", 1, 30, 7)
        chart_type=st.radio("📉 Chart Type", ["Candlestick","Line"], horizontal=True)
        
        run=st.button("🔍 Run Prediction", use_container_width=True, type="primary")
    
    with chart:
        if run:
            with st.spinner(f"Loading {company} ({ticker}) data..."):
                df=load_stock(ticker, period)
                info=get_stock_info(ticker)
            
            if df.empty:
                st.error("⚠️ Could not fetch stock data. Try another ticker.")
            else:
                current=float(df["Close"].iloc[-1])
                prev=float(df["Close"].iloc[-2])
                delta_val=current-prev
                delta_pct=(delta_val/prev)*100
                
                preds,future_dates=run_prediction(df, days)
                pred_final=float(preds[-1])
                pred_pct=((pred_final-current)/current)*100
                
                week_ago=float(df["Close"].iloc[-min(7, len(df)-1)])
                week_pct=((current-week_ago)/week_ago)*100
                
                m1,m2,m3,m4=st.columns(4)
                m1.metric("Current Price", f"${current:.2f}", f"{delta_pct:+.2f}% today")
                m2.metric(f"Forecast +{days}d", f"${pred_final:.2f}", f"{pred_pct:+.2f}%")
                m3.metric("7-Day Change", f"{week_pct:+.2f}%")
                m4.metric("Volume", f"{int(df['Volume'].iloc[-1]):,}")
                
                st.divider()
                
                fig=go.Figure()
                
                if chart_type=="Candlestick":
                    fig.add_trace(go.Candlestick(
                        x=df.index,
                        open=df["Open"],
                        high=df["High"],
                        low=df["Low"],
                        close=df["Close"],
                        name="Price",
                        increasing_line_color="#00ff88",
                        decreasing_line_color="#ff4444"
                    ))
                else:
                    fig.add_trace(go.Scatter(
                        x=df.index,
                        y=df["Close"],
                        mode="lines",
                        name="Close Price",
                        line=dict(color="#4da6ff", width=2)
                    ))
                
                ma20=df["Close"].rolling(20).mean()
                fig.add_trace(go.Scatter(x=df.index, y=ma20, mode="lines", name="MA 20", line=dict(color="yellow", width=1, dash="dot")))
                
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=preds,
                    mode="lines+markers",
                    name=f"Forecast ({days}d)",
                    line=dict(color="orange", dash="dash", width=2),
                    marker=dict(size=5)
                ))
                
                fig.update_layout(
                    title=f"{company} ({ticker}) — {cap_label}",
                    xaxis_title="Date",
                    yaxis_title="Price (USD)",
                    template="plotly_dark",
                    xaxis_rangeslider_visible=False,
                    height=420,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02)
                )
                st.plotly_chart(fig, use_container_width=True)
                
                rsi_series=calc_rsi(df["Close"])
                rsi_fig=go.Figure()
                rsi_fig.add_trace(go.Scatter(x=df.index, y=rsi_series, mode="lines", name="RSI", line=dict(color="#b388ff", width=1.5)))
                rsi_fig.add_hline(y=70, line_color="red", line_dash="dash", annotation_text="Overbought")
                rsi_fig.add_hline(y=30, line_color="green", line_dash="dash", annotation_text="Oversold")
                rsi_fig.update_layout(title="RSI (14)", template="plotly_dark", height=200, yaxis=dict(range=[0,100]))
                st.plotly_chart(rsi_fig, use_container_width=True)
                
                with st.expander("📋 Last 10 Trading Days"):
                    tail_df=df[["Open","High","Low","Close","Volume"]].tail(10).round(2)
                    st.dataframe(tail_df, use_container_width=True)
        else:
            st.info("👈 Configure your settings and click **Run Prediction** to start.")

with t2:
    st.markdown("<h4 style='text-align:center;'>📊 Market Overview by Category</h4>", unsafe_allow_html=True)
    
    selected_cat=st.selectbox("Pick a category to overview", ["Large Cap","Mid Cap","Small Cap"])
    watch_map=large_cap if selected_cat=="Large Cap" else mid_cap if selected_cat=="Mid Cap" else small_cap
    
    picked=st.multiselect("Select companies to compare", list(watch_map.keys()), default=list(watch_map.keys())[:5])
    
    if st.button("📥 Load Market Data", use_container_width=True):
        rows=[]
        prog=st.progress(0)
        for i,name in enumerate(picked):
            tkr=watch_map[name]
            try:
                d=load_stock(tkr, "5d")
                if not d.empty:
                    cur=float(d["Close"].iloc[-1])
                    prv=float(d["Close"].iloc[-2])
                    chg=((cur-prv)/prv)*100
                    vol=int(d["Volume"].iloc[-1])
                    rows.append({"Company":name,"Ticker":tkr,"Price ($)":round(cur,2),"Day Change (%)":round(chg,2),"Volume":f"{vol:,}"})
            except:
                pass
            prog.progress((i+1)/len(picked))
        prog.empty()
        
        if rows:
            overview_df=pd.DataFrame(rows)
            st.dataframe(overview_df, use_container_width=True)
            
            bar_fig=px.bar(overview_df, x="Company", y="Day Change (%)", color="Day Change (%)",
                           color_continuous_scale=["red","gray","green"], template="plotly_dark",
                           title=f"{selected_cat} — Daily % Change Comparison")
            st.plotly_chart(bar_fig, use_container_width=True)
        else:
            st.warning("No data returned. Try again later.")

with t3:
    st.markdown("<h4 style='text-align:center;'>🤖 AI Stock Market Assistant</h4>", unsafe_allow_html=True)
    st.write("Ask anything about stocks, earnings, market trends, or specific companies.")
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history=[]
    
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    prompt=st.chat_input("Ask about any stock, market news, or financial trend...", key="chat1")
    
    if prompt:
        st.session_state.chat_history.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.write(prompt)
        
        with st.spinner("Searching real-time data..."):
            rt=fetch_real_time(prompt+" stock market finance 2025")
        
        msgs=[
            {"role":"system","content":f"You are an expert stock market and financial AI assistant with access to real-time web data. Answer questions about stocks, markets, companies, earnings, economic trends, and investment insights. Be concise, precise, and cite the data when relevant.\n\nUser Question: {prompt}\n\nReal-Time Web Data: {rt}\n\nRespond using the real-time data if it's relevant. Always clarify that predictions are not financial advice."},
            {"role":"user","content":prompt}
        ]
        
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
            res=client.chat.completions.create(
                messages=msgs,
                max_tokens=700,
                temperature=0.7
            )
            
            reply=res.choices[0].message.content
            st.write(reply)
        
        st.session_state.chat_history.append({"role":"assistant","content":reply})
