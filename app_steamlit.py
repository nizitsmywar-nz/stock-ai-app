import os
import json
import math
import time
import re
import warnings
from datetime import datetime, timedelta, timezone

import streamlit as st
import pandas as pd
import pandas_datareader.data as web
import ta
import yfinance as yf
from dotenv import load_dotenv

from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

warnings.filterwarnings("ignore")
load_dotenv()

# 한국 표준시(KST) 정의 (UTC+9)
KST = timezone(timedelta(hours=9))

def get_current_kst_time_str():
    return datetime.now(KST).strftime("%Y-%m-%d %H:%M")

st.set_page_config(
    page_title="AI Stock Valuation Dashboard Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# 📌 모바일 전용 반응형 & 텍스트 넘침 방지 커스텀 CSS 주입
# -------------------------------------------------------------
st.markdown("""
<style>
    /* 1. 전체 화면 가로 스크롤 튀어나감 방지 */
    html, body, [data-testid="stAppViewContainer"] {
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }
    
    /* 2. 텍스트 자동 줄바꿈 및 글자 겹침 방지 */
    div, p, span, h1, h2, h3, h4, h5, h6 {
        overflow-wrap: break-word !important;
        word-break: break-word !important;
    }

    /* 3. 모바일 화면 최적화 미디어 쿼리 (스마트폰 & 태블릿 세로) */
    @media (max-width: 768px) {
        /* 상하/좌우 패딩 축소 */
        .block-container {
            padding-top: 1.2rem !important;
            padding-left: 0.6rem !important;
            padding-right: 0.6rem !important;
            padding-bottom: 2rem !important;
        }

        /* 4열 컬럼을 모바일에서 2열 2행(50%)으로 자동 줄바꿈 */
        [data-testid="stHorizontalBlock"] {
            flex-wrap: wrap !important;
            gap: 8px !important;
        }
        
        [data-testid="stHorizontalBlock"] > [data-testid="column"] {
            flex: 1 1 calc(50% - 10px) !important;
            min-width: calc(50% - 10px) !important;
            max-width: 100% !important;
            margin-bottom: 4px !important;
        }

        /* 메트릭 폰트 크기 모바일 축소 */
        [data-testid="stMetricValue"] {
            font-size: 1.15rem !important;
            line-height: 1.25 !important;
        }
        [data-testid="stMetricLabel"] {
            font-size: 0.72rem !important;
            line-height: 1.2 !important;
        }
        [data-testid="stMetricDelta"] {
            font-size: 0.68rem !important;
            line-height: 1.2 !important;
        }
        
        /* 헤더 폰트 크기 조정 */
        h2 {
            font-size: 1.35rem !important;
        }
        h4, h5 {
            font-size: 0.95rem !important;
        }
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 0. 영구 저장소 (history.json) 관리 함수
# -------------------------------------------------------------
HISTORY_FILE = "history.json"

def load_history():
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}

def save_history(history_data):
    try:
        with open(HISTORY_FILE, "w", encoding="utf-8") as f:
            json.dump(history_data, f, ensure_ascii=False, indent=2)
    except Exception:
        pass

# --- 세션 상태 초기화 ---
if "history" not in st.session_state:
    st.session_state.history = load_history()

if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = "TSLA"

# -------------------------------------------------------------
# 1. RAG 데이터 수집 모듈
# -------------------------------------------------------------
def fetch_stock_technical_data(ticker: str):
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo")
    if df.empty:
        return {}, "N/A", {}
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    df['SMA_120'] = df['Close'].rolling(window=120).mean()
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    try:
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    except Exception:
        df['ATR'] = None

    try:
        df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
    except Exception:
        df['MFI'] = None

    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    
    latest = df.iloc[-1]
    last_date = df.index[-1].strftime("%Y-%m-%d")
    
    high_6m = float(df['High'].max())
    low_6m = float(df['Low'].min())
    diff_hl = high_6m - low_6m
    
    fibonacci_levels = {
        "high_6m": round(high_6m, 2),
        "low_6m": round(low_6m, 2),
        "fib_236": round(high_6m - (0.236 * diff_hl), 2),
        "fib_382": round(high_6m - (0.382 * diff_hl), 2),
        "fib_500": round(high_6m - (0.500 * diff_hl), 2),
        "fib_618": round(high_6m - (0.618 * diff_hl), 2)
    }
    
    atr_val = round(float(latest['ATR']), 2) if pd.notnull(latest['ATR']) else "N/A"
    
    data = {
        "current_price": round(float(latest['Close']), 2),
        "atr_14": atr_val,
        "atr_stop_1_5x": round(float(latest['Close']) - (1.5 * float(latest['ATR'])), 2) if pd.notnull(latest['ATR']) else "N/A",
        "atr_stop_2_0x": round(float(latest['Close']) - (2.0 * float(latest['ATR'])), 2) if pd.notnull(latest['ATR']) else "N/A",
        "sma_20": round(float(latest['SMA_20']), 2) if pd.notnull(latest['SMA_20']) else "N/A",
        "sma_60": round(float(latest['SMA_60']), 2) if pd.notnull(latest['SMA_60']) else "N/A",
        "sma_120": round(float(latest['SMA_120']), 2) if pd.notnull(latest['SMA_120']) else "N/A",
        "rsi_14": round(float(latest['RSI']), 2) if pd.notnull(latest['RSI']) else "N/A",
        "mfi_14": round(float(latest['MFI']), 2) if pd.notnull(latest['MFI']) else "N/A",
        "macd": round(float(latest['MACD']), 2) if pd.notnull(latest['MACD']) else "N/A",
        "macd_signal": round(float(latest['MACD_Signal']), 2) if pd.notnull(latest['MACD_Signal']) else "N/A",
        "macd_hist": round(float(latest['MACD_Hist']), 2) if pd.notnull(latest['MACD_Hist']) else "N/A",
        "bb_upper": round(float(latest['BB_High']), 2) if pd.notnull(latest['BB_High']) else "N/A",
        "bb_lower": round(float(latest['BB_Low']), 2) if pd.notnull(latest['BB_Low']) else "N/A"
    }
    return data, last_date, fibonacci_levels

def fetch_nearest_options_data(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        expirations = stock.options
        if not expirations:
            return None
        
        nearest_exp = expirations[0]
        opt_chain = stock.option_chain(nearest_exp)
        calls = opt_chain.calls
        puts = opt_chain.puts
        
        if calls.empty or puts.empty:
            return None
            
        call_max_oi_row = calls.loc[calls['openInterest'].idxmax()] if calls['openInterest'].notnull().any() and calls['openInterest'].max() > 0 else calls.iloc[0]
        call_max_vol_row = calls.loc[calls['volume'].idxmax()] if calls['volume'].notnull().any() and calls['volume'].max() > 0 else calls.iloc[0]
        
        put_max_oi_row = puts.loc[puts['openInterest'].idxmax()] if puts['openInterest'].notnull().any() and puts['openInterest'].max() > 0 else puts.iloc[0]
        put_max_vol_row = puts.loc[puts['volume'].idxmax()] if puts['volume'].notnull().any() and puts['volume'].max() > 0 else puts.iloc[0]
        
        tot_call_vol = calls['volume'].sum() if calls['volume'].notnull().any() else 0
        tot_put_vol = puts['volume'].sum() if puts['volume'].notnull().any() else 0
        pc_ratio = round(tot_put_vol / tot_call_vol, 2) if tot_call_vol > 0 else "N/A"

        return {
            "expiration_date": nearest_exp,
            "pc_volume_ratio": pc_ratio,
            "call_max_oi": {
                "strike": call_max_oi_row.get("strike", "N/A"),
                "oi": int(call_max_oi_row.get("openInterest", 0)) if pd.notnull(call_max_oi_row.get("openInterest")) else 0,
                "price": round(float(call_max_oi_row.get("lastPrice", 0)), 2)
            },
            "call_max_vol": {
                "strike": call_max_vol_row.get("strike", "N/A"),
                "volume": int(call_max_vol_row.get("volume", 0)) if pd.notnull(call_max_vol_row.get("volume")) else 0,
                "price": round(float(call_max_vol_row.get("lastPrice", 0)), 2)
            },
            "put_max_oi": {
                "strike": put_max_oi_row.get("strike", "N/A"),
                "oi": int(put_max_oi_row.get("openInterest", 0)) if pd.notnull(put_max_oi_row.get("openInterest")) else 0,
                "price": round(float(put_max_oi_row.get("lastPrice", 0)), 2)
            },
            "put_max_vol": {
                "strike": put_max_vol_row.get("strike", "N/A"),
                "volume": int(put_max_vol_row.get("volume", 0)) if pd.notnull(put_max_vol_row.get("volume")) else 0,
                "price": round(float(put_max_vol_row.get("lastPrice", 0)), 2)
            }
        }
    except Exception:
        return None

def fetch_macro_indicators():
    macro_data = {}
    try:
        end = datetime.now()
        start = end - timedelta(days=30)
        fred_res = web.DataReader('DGS10', 'fred', start, end).dropna()
        dgs10 = fred_res.iloc[-1, 0]
        macro_data["us_10y_yield"] = {
            "source": "FRED (Federal Reserve Economic Data)",
            "value": f"{round(float(dgs10), 2)}%",
            "date": fred_res.index[-1].strftime("%Y-%m-%d")
        }
    except Exception:
        macro_data["us_10y_yield"] = {"source": "FRED", "value": "N/A", "date": "N/A"}
        
    asset_tickers = [
        ("vix", "^VIX", "CBOE Volatility Index"),
        ("dollar_index", "DX-Y.NYB", "ICE US Dollar Index"),
        ("wti_oil", "CL=F", "NYMEX WTI Crude Oil"),
        ("gold", "GC=F", "COMEX Gold Futures"),
        ("bitcoin", "BTC-USD", "Binance/Coinbase Crypto Market")
    ]
    for name, ticker, src_name in asset_tickers:
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                macro_data[name] = {
                    "source": src_name,
                    "value": round(float(hist['Close'].iloc[-1]), 2),
                    "date": hist.index[-1].strftime("%Y-%m-%d")
                }
            else:
                macro_data[name] = {"source": src_name, "value": "N/A", "date": "N/A"}
        except Exception:
            macro_data[name] = {"source": src_name, "value": "N/A", "date": "N/A"}
    return macro_data

def format_market_cap(market_cap):
    if not market_cap or market_cap == "N/A":
        return "N/A"
    if market_cap >= 1e12:
        return f"${market_cap / 1e12:.2f}T"
    elif market_cap >= 1e9:
        return f"${market_cap / 1e9:.2f}B"
    elif market_cap >= 1e6:
        return f"${market_cap / 1e6:.2f}M"
    return f"${market_cap:,.0f}"

def fetch_ownership_and_shorts(stock, info):
    data = {
        "insider_own": "N/A",
        "insider_trans": "N/A",
        "inst_own": "N/A",
        "inst_trans": "N/A",
        "short_percent_of_float": "N/A",
        "short_ratio": "N/A"
    }
    ins_own_val = info.get("heldPercentInsiders", None)
    if ins_own_val is not None:
        data["insider_own"] = f"{ins_own_val * 100:.2f}%"
        
    inst_own_val = info.get("heldPercentInstitutions", None)
    if inst_own_val is not None:
        data["inst_own"] = f"{inst_own_val * 100:.2f}%"

    short_float = info.get("shortPercentOfFloat", None)
    if short_float is not None:
        data["short_percent_of_float"] = f"{short_float * 100:.2f}%"
        
    short_rat = info.get("shortRatio", None)
    if short_rat is not None:
        data["short_ratio"] = f"{short_rat:.2f}일"

    try:
        ins_df = stock.insider_transactions
        if ins_df is not None and not ins_df.empty and 'Shares' in ins_df.columns:
            recent_ins = ins_df.head(15)
            net_shares = recent_ins['Shares'].dropna().sum()
            shares_out = info.get("sharesOutstanding", None)
            if shares_out and shares_out > 0:
                trans_pct = (net_shares / shares_out) * 100
                data["insider_trans"] = f"{trans_pct:+.2f}%"
            else:
                data["insider_trans"] = f"{net_shares:+,.0f}주"
    except Exception:
        pass

    try:
        inst_df = stock.institutional_holders
        if inst_df is not None and not inst_df.empty and '% Out' in inst_df.columns:
            tot_pct = inst_df['% Out'].sum() * 100
            data["inst_trans"] = f"{tot_pct:.2f}% (Top10)"
        elif inst_df is not None and not inst_df.empty and 'Shares' in inst_df.columns:
            tot_shares = inst_df['Shares'].sum()
            data["inst_trans"] = f"{tot_shares:,.0f}주 (Top10)"
    except Exception:
        pass

    return data

def fetch_earnings_calendar(stock, info):
    try:
        cal = stock.calendar
        earnings_date_str = "미정"
        d_day_str = ""
        
        if cal is not None and isinstance(cal, dict) and 'Earnings Date' in cal:
            e_dates = cal['Earnings Date']
            if isinstance(e_dates, list) and e_dates:
                e_date = pd.to_datetime(e_dates[0])
                earnings_date_str = e_date.strftime("%Y-%m-%d")
                now = datetime.now()
                days_diff = (e_date - now).days
                if days_diff >= 0:
                    d_day_str = f"D-{days_diff}일"
                else:
                    d_day_str = f"최근 발표완료 ({abs(days_diff)}일 전)"
        elif cal is not None and isinstance(cal, pd.DataFrame) and not cal.empty:
            if 'Earnings Date' in cal.index:
                val = cal.loc['Earnings Date'].iloc[0]
                earnings_date_str = str(val)[:10]
        
        high_52w = info.get("fiftyTwoWeekHigh", "N/A")
        low_52w = info.get("fiftyTwoWeekLow", "N/A")
        
        return {
            "earnings_date": earnings_date_str,
            "d_day": d_day_str,
            "fiftyTwoWeekHigh": high_52w,
            "fiftyTwoWeekLow": low_52w
        }
    except Exception:
        return {"earnings_date": "확인불가", "d_day": "", "fiftyTwoWeekHigh": "N/A", "fiftyTwoWeekLow": "N/A"}

def fetch_fundamentals_and_valuation(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        market_cap = info.get("marketCap", "N/A")
        trailing_pe = info.get("trailingPE", "N/A")
        forward_pe = info.get("forwardPE", "N/A")
        pbr = info.get("priceToBook", "N/A")
        ps_ratio = info.get("priceToSalesTrailing12Months", "N/A")
        
        roe_raw = info.get("returnOnEquity", None)
        roe_pct = round(roe_raw * 100, 2) if roe_raw is not None else "N/A"
        eps = info.get("trailingEps", None)
        forward_eps = info.get("forwardEps", None)
        bps = info.get("bookValue", None)
        revenue_per_share = info.get("revenuePerShare", None)
        target_mean_price = info.get("targetMeanPrice", "N/A")
        
        ownership_and_shorts = fetch_ownership_and_shorts(stock, info)
        earnings_cal = fetch_earnings_calendar(stock, info)
        
        earnings_growth = info.get("earningsGrowth", None)
        if earnings_growth and earnings_growth > 0:
            est_growth = min(earnings_growth * 100, 35.0)
        else:
            est_growth = 20.0
            
        value_models = {}
        if eps and bps and eps > 0 and bps > 0:
            value_models["graham"] = round(math.sqrt(22.5 * eps * bps), 2)
        else:
            value_models["graham"] = "산출불가"
            
        if eps and eps > 0 and roe_raw and roe_raw > 0:
            value_models["peter_lynch"] = round(eps * min(roe_raw * 100, 25.0), 2)
        else:
            value_models["peter_lynch"] = "산출불가"

        if bps and bps > 0 and roe_raw and roe_raw > 0:
            value_models["roe_pbr"] = round(bps * (roe_raw / 0.10), 2)
        else:
            value_models["roe_pbr"] = "산출불가"

        growth_models = {}
        f_eps = forward_eps if forward_eps and forward_eps > 0 else eps
        if f_eps and f_eps > 0:
            growth_models["forward_peg"] = round(f_eps * (est_growth * 1.5), 2)
        else:
            growth_models["forward_peg"] = "산출불가"

        if revenue_per_share and revenue_per_share > 0:
            growth_models["psr_target"] = round(revenue_per_share * 8.5, 2)
        else:
            growth_models["psr_target"] = "산출불가"

        if f_eps and f_eps > 0:
            wacc = 0.09
            g_long = 0.025
            pv_sum = 0
            cur_cf = f_eps
            for y in range(1, 6):
                cur_cf *= (1 + est_growth / 100)
                pv_sum += cur_cf / ((1 + wacc) ** y)
            terminal_val = (cur_cf * (1 + g_long)) / (wacc - g_long)
            pv_terminal = terminal_val / ((1 + wacc) ** 5)
            growth_models["dcf_growth"] = round(pv_sum + pv_terminal, 2)
        else:
            growth_models["dcf_growth"] = "산출불가"

        return {
            "market_cap_fmt": format_market_cap(market_cap),
            "trailing_pe": round(trailing_pe, 2) if isinstance(trailing_pe, (int, float)) else trailing_pe,
            "forward_pe": round(forward_pe, 2) if isinstance(forward_pe, (int, float)) else forward_pe,
            "pbr": round(pbr, 2) if isinstance(pbr, (int, float)) else pbr,
            "ps_ratio": round(ps_ratio, 2) if isinstance(ps_ratio, (int, float)) else ps_ratio,
            "roe": f"{roe_pct}%" if roe_pct != "N/A" else "N/A",
            "target_mean_price": target_mean_price,
            "ownership_and_shorts": ownership_and_shorts,
            "earnings_calendar": earnings_cal,
            "value_models": value_models,
            "growth_models": growth_models
        }
    except Exception:
        return {}

def fetch_sector_performance():
    sector_etfs = ["XLK", "XLF", "XLE", "XLV", "XLI"]
    summary = {}
    for etf in sector_etfs:
        try:
            hist = yf.Ticker(etf).history(period="5d")
            if len(hist) >= 2:
                pct = (hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100
                summary[etf] = f"{pct:+.2f}%"
            else:
                summary[etf] = "N/A"
        except Exception:
            summary[etf] = "N/A"
    return summary

def fetch_news(ticker: str, limit: int = 5):
    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news
        if not raw_news:
            return []
            
        articles = []
        for n in raw_news[:limit]:
            content = n.get("content", {})
            if isinstance(content, dict) and content:
                title = content.get("title", "")
                summary = content.get("summary", "")
                publisher = content.get("provider", {}).get("displayName", "Yahoo Finance")
                click_url = content.get("clickThroughUrl", {})
                link = click_url.get("url", "") if isinstance(click_url, dict) else click_url
                if not link:
                    link = content.get("canonicalUrl", {}).get("url", "")
                pub_date = str(content.get("pubDate", "최근"))[:10]
            else:
                title = n.get("title", "")
                summary = ""
                publisher = n.get("publisher", "Yahoo Finance")
                link = n.get("link", "")
                pub_time = n.get("providerPublishTime", None)
                pub_date = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d") if pub_time else "최근"
            
            if title:
                articles.append({
                    "title": title,
                    "summary": summary,
                    "publisher": publisher,
                    "date": pub_date,
                    "link": link or f"https://finance.yahoo.com/quote/{ticker}"
                })
        return articles
    except Exception:
        return []

def fetch_macro_news(limit: int = 4):
    macro_articles = []
    for sym in ["SPY", "TLT"]:
        try:
            stock = yf.Ticker(sym)
            raw = stock.news
            if raw:
                for n in raw[:2]:
                    content = n.get("content", {})
                    if isinstance(content, dict) and content:
                        title = content.get("title", "")
                        summary = content.get("summary", "")
                        publisher = content.get("provider", {}).get("displayName", "MarketWatch")
                        click_url = content.get("clickThroughUrl", {})
                        link = click_url.get("url", "") if isinstance(click_url, dict) else click_url
                        if not link:
                            link = content.get("canonicalUrl", {}).get("url", "")
                        pub_date = str(content.get("pubDate", "최근"))[:10]
                    else:
                        title = n.get("title", "")
                        summary = ""
                        publisher = n.get("publisher", "MarketWatch")
                        link = n.get("link", "")
                        pub_time = n.get("providerPublishTime", None)
                        pub_date = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d") if pub_time else "최근"
                    
                    if title and not any(a["title"] == title for a in macro_articles):
                        macro_articles.append({
                            "title": title,
                            "summary": summary,
                            "publisher": publisher,
                            "date": pub_date,
                            "link": link or f"https://finance.yahoo.com/quote/{sym}"
                        })
        except Exception:
            pass
    return macro_articles[:limit]

def extract_clean_text(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "\n".join([p["text"] if isinstance(p, dict) and "text" in p else str(p) for p in content])
    return str(content)

def parse_full_trading_scenario(text):
    action = "홀딩"
    buy_band = "분석 리포트 참조"
    take_profit = "분석 리포트 참조"
    sell_target = "분석 리포트 참조"
    stop_loss = "분석 리포트 참조"
    
    if "적극매수" in text or "분할매수" in text or "매수의견" in text or "매수 (" in text:
        action = "매수"
    elif "비중축소" in text or "전량매도" in text or "매도 (" in text:
        action = "매도"
    elif "관망" in text or "보유" in text or "홀딩" in text:
        action = "홀딩"

    for line in text.split("\n"):
        line_clean = line.replace("*", "").replace("-", "").strip()
        if "분할 매수" in line_clean or "매수 밴드" in line_clean:
            parts = line_clean.split(":")
            if len(parts) > 1:
                buy_band = parts[1].strip()
        elif "목표가" in line_clean or "익절" in line_clean:
            parts = line_clean.split(":")
            if len(parts) > 1:
                take_profit = parts[1].strip()
                sell_target = parts[1].strip()
        elif "매도가" in line_clean:
            parts = line_clean.split(":")
            if len(parts) > 1:
                sell_target = parts[1].strip()
        elif "손절" in line_clean or "Stop-loss" in line_clean:
            parts = line_clean.split(":")
            if len(parts) > 1:
                stop_loss = parts[1].strip()
                
    return action, buy_band, take_profit, sell_target, stop_loss

def fetch_recent_upgrades_downgrades(ticker: str, months: int = 2):
    try:
        stock = yf.Ticker(ticker)
        upgrades = stock.upgrades_downgrades
        if upgrades is None or upgrades.empty:
            return []
        
        cutoff_date = datetime.now() - timedelta(days=months * 30)
        
        if isinstance(upgrades.index, pd.DatetimeIndex):
            filtered = upgrades[upgrades.index >= cutoff_date.strftime("%Y-%m-%d")]
        elif 'Date' in upgrades.columns:
            upgrades['Date'] = pd.to_datetime(upgrades['Date'])
            filtered = upgrades[upgrades['Date'] >= cutoff_date]
        else:
            filtered = upgrades.head(5)
            
        records = []
        for idx, row in filtered.head(5).iterrows():
            date_str = idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx)[:10]
            records.append({
                "date": date_str,
                "firm": row.get("Firm", "N/A"),
                "to_grade": row.get("ToGrade", "N/A"),
                "action": row.get("Action", "N/A")
            })
        return records
    except Exception:
        return []

# -------------------------------------------------------------
# 2. 사이드바 UI
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ **AI Stock Analyst**")
    ticker_input = st.text_input("종목 티커 (Ticker)", value=st.session_state.selected_ticker).upper()
    
    st.markdown("---")
    is_holding = st.checkbox("💼 **현재 보유 중인 종목인가요?**", value=False)
    
    user_avg_price = 0.0
    user_shares = 0.0
    
    if is_holding:
        u_col1, u_col2 = st.columns(2)
        with u_col1:
            user_avg_price = st.number_input("내 평단가 ($)", min_value=0.0, value=0.0, step=0.5, format="%.2f")
        with u_col2:
            user_shares = st.number_input("보유 수량 (주)", min_value=0.0, value=0.0, step=1.0, format="%.1f")
            
    st.write("")
    analyze_btn = st.button("🚀 분석 실행", type="primary", use_container_width=True)
    st.divider()

    st.markdown("#### 📌 **종목별 트레이딩 히스토리**")
    
    if st.session_state.history:
        tab_all, tab_buy, tab_sell, tab_hold = st.tabs(["전체", "🟢매수", "🔴매도", "🟡홀딩"])
        
        def render_history_card(tab_prefix, t_code, data):
            action_badge = "🟢 매수" if data['action'] == "매수" else ("🔴 매도" if data['action'] == "매도" else "🟡 홀딩")
            with st.expander(f"**{t_code}** (${data['price']}) | {action_badge}", expanded=False):
                st.markdown(f"- **현재가:** `${data['price']}`")
                if data.get('my_avg', 0) > 0:
                    st.markdown(f"- **💼 내 평단:** `${data['my_avg']}` ({data.get('my_return', 'N/A')})")
                st.markdown(f"- **🎯 익절/목표가:** `{data['take_profit']}`")
                st.markdown(f"- **📤 매도가 밴드:** `{data['sell_target']}`")
                st.markdown(f"- **📥 분할매수 밴드:** `{data['buy_band']}`")
                st.markdown(f"- **🛑 손절선:** `{data['stop_loss']}`")
                st.caption(f"분석 일시(KST): {data.get('time', 'N/A')}")
                if st.button(f"'{t_code}' 다시 분석", key=f"btn_re_{tab_prefix}_{t_code}", use_container_width=True):
                    st.session_state.selected_ticker = t_code
                    st.rerun()

        with tab_all:
            for t_code, data in list(st.session_state.history.items())[::-1]:
                render_history_card("all", t_code, data)
                
        with tab_buy:
            buy_items = [item for item in list(st.session_state.history.items())[::-1] if item[1]['action'] == "매수"]
            if buy_items:
                for t_code, data in buy_items:
                    render_history_card("buy", t_code, data)
            else:
                st.caption("매수 판정 종목이 없습니다.")

        with tab_sell:
            sell_items = [item for item in list(st.session_state.history.items())[::-1] if item[1]['action'] == "매도"]
            if sell_items:
                for t_code, data in sell_items:
                    render_history_card("sell", t_code, data)
            else:
                st.caption("매도 판정 종목이 없습니다.")

        with tab_hold:
            hold_items = [item for item in list(st.session_state.history.items())[::-1] if item[1]['action'] == "홀딩"]
            if hold_items:
                for t_code, data in hold_items:
                    render_history_card("hold", t_code, data)
            else:
                st.caption("홀딩/관망 판정 종목이 없습니다.")
                    
        st.write("")
        if st.button("🗑️ 히스토리 전체 삭제", use_container_width=True):
            st.session_state.history = {}
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.rerun()
    else:
        st.caption("분석을 실행하면 종목별 매수/매도/홀딩 판정 및 목표가/손절가 밴드가 영구 저장됩니다.")

# -------------------------------------------------------------
# 3. 메인 분석 화면
# -------------------------------------------------------------
st.header(f"📊 {ticker_input} 종합 밸류에이션 & 투자전략 리포트")

if analyze_btn:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            api_key = None
            
    if not api_key:
        st.error("GEMINI_API_KEY가 설정되지 않았습니다. Streamlit Secrets에 등록하세요.")
    else:
        with st.spinner(f"🔍 [{ticker_input}] 실시간 재무/옵션체인/피보나치/공매도/실적일정 수집 및 분석 중..."):
            tech_data, stock_date, fib_levels = fetch_stock_technical_data(ticker_input)
            options_data = fetch_nearest_options_data(ticker_input)
            macro_data = fetch_macro_indicators()
            fund_data = fetch_fundamentals_and_valuation(ticker_input)
            sector_data = fetch_sector_performance()
            news_data = fetch_news(ticker_input, limit=5)
            macro_news_data = fetch_macro_news(limit=4)
            analyst_data = fetch_recent_upgrades_downgrades(ticker_input, months=2)
            
            curr_p = tech_data.get('current_price', 0)
            ownership = fund_data.get('ownership_and_shorts', {})
            earnings_info = fund_data.get('earnings_calendar', {})
            
            # 💼 보유 주식 평가손익 전용 컨테이너
            my_return_str = "N/A"
            if is_holding and user_avg_price > 0 and user_shares > 0 and curr_p > 0:
                total_invested = user_avg_price * user_shares
                total_current = curr_p * user_shares
                pnl_dollar = total_current - total_invested
                pnl_pct = ((curr_p - user_avg_price) / user_avg_price) * 100
                my_return_str = f"{pnl_pct:+.2f}%"
                
                with st.container(border=True):
                    st.markdown(f"#### 💼 **내 보유 포지션 분석 ({ticker_input})**")
                    p_c1, p_c2, p_c3, p_c4 = st.columns(4)
                    p_c1.metric("내 매수 평단가", f"${user_avg_price:,.2f}", f"{user_shares:,.1f}주 보유")
                    p_c2.metric("총 매수 원금", f"${total_invested:,.2f}")
                    p_c3.metric("현재 평가 금액", f"${total_current:,.2f}")
                    p_c4.metric("평가 손익 (수익률)", f"${pnl_dollar:+,.2f}", f"{pnl_pct:+.2f}%")

            # 1. 상단 핵심 메트릭 (재무 + 스마트머니 8대 지표 완벽 복원)
            with st.container(border=True):
                st.markdown("**🏢 핵심 시장 및 재무 지표**")
                r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
                r1_c1.metric("현재 주가", f"${curr_p}")
                r1_c2.metric("시가총액", str(fund_data.get('market_cap_fmt', 'N/A')))
                r1_c3.metric("PER (선행/후행)", f"{fund_data.get('forward_pe', 'N/A')} / {fund_data.get('trailing_pe', 'N/A')}")
                r1_c4.metric("PBR / PSR", f"{fund_data.get('pbr', 'N/A')} / {fund_data.get('ps_ratio', 'N/A')}")
                
                st.divider()
                
                # 📌 내부자/기관 지분율 및 최근 매매 변동 (Own & Trans 완벽 복원)
                st.markdown("**👥 스마트머니 수급 분석 (내부자 & 기관 지분 및 최근 매매)**")
                own_c1, own_c2, own_c3, own_c4 = st.columns(4)
                own_c1.metric("Insider Own (내부자 지분)", str(ownership.get('insider_own', 'N/A')))
                own_c2.metric("Insider Trans (내부자 매매)", str(ownership.get('insider_trans', 'N/A')))
                own_c3.metric("Inst Own (기관 지분)", str(ownership.get('inst_own', 'N/A')))
                own_c4.metric("Inst Trans (기관 매매/보유)", str(ownership.get('inst_trans', 'N/A')))

                st.divider()

                # 📌 공매도 수급, 차기 실적 발표일, 52주 최고/최저가
                st.markdown("**🎯 공매도 수급(Shorts) 및 차기 실적 발표 일정**")
                s_c1, s_c2, s_c3, s_c4 = st.columns(4)
                s_c1.metric("공매도 잔고 비율 (Float)", str(ownership.get('short_percent_of_float', 'N/A')))
                s_c2.metric("공매도 상환 소요 일수", str(ownership.get('short_ratio', 'N/A')))
                s_c3.metric("차기 실적 발표일", str(earnings_info.get('earnings_date', 'N/A')), str(earnings_info.get('d_day', '')))
                
                high_52 = earnings_info.get('fiftyTwoWeekHigh', 'N/A')
                low_52 = earnings_info.get('fiftyTwoWeekLow', 'N/A')
                diff_52h = round(((curr_p - high_52) / high_52) * 100, 1) if isinstance(high_52, (int, float)) and curr_p else None
                s_c4.metric("52주 최고 / 최저가", f"${high_52} / ${low_52}", f"최고가 대비 {diff_52h:+.1f}%" if diff_52h is not None else None)

                st.divider()
                
                # 변동성(ATR), 기술지표 및 거시지표
                st.markdown("**📈 변동성(ATR), 기술적 모멘텀 및 거시 지표**")
                r2_c1, r2_c2, r2_c3, r2_c4 = st.columns(4)
                r2_c1.metric(
                    "MACD (Signal)", 
                    f"{tech_data.get('macd', 'N/A')} ({tech_data.get('macd_signal', 'N/A')})", 
                    f"Hist: {tech_data.get('macd_hist', 'N/A'):+}" if isinstance(tech_data.get('macd_hist'), (int, float)) else None
                )
                r2_c2.metric("14일 ATR (일일 변동폭)", f"${tech_data.get('atr_14', 'N/A')}", f"1.5x 손절가: ${tech_data.get('atr_stop_1_5x', 'N/A')}")
                r2_c3.metric("RSI(14) / MFI 수급", f"{tech_data.get('rsi_14', 'N/A')} / {tech_data.get('mfi_14', 'N/A')}")
                r2_c4.metric("미 10년물 금리 / 달러", f"{macro_data.get('us_10y_yield', {}).get('value', 'N/A')} / {macro_data.get('dollar_index', {}).get('value', 'N/A')}")

            # 📌 피보나치 되돌림 지지/저항 밴드 컨테이너
            with st.container(border=True):
                st.markdown(f"##### 📐 **최근 6개월 피보나치 되돌림 지지/저항 밴드** (최고: `${fib_levels.get('high_6m', 'N/A')}` / 최저: `${fib_levels.get('low_6m', 'N/A')}`)")
                fb1, fb2, fb3, fb4 = st.columns(4)
                
                f236 = fib_levels.get('fib_236', 'N/A')
                f382 = fib_levels.get('fib_382', 'N/A')
                f500 = fib_levels.get('fib_500', 'N/A')
                f618 = fib_levels.get('fib_618', 'N/A')
                
                fb1.metric("23.6% 되돌림 (단기 지지)", f"${f236}", f"{round(((f236-curr_p)/curr_p)*100, 1):+.1f}%" if isinstance(f236, (int, float)) and curr_p else None)
                fb2.metric("38.2% 되돌림 (1차 매수 지지)", f"${f382}", f"{round(((f382-curr_p)/curr_p)*100, 1):+.1f}%" if isinstance(f382, (int, float)) and curr_p else None)
                fb3.metric("50.0% 하프라인 (추세 기준선)", f"${f500}", f"{round(((f500-curr_p)/curr_p)*100, 1):+.1f}%" if isinstance(f500, (int, float)) and curr_p else None)
                fb4.metric("61.8% 되돌림 (강력한 2차 지지)", f"${f618}", f"{round(((f618-curr_p)/curr_p)*100, 1):+.1f}%" if isinstance(f618, (int, float)) and curr_p else None)

            # 📌 옵션 체인 스마트머니 포지션 카드
            with st.container(border=True):
                if options_data:
                    exp_date = options_data['expiration_date']
                    pc_rat = options_data['pc_volume_ratio']
                    st.markdown(f"##### 🎯 **가장 빠른 만기 옵션 체인 스마트머니 포지션** `만기일: {exp_date}` `P/C Ratio: {pc_rat}`")
                    
                    op_c1, op_c2, op_c3, op_c4 = st.columns(4)
                    c_oi = options_data['call_max_oi']
                    diff_c_oi = round(((c_oi['strike'] - curr_p) / curr_p) * 100, 1) if curr_p and isinstance(c_oi['strike'], (int, float)) else None
                    op_c1.metric(
                        "콜옵션 Max OI (상방 저항벽)",
                        f"${c_oi['strike']}",
                        f"{diff_c_oi:+.1f}% (OI: {c_oi['oi']:,} / ${c_oi['price']})" if diff_c_oi is not None else f"OI: {c_oi['oi']:,}"
                    )
                    
                    c_vol = options_data['call_max_vol']
                    diff_c_vol = round(((c_vol['strike'] - curr_p) / curr_p) * 100, 1) if curr_p and isinstance(c_vol['strike'], (int, float)) else None
                    op_c2.metric(
                        "콜옵션 Max Vol (당일 상방 수급)",
                        f"${c_vol['strike']}",
                        f"{diff_c_vol:+.1f}% (Vol: {c_vol['volume']:,} / ${c_vol['price']})" if diff_c_vol is not None else f"Vol: {c_vol['volume']:,}"
                    )
                    
                    p_oi = options_data['put_max_oi']
                    diff_p_oi = round(((p_oi['strike'] - curr_p) / curr_p) * 100, 1) if curr_p and isinstance(p_oi['strike'], (int, float)) else None
                    op_c3.metric(
                        "풋옵션 Max OI (하방 지지벽)",
                        f"${p_oi['strike']}",
                        f"{diff_p_oi:+.1f}% (OI: {p_oi['oi']:,} / ${p_oi['price']})" if diff_p_oi is not None else f"OI: {p_oi['oi']:,}"
                    )
                    
                    p_vol = options_data['put_max_vol']
                    diff_p_vol = round(((p_vol['strike'] - curr_p) / curr_p) * 100, 1) if curr_p and isinstance(p_vol['strike'], (int, float)) else None
                    op_c4.metric(
                        "풋옵션 Max Vol (당일 하방 헤지)",
                        f"${p_vol['strike']}",
                        f"{diff_p_vol:+.1f}% (Vol: {p_vol['volume']:,} / ${p_vol['price']})" if diff_p_vol is not None else f"Vol: {p_vol['volume']:,}"
                    )
                else:
                    st.markdown("##### 🎯 **옵션 체인 스마트머니 포지션**")
                    st.info("해당 종목은 옵션 체인 거래 데이터가 없거나 수집되지 않았습니다.")

            # 2. 성장주 3대 밸류에이션 모델
            with st.container(border=True):
                st.markdown("##### 🚀 **성장주/빅테크 맞춤형 밸류에이션 모델 (Growth Models)**")
                g_models = fund_data.get('growth_models', {})
                g1, g2, g3, g4 = st.columns(4)
                
                target_p = fund_data.get('target_mean_price', 'N/A')
                diff_t = round(((target_p - curr_p) / curr_p) * 100, 1) if isinstance(target_p, (int, float)) and curr_p else None
                g1.metric("IB 컨센서스 목표가", f"${target_p}" if isinstance(target_p, (int, float)) else str(target_p), f"{diff_t:+.1f}%" if diff_t is not None else None)
                
                peg_p = g_models.get('forward_peg', 'N/A')
                diff_peg = round(((peg_p - curr_p) / curr_p) * 100, 1) if isinstance(peg_p, (int, float)) and curr_p else None
                g2.metric("Forward PEG 1.5 모델", f"${peg_p}" if isinstance(peg_p, (int, float)) else str(peg_p), f"{diff_peg:+.1f}%" if diff_peg is not None else None)
                
                psr_p = g_models.get('psr_target', 'N/A')
                diff_psr = round(((psr_p - curr_p) / curr_p) * 100, 1) if isinstance(psr_p, (int, float)) and curr_p else None
                g3.metric("PSR 타깃 매출가치 (8.5배)", f"${psr_p}" if isinstance(psr_p, (int, float)) else str(psr_p), f"{diff_psr:+.1f}%" if diff_psr is not None else None)
                
                dcf_p = g_models.get('dcf_growth', 'N/A')
                diff_dcf = round(((dcf_p - curr_p) / curr_p) * 100, 1) if isinstance(dcf_p, (int, float)) and curr_p else None
                g4.metric("2단계 DCF 현금흐름 모델", f"${dcf_p}" if isinstance(dcf_p, (int, float)) else str(dcf_p), f"{diff_dcf:+.1f}%" if diff_dcf is not None else None)

            # 3. 전통 가치투자 모델
            with st.container(border=True):
                st.markdown("##### 🏛️ **전통 제조업/자산가치 기반 3대 모델 (Value Models - 청산/장부가치 기준)**")
                v_models = fund_data.get('value_models', {})
                v1, v2, v3 = st.columns(3)
                
                g_val = v_models.get('graham', 'N/A')
                diff_g = round(((g_val - curr_p) / curr_p) * 100, 1) if isinstance(g_val, (int, float)) and curr_p else None
                v1.metric("그레이엄 청산가치", f"${g_val}" if isinstance(g_val, (int, float)) else str(g_val), f"{diff_g:+.1f}%" if diff_g is not None else None)
                
                l_val = v_models.get('peter_lynch', 'N/A')
                diff_l = round(((l_val - curr_p) / curr_p) * 100, 1) if isinstance(l_val, (int, float)) and curr_p else None
                v2.metric("피터 린치 가치모델", f"${l_val}" if isinstance(l_val, (int, float)) else str(l_val), f"{diff_l:+.1f}%" if diff_l is not None else None)
                
                r_val = v_models.get('roe_pbr', 'N/A')
                diff_r = round(((r_val - curr_p) / curr_p) * 100, 1) if isinstance(r_val, (int, float)) and curr_p else None
                v3.metric("ROE-PBR 자본가치", f"${r_val}" if isinstance(r_val, (int, float)) else str(r_val), f"{diff_r:+.1f}%" if diff_r is not None else None)

            st.caption(f"🕒 데이터 수집 기준일자: 주가/재무제표 ({stock_date}) | FRED 국채금리 ({macro_data.get('us_10y_yield', {}).get('date', 'N/A')})")

            # 4. Gemini 3.6 Flash 심층 분석
            user_position_text = (
                f"사용자 현재 보유 정보: 평단가 ${user_avg_price}, 보유 수량 {user_shares}주 (현재 수익률: {my_return_str})"
                if is_holding and user_avg_price > 0 else "사용자 미보유 종목 (신규 진입 검토 관점)"
            )

            template = """
[RAG 심층 주입 데이터]
1. 기술적/수급 및 ATR 변동성 데이터 ({ticker}) (기준일: {stock_date}):
{tech_json}

2. 최근 6개월 피보나치 되돌림 밴드:
{fib_json}

3. 가장 빠른 만기 옵션 체인 수급 (콜/풋 Max OI & Volume):
{options_json}

4. 내부자/기관 지분율 및 최근 매매, 공매도 지표 (Short Interest):
{ownership_json}

5. 실적 발표 일정 및 52주 고저:
{earnings_json}

6. 매크로 및 6대 자산 실시간 지표 (출처 및 기준일 포함):
{macro_json}

7. 글로벌 거시/시장 주요 뉴스:
{macro_news_json}

8. 주요 섹터 5일 등락률:
{sector_json}

9. 펀더멘털 및 6대 밸류에이션:
{fund_json}

10. 사용자 보유 현황:
{user_position}

11. 종목 최신 주요 기사:
{news_json}

12. 최근 2개월 증권가 투자의견 변동:
{analyst_json}

---

[지시사항]
위 데이터를 바탕으로 최고 수준의 금융 애널리스트 관점에서 정밀 리포트를 작성할 것:

1. 거시환경 및 시장 국면
- **[참고자료 및 기준일자]**: 분석에 활용된 핵심 매크로 지표의 **출처 및 수집 기준일자**를 요약 명시할 것.
- 경기 국면 (회복 / 활황 / 둔화 / 침체 판정) 및 단기 변동성 촉발 요인
- 최신 매크로 지표 및 글로벌 거시 뉴스를 직접 인용하여 [6대 유동성 자산 변동 예측]:
  * 현금 (달러), 채권 (미 국채), 주식 (위험자산), 코인 (가상자산), 금 (안전자산), 원유 (에너지) 각각의 전망 및 사유
- 권장 자산 배분 비중 (주식 : 채권 : 대체자산 : 현금)

2. 섹터 전망 및 순환매
- 상대적 강세/약세 섹터 요약 및 자금 순환매 방향

3. 밸류에이션 및 스마트머니/공매도/옵션 수급 분석 ({ticker})
- 전통 가치모델 vs 성장주 모델 괴리 분석 및 적정주가 밴드 제시
- **스마트머니 & 공매도 분석**: 내부자/기관 지분 및 최근 매매 동향, 공매도 잔고율(Short % of Float) 및 숏스퀴즈 잠재력 평가
- **옵션 체인 분석**: 콜/풋 Max OI 벽 및 P/C Ratio 기반 스마트머니 배팅 평가

4. 종목 종합 평가 및 [정밀 매매 시나리오] ({ticker})
- 기술적 분석: 피보나치 되돌림 레벨(38.2%, 61.8%), 이평선, MACD, MFI, ATR 변동폭
- 실적 발표 D-Day 리스크 점검 및 대응 방안
- 스코어카드 (각 10점 만점): 성장성, 수익성, 밸류에이션, 해자, 리스크
- 종합 평점 및 최종 투자 의견 (적극매수 / 분할매수 / 관망 / 비중축소)
- **사용자 맞춤 포지션 대응 전략**:
  * (보유 중인 경우) 현재 평단가 대비 물타기(추가 매수) 유효성, 불타기 시점, 부분 익절 전략
  * (미보유인 경우) 신규 진입 시 매수 타이밍 검토
- **정밀 매매 시나리오**:
  * 분할 매수 밴드: [피보나치 38.2%~61.8% 및 풋옵션 지지선을 결합한 구체적 달러 밴드 제시]
  * 목표가/익절 라인: [1차 및 2차 구체적 달러 가격대 제시]
  * 매도가 밴드: [차익실현 또는 비중축소 구체적 달러 가격대 제시]
  * 손절(Stop-loss) 기준선: [1.5x~2.0x ATR 및 핵심 추세선을 반영한 구체적 달러 가격대 제시]
"""
            prompt = PromptTemplate(
                input_variables=["ticker", "stock_date", "tech_json", "fib_json", "options_json", "ownership_json", "earnings_json", "macro_json", "macro_news_json", "sector_json", "fund_json", "user_position", "news_json", "analyst_json"],
                template=template
            )
            llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=api_key)
            chain = prompt | llm
            
            payload = {
                "ticker": ticker_input,
                "stock_date": stock_date,
                "tech_json": json.dumps(tech_data, indent=2, ensure_ascii=False),
                "fib_json": json.dumps(fib_levels, indent=2, ensure_ascii=False),
                "options_json": json.dumps(options_data, indent=2, ensure_ascii=False) if options_data else "옵션 데이터 없음",
                "ownership_json": json.dumps(ownership, indent=2, ensure_ascii=False),
                "earnings_json": json.dumps(earnings_info, indent=2, ensure_ascii=False),
                "macro_json": json.dumps(macro_data, indent=2, ensure_ascii=False),
                "macro_news_json": json.dumps(macro_news_data, indent=2, ensure_ascii=False),
                "sector_json": json.dumps(sector_data, indent=2, ensure_ascii=False),
                "fund_json": json.dumps(fund_data, indent=2, ensure_ascii=False),
                "user_position": user_position_text,
                "news_json": json.dumps(news_data, indent=2, ensure_ascii=False),
                "analyst_json": json.dumps(analyst_data, indent=2, ensure_ascii=False)
            }
            
            response_content = None
            for delay in [0, 5, 10]:
                if delay > 0:
                    time.sleep(delay)
                try:
                    res = chain.invoke(payload)
                    response_content = extract_clean_text(res.content)
                    break
                except Exception:
                    continue
                    
            if not response_content:
                response_content = "⚠️ Gemini API 일시적 지연이 발생했습니다. 잠시 후 다시 [분석 실행]을 눌러주세요."

            act, buy_b, tp_b, sell_b, sl_b = parse_full_trading_scenario(response_content)
            st.session_state.history[ticker_input] = {
                "action": act,
                "price": curr_p,
                "my_avg": user_avg_price if is_holding else 0,
                "my_return": my_return_str if is_holding else "미보유",
                "buy_band": buy_b,
                "take_profit": tp_b,
                "sell_target": sell_b,
                "stop_loss": sl_b,
                "time": get_current_kst_time_str()
            }
            save_history(st.session_state.history)

            with st.container(border=True):
                st.markdown("### 📝 **Gemini 3.6 Flash 종합 분석 브리핑**")
                st.markdown(response_content)

            with st.expander("🌐 **6대 유동성 자산 분석 참고 거시 기사 & 원문 링크 (클릭하여 접기/펼치기)**", expanded=False):
                if macro_news_data:
                    for m_item in macro_news_data:
                        st.markdown(f"- **[{m_item['title']}]({m_item['link']})**")
                        if m_item.get("summary"):
                            st.caption(f"> {m_item['summary']}")
                        st.caption(f"출처: {m_item['publisher']} | 게시일: {m_item['date']}")
                        st.write("")
                else:
                    st.info("수집된 거시경제 기사가 없습니다.")
            
            st.write("")

            col_left, col_right = st.columns([1.1, 0.9])
            
            with col_left:
                with st.container(border=True):
                    st.markdown(f"##### 📰 **{ticker_input} 최신 주요 뉴스 및 기사 원문**")
                    if news_data:
                        for item in news_data:
                            st.markdown(f"**[{item['title']}]({item['link']})**")
                            if item.get("summary"):
                                st.markdown(f"> *{item['summary']}*")
                            st.caption(f"출처: {item['publisher']} | {item['date']}")
                            st.divider()
                    else:
                        st.info("수집된 최신 뉴스가 없습니다.")
                        
            with col_right:
                with st.container(border=True):
                    st.markdown(f"##### 🏛️ **{ticker_input} 최근 2개월 증권가 투자의견 변동**")
                    if analyst_data:
                        df_analyst = pd.DataFrame(analyst_data)
                        df_analyst.columns = ["일자", "증권사", "투자의견", "액션"]
                        st.dataframe(df_analyst, use_container_width=True, hide_index=True)
                    else:
                        st.info("최근 2개월간 등록된 투자의견 변동 내역이 없습니다.")