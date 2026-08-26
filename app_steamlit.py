import os
import json
import math
import time
import re
import warnings
from datetime import datetime, timedelta, timezone

import streamlit as st
import pandas as pd
import numpy as np
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
# 📌 모바일 반응형 & 텍스트 넘침 방지 커스텀 CSS 주입
# -------------------------------------------------------------
st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"] {
        overflow-x: hidden !important;
        max-width: 100vw !important;
    }
    
    div, p, span, h1, h2, h3, h4, h5, h6 {
        overflow-wrap: break-word !important;
        word-break: break-word !important;
    }

    @media (max-width: 768px) {
        .block-container {
            padding-top: 1.2rem !important;
            padding-left: 0.6rem !important;
            padding-right: 0.6rem !important;
            padding-bottom: 2rem !important;
        }

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

if "last_analysis_result" not in st.session_state:
    st.session_state.last_analysis_result = None

# -------------------------------------------------------------
# 1. RAG 데이터 수집 모듈 (기술적 지표, BB Squeeze, VWAP 등)
# -------------------------------------------------------------
def get_stock_info_with_retry(stock, retries=3):
    for attempt in range(retries):
        try:
            info = stock.info
            if isinstance(info, dict) and len(info) > 10 and any(k in info for k in ['marketCap', 'trailingPE', 'forwardPE', 'trailingEps', 'bookValue', 'currentPrice']):
                return info, "stock.info"
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(1.0 * (attempt + 1))
            
    try:
        fallback_info = stock.info or {}
        if fallback_info and len(fallback_info) > 5:
            return fallback_info, "stock.info"
    except Exception:
        pass
    return {}, "stock.fast_info"

def fetch_stock_technical_data(ticker: str):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    if df.empty:
        df = stock.history(period="6mo")
    if df.empty:
        return {}, "N/A", {}, "N/A", "N/A", pd.DataFrame()
    
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
    df['BB_Mid'] = bb.bollinger_mavg()
    df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid'] * 100

    if df['ATR'] is not None:
        df['KC_High'] = df['SMA_20'] + (1.5 * df['ATR'])
        df['KC_Low'] = df['SMA_20'] - (1.5 * df['ATR'])
        df['BB_Squeeze_On'] = (df['BB_High'] < df['KC_High']) & (df['BB_Low'] > df['KC_Low'])
    else:
        df['BB_Squeeze_On'] = False

    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_Vol'] = df['Typical_Price'] * df['Volume']
    df['Cumulative_VWAP'] = df['TP_Vol'].cumsum() / df['Volume'].cumsum()
    df['Rolling_VWAP_20'] = df['TP_Vol'].rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) >= 2 else latest
    last_date = df.index[-1].strftime("%Y-%m-%d")
    
    high_52w_calc = round(float(df['High'].max()), 2)
    low_52w_calc = round(float(df['Low'].min()), 2)
    
    df_6m = df.tail(126) if len(df) >= 126 else df
    high_6m = float(df_6m['High'].max())
    low_6m = float(df_6m['Low'].min())
    diff_hl = high_6m - low_6m
    
    fibonacci_levels = {
        "high_6m": round(high_6m, 2),
        "low_6m": round(low_6m, 2),
        "fib_23.6%": round(high_6m - (0.236 * diff_hl), 2),
        "fib_38.2%": round(high_6m - (0.382 * diff_hl), 2),
        "fib_50.0%": round(high_6m - (0.500 * diff_hl), 2),
        "fib_61.8%": round(high_6m - (0.618 * diff_hl), 2)
    }
    
    atr_val = round(float(latest['ATR']), 2) if pd.notnull(latest['ATR']) else "N/A"
    
    is_sqz_now = bool(latest['BB_Squeeze_On'])
    is_sqz_prev = bool(prev['BB_Squeeze_On'])
    if is_sqz_now:
        squeeze_status = "⚠️ 스퀴즈 진행 중 (에너지 응축/변동성 폭발 임박)"
    elif is_sqz_prev and not is_sqz_now:
        if latest['Close'] > latest['BB_Mid']:
            squeeze_status = "🚀 상방 스퀴즈 분출 (상승 랠리 가속)"
        else:
            squeeze_status = "📉 하방 스퀴즈 이탈 (하락 가속 경보)"
    else:
        squeeze_status = "정상 변동성 구간 (스퀴즈 해제 상태)"

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
        "bb_middle": round(float(latest['BB_Mid']), 2) if pd.notnull(latest['BB_Mid']) else "N/A",
        "bb_lower": round(float(latest['BB_Low']), 2) if pd.notnull(latest['BB_Low']) else "N/A",
        "bb_width_pct": round(float(latest['BB_Width']), 2) if pd.notnull(latest['BB_Width']) else "N/A",
        "bb_squeeze_status": squeeze_status,
        "vwap_1y": round(float(latest['Cumulative_VWAP']), 2) if pd.notnull(latest['Cumulative_VWAP']) else "N/A",
        "vwap_20d": round(float(latest['Rolling_VWAP_20']), 2) if pd.notnull(latest['Rolling_VWAP_20']) else "N/A"
    }
    return data, last_date, fibonacci_levels, high_52w_calc, low_52w_calc, df

# -------------------------------------------------------------
# 📌 정밀 전략 백테스팅 모듈 (1Y 일봉 기반)
# -------------------------------------------------------------
def run_strategy_backtest(df: pd.DataFrame):
    if df is None or len(df) < 60:
        return None

    b_df = df.copy().dropna(subset=['Close', 'SMA_20', 'MACD', 'MACD_Signal', 'ATR', 'Cumulative_VWAP'])
    if len(b_df) < 30:
        return None

    bh_return = (b_df['Close'].iloc[-1] - b_df['Close'].iloc[0]) / b_df['Close'].iloc[0] * 100

    pos1 = 0
    entry_p1 = 0
    trades1 = []

    for i in range(1, len(b_df)):
        cur = b_df.iloc[i]
        prev = b_df.iloc[i-1]
        
        if pos1 == 1:
            stop_price = entry_p1 - (1.5 * cur['ATR']) if pd.notnull(cur['ATR']) else entry_p1 * 0.93
            if cur['Close'] < stop_price or cur['MACD_Hist'] < 0:
                ret = (cur['Close'] - entry_p1) / entry_p1
                trades1.append(ret)
                pos1 = 0
                entry_p1 = 0
        
        if pos1 == 0:
            cond_macd = (prev['MACD_Hist'] <= 0 and cur['MACD_Hist'] > 0)
            cond_trend = cur['Close'] > cur['SMA_20']
            cond_vwap = cur['Close'] > cur['Rolling_VWAP_20'] if pd.notnull(cur['Rolling_VWAP_20']) else True
            if cond_macd and cond_trend and cond_vwap:
                pos1 = 1
                entry_p1 = cur['Close']

    if pos1 == 1:
        ret = (b_df['Close'].iloc[-1] - entry_p1) / entry_p1
        trades1.append(ret)

    pos2 = 0
    entry_p2 = 0
    trades2 = []

    for i in range(1, len(b_df)):
        cur = b_df.iloc[i]
        
        if pos2 == 1:
            if cur['Close'] >= cur['Cumulative_VWAP'] or cur['RSI'] >= 65 or cur['Close'] < (cur['BB_Low'] * 0.97):
                ret = (cur['Close'] - entry_p2) / entry_p2
                trades2.append(ret)
                pos2 = 0
                entry_p2 = 0
        
        if pos2 == 0:
            cond_val = cur['Close'] < cur['Cumulative_VWAP']
            cond_rsi = cur['RSI'] < 42
            cond_bb = cur['Close'] > cur['BB_Low']
            if cond_val and cond_rsi and cond_bb:
                pos2 = 1
                entry_p2 = cur['Close']

    if pos2 == 1:
        ret = (b_df['Close'].iloc[-1] - entry_p2) / entry_p2
        trades2.append(ret)

    def calc_stats(trades):
        if not trades:
            return {"total_ret": 0.0, "win_rate": 0.0, "trades_count": 0, "profit_factor": 0.0, "mdd": 0.0}
        
        cum = 1.0
        peak = 1.0
        mdd = 0.0
        wins = [t for t in trades if t > 0]
        losses = [t for t in trades if t < 0]
        
        for t in trades:
            cum *= (1.0 + t)
            if cum > peak:
                peak = cum
            dd = (peak - cum) / peak
            if dd > mdd:
                mdd = dd
                
        tot_ret = (cum - 1.0) * 100
        win_rate = (len(wins) / len(trades)) * 100 if trades else 0
        sum_win = sum(wins) if wins else 0
        sum_loss = abs(sum(losses)) if losses else 0
        pf = round(sum_win / sum_loss, 2) if sum_loss > 0 else (99.9 if sum_win > 0 else 0.0)

        return {
            "total_ret": round(tot_ret, 2),
            "win_rate": round(win_rate, 1),
            "trades_count": len(trades),
            "profit_factor": pf,
            "mdd": round(mdd * 100, 2)
        }

    return {
        "benchmark_buy_and_hold": round(bh_return, 2),
        "strategy_1_momentum_squeeze": calc_stats(trades1),
        "strategy_2_vwap_mean_reversion": calc_stats(trades2)
    }

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
    try:
        mc = float(market_cap)
        if mc >= 1e12:
            return f"${mc / 1e12:.2f}T"
        elif mc >= 1e9:
            return f"${mc / 1e9:.2f}B"
        elif mc >= 1e6:
            return f"${mc / 1e6:.2f}M"
        return f"${mc:,.0f}"
    except Exception:
        return str(market_cap)

# -------------------------------------------------------------
# 📌 유명 헤지펀드 보유 내역 & 공매도 세력 인텔 수집기
# -------------------------------------------------------------
def fetch_hedge_funds_and_short_intel(stock, info):
    intel = {
        "top_holders": [],
        "short_intel": {}
    }
    
    try:
        inst_df = stock.institutional_holders
        if inst_df is not None and not inst_df.empty:
            for _, row in inst_df.head(6).iterrows():
                holder_name = str(row.get("Holder", "N/A")).strip()
                shares_val = row.get("Shares", 0)
                pct_out_val = row.get("% Out", 0)
                val_val = row.get("Value", 0)
                
                pct_str = f"{pct_out_val * 100:.2f}%" if pd.notnull(pct_out_val) and pct_out_val < 1.0 else f"{pct_out_val:.2f}%"
                val_str = f"${val_val / 1e9:.2f}B" if pd.notnull(val_val) and val_val >= 1e9 else (f"${val_val / 1e6:.1f}M" if pd.notnull(val_val) and val_val >= 1e6 else "-")
                
                intel["top_holders"].append({
                    "holder": holder_name,
                    "shares": f"{int(shares_val):,}" if pd.notnull(shares_val) else "-",
                    "percent_out": pct_str,
                    "value": val_str
                })
    except Exception:
        pass

    try:
        short_float = info.get("shortPercentOfFloat", None)
        short_ratio = info.get("shortRatio", None)
        shares_short = info.get("sharesShort", None)
        shares_short_prior = info.get("sharesShortPriorMonth", None)

        short_float_pct = round(short_float * 100, 2) if short_float is not None else None
        short_ratio_days = round(short_ratio, 2) if short_ratio is not None else None
        
        short_mom_pct = None
        if shares_short and shares_short_prior and shares_short_prior > 0:
            short_mom_pct = round(((shares_short - shares_short_prior) / shares_short_prior) * 100, 2)

        squeeze_risk = "🟢 안정 (Low Risk)"
        if short_float_pct is not None and short_ratio_days is not None:
            if short_float_pct >= 15.0 and short_ratio_days >= 5.0:
                squeeze_risk = "🚨 숏스퀴즈 고위험 (High Squeeze Potential)"
            elif short_float_pct >= 10.0 or short_ratio_days >= 4.0:
                squeeze_risk = "⚠️ 숏스퀴즈 주의 (Moderate Potential)"
            elif short_float_pct >= 5.0:
                squeeze_risk = "💡 모니터링 구간 (Low-Moderate)"

        intel["short_intel"] = {
            "short_percent_of_float": f"{short_float_pct:.2f}%" if short_float_pct is not None else "N/A",
            "short_ratio_days": f"{short_ratio_days:.2f}일" if short_ratio_days is not None else "N/A",
            "shares_short_formatted": f"{shares_short:,.0f}주" if shares_short else "N/A",
            "short_mom_change": f"{short_mom_pct:+.2f}%" if short_mom_pct is not None else "N/A",
            "squeeze_risk_level": squeeze_risk
        }
    except Exception:
        intel["short_intel"] = {
            "short_percent_of_float": "N/A",
            "short_ratio_days": "N/A",
            "shares_short_formatted": "N/A",
            "short_mom_change": "N/A",
            "squeeze_risk_level": "N/A"
        }

    return intel

def fetch_ownership_and_shorts(stock, info):
    data = {
        "insider_own": "N/A",
        "insider_trans": "N/A",
        "inst_own": "N/A",
        "inst_trans": "N/A"
    }
    try:
        ins_own_val = info.get("heldPercentInsiders", None)
        if ins_own_val is not None:
            data["insider_own"] = f"{ins_own_val * 100:.2f}%"
            
        inst_own_val = info.get("heldPercentInstitutions", None)
        if inst_own_val is not None:
            data["inst_own"] = f"{inst_own_val * 100:.2f}%"
    except Exception:
        pass

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

def fetch_earnings_calendar(stock, info, high_52_calc, low_52_calc):
    earnings_date_str = "미정"
    d_day_str = ""
    try:
        cal = stock.calendar
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
    except Exception:
        pass

    high_52w = info.get("fiftyTwoWeekHigh", None) or high_52_calc
    low_52w = info.get("fiftyTwoWeekLow", None) or low_52_calc

    return {
        "earnings_date": earnings_date_str,
        "d_day": d_day_str,
        "fiftyTwoWeekHigh": high_52w,
        "fiftyTwoWeekLow": low_52w
    }

def fetch_fundamentals_and_valuation(ticker: str, curr_price: float, high_52_calc, low_52_calc):
    stock = yf.Ticker(ticker)
    info, info_source = get_stock_info_with_retry(stock, retries=3)

    fast_info = {}
    try:
        if hasattr(stock, 'fast_info') and stock.fast_info:
            fast_info = stock.fast_info
    except Exception:
        pass

    market_cap = info.get("marketCap", None)
    if not market_cap and fast_info:
        market_cap = getattr(fast_info, 'market_cap', None) or fast_info.get('market_cap', "N/A")

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
    hedge_and_short_intel = fetch_hedge_funds_and_short_intel(stock, info)
    earnings_cal = fetch_earnings_calendar(stock, info, high_52_calc, low_52_calc)

    earnings_growth = info.get("earningsGrowth", None)
    if earnings_growth and earnings_growth > 0:
        est_growth = min(earnings_growth * 100, 35.0)
    else:
        est_growth = 15.0

    value_models = {}
    try:
        if eps and bps and eps > 0 and bps > 0:
            value_models["graham"] = round(math.sqrt(22.5 * float(eps) * float(bps)), 2)
        else:
            value_models["graham"] = "산출불가"
    except Exception:
        value_models["graham"] = "산출불가"

    try:
        if eps and eps > 0 and roe_raw and roe_raw > 0:
            value_models["peter_lynch"] = round(float(eps) * min(float(roe_raw) * 100, 25.0), 2)
        else:
            value_models["peter_lynch"] = "산출불가"
    except Exception:
        value_models["peter_lynch"] = "산출불가"

    try:
        if bps and bps > 0 and roe_raw and roe_raw > 0:
            value_models["roe_pbr"] = round(float(bps) * (float(roe_raw) / 0.10), 2)
        else:
            value_models["roe_pbr"] = "산출불가"
    except Exception:
        value_models["roe_pbr"] = "산출불가"

    growth_models = {}
    f_eps = forward_eps if forward_eps and forward_eps > 0 else eps
    try:
        if f_eps and f_eps > 0:
            growth_models["forward_peg"] = round(float(f_eps) * (est_growth * 1.5), 2)
        else:
            growth_models["forward_peg"] = "산출불가"
    except Exception:
        growth_models["forward_peg"] = "산출불가"

    try:
        if revenue_per_share and revenue_per_share > 0:
            growth_models["psr_target"] = round(float(revenue_per_share) * 5.0, 2)
        else:
            growth_models["psr_target"] = "산출불가"
    except Exception:
        growth_models["psr_target"] = "산출불가"

    try:
        if f_eps and f_eps > 0:
            wacc = 0.09
            g_long = 0.025
            pv_sum = 0
            cur_cf = float(f_eps)
            for y in range(1, 6):
                cur_cf *= (1 + est_growth / 100)
                pv_sum += cur_cf / ((1 + wacc) ** y)
            terminal_val = (cur_cf * (1 + g_long)) / (wacc - g_long)
            pv_terminal = terminal_val / ((1 + wacc) ** 5)
            growth_models["dcf_growth"] = round(pv_sum + pv_terminal, 2)
        else:
            growth_models["dcf_growth"] = "산출불가"
    except Exception:
        growth_models["dcf_growth"] = "산출불가"

    return {
        "info_source": info_source,
        "market_cap_fmt": format_market_cap(market_cap),
        "trailing_pe": round(trailing_pe, 2) if isinstance(trailing_pe, (int, float)) else trailing_pe,
        "forward_pe": round(forward_pe, 2) if isinstance(forward_pe, (int, float)) else forward_pe,
        "pbr": round(pbr, 2) if isinstance(pbr, (int, float)) else pbr,
        "ps_ratio": round(ps_ratio, 2) if isinstance(ps_ratio, (int, float)) else ps_ratio,
        "roe": f"{roe_pct}%" if roe_pct != "N/A" else "N/A",
        "target_mean_price": target_mean_price,
        "ownership_and_shorts": ownership_and_shorts,
        "hedge_and_short_intel": hedge_and_short_intel,
        "earnings_calendar": earnings_cal,
        "value_models": value_models,
        "growth_models": growth_models
    }

# -------------------------------------------------------------
# 📌 S&P 500 11개 전 섹터 5일/1개월 수익률 수집
# -------------------------------------------------------------
def fetch_sector_performance():
    sector_etfs = [
        ("XLK", "IT/기술 (Technology)"),
        ("XLC", "커뮤니케이션 (Communication Services)"),
        ("XLY", "임의소비재 (Consumer Discretionary)"),
        ("XLP", "필수소비재 (Consumer Staples)"),
        ("XLF", "금융 (Financials)"),
        ("XLV", "헬스케어 (Health Care)"),
        ("XLI", "산업재 (Industrials)"),
        ("XLE", "에너지 (Energy)"),
        ("XLB", "소재 (Materials)"),
        ("XLU", "유틸리티 (Utilities)"),
        ("XLRE", "부동산 (Real Estate)")
    ]
    summary = {}
    for etf, name in sector_etfs:
        try:
            hist = yf.Ticker(etf).history(period="1mo")
            if len(hist) >= 2:
                pct_5d = ((hist['Close'].iloc[-1] - hist['Close'].iloc[-5]) / hist['Close'].iloc[-5] * 100) if len(hist) >= 5 else ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100)
                pct_1m = ((hist['Close'].iloc[-1] - hist['Close'].iloc[0]) / hist['Close'].iloc[0] * 100)
                summary[etf] = {
                    "sector_name": name,
                    "return_5d": f"{pct_5d:+.2f}%",
                    "return_1m": f"{pct_1m:+.2f}%",
                    "latest_close": round(float(hist['Close'].iloc[-1]), 2)
                }
            else:
                summary[etf] = {"sector_name": name, "return_5d": "N/A", "return_1m": "N/A", "latest_close": "N/A"}
        except Exception:
            summary[etf] = {"sector_name": name, "return_5d": "N/A", "return_1m": "N/A", "latest_close": "N/A"}
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

# -------------------------------------------------------------
# 📌 히스토리용 전략 요약 함수
# -------------------------------------------------------------
def summarize_user_strategy(raw_text: str) -> str:
    if not raw_text or raw_text == "분석 리포트 참조":
        return "분석 리포트 참조"
        
    text = raw_text.replace("\n", " ").strip()
    
    if "보유자:" in text or "미보유자:" in text:
        parts = []
        if "보유자:" in text:
            holder_part = text.split("보유자:")[1].split("미보유자:")[0].strip()
            first_sen = re.split(r'[.!?]\s+', holder_part)[0].strip()
            if first_sen:
                parts.append(f"보유: {first_sen}")
        if "미보유자:" in text:
            non_holder_part = text.split("미보유자:")[1].strip()
            first_sen = re.split(r'[.!?]\s+', non_holder_part)[0].strip()
            if first_sen:
                parts.append(f"신규: {first_sen}")
        if parts:
            res = " | ".join(parts)
            return res[:130] + "..." if len(res) > 130 else res

    sentences = [s.strip() for s in re.split(r'[.!?]\s+', text) if len(s.strip()) > 5]
    if not sentences:
        return text[:110]
        
    summary = ". ".join(sentences[:2])
    if len(summary) > 120:
        summary = summary[:120] + "..."
    elif not summary.endswith("."):
        summary += "."
    return summary

# -------------------------------------------------------------
# 📌 신규 진입 적격성 평가 파서
# -------------------------------------------------------------
def parse_full_trading_scenario(text):
    action = "홀딩"
    entry_grade = "분석 리포트 참조"
    entry_rr = "분석 리포트 참조"
    target_1 = "분석 리포트 참조"
    target_2 = ""
    sell_target = "분석 리포트 참조"
    buy_band = "분석 리포트 참조"
    stop_loss = "분석 리포트 참조"
    pyramiding = ""
    user_strategy_raw = ""

    match_action = re.search(r"\[최종\s*투자의견\s*[:\-]?\s*([^\]]+)\]", text)
    if match_action:
        op_text = match_action.group(1).strip()
        if "매수" in op_text and "관망" not in op_text and "보유" not in op_text:
            action = "매수"
        elif "매도" in op_text or "비중축소" in op_text or "차익실현" in op_text:
            action = "매도"
        elif "홀딩" in op_text or "보유" in op_text or "관망" in op_text:
            action = "홀딩"
    else:
        for line in text.split("\n"):
            line_str = line.replace("*", "").replace("#", "").replace("-", "").replace("•", "").strip()
            if "최종 투자 의견" in line_str or "최종 투자의견" in line_str:
                if "적극매수" in line_str or "분할매수" in line_str:
                    action = "매수"
                elif "비중축소" in line_str or "매도" in line_str or "차익실현" in line_str:
                    action = "매도"
                elif "홀딩" in line_str or "보유" in line_str or "관망" in line_str:
                    action = "홀딩"
                break

    match_entry_grade = re.search(r"\[신규\s*진입\s*적격성\s*평가\s*[:\-]?\s*([^\]]+)\]", text)
    if match_entry_grade:
        entry_grade = match_entry_grade.group(1).strip()
    
    for line in text.split("\n"):
        line_clean = line.replace("*", "").replace("-", "").replace("•", "").strip()
        if ("신규 진입 등급" in line_clean or "진입 등급" in line_clean) and entry_grade == "분석 리포트 참조":
            if ":" in line_clean:
                entry_grade = ":".join(line_clean.split(":")[1:]).strip()
        elif "예상 손익비" in line_clean or "손익비" in line_clean:
            if ":" in line_clean:
                entry_rr = ":".join(line_clean.split(":")[1:]).strip()
        elif "사용자 대응 전략" in line_clean or "사용자대응전략" in line_clean:
            if ":" in line_clean:
                user_strategy_raw = ":".join(line_clean.split(":")[1:]).strip()
            else:
                user_strategy_raw = line_clean.replace("사용자 대응 전략", "").replace("사용자대응전략", "").strip(" -:\t")

    scenario_block = text
    if "[정밀 매매 시나리오]" in text:
        after_header = text.split("[정밀 매매 시나리오]")[1]
        if "[최종 투자의견" in after_header:
            scenario_block = after_header.split("[최종 투자의견")[0]
        else:
            scenario_block = after_header

    for line in scenario_block.split("\n"):
        line_clean = line.replace("*", "").replace("-", "").replace("•", "").strip()
        
        if "1차 목표가" in line_clean or "1차목표가" in line_clean:
            if ":" in line_clean:
                target_1 = ":".join(line_clean.split(":")[1:]).strip()
        elif "2차 목표가" in line_clean or "2차목표가" in line_clean:
            if ":" in line_clean:
                target_2 = ":".join(line_clean.split(":")[1:]).strip()
        elif ("목표가" in line_clean or "익절 라인" in line_clean or "익절/" in line_clean) and target_1 == "분석 리포트 참조":
            if ":" in line_clean:
                target_1 = ":".join(line_clean.split(":")[1:]).strip()
        elif "매도가 밴드" in line_clean or "비중축소(익절) 밴드" in line_clean or "비중축소 밴드" in line_clean or "매도가" in line_clean:
            if ":" in line_clean:
                sell_target = ":".join(line_clean.split(":")[1:]).strip()
        elif "분할 매수 밴드" in line_clean or "분할매수 밴드" in line_clean or "분할 매수" in line_clean:
            if ":" in line_clean:
                buy_band = ":".join(line_clean.split(":")[1:]).strip()
        elif "손절" in line_clean or "Stop-loss" in line_clean:
            if ":" in line_clean:
                stop_loss = ":".join(line_clean.split(":")[1:]).strip()
        elif "불타기 조건" in line_clean or "불타기" in line_clean or "추가 매수 조건" in line_clean:
            if ":" in line_clean:
                pyramiding = ":".join(line_clean.split(":")[1:]).strip()

    user_strategy_summary = summarize_user_strategy(user_strategy_raw)
    return action, entry_grade, entry_rr, target_1, target_2, sell_target, buy_band, stop_loss, pyramiding, user_strategy_summary

# -------------------------------------------------------------
# 📌 증권사 투자의견, 목표가 & 티어 분류기
# -------------------------------------------------------------
TIER_1_FIRMS = [
    "goldman", "morgan stanley", "jpmorgan", "jp morgan", "citi", "citigroup",
    "bank of america", "bofa", "merrill", "ubs", "barclays", "deutsche bank",
    "hsbc", "bernstein", "credit suisse", "bnp paribas"
]

TIER_2_FIRMS = [
    "wells fargo", "rbc", "mizuho", "jefferies", "piper sandler", "wedbush",
    "baird", "oppenheimer", "bmo", "stifel", "td cowen", "cowen", "wolfe",
    "keybanc", "raymond james", "canaccord", "evercore", "truist", "guggenheim",
    "btig", "da davidson", "needham", "mmpm", "loop capital", "roth mkm", "bernstein"
]

def classify_analyst_tier(firm_name: str):
    f_lower = firm_name.lower()
    if any(k in f_lower for k in TIER_1_FIRMS):
        return "🌟 Tier 1 (글로벌 탑티어)", 1
    elif any(k in f_lower for k in TIER_2_FIRMS):
        return "✨ Tier 2 (주요 전문리서치)", 2
    else:
        return "🔎 Tier 3 (독립/부티크)", 3

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
            filtered = upgrades.head(8)
            
        records = []
        for idx, row in filtered.head(8).iterrows():
            date_str = idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx)[:10]
            firm_name = str(row.get("Firm", "N/A")).strip()
            from_g = str(row.get("FromGrade", "")).strip()
            to_g = str(row.get("ToGrade", "")).strip()
            action_raw = str(row.get("Action", "N/A")).strip()
            
            if from_g and from_g.lower() != "nan" and from_g != to_g:
                grade_str = f"{from_g} ➡️ {to_g}"
            else:
                grade_str = to_g if to_g and to_g.lower() != "nan" else "N/A"

            target_val = row.get("currentPriceTarget", None) or row.get("priceTarget", None) or row.get("TargetPrice", None)
            target_str = f"${float(target_val):.2f}" if pd.notnull(target_val) and target_val != "" else "-"

            tier_badge, tier_num = classify_analyst_tier(firm_name)

            records.append({
                "date": date_str,
                "firm": firm_name,
                "tier": tier_badge,
                "tier_num": tier_num,
                "action": action_raw,
                "grade_change": grade_str,
                "target_price": target_str
            })
        return records
    except Exception:
        return []

# -------------------------------------------------------------
# 2. 사이드바 UI
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ **AI Stock Analyst Pro**")
    
    MODEL_OPTIONS = {
        "Gemini 3.1 Flash Lite": "gemini-3.1-flash-lite",
        "Gemini 3.6 Flash Lite": "gemini-3.6-flash-lite",
        "Gemini 3.6 Flash": "gemini-3.6-flash"
    }
    
    selected_model_label = st.selectbox(
        "🤖 **AI 추론 모델 선택**",
        options=list(MODEL_OPTIONS.keys()),
        index=0
    )
    selected_model_id = MODEL_OPTIONS[selected_model_label]
    
    st.markdown("---")
    ticker_input = st.text_input("종목 티커 (Ticker)", value=st.session_state.selected_ticker).upper()
    
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
    analyze_btn = st.button("🚀 분석 & 백테스팅 실행", type="primary", use_container_width=True)
    st.divider()

    # 📌 종목별 트레이딩 히스토리
    st.markdown("#### 📌 **종목별 트레이딩 히스토리**")
    
    if st.session_state.history:
        tab_all, tab_buy, tab_sell, tab_hold = st.tabs(["전체", "🟢매수", "🔴매도", "🟡홀딩"])
        
        def render_history_card(t_code, data):
            action_badge = "🟢 매수" if data['action'] == "매수" else ("🔴 매도" if data['action'] == "매도" else "🟡 홀딩")
            with st.expander(f"**{t_code}** (${data['price']}) | {action_badge}", expanded=False):
                st.markdown(f"- **현재가:** `${data['price']}`")
                
                entry_grade_val = data.get('entry_grade', '')
                if entry_grade_val and entry_grade_val != "분석 리포트 참조":
                    st.markdown(f"- **🆕 신규 진입 판정:** `{entry_grade_val}`")
                
                if data.get('my_avg', 0) > 0:
                    st.markdown(f"- **💼 내 평단:** `${data['my_avg']}` ({data.get('my_return', 'N/A')})")
                
                t1 = data.get('target_1') or data.get('take_profit', '분석 리포트 참조')
                t2 = data.get('target_2', '')
                st.markdown(f"- **🎯 1차 목표가:** `{t1}`")
                if t2:
                    st.markdown(f"- **🎯 2차 목표가:** `{t2}`")
                    
                st.markdown(f"- **📤 매도가 밴드:** `{data.get('sell_target', '분석 리포트 참조')}`")
                st.markdown(f"- **📥 분할매수 밴드:** `{data.get('buy_band', '분석 리포트 참조')}`")
                st.markdown(f"- **🛑 손절선:** `{data.get('stop_loss', '분석 리포트 참조')}`")
                
                if data.get('pyramiding'):
                    st.markdown(f"- **🔥 불타기 조건:** `{data['pyramiding']}`")
                
                strat_text = data.get('user_strategy', '')
                if strat_text and strat_text != "분석 리포트 참조":
                    st.markdown(f"- **💡 대응 전략:** `{strat_text}`")
                else:
                    st.markdown(f"- **💡 대응 전략:** `분석 리포트 참조`")
                    
                st.caption(f"분석 일시(KST): {data.get('time', 'N/A')}")

        with tab_all:
            for t_code, data in list(st.session_state.history.items())[::-1]:
                render_history_card(t_code, data)
                
        with tab_buy:
            buy_items = [item for item in list(st.session_state.history.items())[::-1] if item[1]['action'] == "매수"]
            if buy_items:
                for t_code, data in buy_items:
                    render_history_card(t_code, data)
            else:
                st.caption("매수 판정 종목이 없습니다.")

        with tab_sell:
            sell_items = [item for item in list(st.session_state.history.items())[::-1] if item[1]['action'] == "매도"]
            if sell_items:
                for t_code, data in sell_items:
                    render_history_card(t_code, data)
            else:
                st.caption("매도 판정 종목이 없습니다.")

        with tab_hold:
            hold_items = [item for item in list(st.session_state.history.items())[::-1] if item[1]['action'] == "홀딩"]
            if hold_items:
                for t_code, data in hold_items:
                    render_history_card(t_code, data)
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
# 3. 메인 분석 화면 & 분석 완료 즉시 렌더링 로직
# -------------------------------------------------------------
st.header(f"📊 {ticker_input} 종합 밸류에이션 & 정밀 트레이딩 리포트")

# [1] 분석 버튼 클릭 시 실행
if analyze_btn:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            api_key = None
            
    with st.spinner(f"🔍 [{ticker_input}] 헤지펀드 지분/공매도 세력 인텔/11개 섹터 수급/VWAP 분석 및 백테스팅 실행 중..."):
        tech_data, stock_date, fib_levels, high_52_calc, low_52_calc, raw_df = fetch_stock_technical_data(ticker_input)
        backtest_results = run_strategy_backtest(raw_df)
        options_data = fetch_nearest_options_data(ticker_input)
        macro_data = fetch_macro_indicators()
        
        curr_p = tech_data.get('current_price', 0)
        fund_data = fetch_fundamentals_and_valuation(ticker_input, curr_p, high_52_calc, low_52_calc)
        sector_data = fetch_sector_performance()
        news_data = fetch_news(ticker_input, limit=5)
        macro_news_data = fetch_macro_news(limit=4)
        analyst_data = fetch_recent_upgrades_downgrades(ticker_input, months=2)
        
        info_source_flag = fund_data.get('info_source', 'stock.info')
        ownership = fund_data.get('ownership_and_shorts', {})
        hedge_short_intel = fund_data.get('hedge_and_short_intel', {})
        earnings_info = fund_data.get('earnings_calendar', {})
        
        my_return_str = "N/A"
        if is_holding and user_avg_price > 0 and user_shares > 0 and curr_p > 0:
            total_invested = user_avg_price * user_shares
            total_current = curr_p * user_shares
            pnl_dollar = total_current - total_invested
            pnl_pct = ((curr_p - user_avg_price) / user_avg_price) * 100
            my_return_str = f"{pnl_pct:+.2f}%"

        user_position_text = (
            f"사용자 보유 현황: 평단가 ${user_avg_price:.2f}, 보유수량 {user_shares:.1f}주, 평가수익률 {my_return_str}"
            if is_holding and user_avg_price > 0 else "사용자 미보유 종목 (신규 진입 검토 관점)"
        )

        full_rag_payload = {
            "meta": {
                "ticker": ticker_input,
                "data_source": info_source_flag,
                "model_used": selected_model_label,
                "analysis_requested_at": get_current_kst_time_str(),
                "stock_data_date": stock_date
            },
            "technical_vwap_and_squeeze": tech_data,
            "one_year_backtesting": backtest_results,
            "fibonacci_retracement_6m": fib_levels,
            "options_chain_nearest": options_data,
            "ownership_and_short_interest": ownership,
            "hedge_funds_and_short_intel": hedge_short_intel,
            "earnings_calendar_and_52w": earnings_info,
            "macro_6_assets": macro_data,
            "global_macro_news": macro_news_data,
            "sector_performance_11_sectors": sector_data,
            "fundamentals_and_6_valuations": fund_data,
            "user_portfolio_status": user_position_text,
            "stock_recent_news": news_data,
            "analyst_upgrades_downgrades_2m": analyst_data
        }

        full_json_str = json.dumps(full_rag_payload, indent=2, ensure_ascii=False)

        response_content = None
        if not api_key:
            response_content = "⚠️ GEMINI_API_KEY가 등록되지 않았습니다. 아래 [분석용 JSON 데이터 다운로드] 버튼으로 JSON을 내려받아 분석을 요청하세요."
        else:
            # 📌 헤지펀드 및 공매도 세력 인텔 분석 프롬프트
            template = """
[RAG 심층 주입 데이터]
1. 기술적/수급 및 VWAP, 볼린저 밴드 스퀴즈 ({ticker}) (기준일: {stock_date}):
{tech_json}

2. 최근 1년 과거 데이터 기반 듀얼 전략 백테스팅 결과:
{backtest_json}

3. 최근 6개월 피보나치 되돌림 밴드:
{fib_json}

4. 가장 빠른 만기 옵션 체인 수급 (콜/풋 Max OI & Volume):
{options_json}

5. 내부자/기관 지분율 및 유명 헤지펀드/공매도 세력 인텔 (Short Squeeze Intel):
{hedge_short_json}

6. 실적 발표 일정 및 52주 고저:
{earnings_json}

7. 매크로 및 6대 자산 실시간 지표 (출처 및 기준일 포함):
{macro_json}

8. 글로벌 거시/시장 주요 뉴스:
{macro_news_json}

9. S&P 500 11개 전 섹터 실시간 등락률 및 모멘텀:
{sector_json}

10. 펀더멘털 및 6대 밸류에이션:
{fund_json}

11. 사용자 보유 현황:
{user_position}

12. 종목 최신 주요 기사:
{news_json}

13. 최근 2개월 증권가 투자의견 및 목표가 변동 (기관 신뢰도 티어 포함):
{analyst_json}

---

[지시사항 - 분석 정합성, 11개 섹터 전수 분석 및 헤지펀드/공매도 수급 규칙]
위 데이터를 바탕으로 최고 수준의 퀀트/금융 애널리스트 관점에서 정밀 리포트를 작성할 것:

1. 거시환경 및 시장 국면
- **[참고자료 및 기준일자]**: 분석에 활용된 핵심 매크로 지표의 **출처 및 수집 기준일자**를 요약 명시할 것.
- 경기 국면 판정 및 최신 매크로 지표/뉴스를 직접 인용하여 [6대 유동성 자산 변동 예측] (현금, 채권, 주식, 코인, 금, 원유).
- 권장 자산 배분 비중 (주식 : 채권 : 대체자산 : 현금)

2. 11개 전 섹터 전망 및 자금 순환매 심층 분석 (서식 엄격 준수)
- **11개 섹터 전수 리스트 작성**: 주입된 11개 섹터 데이터(XLK, XLC, XLY, XLP, XLF, XLV, XLI, XLE, XLB, XLU, XLRE) 각각에 대해 5일/1개월 등락률을 바탕으로 현재 상태를 11개 모두 글머리 기호(*)로 작성할 것.
- **자금 순환매 결론 분리 (필수)**: 11개 섹터 글머리 기호 작성이 끝난 후, **반드시 빈 줄(한 줄 공백)을 삽입**하여 아래와 같이 독립된 글머리 기호로 작성할 것:

  * **자금 순환매 결론**: [방어주 vs 성장주 순환매 방향성과 {ticker}가 속한 섹터의 수혜/소외 여부 및 상대 강도를 명확히 도출]

3. 밸류에이션, 스마트머니(헤지펀드) 및 공매도 세력/옵션 분석 ({ticker})
- **유명 헤지펀드 포지션**: 13F 주요 보유 기관(Top Holders)의 지분 집중도와 스마트머니 매집 특성을 분석할 것.
- **공매도 세력 및 숏스퀴즈 리스크**: Short Float, Days to Cover(상환 소요 일수), 월간 공매도 증감율을 결합하여 공매도 세력의 하방 압력 강도 및 숏스퀴즈 촉발 가능성을 평가할 것.
- **IB 투자의견 신뢰도 가중**: Tier 1/2 투자은행의 목표가 변동을 가중 평가할 것.

4. 정밀 기술적 지표, VWAP 및 백테스팅 평가 ({ticker})
- **VWAP & 볼린저 밴드 스퀴즈 판정**: 현재가의 1Y/20D VWAP 위치 및 스퀴즈 상태(Squeeze On/Off)에 따른 폭발 방향성 제시.
- **백테스팅 시사점**: 과거 1년간 모멘텀 스퀴즈 전략과 VWAP 평균회귀 전략 중 어떤 로직이 우세했는지 비교 평가.
- 스코어카드 (각 10점 만점): 성장성, 수익성, 밸류에이션, 해자, 퀀트/모멘텀 | 종합 평점 제시

5. [신규 진입 적격성 평가 (미보유자 관점 핵심 진단)]
- **신규 진입 등급**: [적극 진입 추천 | 조정 시 분할 진입 | 돌파 확인 후 진입 | 진입 부적합(관망/리스크 과다) 중 택1]
- **진입 적합성 종합 판정**: 미보유자 입장에서 현재 시점에 무조건 하방 매수를 기다려야 하는지, 아니면 현재가 부근에서 즉시/분할 진입할 만한 모멘텀과 밸류에이션 메리트가 있는지 객관적이고 냉정하게 평가할 것.
- **예상 손익비 (Risk/Reward Ratio)**: 1차 목표가까지의 기대 상승률 대비 손절선까지의 하방 리스크 비율을 수치로 명시할 것.

6. [정밀 매매 시나리오]
- **[매수 밴드 및 진입 가격의 상·하단 논리 일치 규칙 (필수)]**:
  * **분할 매수 밴드 설정**: 피보나치 지지선, 20일 VWAP, 풋옵션 지지선을 결합하여 실질적 달러 범위를 도출할 것.
  * **하단 [사용자 대응 전략]과의 가격 일치**: 미보유자의 신규 진입가는 반드시 상단 **[분할 매수 밴드]**에서 제시한 가격대와 100% 동일한 수치를 인용할 것.

- **[반드시 아래의 구조 및 순서로 완벽히 동일하게 작성할 것]**:
- **[서식 주의]**: 
  * 비중을 언급할 때는 반드시 '30%'처럼 퍼센트(%) 기호를 붙이고, 단어와 숫자 사이에 반드시 공백을 둘 것.
  * **피보나치 수치 표기 필수 규칙**: 반드시 **'피보나치 50.0%', '피보나치 38.2%', '피보나치 23.6%', '피보나치 61.8%'**와 같이 소수점과 퍼센트(%) 기호를 붙여 정확히 표기할 것.

[신규 진입 적격성 평가]
* **신규 진입 등급**: [적극 진입 추천 | 조정 시 분할 진입 | 돌파 확인 후 진입 | 진입 부적합(관망) 중 택1]
* **진입 적합성 분석**: [미보유자 관점에서 현재 진입의 타당성, 추세 지속성 및 밸류에이션 매력도 분석]
* **예상 손익비 (Risk/Reward)**: [기대수익 % : 손실 % (비율)]

[정밀 매매 시나리오]

* **분할 매수 밴드**: [피보나치 지지선 및 VWAP/옵션 지지선을 결합한 구체적 달러 범위와 근거]
* **1차 목표가**: [피보나치 38.2% 또는 50.0% 구간 구체적 달러 가격대와 근거]
* **2차 목표가**: [피보나치 23.6% 또는 52주 고점 인근 저항 구체적 달러 가격대와 근거]
* **매도가 밴드**: [볼린저밴드 상단 및 피보나치 결합 분할 차익실현 구체적 달러 밴드]
* **손절(Stop-loss) 기준선**: [2.0x ATR 또는 1Y VWAP 이탈 시 추세 훼손 구체적 달러 가격대]
* **불타기 조건**: [스퀴즈 상방 돌파 및 상방 저항선 안착 시 추가 매수 검토 기준]

[최종 투자의견: 적극매수 | 분할매수 | 홀딩(보유) | 비중축소 | 관망 중 택1]

* **사용자 대응 전략**: [보유자의 경우 평단가 및 수익률에 따른 부분 익절/비중관리/홀딩 유지 방안을 구체적 비중(예: 물량의 30% 매도 등)과 지지선/저항선 달러 수치를 공백을 두어 명확히 작성할 것. 미보유자의 경우 상단 '신규 진입 적격성 평가' 및 '분할 매수 밴드'와 정확히 일치하는 진입 가격을 명시할 것]
"""
            prompt = PromptTemplate(
                input_variables=["ticker", "stock_date", "tech_json", "backtest_json", "fib_json", "options_json", "hedge_short_json", "earnings_json", "macro_json", "macro_news_json", "sector_json", "fund_json", "user_position", "news_json", "analyst_json"],
                template=template
            )
            llm = ChatGoogleGenerativeAI(model=selected_model_id, google_api_key=api_key)
            chain = prompt | llm
            
            payload = {
                "ticker": ticker_input,
                "stock_date": stock_date,
                "tech_json": json.dumps(tech_data, indent=2, ensure_ascii=False),
                "backtest_json": json.dumps(backtest_results, indent=2, ensure_ascii=False) if backtest_results else "백테스팅 데이터 부족",
                "fib_json": json.dumps(fib_levels, indent=2, ensure_ascii=False),
                "options_json": json.dumps(options_data, indent=2, ensure_ascii=False) if options_data else "옵션 데이터 없음",
                "hedge_short_json": json.dumps(hedge_short_intel, indent=2, ensure_ascii=False),
                "earnings_json": json.dumps(earnings_info, indent=2, ensure_ascii=False),
                "macro_json": json.dumps(macro_data, indent=2, ensure_ascii=False),
                "macro_news_json": json.dumps(macro_news_data, indent=2, ensure_ascii=False),
                "sector_json": json.dumps(sector_data, indent=2, ensure_ascii=False),
                "fund_json": json.dumps(fund_data, indent=2, ensure_ascii=False),
                "user_position": user_position_text,
                "news_json": json.dumps(news_data, indent=2, ensure_ascii=False),
                "analyst_json": json.dumps(analyst_data, indent=2, ensure_ascii=False)
            }
            
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
                response_content = "⚠️ Gemini API 일시적 지연이 발생했습니다. [분석용 JSON 데이터 다운로드]를 통해 확인하세요."

        if response_content and not response_content.startswith("⚠️"):
            act, ent_grade, ent_rr, t1, t2, sell_b, buy_b, sl_b, pyr, u_strat_summary = parse_full_trading_scenario(response_content)
            st.session_state.history[ticker_input] = {
                "action": act,
                "entry_grade": ent_grade,
                "entry_rr": ent_rr,
                "price": curr_p,
                "my_avg": user_avg_price if is_holding else 0,
                "my_return": my_return_str if is_holding else "미보유",
                "target_1": t1,
                "target_2": t2,
                "take_profit": t1,
                "sell_target": sell_b,
                "buy_band": buy_b,
                "stop_loss": sl_b,
                "pyramiding": pyr,
                "user_strategy": u_strat_summary,
                "time": get_current_kst_time_str()
            }
            save_history(st.session_state.history)

        st.session_state.last_analysis_result = {
            "ticker": ticker_input,
            "info_source": info_source_flag,
            "model_label": selected_model_label,
            "curr_p": curr_p,
            "is_holding": is_holding,
            "user_avg_price": user_avg_price,
            "user_shares": user_shares,
            "my_return_str": my_return_str,
            "tech_data": tech_data,
            "backtest_results": backtest_results,
            "stock_date": stock_date,
            "fib_levels": fib_levels,
            "options_data": options_data,
            "macro_data": macro_data,
            "fund_data": fund_data,
            "sector_data": sector_data,
            "ownership": ownership,
            "hedge_short_intel": hedge_short_intel,
            "earnings_info": earnings_info,
            "response_content": response_content,
            "full_json_str": full_json_str,
            "macro_news_data": macro_news_data,
            "news_data": news_data,
            "analyst_data": analyst_data
        }
        st.rerun()

# [2] 캐시된 분석 결과 렌더링
if st.session_state.last_analysis_result:
    res = st.session_state.last_analysis_result
    curr_p = res["curr_p"]
    ownership = res["ownership"]
    hedge_short_intel = res.get("hedge_short_intel", {})
    earnings_info = res["earnings_info"]
    tech_data = res["tech_data"]
    backtest_results = res.get("backtest_results", None)
    macro_data = res["macro_data"]
    fib_levels = res["fib_levels"]
    options_data = res["options_data"]
    fund_data = res["fund_data"]
    sector_data = res.get("sector_data", {})
    info_source = res.get("info_source", "stock.info")

    if info_source == "stock.info":
        st.markdown("📡 **데이터 소스:** `🟢 Yahoo Finance stock.info` (상세 펀더멘털 & 밸류에이션 정상 수집)")
    else:
        st.markdown("📡 **데이터 소스:** `🟡 Yahoo Finance stock.fast_info` (야후 서버 지연으로 인한 간이 시세 백업 데이터 적용)")

    # 💼 보유 주식 평가손익 컨테이너
    if res["is_holding"] and res["user_avg_price"] > 0 and res["user_shares"] > 0 and curr_p > 0:
        total_invested = res["user_avg_price"] * res["user_shares"]
        total_current = curr_p * res["user_shares"]
        pnl_dollar = total_current - total_invested
        pnl_pct = ((curr_p - res["user_avg_price"]) / res["user_avg_price"]) * 100
        
        with st.container(border=True):
            st.markdown(f"#### 💼 **내 보유 포지션 분석 ({res['ticker']})**")
            p_c1, p_c2, p_c3, p_c4 = st.columns(4)
            p_c1.metric("내 매수 평단가", f"${res['user_avg_price']:,.2f}", f"{res['user_shares']:,.1f}주 보유")
            p_c2.metric("총 매수 원금", f"${total_invested:,.2f}")
            p_c3.metric("현재 평가 금액", f"${total_current:,.2f}")
            p_c4.metric("평가 손익 (수익률)", f"${pnl_dollar:+,.2f}", f"{pnl_pct:+.2f}%")

    # 1. 상단 핵심 시장/재무 메트릭
    with st.container(border=True):
        st.markdown("**🏢 핵심 시장 및 재무 지표**")
        r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
        r1_c1.metric("현재 주가", f"${curr_p}")
        r1_c2.metric("시가총액", str(fund_data.get('market_cap_fmt', 'N/A')))
        r1_c3.metric("PER (선행/후행)", f"{fund_data.get('forward_pe', 'N/A')} / {fund_data.get('trailing_pe', 'N/A')}")
        r1_c4.metric("PBR / PSR", f"{fund_data.get('pbr', 'N/A')} / {fund_data.get('ps_ratio', 'N/A')}")
        
        st.divider()
        
        st.markdown("**👥 스마트머니 기본 지분 (내부자 & 기관 지분율 및 내부자 매매)**")
        own_c1, own_c2, own_c3, own_c4 = st.columns(4)
        own_c1.metric("Insider Own (내부자 지분)", str(ownership.get('insider_own', 'N/A')))
        own_c2.metric("Insider Trans (내부자 매매)", str(ownership.get('insider_trans', 'N/A')))
        own_c3.metric("Inst Own (기관 지분)", str(ownership.get('inst_own', 'N/A')))
        own_c4.metric("Inst Trans (기관 매매/보유)", str(ownership.get('inst_trans', 'N/A')))

        st.divider()

        st.markdown("**📅 차기 실적 발표 일정, 52주 가격 범위 & ATR/모멘텀**")
        s_c1, s_c2, s_c3, s_c4 = st.columns(4)
        s_c1.metric("차기 실적 발표일", str(earnings_info.get('earnings_date', 'N/A')), str(earnings_info.get('d_day', '')))
        
        high_52 = earnings_info.get('fiftyTwoWeekHigh', 'N/A')
        low_52 = earnings_info.get('fiftyTwoWeekLow', 'N/A')
        diff_52h = round(((curr_p - high_52) / high_52) * 100, 1) if isinstance(high_52, (int, float)) and curr_p else None
        s_c2.metric("52주 최고 / 최저가", f"${high_52} / ${low_52}", f"최고가 대비 {diff_52h:+.1f}%" if diff_52h is not None else None)
        
        s_c3.metric("14일 ATR (일일 변동폭)", f"${tech_data.get('atr_14', 'N/A')}", f"2.0x 손절: ${tech_data.get('atr_stop_2_0x', 'N/A')}")
        s_c4.metric(
            "MACD (Signal)", 
            f"{tech_data.get('macd', 'N/A')} ({tech_data.get('macd_signal', 'N/A')})", 
            f"Hist: {tech_data.get('macd_hist', 'N/A'):+}" if isinstance(tech_data.get('macd_hist'), (int, float)) else None
        )

    # S&P 500 11개 전 섹터 실시간 등락 현황판 (기본 접기: expanded=False)
    if sector_data:
        with st.expander("🧭 **S&P 500 11개 전 섹터 실시간 등락 및 순환매 현황 (11 Sectors Rotation) [클릭하여 펼치기]**", expanded=False):
            s_rows = []
            for etf, s_info in sector_data.items():
                s_rows.append({
                    "티커": etf,
                    "섹터명": s_info.get("sector_name", ""),
                    "5일 등락률": s_info.get("return_5d", "N/A"),
                    "1개월 등락률": s_info.get("return_1m", "N/A"),
                    "현재가 ($)": f"${s_info.get('latest_close', 'N/A')}"
                })
            
            df_sec = pd.DataFrame(s_rows)
            st.dataframe(df_sec, use_container_width=True, hide_index=True)

    # 2. 퀀트 모멘텀 & VWAP / 볼린저 밴드 스퀴즈 컨테이너
    with st.container(border=True):
        st.markdown("##### 🧪 **퀀트 모멘텀, 스마트머니 VWAP & 볼린저 밴드 스퀴즈**")
        
        q_c1, q_c2, q_c3, q_c4 = st.columns(4)
        
        vwap_1y = tech_data.get('vwap_1y', 'N/A')
        diff_vwap1y = round(((curr_p - vwap_1y) / vwap_1y) * 100, 1) if isinstance(vwap_1y, (int, float)) and curr_p else None
        q_c1.metric("1Y 누적 VWAP (장기 평단)", f"${vwap_1y}", f"현재가 {diff_vwap1y:+.1f}%" if diff_vwap1y is not None else None)

        vwap_20d = tech_data.get('vwap_20d', 'N/A')
        diff_vwap20 = round(((curr_p - vwap_20d) / vwap_20d) * 100, 1) if isinstance(vwap_20d, (int, float)) and curr_p else None
        q_c2.metric("20일 단기 VWAP (스마트머니)", f"${vwap_20d}", f"현재가 {diff_vwap20:+.1f}%" if diff_vwap20 is not None else None)

        bb_w = tech_data.get('bb_width_pct', 'N/A')
        q_c3.metric("볼린저 밴드폭 (Bandwidth)", f"{bb_w}%" if bb_w != "N/A" else "N/A", "변동성 압축도")
        
        q_c4.metric("RSI(14) / MFI 수급", f"{tech_data.get('rsi_14', 'N/A')} / {tech_data.get('mfi_14', 'N/A')}")
        
        st.markdown(f"**⚡ 변동성 국면 판정:** `{tech_data.get('bb_squeeze_status', 'N/A')}`")

    # 3. 1년 과거 데이터 기반 듀얼 전략 백테스팅 컨테이너
    if backtest_results:
        with st.container(border=True):
            st.markdown("##### 🔬 **과거 1년 퀀트 전략 백테스팅 시뮬레이션 (1-Year Backtest)**")
            
            bh_ret = backtest_results.get("benchmark_buy_and_hold", 0.0)
            st.caption(f"📌 **벤치마크 (단순 보유 Buy & Hold 1년 수익률):** `{bh_ret:+.2f}%`")
            
            bt_col1, bt_col2 = st.columns(2)
            
            with bt_col1:
                st.markdown("**🚀 전략 A: 모멘텀 스퀴즈 돌파 (Momentum Squeeze Breakout)**")
                st.caption("진입: MACD 상방전환 + 20일 이평 및 20일 VWAP 상회 시 | 청산: MACD 꺾임 또는 1.5x ATR 이탈")
                s1 = backtest_results.get("strategy_1_momentum_squeeze", {})
                
                m1_1, m1_2, m1_3 = st.columns(3)
                m1_1.metric("총 누적 수익률", f"{s1.get('total_ret', 0):+.2f}%", f"B&H 대비 {round(s1.get('total_ret', 0) - bh_ret, 2):+.2f}%p")
                m1_2.metric("승률 (Win Rate)", f"{s1.get('win_rate', 0)}%", f"총 {s1.get('trades_count', 0)}회 매매")
                m1_3.metric("Profit Factor / MDD", f"{s1.get('profit_factor', 0)}", f"MDD: -{s1.get('mdd', 0)}%")

            with bt_col2:
                st.markdown("**🔄 전략 B: 1Y VWAP + RSI 밸류 되돌림 (Mean Reversion)**")
                st.caption("진입: 1Y 누적 VWAP 하회 + RSI 42 이하 + 볼린저 하단 지지 | 청산: VWAP 도달 또는 RSI 65")
                s2 = backtest_results.get("strategy_2_vwap_mean_reversion", {})
                
                m2_1, m2_2, m2_3 = st.columns(3)
                m2_1.metric("총 누적 수익률", f"{s2.get('total_ret', 0):+.2f}%", f"B&H 대비 {round(s2.get('total_ret', 0) - bh_ret, 2):+.2f}%p")
                m2_2.metric("승률 (Win Rate)", f"{s2.get('win_rate', 0)}%", f"총 {s2.get('trades_count', 0)}회 매매")
                m2_3.metric("Profit Factor / MDD", f"{s2.get('profit_factor', 0)}", f"MDD: -{s2.get('mdd', 0)}%")

    # 4. 피보나치 되돌림 컨테이너
    with st.container(border=True):
        st.markdown(f"##### 📐 **최근 6개월 피보나치 되돌림 지지/저항 밴드** (최고: `${fib_levels.get('high_6m', 'N/A')}` / 최저: `${fib_levels.get('low_6m', 'N/A')}`)")
        fb1, fb2, fb3, fb4 = st.columns(4)
        
        f236 = fib_levels.get('fib_23.6%', 'N/A')
        f382 = fib_levels.get('fib_38.2%', 'N/A')
        f500 = fib_levels.get('fib_50.0%', 'N/A')
        f618 = fib_levels.get('fib_61.8%', 'N/A')
        
        fb1.metric("23.6% 되돌림 (단기 지지)", f"${f236}", f"{round(((f236-curr_p)/curr_p)*100, 1):+.1f}%" if isinstance(f236, (int, float)) and curr_p else None)
        fb2.metric("38.2% 되돌림 (1차 매수 지지)", f"${f382}", f"{round(((f382-curr_p)/curr_p)*100, 1):+.1f}%" if isinstance(f382, (int, float)) and curr_p else None)
        fb3.metric("50.0% 하프라인 (추세 기준선)", f"${f500}", f"{round(((f500-curr_p)/curr_p)*100, 1):+.1f}%" if isinstance(f500, (int, float)) and curr_p else None)
        fb4.metric("61.8% 되돌림 (강력한 2차 지지)", f"${f618}", f"{round(((f618-curr_p)/curr_p)*100, 1):+.1f}%" if isinstance(f618, (int, float)) and curr_p else None)

    # 5. 옵션 체인 컨테이너
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

    # 6. 성장주 3대 밸류에이션 모델
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
        g3.metric("PSR 타깃 매출가치 (5배)", f"${psr_p}" if isinstance(psr_p, (int, float)) else str(psr_p), f"{diff_psr:+.1f}%" if diff_psr is not None else None)
        
        dcf_p = g_models.get('dcf_growth', 'N/A')
        diff_dcf = round(((dcf_p - curr_p) / curr_p) * 100, 1) if isinstance(dcf_p, (int, float)) and curr_p else None
        g4.metric("2단계 DCF 현금흐름 모델", f"${dcf_p}" if isinstance(dcf_p, (int, float)) else str(dcf_p), f"{diff_dcf:+.1f}%" if diff_dcf is not None else None)

    # 7. 전통 가치투자 모델
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

    st.caption(f"🕒 데이터 수집 기준일자: 주가/재무제표 ({res['stock_date']}) | FRED 국채금리 ({macro_data.get('us_10y_yield', {}).get('date', 'N/A')})")

    # 8. AI 종합 분석 브리핑 & JSON 다운로드 버튼
    with st.container(border=True):
        head_col1, head_col2 = st.columns([0.65, 0.35])
        with head_col1:
            st.markdown(f"### 📝 **{res['model_label']} 종합 분석 브리핑**")
        with head_col2:
            st.download_button(
                label=f"📥 {res['ticker']} 분석용 JSON 다운로드",
                data=res["full_json_str"],
                file_name=f"{res['ticker']}_rag_analysis_data_{datetime.now(KST).strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )
        clean_rendered_content = re.sub(r'(?<!\\)\$', r'\$', res["response_content"])
        st.markdown(clean_rendered_content)

    with st.expander("🌐 **6대 유동성 자산 분석 참고 거시 기사 & 원문 링크 (클릭하여 접기/펼치기)**", expanded=False):
        if res.get("macro_news_data"):
            for m_item in res.get("macro_news_data", []):
                st.markdown(f"- **[{m_item['title']}]({m_item['link']})**")
                if m_item.get("summary"):
                    st.caption(f"> {m_item['summary']}")
                st.caption(f"출처: {m_item['publisher']} | 게시일: {m_item['date']}")
                st.write("")
        else:
            st.info("수집된 거시경제 기사가 없습니다.")
    
    st.write("")

    col_left, col_right = st.columns([0.9, 1.1])
    
    with col_left:
        with st.container(border=True):
            st.markdown(f"##### 📰 **{res['ticker']} 최신 주요 뉴스 및 기사 원문**")
            if res.get("news_data"):
                for item in res.get("news_data", []):
                    st.markdown(f"**[{item['title']}]({item['link']})**")
                    if item.get("summary"):
                        st.markdown(f"> *{item['summary']}*")
                    st.caption(f"출처: {item['publisher']} | {item['date']}")
                    st.divider()
            else:
                st.info("수집된 최신 뉴스가 없습니다.")
                
    with col_right:
        # 1. 최근 2개월 증권가 투자의견 변동
        with st.container(border=True):
            st.markdown(f"##### 🏛️ **{res['ticker']} 최근 2개월 증권가 투자의견 및 목표가 변동**")
            if res.get("analyst_data"):
                df_analyst = pd.DataFrame(res.get("analyst_data", []))
                display_cols = ["date", "firm", "tier", "action", "grade_change", "target_price"]
                df_analyst = df_analyst[[c for c in display_cols if c in df_analyst.columns]]
                df_analyst.columns = ["일자", "증권사", "기관 신뢰도 등급", "구분", "투자의견 변동", "제시 목표가"]
                st.dataframe(df_analyst, use_container_width=True, hide_index=True)
            else:
                st.info("최근 2개월간 등록된 투자의견 변동 내역이 없습니다.")

        # 📌 [통합 완료] 2. 헤지펀드 13F 보유 현황 & 공매도(Shorts) 수급 정밀 인텔
        with st.container(border=True):
            st.markdown(f"##### 🐋 **{res['ticker']} 13F 헤지펀드 지분 & 공매도(Shorts) 수급 정밀 인텔**")
            
            s_intel = hedge_short_intel.get("short_intel", {})
            
            sk1, sk2, sk3 = st.columns(3)
            sk1.metric("공매도 잔고 (Float)", s_intel.get("short_percent_of_float", "N/A"), f"MoM: {s_intel.get('short_mom_change', 'N/A')}")
            sk2.metric("상환 소요 일수 (DTC)", s_intel.get("short_ratio_days", "N/A"), "Days to Cover")
            sk3.metric("공매도 총 주수", s_intel.get("shares_short_formatted", "N/A"))
            
            st.markdown(f"**🎯 숏스퀴즈 리스크 등급:** `{s_intel.get('squeeze_risk_level', 'N/A')}`")
            
            # 실적 발표 및 52주 고점 괴리율 캡션 연계
            e_date = earnings_info.get('earnings_date', '미정')
            e_dday = earnings_info.get('d_day', '')
            st.caption(f"📅 차기 실적 발표: **{e_date} ({e_dday})** | 52주 고점 괴리율: **{diff_52h:+.1f}%**" if diff_52h is not None else f"📅 차기 실적 발표: **{e_date}**")
            
            st.divider()
            
            st.markdown("**📋 13F 상위 기관 & 헤지펀드 보유 내역 (Top Holders)**")
            holders_list = hedge_short_intel.get("top_holders", [])
            if holders_list:
                df_holders = pd.DataFrame(holders_list)
                df_holders.columns = ["기관/펀드명", "보유 주식수", "지분율 (% Out)", "평가 가치"]
                st.dataframe(df_holders, use_container_width=True, hide_index=True)
            else:
                st.caption("수집된 상위 기관/헤지펀드 13F 지분 데이터가 없습니다.")