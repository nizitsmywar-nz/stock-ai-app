# =============================================================================
# [BLOCK 01] 라이브러리 임포트 및 전역 환경 설정
# =============================================================================
import os
import json
import math
import time
import re
import warnings
from datetime import datetime, timedelta, timezone

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

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

# -------------------------------------------------------------
# 💡 [신규 추가] 커넥션 풀링 & User-Agent 세션 생성 (야후 봇 차단 방어)
# -------------------------------------------------------------
def init_global_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5"
    })
    retries = Retry(
        total=2,
        backoff_factor=0.3,
        status_forcelist=[429, 500, 502, 503, 504],
        raise_on_status=False
    )
    adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20, max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

GLOBAL_SESSION = init_global_session()

st.set_page_config(
    page_title="AI Stock Valuation Dashboard Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# [BLOCK 02] 커스텀 CSS 및 UI 스타일링
# =============================================================================
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

# =============================================================================
# [BLOCK 03] 데이터 영구 저장소 및 세션 상태 관리
# =============================================================================
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

# =============================================================================
# [BLOCK 04] 기술적 지표 & 퀀트 매물대 연산 엔진
# =============================================================================
def get_stock_info_with_retry(stock, retries=2):
    for attempt in range(retries):
        try:
            info = stock.info
            if isinstance(info, dict) and len(info) > 10 and any(k in info for k in ['marketCap', 'trailingPE', 'forwardPE', 'trailingEps', 'bookValue', 'currentPrice']):
                return info, "stock.info"
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(0.5 * (attempt + 1))
            
    try:
        fallback_info = stock.info or {}
        if fallback_info and len(fallback_info) > 5:
            return fallback_info, "stock.info"
    except Exception:
        pass
    return {}, "stock.fast_info"

def fetch_stock_technical_data(ticker: str):
    stock = yf.Ticker(ticker, session=GLOBAL_SESSION)
    df = stock.history(period="1y")
    if df.empty:
        df = stock.history(period="6mo")
    if df.empty:
        return {}, "N/A", {}, "N/A", "N/A", pd.DataFrame(), {}
    
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
        has_valid_atr = df['ATR'].notna().any()
    except Exception:
        df['ATR'] = np.nan
        has_valid_atr = False

    try:
        df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
    except Exception:
        df['MFI'] = np.nan

    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid'] * 100

    if has_valid_atr:
        df['KC_High'] = df['SMA_20'] + (1.5 * df['ATR'])
        df['KC_Low'] = df['SMA_20'] - (1.5 * df['ATR'])
        df['BB_Squeeze_On'] = (df['BB_High'] < df['KC_High']) & (df['BB_Low'] > df['KC_Low'])
    else:
        df['KC_High'] = np.nan
        df['KC_Low'] = np.nan
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
    
    volume_profile = {}
    try:
        num_bins = 30
        price_bins = np.linspace(low_6m, high_6m, num_bins + 1)
        bin_indices = np.digitize(df_6m['Close'], price_bins) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        
        vol_by_bin = np.zeros(num_bins)
        for idx_val, vol_val in zip(bin_indices, df_6m['Volume']):
            vol_by_bin[idx_val] += vol_val
            
        poc_idx = int(np.argmax(vol_by_bin))
        poc_price = round(float((price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2), 2)
        
        tot_vol = np.sum(vol_by_bin)
        target_va_vol = tot_vol * 0.70
        sorted_indices = np.argsort(vol_by_bin)[::-1]
        cum_va = 0.0
        va_bins = []
        for s_idx in sorted_indices:
            cum_va += vol_by_bin[s_idx]
            va_bins.append(s_idx)
            if cum_va >= target_va_vol:
                break
        min_va_bin = min(va_bins)
        max_va_bin = max(va_bins)
        val_price = round(float(price_bins[min_va_bin]), 2)
        vah_price = round(float(price_bins[max_va_bin + 1]), 2)
        
        volume_profile = {
            "poc_price": poc_price,
            "vah_price": vah_price,
            "val_price": val_price,
            "value_area_range": f"${val_price} ~ ${vah_price}"
        }
    except Exception:
        volume_profile = {
            "poc_price": "N/A",
            "vah_price": "N/A",
            "val_price": "N/A",
            "value_area_range": "N/A"
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
        "vwap_20d": round(float(latest['Rolling_VWAP_20']), 2) if pd.notnull(latest['Rolling_VWAP_20']) else "N/A",
        "poc_price_6m": volume_profile.get("poc_price", "N/A"),
        "value_area_range_6m": volume_profile.get("value_area_range", "N/A")
    }
    return data, last_date, fibonacci_levels, high_52w_calc, low_52w_calc, df, volume_profile

# =============================================================================
# [BLOCK 05] 퀀트 전략 백테스팅 엔진
# =============================================================================
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
            return {"total_ret": 0.0, "win_rate": 0.0, "trades_count": 0, "profit_factor": 0.0, "mdd": 0.0, "sharpe_ratio": 0.0}
        
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

        # 📌 샤프비율 연산 (무위험 수익률 0 가정, 단위 매매당 표준편차 기반)
        if len(trades) > 1:
            std_dev = np.std(trades, ddof=1)
            sharpe = round((np.mean(trades) / std_dev), 2) if std_dev > 0 else 0.0
        elif len(trades) == 1 and trades[0] > 0:
            sharpe = 1.0 
        else:
            sharpe = 0.0

        return {
            "total_ret": round(tot_ret, 2),
            "win_rate": round(win_rate, 1),
            "trades_count": len(trades),
            "profit_factor": pf,
            "mdd": round(mdd * 100, 2),
            "sharpe_ratio": sharpe
        }

    return {
        "benchmark_buy_and_hold": round(bh_return, 2),
        "strategy_1_momentum_squeeze": calc_stats(trades1),
        "strategy_2_vwap_mean_reversion": calc_stats(trades2)
    }

# =============================================================================
# [BLOCK 06] 시장 수급 & 외부 데이터 수집기 (옵션/매크로/섹터/뉴스/지분)
# =============================================================================
# -------------------------------------------------------------
# 📌 옵션 체인 스마트머니 수급 수집기
# -------------------------------------------------------------
def fetch_nearest_options_data(ticker: str, retries: int = 2):
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker, session=GLOBAL_SESSION)
            expirations = getattr(stock, 'options', None)
            if not expirations:
                return None
            
            nearest_exp = expirations[0]
            opt_chain = stock.option_chain(nearest_exp)
            calls = opt_chain.calls
            puts = opt_chain.puts
            
            if calls is None or puts is None or calls.empty or puts.empty:
                if attempt < retries - 1:
                    time.sleep(0.5 * (attempt + 1))
                    continue
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
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            return None
    return None

# -------------------------------------------------------------
# 📌 매크로 지표 수집기 (5분 캐싱 복구 + 일괄 다운로드 최적화)
# -------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_macro_indicators():
    macro_data = {}
    try:
        end = datetime.now()
        start = end - timedelta(days=30)
        fred_res = web.DataReader('DGS10', 'fred', start, end, session=GLOBAL_SESSION).dropna()
        dgs10 = fred_res.iloc[-1, 0]
        macro_data["us_10y_yield"] = {
            "source": "FRED (Federal Reserve Economic Data)",
            "value": f"{round(float(dgs10), 2)}%",
            "date": fred_res.index[-1].strftime("%Y-%m-%d")
        }
    except Exception:
        macro_data["us_10y_yield"] = {"source": "FRED", "value": "N/A", "date": "N/A"}
        
    asset_map = {
        "^VIX": ("vix", "CBOE Volatility Index"),
        "DX-Y.NYB": ("dollar_index", "ICE US Dollar Index"),
        "CL=F": ("wti_oil", "NYMEX WTI Crude Oil"),
        "GC=F": ("gold", "COMEX Gold Futures"),
        "BTC-USD": ("bitcoin", "Binance/Coinbase Crypto Market")
    }
    
    try:
        tickers = list(asset_map.keys())
        df = yf.download(tickers, period="5d", progress=False, session=GLOBAL_SESSION)['Close']
        
        for ticker, (name, src_name) in asset_map.items():
            try:
                hist = df[ticker].dropna()
                if not hist.empty:
                    macro_data[name] = {
                        "source": src_name,
                        "value": round(float(hist.iloc[-1]), 2),
                        "date": hist.index[-1].strftime("%Y-%m-%d")
                    }
                else:
                    macro_data[name] = {"source": src_name, "value": "N/A", "date": "N/A"}
            except Exception:
                macro_data[name] = {"source": src_name, "value": "N/A", "date": "N/A"}
    except Exception:
        for ticker, (name, src_name) in asset_map.items():
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
# 📌 유명 헤지펀드 보유 내역 & 공매도 세력 분석 데이터 수집기
# -------------------------------------------------------------
def fetch_hedge_funds_and_short_intel(stock, info):
    intel = {
        "top_holders": [],
        "short_intel": {}
    }
    
    try:
        inst_df = getattr(stock, 'institutional_holders', None)
        if inst_df is not None and isinstance(inst_df, pd.DataFrame) and not inst_df.empty:
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
        short_float = info.get("shortPercentOfFloat", None) if isinstance(info, dict) else None
        short_ratio = info.get("shortRatio", None) if isinstance(info, dict) else None
        shares_short = info.get("sharesShort", None) if isinstance(info, dict) else None
        shares_short_prior = info.get("sharesShortPriorMonth", None) if isinstance(info, dict) else None

        short_float_pct = round(short_float * 100, 2) if short_float is not None else None
        short_ratio_days = round(short_ratio, 2) if short_ratio is not None else None
        
        short_mom_pct = None
        if shares_short and shares_short_prior and shares_short_prior > 0:
            short_mom_pct = round(((shares_short - shares_short_prior) / shares_short_prior) * 100, 2)

        squeeze_risk = "해당없음 (원자재/코인/지수)" if not short_float_pct and not short_ratio_days else "🟢 안정 (Low Risk)"
        if short_float_pct is not None and short_ratio_days is not None:
            if short_float_pct >= 20.0 and short_ratio_days >= 5.0:
                squeeze_risk = "🚨 숏스퀴즈 고위험 (High Squeeze Potential)"
            elif short_float_pct >= 10.0 and short_ratio_days >= 3.0:
                squeeze_risk = "⚠️ 숏스퀴즈 주의 (Moderate Potential)"
            elif short_float_pct >= 5.0:
                squeeze_risk = "💡 모니터링 구간 (Low-Moderate)"
        elif short_float_pct is not None:
            if short_float_pct >= 20.0:
                squeeze_risk = "🚨 숏스퀴즈 고위험 (High Squeeze Potential)"
            elif short_float_pct >= 10.0:
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
        if isinstance(info, dict):
            ins_own_val = info.get("heldPercentInsiders", None)
            if ins_own_val is not None:
                data["insider_own"] = f"{ins_own_val * 100:.2f}%"
                
            inst_own_val = info.get("heldPercentInstitutions", None)
            if inst_own_val is not None:
                data["inst_own"] = f"{inst_own_val * 100:.2f}%"
    except Exception:
        pass

    try:
        ins_df = getattr(stock, 'insider_transactions', None)
        if ins_df is not None and isinstance(ins_df, pd.DataFrame) and not ins_df.empty and 'Shares' in ins_df.columns:
            recent_ins = ins_df.head(15)
            net_shares = recent_ins['Shares'].dropna().sum()
            shares_out = info.get("sharesOutstanding", None) if isinstance(info, dict) else None
            if shares_out and shares_out > 0:
                trans_pct = (net_shares / shares_out) * 100
                data["insider_trans"] = f"{trans_pct:+.2f}%"
            else:
                data["insider_trans"] = f"{net_shares:+,.0f}주"
    except Exception:
        pass

    try:
        inst_df = getattr(stock, 'institutional_holders', None)
        if inst_df is not None and isinstance(inst_df, pd.DataFrame) and not inst_df.empty and '% Out' in inst_df.columns:
            tot_pct = inst_df['% Out'].sum() * 100
            data["inst_trans"] = f"{tot_pct:.2f}% (Top10)"
        elif inst_df is not None and isinstance(inst_df, pd.DataFrame) and not inst_df.empty and 'Shares' in inst_df.columns:
            tot_shares = inst_df['Shares'].sum()
            data["inst_trans"] = f"{tot_shares:,.0f}주 (Top10)"
    except Exception:
        pass

    return data

def fetch_earnings_calendar(stock, info, high_52_calc, low_52_calc):
    earnings_date_str = "해당없음 (원자재/코인/지수)"
    d_day_str = ""
    try:
        cal = getattr(stock, 'calendar', None)
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

    high_52w = (info.get("fiftyTwoWeekHigh", None) if isinstance(info, dict) else None) or high_52_calc
    low_52w = (info.get("fiftyTwoWeekLow", None) if isinstance(info, dict) else None) or low_52_calc

    return {
        "earnings_date": earnings_date_str,
        "d_day": d_day_str,
        "fiftyTwoWeekHigh": high_52w,
        "fiftyTwoWeekLow": low_52w
    }

def fetch_fundamentals_and_valuation(ticker: str, curr_price: float, high_52_calc, low_52_calc):
    stock = yf.Ticker(ticker, session=GLOBAL_SESSION)
    info, info_source = get_stock_info_with_retry(stock, retries=2)

    fast_info = {}
    try:
        if hasattr(stock, 'fast_info') and stock.fast_info:
            fast_info = stock.fast_info
    except Exception:
        pass

    market_cap = info.get("marketCap", None) if isinstance(info, dict) else None
    if not market_cap and fast_info:
        market_cap = getattr(fast_info, 'market_cap', None) or (fast_info.get('market_cap', "N/A") if isinstance(fast_info, dict) else "N/A")

    trailing_pe = info.get("trailingPE", "N/A") if isinstance(info, dict) else "N/A"
    forward_pe = info.get("forwardPE", "N/A") if isinstance(info, dict) else "N/A"
    pbr = info.get("priceToBook", "N/A") if isinstance(info, dict) else "N/A"
    ps_ratio = info.get("priceToSalesTrailing12Months", "N/A") if isinstance(info, dict) else "N/A"
    
    roe_raw = info.get("returnOnEquity", None) if isinstance(info, dict) else None
    roe_pct = round(roe_raw * 100, 2) if roe_raw is not None else "N/A"
    eps = info.get("trailingEps", None) if isinstance(info, dict) else None
    forward_eps = info.get("forwardEps", None) if isinstance(info, dict) else None
    bps = info.get("bookValue", None) if isinstance(info, dict) else None
    revenue_per_share = info.get("revenuePerShare", None) if isinstance(info, dict) else None
    target_mean_price = info.get("targetMeanPrice", "N/A") if isinstance(info, dict) else "N/A"

    rnd_ratio_fmt = "N/A"
    try:
        inc = stock.financials
        if not inc.empty:
            rnd_val = 0
            for col in ["Research And Development", "Research & Development", "Research Expense"]:
                if col in inc.index:
                    rnd_val = inc.loc[col].iloc[0]
                    break
            tot_rev = inc.loc["Total Revenue"].iloc[0] if "Total Revenue" in inc.index else 0
            if tot_rev > 0 and pd.notnull(rnd_val):
                rnd_ratio_fmt = f"{(rnd_val / tot_rev) * 100:.2f}%"
    except Exception:
        pass

    fcf_raw = info.get("freeCashflow", None) if isinstance(info, dict) else None
    fcf_fmt = format_market_cap(fcf_raw) if fcf_raw else "N/A"
    
    de_ratio = info.get("debtToEquity", None) if isinstance(info, dict) else None
    de_fmt = f"{de_ratio:.2f}%" if isinstance(de_ratio, (int, float)) else "N/A"
    
    gross_margin = info.get("grossMargins", None) if isinstance(info, dict) else None
    gross_margin_fmt = f"{gross_margin * 100:.2f}%" if isinstance(gross_margin, (int, float)) else "N/A"
    
    op_margin = info.get("operatingMargins", None) if isinstance(info, dict) else None
    op_margin_fmt = f"{op_margin * 100:.2f}%" if isinstance(op_margin, (int, float)) else "N/A"

    quality_factors = {
        "free_cash_flow": fcf_fmt,
        "debt_to_equity": de_fmt,
        "gross_margin": gross_margin_fmt,
        "operating_margin": op_margin_fmt,
        "rnd_to_revenue": rnd_ratio_fmt
    }

    long_term_quality = {
        "3y_fcf_status": "N/A",
        "roic": "N/A",
        "shareholder_yield": "N/A",
        "shares_change_pct": "N/A"
    }
    try:
        cf = stock.cashflow
        if not cf.empty and "Free Cash Flow" in cf.index:
            fcf_data = cf.loc["Free Cash Flow"].dropna()
            if len(fcf_data) >= 3:
                fcf_3yr = fcf_data.iloc[:3]
                if all(val > 0 for val in fcf_3yr):
                    long_term_quality["3y_fcf_status"] = "3년 연속 흑자 (+)"
                elif all(val < 0 for val in fcf_3yr):
                    long_term_quality["3y_fcf_status"] = "3년 연속 적자 (-)"
                else:
                    long_term_quality["3y_fcf_status"] = "흑자/적자 혼조"
                    
        bs = stock.balance_sheet
        inc = stock.financials
        if not bs.empty and not inc.empty:
            try:
                operating_income = inc.loc["Operating Income"].iloc[0] if "Operating Income" in inc.index else 0
                tax_provision = inc.loc["Tax Provision"].iloc[0] if "Tax Provision" in inc.index else 0
                pretax_income = inc.loc["Pretax Income"].iloc[0] if "Pretax Income" in inc.index else 1
                tax_rate = tax_provision / pretax_income if pretax_income > 0 else 0.21
                nopat = operating_income * (1 - tax_rate)
                
                total_debt = bs.loc["Total Debt"].iloc[0] if "Total Debt" in bs.index else 0
                equity = bs.loc["Stockholders Equity"].iloc[0] if "Stockholders Equity" in bs.index else (bs.loc["Total Equity Gross Minority Interest"].iloc[0] if "Total Equity Gross Minority Interest" in bs.index else 0)
                cash = bs.loc["Cash And Cash Equivalents"].iloc[0] if "Cash And Cash Equivalents" in bs.index else 0
                
                invested_capital = total_debt + equity - cash
                if invested_capital > 0:
                    roic = nopat / invested_capital
                    long_term_quality["roic"] = f"{roic * 100:.2f}%"
            except Exception:
                pass
        
        payout_ratio = info.get("payoutRatio", None) if isinstance(info, dict) else None
        if payout_ratio is not None:
            long_term_quality["shareholder_yield"] = f"배당성향 {payout_ratio * 100:.1f}%"
            
        if not inc.empty and "Basic Average Shares" in inc.index:
            shares = inc.loc["Basic Average Shares"].dropna()
            if len(shares) >= 2:
                recent_shares = shares.iloc[0]
                prev_shares = shares.iloc[1]
                if prev_shares > 0:
                    change_pct = ((recent_shares - prev_shares) / prev_shares) * 100
                    long_term_quality["shares_change_pct"] = f"{change_pct:+.2f}%"
    except Exception:
        pass

    ownership_and_shorts = fetch_ownership_and_shorts(stock, info)
    hedge_and_short_intel = fetch_hedge_funds_and_short_intel(stock, info)
    earnings_cal = fetch_earnings_calendar(stock, info, high_52_calc, low_52_calc)

    earnings_growth = info.get("earningsGrowth", None) if isinstance(info, dict) else None
    revenue_growth = info.get("revenueGrowth", None) if isinstance(info, dict) else None
    
    if earnings_growth and earnings_growth > 0:
        est_growth = min(earnings_growth * 100, 35.0)
    else:
        est_growth = 15.0

    growth_factors = {
        "revenue_growth_yoy": f"{revenue_growth * 100:.2f}%" if revenue_growth is not None else "N/A",
        "earnings_growth_yoy": f"{earnings_growth * 100:.2f}%" if earnings_growth is not None else "N/A"
    }

    def _value_model_sanity(value, label):
        try:
            if not isinstance(value, (int, float)):
                return "산출불가 (재무제표 미존재/해당없음)"
            if not isinstance(curr_price, (int, float)) or curr_price <= 0:
                return "산출불가"
            deviation = abs(value - curr_price) / curr_price
            high_per = isinstance(trailing_pe, (int, float)) and trailing_pe >= 60.0
            if deviation > 0.6 and high_per:
                return f"산출불가 (고PER 성장주 - 자산가치 모델 부적합, PER {trailing_pe:.1f}배)"
            if deviation > 0.6:
                return f"산출불가 (모델 괴리율 과다: {deviation*100:.0f}%)"
            return value
        except Exception:
            return "산출불가 (해당없음)"

    value_models = {}
    try:
        if eps and bps and eps > 0 and bps > 0:
            raw_graham = round(math.sqrt(22.5 * float(eps) * float(bps)), 2)
            value_models["graham"] = _value_model_sanity(raw_graham, "graham")
        else:
            value_models["graham"] = "산출불가 (해당없음)"
    except Exception:
        value_models["graham"] = "산출불가 (해당없음)"

    try:
        if eps and eps > 0 and roe_raw and roe_raw > 0:
            raw_lynch = round(float(eps) * min(float(roe_raw) * 100, 25.0), 2)
            value_models["peter_lynch"] = _value_model_sanity(raw_lynch, "peter_lynch")
        else:
            value_models["peter_lynch"] = "산출불가 (해당없음)"
    except Exception:
        value_models["peter_lynch"] = "산출불가 (해당없음)"

    try:
        if bps and bps > 0 and roe_raw and roe_raw > 0:
            raw_roe_pbr = round(float(bps) * (float(roe_raw) / 0.10), 2)
            value_models["roe_pbr"] = _value_model_sanity(raw_roe_pbr, "roe_pbr")
        else:
            value_models["roe_pbr"] = "산출불가 (해당없음)"
    except Exception:
        value_models["roe_pbr"] = "산출불가 (해당없음)"

    used_growth_fallback = not (earnings_growth and earnings_growth > 0)

    def _sanity_capped(value, label):
        try:
            if not isinstance(value, (int, float)):
                return "산출불가 (해당없음)"
            if not isinstance(curr_price, (int, float)) or curr_price <= 0:
                return "산출불가"
            deviation = abs(value - curr_price) / curr_price
            if deviation > 0.6:
                return f"산출불가 (모델 괴리율 과다: {deviation*100:.0f}%)"
            if used_growth_fallback:
                return f"{value} (참고용·추정성장률 가정치)"
            return value
        except Exception:
            return "산출불가 (해당없음)"

    growth_models = {}
    f_eps = forward_eps if forward_eps and forward_eps > 0 else eps
    try:
        if f_eps and f_eps > 0:
            raw_peg = round(float(f_eps) * (est_growth * 1.5), 2)
            growth_models["forward_peg"] = _sanity_capped(raw_peg, "forward_peg")
        else:
            growth_models["forward_peg"] = "산출불가 (해당없음)"
    except Exception:
        growth_models["forward_peg"] = "산출불가 (해당없음)"

    try:
        if revenue_per_share and revenue_per_share > 0:
            raw_psr = round(float(revenue_per_share) * 5.0, 2)
            growth_models["psr_target"] = _sanity_capped(raw_psr, "psr_target")
        else:
            growth_models["psr_target"] = "산출불가 (해당없음)"
    except Exception:
        growth_models["psr_target"] = "산출불가 (해당없음)"

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
            raw_dcf = round(pv_sum + pv_terminal, 2)
            growth_models["dcf_growth"] = _sanity_capped(raw_dcf, "dcf_growth")
        else:
            growth_models["dcf_growth"] = "산출불가 (해당없음)"
    except Exception:
        growth_models["dcf_growth"] = "산출불가 (해당없음)"

    return {
        "info_source": info_source,
        "market_cap_fmt": format_market_cap(market_cap),
        "trailing_pe": round(trailing_pe, 2) if isinstance(trailing_pe, (int, float)) else trailing_pe,
        "forward_pe": round(forward_pe, 2) if isinstance(forward_pe, (int, float)) else forward_pe,
        "pbr": round(pbr, 2) if isinstance(pbr, (int, float)) else pbr,
        "ps_ratio": round(ps_ratio, 2) if isinstance(ps_ratio, (int, float)) else ps_ratio,
        "roe": f"{roe_pct}%" if roe_pct != "N/A" else "N/A",
        "target_mean_price": target_mean_price,
        "quality_factors": quality_factors,
        "long_term_quality": long_term_quality,
        "growth_factors": growth_factors,
        "ownership_and_shorts": ownership_and_shorts,
        "hedge_and_short_intel": hedge_and_short_intel,
        "earnings_calendar": earnings_cal,
        "value_models": value_models,
        "growth_models": growth_models
    }

# -------------------------------------------------------------
# 📌 S&P 500 11개 전 섹터 수익률 수집 (5분 캐싱 적용)
# -------------------------------------------------------------
@st.cache_data(ttl=300)
def fetch_sector_performance():
    sector_etfs = {
        "XLK": "IT/기술 (Technology)", "XLC": "커뮤니케이션 (Communication Services)",
        "XLY": "임의소비재 (Consumer Discretionary)", "XLP": "필수소비재 (Consumer Staples)",
        "XLF": "금융 (Financials)", "XLV": "헬스케어 (Health Care)",
        "XLI": "산업재 (Industrials)", "XLE": "에너지 (Energy)",
        "XLB": "소재 (Materials)", "XLU": "유틸리티 (Utilities)",
        "XLRE": "부동산 (Real Estate)"
    }
    summary = {}
    
    try:
        tickers = list(sector_etfs.keys())
        df = yf.download(tickers, period="1mo", progress=False, session=GLOBAL_SESSION)['Close']
        
        for etf, name in sector_etfs.items():
            try:
                hist = df[etf].dropna()
                if len(hist) >= 2:
                    pct_5d = ((hist.iloc[-1] - hist.iloc[-5]) / hist.iloc[-5] * 100) if len(hist) >= 5 else ((hist.iloc[-1] - hist.iloc[0]) / hist.iloc[0] * 100)
                    pct_1m = ((hist.iloc[-1] - hist.iloc[0]) / hist.iloc[0] * 100)
                    summary[etf] = {
                        "sector_name": name,
                        "return_5d": f"{pct_5d:+.2f}%",
                        "return_1m": f"{pct_1m:+.2f}%",
                        "latest_close": round(float(hist.iloc[-1]), 2)
                    }
                else:
                    summary[etf] = {"sector_name": name, "return_5d": "N/A", "return_1m": "N/A", "latest_close": "N/A"}
            except Exception:
                summary[etf] = {"sector_name": name, "return_5d": "N/A", "return_1m": "N/A", "latest_close": "N/A"}
    except Exception:
        for etf, name in sector_etfs.items():
            summary[etf] = {"sector_name": name, "return_5d": "N/A", "return_1m": "N/A", "latest_close": "N/A"}
            
    return summary

@st.cache_data(ttl=300)
def fetch_news(ticker: str, limit: int = 5):
    try:
        stock = yf.Ticker(ticker, session=GLOBAL_SESSION)
        raw_news = getattr(stock, 'news', None)
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

@st.cache_data(ttl=300)
def fetch_macro_news(limit: int = 4):
    macro_articles = []
    for sym in ["SPY", "TLT"]:
        try:
            stock = yf.Ticker(sym, session=GLOBAL_SESSION)
            raw = getattr(stock, 'news', None)
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
    
# =============================================================================
# [BLOCK 07] 증권사 투자의견 & LLM 응답 파서
# =============================================================================
# -------------------------------------------------------------
# 📌 히스토리용 전략 요약 함수
# -------------------------------------------------------------
def summarize_user_strategy(raw_text: str) -> str:
    if not raw_text or raw_text == "분석 리포트 참조":
        return "분석 리포트 참조"
        
    text = raw_text.replace("\n", " ").strip()
    text = re.sub(r'\s+', ' ', text)
    
    if len(text) > 300:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 3]
        collected = []
        cur_len = 0
        for s in sentences:
            collected.append(s)
            cur_len += len(s)
            if cur_len >= 220:
                break
        res = " ".join(collected)
        return res if res.endswith((".", "!", "?")) else res + "..."
    return text

# -------------------------------------------------------------
# 📌 신규 진입 적격성 평가 및 매매 시나리오 파서
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
    quality_badge = ""

    # 0. 스코어카드 종합 평점 및 우량성 이모지 추출
    if "최상위 핵심 우량주" in text or "👑" in text:
        quality_badge = "👑 "
    elif "적격 우량주" in text or "🥇" in text:
        quality_badge = "🥇 "
    elif "조건부 종목" in text or "⚠️" in text:
        quality_badge = "⚠️ "
    elif "비우량주" in text or "🚨" in text:
        quality_badge = "🚨 "

    # 1. 최종 투자의견 액션 판정
    match_action = re.search(r"(?:최종\s*투자의견|최종투자\s*의견)[^:\n]*[:\-]?\s*([^\n\r]+)", text)
    if match_action:
        op_text = match_action.group(1).replace("*", "").replace("[", "").replace("]", "").strip()
        if "적극매수" in op_text or ("매수" in op_text and "관망" not in op_text and "보유" not in op_text and "홀딩" not in op_text):
            action = "매수"
        elif "매도" in op_text or "비중축소" in op_text or "차익실현" in op_text or "손절" in op_text:
            action = "매도"
        elif "홀딩" in op_text or "보유" in op_text or "관망" in op_text:
            action = "홀딩"

    # 2. [신규 진입 적격성 평가] 전용 섹션 파싱
    match_entry_sec = re.search(r"\[신규\s*진입\s*적격성\s*평가\](.*?)(?=\[(?:정밀\s*매매\s*시나리오|최종\s*투자의견)|\Z)", text, re.DOTALL)
    if match_entry_sec:
        entry_block = match_entry_sec.group(1)
        for line in entry_block.split("\n"):
            line_clean = line.replace("*", "").replace("-", "").replace("•", "").strip()
            if "신규 진입 등급" in line_clean or "진입 등급" in line_clean:
                if ":" in line_clean:
                    entry_grade = ":".join(line_clean.split(":")[1:]).strip()
            elif "예상 손익비" in line_clean or "손익비" in line_clean:
                if ":" in line_clean:
                    entry_rr = ":".join(line_clean.split(":")[1:]).strip()

    # 3. [정밀 매매 시나리오] 전용 섹션 엄격 분리 파싱
    match_scen_sec = re.search(r"\[정밀\s*매매\s*시나리오\](.*?)(?=\[(?:최종\s*투자의견)|\Z)", text, re.DOTALL)
    scenario_block = match_scen_sec.group(1) if match_scen_sec else text

    for line in scenario_block.split("\n"):
        line_clean = line.replace("*", "").replace("-", "").replace("•", "").strip()
        
        if ("1차 목표가" in line_clean or "1차목표가" in line_clean) and target_1 == "분석 리포트 참조":
            if ":" in line_clean:
                target_1 = ":".join(line_clean.split(":")[1:]).strip()
        elif ("2차 목표가" in line_clean or "2차목표가" in line_clean) and not target_2:
            if ":" in line_clean:
                target_2 = ":".join(line_clean.split(":")[1:]).strip()
        elif ("매도가 밴드" in line_clean or "비중축소" in line_clean or "매도가" in line_clean) and sell_target == "분석 리포트 참조":
            if ":" in line_clean:
                sell_target = ":".join(line_clean.split(":")[1:]).strip()
        elif ("분할 매수 밴드" in line_clean or "분할매수 밴드" in line_clean) and buy_band == "분석 리포트 참조":
            if ":" in line_clean:
                buy_band = ":".join(line_clean.split(":")[1:]).strip()
        elif ("손절" in line_clean or "Stop-loss" in line_clean) and stop_loss == "분석 리포트 참조":
            if ":" in line_clean:
                stop_loss = ":".join(line_clean.split(":")[1:]).strip()
        elif ("불타기 조건" in line_clean or "불타기" in line_clean) and not pyramiding:
            if ":" in line_clean:
                pyramiding = ":".join(line_clean.split(":")[1:]).strip()

    # 4. [최종 투자의견 / 사용자 대응 전략] 전용 섹션 파싱
    match_opinion_sec = re.search(r"\[최종\s*투자의견\](.*?)(?=\Z|\[|\n\n#)", text, re.DOTALL)
    opinion_block = match_opinion_sec.group(1) if match_opinion_sec else text

    strategy_lines = []
    collecting_strategy = False

    for line in opinion_block.split("\n"):
        line_clean = line.replace("*", "").replace("-", "").replace("•", "").replace("◦", "").strip()
        
        if "사용자 대응 전략" in line_clean or "사용자대응전략" in line_clean:
            content_after = ":".join(line_clean.split(":")[1:]).strip() if ":" in line_clean else ""
            if content_after:
                strategy_lines.append(content_after)
            collecting_strategy = True
            continue
            
        if collecting_strategy:
            if line.strip().startswith("[") or line.strip().startswith("#"):
                collecting_strategy = False
            elif line_clean:
                strategy_lines.append(line_clean)

    if strategy_lines:
        user_strategy_raw = " ".join(strategy_lines)

    user_strategy_summary = summarize_user_strategy(user_strategy_raw)
    return action, entry_grade, entry_rr, target_1, target_2, sell_target, buy_band, stop_loss, pyramiding, user_strategy_summary, quality_badge

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
        
# =============================================================================
# [BLOCK 08] 사이드바 UI 컴포넌트 & 히스토리 카드 렌더링
# =============================================================================
# -------------------------------------------------------------
# 2. 사이드바 UI
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ **AI Stock Analyst Pro**")
    
    MODEL_OPTIONS = {
        "Gemini 3.5 Flash Lite": "gemini-3.5-flash-lite",
        "Gemini 3.1 Flash Lite": "gemini-3.1-flash-lite",
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

    # 📌 트레이딩 히스토리
    st.markdown("#### 📌 **트레이딩 히스토리**")
    
    if st.session_state.history:
        tab_all, tab_buy, tab_sell, tab_hold = st.tabs(["전체", "🟢매수", "🔴매도", "🟡홀딩"])
        
        def render_history_card(t_code, data):
            action_badge = "🟢 매수" if data['action'] == "매수" else ("🔴 매도" if data['action'] == "매도" else "🟡 홀딩")
            q_badge = data.get('quality_badge', '')
            
            with st.expander(f"{q_badge}**{t_code}** (${data['price']}) | {action_badge}", expanded=False):
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
                st.caption("매수 판정 내역이 없습니다.")

        with tab_sell:
            sell_items = [item for item in list(st.session_state.history.items())[::-1] if item[1]['action'] == "매도"]
            if sell_items:
                for t_code, data in sell_items:
                    render_history_card(t_code, data)
            else:
                st.caption("매도 판정 내역이 없습니다.")

        with tab_hold:
            hold_items = [item for item in list(st.session_state.history.items())[::-1] if item[1]['action'] == "홀딩"]
            if hold_items:
                for t_code, data in hold_items:
                    render_history_card(t_code, data)
            else:
                st.caption("홀딩 판정 내역이 없습니다.")
                    
        st.write("")
        if st.button("🗑️ 히스토리 전체 삭제", use_container_width=True):
            st.session_state.history = {}
            if os.path.exists(HISTORY_FILE):
                os.remove(HISTORY_FILE)
            st.rerun()
    else:
        st.caption("분석 내역이 여기에 영구 저장됩니다.")

# =============================================================================
# [BLOCK 09] LLM RAG 추론 파이프라인 & 실행 트리거 (analyze_btn)
# =============================================================================
# -------------------------------------------------------------
# 3. 메인 분석 화면 & 분석 완료 즉시 렌더링 로직
# -------------------------------------------------------------
st.header(f"📊 {ticker_input} 종합 밸류에이션 & 정밀 트레이딩 리포트")

if analyze_btn:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            api_key = None
            
    with st.spinner(f"🔍 [{ticker_input}] POC 매물대/헤지펀드 지분/공매도 세력 분석/11개 섹터 수급/VWAP 분석 및 백테스팅 실행 중..."):
        tech_data, stock_date, fib_levels, high_52_calc, low_52_calc, raw_df, vol_profile = fetch_stock_technical_data(ticker_input)
        backtest_results = run_strategy_backtest(raw_df)
        options_data = fetch_nearest_options_data(ticker_input, retries=3)
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
        if is_holding and user_avg_price > 0 and user_shares > 0 and isinstance(curr_p, (int, float)) and curr_p > 0:
            total_invested = user_avg_price * user_shares
            total_current = curr_p * user_shares
            pnl_dollar = total_current - total_invested
            pnl_pct = ((curr_p - user_avg_price) / user_avg_price) * 100
            my_return_str = f"{pnl_pct:+.2f}%"

        user_position_text = (
            f"사용자 보유 현황: 평단가 ${user_avg_price:.2f}, 보유수량 {user_shares:.1f}주, 평가수익률 {my_return_str}"
            if is_holding and user_avg_price > 0 else "사용자 미보유 종목 (신규 진입 검토 관점)"
        )

        if is_holding and user_avg_price > 0:
            strategy_instruction_text = f"""* **사용자 대응 전략**: [현재 사용자가 평단가 ${user_avg_price:.2f}, 평가수익률 {my_return_str}로 주식을 보유 중인 상태입니다. 반드시 '보유자 관점'의 전략만 단독 작성할 것. 미보유자나 신규 진입 관련 문구는 일절 작성하지 말 것. 1차/2차 목표가 도달 시 구체적인 부분 익절/비중축소 비중(예: 30% 매도) 및 손절선 이탈 시 전량 손절 계획을 명시할 것.]"""
        else:
            strategy_instruction_text = """* **사용자 대응 전략**: [현재 사용자가 주식을 보유하지 않은 '미보유 상태'입니다. 반드시 '미보유자 신규 진입 관점'의 전략만 단독 작성할 것. 보유자 관련 문구는 일절 작성하지 말 것. 상단 [신규 진입 적격성 평가] 및 [분할 매수 밴드]와 100% 일치하는 진입 가격대와 진입 비중(예: 1차 30% 분할 진입 등)을 명확히 제시할 것.]"""

        # =====================================================================
        # 💡 [신규 추가] Python 기반 정량 지표 & 손익비 사전 연산 엔진 (할루시네이션 원천 차단)
        # =====================================================================
        def calculate_pre_scores(fund, tech, bt, curr_price):
            def parse_num(v):
                if isinstance(v, str):
                    try: return float(re.sub(r'[^0-9.-]', '', v))
                    except: return 0.0
                return float(v) if v else 0.0

            # 1) 성장성 (20%): 실제 매출 및 EPS 성장률 기반 보수적 평균 채점 (체리피킹 방지)
            earn_g_str = fund.get('growth_factors', {}).get('earnings_growth_yoy', 'N/A')
            rev_g_str = fund.get('growth_factors', {}).get('revenue_growth_yoy', 'N/A')
            
            if earn_g_str == 'N/A' and rev_g_str == 'N/A':
                s_growth = 5.5  # 두 데이터가 모두 없을 때만 예외 처리
            else:
                earn_g = parse_num(earn_g_str) if earn_g_str != 'N/A' else None
                rev_g = parse_num(rev_g_str) if rev_g_str != 'N/A' else None
                
                # 각 지표별 점수 산출 함수
                def get_g_score(val):
                    if val is None: return None
                    if val >= 30.0: return 9.5
                    elif val >= 15.0: return 7.5
                    elif val >= 5.0: return 5.5
                    elif val > 0.0: return 3.5
                    else: return 1.5 # 역성장 (0 이하)

                s_earn = get_g_score(earn_g)
                s_rev = get_g_score(rev_g)
                
                # 두 지표가 모두 존재하면 평균을 내어 외형성장과 이익감소의 괴리를 정확히 반영
                if s_earn is not None and s_rev is not None:
                    s_growth = (s_earn + s_rev) / 2.0
                elif s_earn is not None:
                    s_growth = s_earn
                elif s_rev is not None:
                    s_growth = s_rev
                else:
                    s_growth = 5.5

            # 2) 수익성 (25%)
            opm = parse_num(fund.get('quality_factors', {}).get('operating_margin', 0))
            fcf_str = str(fund.get('quality_factors', {}).get('free_cash_flow', ''))
            roic = parse_num(fund.get('long_term_quality', {}).get('roic', 0))
            
            s_prof = 1.5
            if opm >= 20 and '-' not in fcf_str and fcf_str != 'N/A': s_prof = 9.5
            elif opm >= 15: s_prof = 7.5
            elif opm >= 8: s_prof = 5.5
            elif opm >= 0: s_prof = 3.5
            if roic >= 15.0: s_prof = min(10.0, s_prof + 1.0)

            # 3) 해자 (25%)
            s_moat = 0.0
            if parse_num(fund.get('roe', 0)) >= 20.0: s_moat += 2.5
            if parse_num(fund.get('quality_factors', {}).get('gross_margin', 0)) >= 50.0: s_moat += 2.5
            if parse_num(fund.get('quality_factors', {}).get('rnd_to_revenue', 0)) >= 5.0: s_moat += 2.5
            if parse_num(fund.get('ownership_and_shorts', {}).get('inst_own', 0)) >= 70.0: s_moat += 1.5
            if parse_num(fund.get('quality_factors', {}).get('debt_to_equity', 999)) <= 100.0: s_moat += 1.0
            if parse_num(fund.get('long_term_quality', {}).get('shares_change_pct', 0)) < 0: s_moat += 1.0
            s_moat = min(10.0, max(1.5, s_moat))

            # 4) 밸류에이션 (20%) - 4개 지표 개별 채점 후 정확한 산술 평균
            pe = fund.get('trailing_pe'); fpe = fund.get('forward_pe')
            ps = fund.get('ps_ratio'); pbr = fund.get('pbr')
            def score_v(val, b1, b2, b3, b4):
                if not isinstance(val, (int, float)) or val < 0: return 1.5
                if val <= b1: return 9.5
                elif val <= b2: return 7.5
                elif val <= b3: return 5.5
                elif val <= b4: return 3.5
                return 1.5
            
            s_val = (score_v(pe, 15, 25, 40, 60) + score_v(fpe, 12, 20, 28, 35) + 
                     score_v(ps, 3, 6, 10, 15) + score_v(pbr, 3, 6, 10, 15)) / 4.0

            # 5) 모멘텀 (10%) - 💡 1년 VWAP이 아닌 정확히 '20일 단기 VWAP' 기준으로 원복
            vwap_20d = parse_num(tech.get('vwap_20d')) # vwap_1y -> vwap_20d 로 변경
            vwap_dev = ((curr_price - vwap_20d) / vwap_20d * 100) if vwap_20d > 0 else 0
            
            if vwap_dev >= 5: s_mom1 = 9.5
            elif vwap_dev >= 0: s_mom1 = 7.5
            elif vwap_dev >= -2: s_mom1 = 5.5
            elif vwap_dev >= -5: s_mom1 = 3.5
            else: s_mom1 = 1.5

            rsi = parse_num(tech.get('rsi_14', 50))
            if 55 <= rsi <= 70: s_mom2 = 9.0
            elif 45 <= rsi < 55: s_mom2 = 7.0
            elif 30 <= rsi < 45: s_mom2 = 5.0
            elif rsi > 70: s_mom2 = 4.0
            else: s_mom2 = 2.0

            sharpe = bt.get('strategy_2_vwap_mean_reversion', {}).get('sharpe_ratio', 0) if bt else 0
            if sharpe >= 1.5: s_mom3 = 9.5
            elif sharpe >= 1.0: s_mom3 = 7.5
            elif sharpe >= 0.5: s_mom3 = 5.5
            elif sharpe >= 0.0: s_mom3 = 3.5
            else: s_mom3 = 1.5
            
            s_mom = (s_mom1 + s_mom2 + s_mom3) / 3.0

            # ✅ 총점 및 등급 산출
            total_score = (s_growth * 0.2) + (s_prof * 0.25) + (s_moat * 0.25) + (s_val * 0.2) + (s_mom * 0.1)
            
            if total_score >= 8.5: badge = "👑 최상위 핵심 우량주"
            elif total_score >= 7.5: badge = "🥇 적격 우량주"
            elif total_score >= 6.0: badge = "⚠️ 조건부 종목"
            else: badge = "🚨 비우량주"
            
            return f"성장성({s_growth:.1f}), 수익성({s_prof:.1f}), 밸류에이션({s_val:.1f}), 해자({s_moat:.1f}), 퀀트/모멘텀({s_mom:.1f}) | 종합 평점: {total_score:.2f} / 10 ({badge})"

        def calculate_pre_risk_reward(curr_price, fib, tech):
            # 1차 목표가를 피보나치 38.2% 또는 현재가 +10%로 설정
            target_p = fib.get('fib_38.2%')
            if not isinstance(target_p, (int, float)) or target_p <= curr_price:
                target_p = curr_price * 1.10
            
            # 손절가를 2.0x ATR Stop으로 설정
            stop_p = tech.get('atr_stop_2_0x')
            if not isinstance(stop_p, (int, float)) or stop_p >= curr_price:
                stop_p = curr_price * 0.90
                
            up_pct = (target_p - curr_price) / curr_price * 100
            down_pct = (curr_price - stop_p) / curr_price * 100
            ratio = abs(up_pct / down_pct) if down_pct != 0 else 0
            
            return f"기대수익 {up_pct:+.2f}% : 예상손실 {down_pct:+.2f}% (손익비 {ratio:.2f} : 1)"

        # 실행 및 변수 할당
        precalc_scorecard = calculate_pre_scores(fund_data, tech_data, backtest_results, curr_p)
        precalc_rr = calculate_pre_risk_reward(curr_p, fib_levels, tech_data)
        # =====================================================================

        full_rag_payload = {
            "meta": {
                "ticker": ticker_input,
                "data_source": info_source_flag,
                "model_used": selected_model_label,
                "analysis_requested_at": get_current_kst_time_str(),
                "stock_data_date": stock_date
            },
            "technical_vwap_and_squeeze": tech_data,
            "volume_profile_poc_6m": vol_profile,
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
            "analyst_upgrades_downgrades_2m": analyst_data,
            "precalculated_scorecard": precalc_scorecard, 
            "precalculated_risk_reward": precalc_rr
        }

        full_json_str = json.dumps(full_rag_payload, indent=2, ensure_ascii=False)

        # 📌 Flash-Lite 모델의 주의력 집중을 위한 뉴스 타이틀 압축 정제
        compact_news = [{"title": item.get("title", ""), "publisher": item.get("publisher", ""), "date": item.get("date", "")} for item in news_data] if news_data else []
        compact_macro_news = [{"title": item.get("title", ""), "publisher": item.get("publisher", ""), "date": item.get("date", "")} for item in macro_news_data] if macro_news_data else []

        response_content = None
        if not api_key:
            response_content = "⚠️ GEMINI_API_KEY가 등록되지 않았습니다. 아래 [분석용 JSON 데이터 다운로드] 버튼으로 JSON을 내려받아 분석을 요청하세요."
        else:
            template = """
[RAG 심층 주입 데이터]
1. 기술적/수급, VWAP, 볼린저 밴드 스퀴즈 및 6개월 매물대 POC ({ticker}) (기준일: {stock_date}):
{tech_json}

2. 최근 6개월 최다 매물대(POC) 및 70% 핵심 매물대(Value Area):
{poc_json}

3. 최근 1년 과거 데이터 기반 듀얼 전략 백테스팅 결과:
{backtest_json}

4. 최근 6개월 피보나치 되돌림 밴드:
{fib_json}

5. 가장 빠른 만기 옵션 체인 수급 (콜/풋 Max OI & Volume):
{options_json}

6. 내부자/기관 지분율 및 유명 헤지펀드/공매도 세력 분석 (Short Squeeze Analysis):
{hedge_short_json}

7. 실적 발표 일정 및 52주 고저:
{earnings_json}

8. 매크로 및 6대 자산 실시간 지표 (출처 및 기준일 포함):
{macro_json}

9. 글로벌 거시/시장 주요 뉴스:
{macro_news_json}

10. S&P 500 11개 전 섹터 실시간 등락률 및 모멘텀:
{sector_json}

11. 펀더멘털 및 6대 밸류에이션 (장기 퀄리티 지표 및 R&D 비중 포함):
{fund_json}

12. 사용자 보유 현황:
{user_position}

13. 종목 최신 주요 기사:
{news_json}

14. 최근 2개월 증권가 투자의견 및 목표가 변동 (기관 신뢰도 티어 포함):
{analyst_json}

15. 파이썬 알고리즘 사전 연산 스코어카드 (절대 임의 수정 금지):
{score_json}

16. 파이썬 알고리즘 사전 연산 예상 손익비 (절대 임의 수정 금지):
{rr_json}

---

[지시사항 - 분석 정합성, 11개 섹터 전수 분석 및 POC 매물벽 검증 규칙]
위 데이터를 바탕으로 최고 수준의 퀀트/금융 애널리스트 관점에서 정밀 리포트를 작성할 것:

1. 거시환경 및 시장 국면
- **[참고자료 및 기준일자]**: 분석에 활용된 핵심 매크로 지표의 **출처 및 수집 기준일자**를 명시할 것. (⚠️경고: 각 자산별 수집 기준일자(date)가 다를 수 있으므로, 임의로 하나의 날짜로 통일하거나 왜곡하지 말고 JSON에 명시된 날짜를 개별적으로 정확히 기재할 것. **또한 JSON 값에 `NaN`, `N/A` 등으로 데이터가 누락되어 있다면 절대 임의의 숫자를 창작하지 말고 '수집 불가'로 있는 그대로 표기할 것.**)
- 경기 국면 판정 및 최신 매크로 지표/뉴스를 직접 인용하여 [6대 유동성 자산 변동 예측] (현금, 채권, 주식, 코인, 금, 원유).
- **[자산배분 코멘트 – 근거 없는 수치 생성 금지 (필수)]**: 주입된 데이터(10년물 국채금리, VIX, 달러인덱스, 유가, 금, 비트코인)만을 근거로 방향성(예: "국채 비중 확대 고려" 등)을 서술할 것. "주식 40% : 채권 30%"처럼 **구체적인 퍼센트 배분 수치는 절대 임의로 생성하지 말 것** — 그런 수치를 뒷받침할 데이터가 제공되지 않았으므로, 정량 배분표 대신 정성적 방향성만 제시할 것.

2. 11개 전 섹터 전망 및 자금 순환매 심층 분석 (서식 엄격 준수)
- **11개 섹터 전수 리스트 작성**: 주입된 11개 섹터 데이터(XLK, XLC, XLY, XLP, XLF, XLV, XLI, XLE, XLB, XLU, XLRE) 각각에 대해 5일/1개월 등락률을 바탕으로 현재 상태를 11개 모두 글머리 기호(*)로 작성할 것.
- **자금 순환매 결론 분리 (필수)**: 11개 섹터 글머리 기호 작성이 끝난 후, **반드시 빈 줄(한 줄 공백)을 삽입**하여 아래와 같이 독립된 글머리 기호로 작성할 것:
  * **자금 순환매 결론**: [방어주 vs 성장주 순환매 방향성과 {ticker}가 속한 섹터의 수혜/소외 여부 및 상대 강도를 명확히 도출. 만약 {ticker}가 원자재(GC=F 등), 가상자산(BTC-USD 등), 지수인 경우 주식 시장 전반 대비 해당 대체 자산으로의 자금 이동/선호도 관점에서 서술할 것]

3. 밸류에이션, 스마트머니(헤지펀드) 및 공매도 세력/옵션 분석 ({ticker})
- **[장기 복리 체력: 3개년 FCF·ROIC·주주환원 분석]**: 제공된 장기 퀄리티 데이터를 바탕으로, 해당 종목이 장기적으로 복리 수익을 창출할 수 있는 기초 체력을 갖추었는지 정량적으로 해석할 것.
- **유명 헤지펀드 포지션**: 13F 주요 보유 기관(Top Holders)의 지분 집중도와 스마트머니 매집 특성을 분석할 것.
- **공매도 세력 및 숏스퀴즈 리스크**: Short Float, Days to Cover, 월간 공매도 증감율을 결합하여 공매도 세력의 하방 압력 강도 및 숏스퀴즈 촉발 가능성을 평가할 것.
- **IB 투자의견 신뢰도 가중**: Tier 1/2 투자은행의 목표가 변동을 가중 평가하되, 기관 목표가는 중장기 상방 여력 참고용으로만 활용할 것.

4. 정밀 기술적 지표, VWAP, POC 매물대 및 백테스팅 평가 ({ticker})
- **스코어카드 산출**: 
  * ⚠️ [매우 중요]: LLM 본인이 직접 점수나 수식을 계산하지 마십시오. 반드시 주입된 데이터 [15. 파이썬 알고리즘 사전 연산 스코어카드]의 텍스트 결과를 **토씨 하나 틀리지 않고 그대로 복사하여 출력**할 것.

5. [신규 진입 적격성 평가 (미보유자 관점 핵심 진단)]
- **신규 진입 등급**: [적극 진입 추천 | 조정 시 분할 진입 | 돌파 확인 후 진입 | 진입 부적합(관망/리스크 과다) 중 택1]
- **진입 적합성 종합 판정**: 미보유자 입장에서 현재 시점에 무조건 하방 매수를 기다려야 하는지, 아니면 현재가 부근에서 즉시/분할 진입할 만한 모멘텀과 밸류에이션 메리트가 있는지 객관적이고 냉정하게 평가할 것. 단, 장기 퀄리티(3년 연속 흑자+고ROIC 등)가 입증된 우량 기업은 단기 눌림목 발생 시 '적극 진입' 또는 '분할 진입' 명분을 강력하게 부여할 것.
- **예상 손익비 (Risk/Reward Ratio)**: ⚠️ 직접 퍼센트를 계산하지 말고 주입된 [16. 파이썬 알고리즘 사전 연산 예상 손익비]의 텍스트를 그대로 출력할 것.

6. [정밀 매매 시나리오]
- **[매수 밴드 및 진입 가격의 상·하단 논리 일치 규칙 (필수)]**:
  * **분할 매수 밴드 설정**: 피보나치 지지선, 20일 VWAP, POC 매물대 지지선을 결합하여 실질적 달러 범위를 도출할 것. (장기 퀄리티 지표는 실시간 진입가/손절가 밴드 설정에 왜곡을 주지 않도록 매매가 산정에는 배제할 것)
  * **하단 [사용자 대응 전략]과의 가격 일치**: 사용자 대응 전략에서 언급하는 진입가는 반드시 상단 **[분할 매수 밴드]**에서 제시한 가격대와 100% 동일한 수치를 인용할 것.
- **[가격 표기 및 서식 엄격 준수 규칙]**: 
  * **가격 밴드 오름차순 표기 (필수)**: 분할 매수 밴드, 매도가 밴드 등 모든 가격 범위는 반드시 **'낮은 가격 ~ 높은 가격 (하단 ~ 상단)' 오름차순 순서로 정렬**하여 작성할 것 (예: $332.49 ~ $356.98). 절대로 높은 가격을 앞에 적지 말 것.
  * **보수적 매도가 밴드 설정**: 증권사 기관 목표가 대신 **볼린저 밴드 상단, 52주 고점 저항선, POC 매물대 저항선, 콜옵션 Max OI 저항벽 등 실제 차트/수급상의 실시간 저항선**을 최우선 기준으로 하여 현실적인 차익실현 달러 밴드를 도출할 것.
  * **피보나치 수치 표기 규칙**: 반드시 **'피보나치 50.0%', '피보나치 38.2%', '피보나치 23.6%', '피보나치 61.8%'**와 같이 소수점과 퍼센트(%) 기호를 붙여 표기할 것.
  * 비중 언급 시 '30%'처럼 퍼센트(%) 기호를 붙이고 단어와 숫자 사이에 공백을 둘 것.
- **손절(Stop-loss) 기준선**: [2.0x ATR 또는 1Y VWAP 이탈 시 추세 훼손 구체적 달러 가격대]
- **불타기 조건**: [스퀴즈 상방 돌파 및 상방 저항선 안착 시 추가 매수 검토 기준 (⚠️주의: '분할매수 밴드 상단'이라는 잘못된 명칭을 쓰지 말고, '볼린저 밴드 상단($가격)' 또는 'POC 저항선($가격)'이라고 명확한 지표명을 명시할 것)]

7. [최종 투자의견 규칙 (엄격 준수)]
- 만약 현재 과열권이거나 추격 매수를 지양해야 하는 상황, 또는 관망/보류가 유리한 국면이라면 **최종 투자의견을 절대 '매수'나 '분할매수'로 적지 말고, 반드시 '관망' 또는 '홀딩'으로 명시할 것.**
- **[최종 투자의견 톤앤매너 분리]**: 장기 퀄리티가 입증된 종목은 사용자 대응 전략 작성 시 "단기 변동성에 흔들리지 않는 장기 복리 홀딩 및 분할 차익실현"의 톤으로, 장기 지표가 부실한 종목은 "철저한 단기 스윙/트레이딩 관점의 빠른 손절 및 익절"로 전략의 성격을 명확히 갈라줄 것.

---

[리포트 최종 출력 포맷 - 반드시 아래의 구조 및 순서로 완벽히 동일하게 작성할 것]:

[1. 거시환경 및 시장 국면]
* **참고자료 및 기준일자**: [내용]
* **6대 유동성 자산 변동 및 자산배분 전략**: [내용]

[2. 11개 전 섹터 전망 및 자금 순환매 심층 분석]
* [11개 섹터 리스트 전수 나열]
* **자금 순환매 결론**: [내용]

[3. 밸류에이션, 스마트머니 및 공매도 세력/옵션 분석]
* **장기 복리 체력**: [내용]
* **유명 헤지펀드 포지션**: [내용]
* **공매도 세력 및 숏스퀴즈 리스크**: [내용]
* **IB 투자의견 신뢰도 가중**: [내용]

[4. 정밀 기술적 지표, VWAP, POC 매물대 및 백테스팅 평가]
* **스코어카드 관점**: [15번 파이썬 알고리즘 사전 연산 스코어카드 결과값 그대로 복사 출력]

[5. 신규 진입 적격성 평가]
* **신규 진입 등급**: [적극 진입 추천 | 조정 시 분할 진입 | 돌파 확인 후 진입 | 진입 부적합(관망) 중 택1]
* **진입 적합성 분석**: [미보유자 관점에서 현재 진입의 타당성, 추세 지속성 및 밸류에이션 매력도 분석]
* **예상 손익비 (Risk/Reward)**: [16번 파이썬 알고리즘 사전 연산 예상 손익비 결과값 그대로 복사 출력]

[6. 정밀 매매 시나리오]
* **분할 매수 밴드**: [피보나치 지지선, VWAP 및 POC 지지선을 결합한 구체적 달러 범위와 근거 (반드시 '낮은 가격 ~ 높은 가격' 순 정렬)]
* **1차 목표가**: [피보나치 38.2% 또는 50.0% 구간 구체적 달러 가격대와 근거]
* **2차 목표가**: [피보나치 23.6% 또는 52주 고점 인근 저항 구체적 달러 가격대와 근거]
* **매도가 밴드**: [볼린저 밴드 상단, POC 저항 및 콜옵션 저항벽을 최우선 반영한 실전 분할 차익실현 구체적 달러 밴드 (반드시 '낮은 가격 ~ 높은 가격' 순 정렬)]
* **손절(Stop-loss) 기준선**: [2.0x ATR 또는 1Y VWAP 이탈 시 추세 훼손 가격대]
* **불타기 조건**: [스퀴즈 상방 돌파 및 상방 저항선 안착 시 추가 매수 검토 기준]

[7. 최종 투자의견]
* **최종 투자의견**: [적극매수 | 분할매수 | 홀딩 | 비중축소 | 관망 | 손절 중 택1]
{strategy_guide}
"""
            prompt = PromptTemplate(
                input_variables=[
                    "ticker", "stock_date", "tech_json", "poc_json", "backtest_json", 
                    "fib_json", "options_json", "hedge_short_json", "earnings_json", 
                    "macro_json", "macro_news_json", "sector_json", "fund_json", 
                    "user_position", "strategy_guide", "news_json", "analyst_json",
                    "score_json", "rr_json"
                ],
                template=template
            )
            llm = ChatGoogleGenerativeAI(model=selected_model_id, google_api_key=api_key)
            chain = prompt | llm
            
            payload = {
                "ticker": ticker_input,
                "stock_date": stock_date,
                "tech_json": json.dumps(tech_data, indent=2, ensure_ascii=False),
                "poc_json": json.dumps(vol_profile, indent=2, ensure_ascii=False),
                "backtest_json": json.dumps(backtest_results, indent=2, ensure_ascii=False) if backtest_results else "백테스팅 데이터 부족",
                "fib_json": json.dumps(fib_levels, indent=2, ensure_ascii=False),
                "options_json": json.dumps(options_data, indent=2, ensure_ascii=False) if options_data else "옵션 데이터 없음",
                "hedge_short_json": json.dumps(hedge_short_intel, indent=2, ensure_ascii=False),
                "earnings_json": json.dumps(earnings_info, indent=2, ensure_ascii=False),
                "macro_json": json.dumps(macro_data, indent=2, ensure_ascii=False),
                "macro_news_json": json.dumps(compact_macro_news, indent=2, ensure_ascii=False),
                "sector_json": json.dumps(sector_data, indent=2, ensure_ascii=False),
                "fund_json": json.dumps(fund_data, indent=2, ensure_ascii=False),
                "user_position": user_position_text,
                "strategy_guide": strategy_instruction_text,
                "news_json": json.dumps(compact_news, indent=2, ensure_ascii=False),
                "analyst_json": json.dumps(analyst_data, indent=2, ensure_ascii=False),
                "score_json": precalc_scorecard,
                "rr_json": precalc_rr
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
            act, ent_grade, ent_rr, t1, t2, sell_b, buy_b, sl_b, pyr, u_strat_summary, q_badge = parse_full_trading_scenario(response_content)
            st.session_state.history[ticker_input] = {
                "action": act,
                "entry_grade": ent_grade,
                "entry_rr": ent_rr,
                "quality_badge": q_badge,
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
            "vol_profile": vol_profile,
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
        
# =============================================================================
# [BLOCK 10] 메인 대시보드 시각화 렌더링 (지표 카드/테이블/리포트 출력)
# =============================================================================
if st.session_state.last_analysis_result:
    res = st.session_state.last_analysis_result
    curr_p = res["curr_p"]
    ownership = res["ownership"]
    hedge_short_intel = res.get("hedge_short_intel", {})
    earnings_info = res["earnings_info"]
    tech_data = res["tech_data"]
    vol_profile = res.get("vol_profile", {})
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

    with st.container(border=True):
        st.markdown("**🏢 핵심 시장 및 재무 지표**")
        r1_c1, r1_c2, r1_c3, r1_c4 = st.columns(4)
        r1_c1.metric("현재 주가", f"${curr_p}")
        r1_c2.metric("시가총액", str(fund_data.get('market_cap_fmt', 'N/A')))
        r1_c3.metric("PER (선행/후행)", f"{fund_data.get('forward_pe', 'N/A')} / {fund_data.get('trailing_pe', 'N/A')}")
        r1_c4.metric("PBR / PSR", f"{fund_data.get('pbr', 'N/A')} / {fund_data.get('ps_ratio', 'N/A')}")
        
        # 📌 4대 우량성 & 펀더멘털 건전성 메트릭
        q_factors = fund_data.get('quality_factors', {})
        st.divider()
        st.markdown("**💎 펀더멘털 우량성 & 현금창출력 (Quality Factors)**")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("잉여현금흐름 (FCF)", str(q_factors.get('free_cash_flow', 'N/A')), "순수 현금창출")
        q2.metric("부채비율 (D/E)", str(q_factors.get('debt_to_equity', 'N/A')), "재무 건전성")
        q3.metric("매출총이익률 (GM)", str(q_factors.get('gross_margin', 'N/A')), "가격 결정력/해자")
        q4.metric("영업이익률 (OPM)", str(q_factors.get('operating_margin', 'N/A')), f"ROE: {fund_data.get('roe', 'N/A')}")

        # 📌 [신규 추가] 장기 퀄리티 & 주주환원 서브 섹션
        l_quality = fund_data.get('long_term_quality', {})
        st.markdown("**🏛️ 장기 퀄리티 & 주주환원 (Long-term Quality & Returns)**")
        lq1, lq2, lq3, lq4 = st.columns(4)
        lq1.metric("3개년 FCF 연속성", str(l_quality.get('3y_fcf_status', 'N/A')))
        lq2.metric("투하자본수익률 (ROIC)", str(l_quality.get('roic', 'N/A')))
        
        shares_chg = str(l_quality.get('shares_change_pct', 'N/A'))
        lq3.metric("주식 수 증감률 (YoY)", shares_chg, "-감소는 자사주 소각(호재)" if shares_chg != "N/A" else None)
        lq4.metric("주주환원/배당성향", str(l_quality.get('shareholder_yield', 'N/A')))

        st.divider()
        
        st.markdown("**👥 스마트머니 기본 지분 (내부자 & 기관 지분율 및 내부자 매매)**")
        own_c1, own_c2, own_c3, own_c4 = st.columns(4)
        own_c1.metric("Insider Own (내부자 지분)", str(ownership.get('insider_own', 'N/A')))
        own_c2.metric("Insider Trans (내부자 매매)", str(ownership.get('insider_trans', 'N/A')))
        own_c3.metric("Inst Own (기관 지분)", str(ownership.get('inst_own', 'N/A')))
        own_c4.metric("Inst Trans (기관 매매/보유)", str(ownership.get('inst_trans', 'N/A')))

        st.divider()

        st.markdown(f"**🐋 13F 헤지펀드 지분 & 공매도(Shorts) 수급 정밀 분석 ({res['ticker']})**")
        s_intel = hedge_short_intel.get("short_intel", {})
        
        sk1, sk2, sk3 = st.columns(3)
        sk1.metric("공매도 잔고 (Float)", s_intel.get("short_percent_of_float", "N/A"), f"MoM: {s_intel.get('short_mom_change', 'N/A')}")
        sk2.metric("상환 소요 일수 (DTC)", s_intel.get("short_ratio_days", "N/A"), "Days to Cover")
        sk3.metric("공매도 총 주수", s_intel.get("shares_short_formatted", "N/A"))
        
        st.markdown(f"**🎯 숏스퀴즈 리스크 등급:** `{s_intel.get('squeeze_risk_level', 'N/A')}`")
        
        high_52 = earnings_info.get('fiftyTwoWeekHigh', 'N/A')
        diff_52h = round(((curr_p - high_52) / high_52) * 100, 1) if isinstance(high_52, (int, float)) and curr_p else None
        e_date = earnings_info.get('earnings_date', '미정')
        e_dday = earnings_info.get('d_day', '')
        st.caption(f"📅 차기 실적 발표: **{e_date} ({e_dday})** | 52주 고점 괴리율: **{diff_52h:+.1f}%**" if diff_52h is not None else f"📅 차기 실적 발표: **{e_date}**")
        
        holders_list = hedge_short_intel.get("top_holders", [])
        if holders_list:
            df_holders = pd.DataFrame(holders_list)
            df_holders.columns = ["기관/펀드명", "보유 주식수", "지분율 (% Out)", "평가 가치"]
            st.dataframe(df_holders, use_container_width=True, hide_index=True)

        st.divider()

        st.markdown("**📅 차기 실적 발표 일정, 52주 가격 범위 & ATR/모멘텀**")
        s_c1, s_c2, s_c3, s_c4 = st.columns(4)
        s_c1.metric("차기 실적 발표일", str(earnings_info.get('earnings_date', 'N/A')), str(earnings_info.get('d_day', '')))
        
        low_52 = earnings_info.get('fiftyTwoWeekLow', 'N/A')
        s_c2.metric("52주 최고 / 최저가", f"${high_52} / ${low_52}", f"최고가 대비 {diff_52h:+.1f}%" if diff_52h is not None else None)
        
        s_c3.metric("14일 ATR (일일 변동폭)", f"${tech_data.get('atr_14', 'N/A')}", f"2.0x 손절: ${tech_data.get('atr_stop_2_0x', 'N/A')}")
        s_c4.metric(
            "MACD (Signal)", 
            f"{tech_data.get('macd', 'N/A')} ({tech_data.get('macd_signal', 'N/A')})", 
            f"Hist: {tech_data.get('macd_hist', 'N/A'):+}" if isinstance(tech_data.get('macd_hist'), (int, float)) else None
        )

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

    with st.container(border=True):
        st.markdown("##### 🧪 **퀀트 모멘텀, 스마트머니 VWAP & 6개월 최다 매물대 (POC)**")
        
        q_c1, q_c2, q_c3, q_c4 = st.columns(4)
        
        vwap_1y = tech_data.get('vwap_1y', 'N/A')
        diff_vwap1y = round(((curr_p - vwap_1y) / vwap_1y) * 100, 1) if isinstance(vwap_1y, (int, float)) and curr_p else None
        q_c1.metric("1Y 누적 VWAP (장기 평단)", f"${vwap_1y}", f"현재가 {diff_vwap1y:+.1f}%" if diff_vwap1y is not None else None)

        vwap_20d = tech_data.get('vwap_20d', 'N/A')
        diff_vwap20 = round(((curr_p - vwap_20d) / vwap_20d) * 100, 1) if isinstance(vwap_20d, (int, float)) and curr_p else None
        q_c2.metric("20일 단기 VWAP (스마트머니)", f"${vwap_20d}", f"현재가 {diff_vwap20:+.1f}%" if diff_vwap20 is not None else None)

        poc_val = tech_data.get('poc_price_6m', 'N/A')
        diff_poc = round(((poc_val - curr_p) / curr_p) * 100, 1) if isinstance(poc_val, (int, float)) and curr_p else None
        q_c3.metric("6M 최다 매물대 (POC)", f"${poc_val}", f"현재가 대비 {diff_poc:+.1f}%" if diff_poc is not None else "최대 거래량 구간")

        bb_w = tech_data.get('bb_width_pct', 'N/A')
        q_c4.metric("볼린저 밴드폭 (Bandwidth)", f"{bb_w}%" if bb_w != "N/A" else "N/A", "변동성 압축도")
        
        st.markdown(f"**⚡ 변동성 국면 판정:** `{tech_data.get('bb_squeeze_status', 'N/A')}` | **🧱 70% 핵심 매물대 밴드:** `{tech_data.get('value_area_range_6m', 'N/A')}`")

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