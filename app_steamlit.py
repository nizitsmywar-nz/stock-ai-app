# =============================================================================
# [BLOCK 01] 라이브러리 임포트 및 전역 환경 설정
# =============================================================================
import os
import json
import math
import time
import re
import logging
import warnings
from datetime import datetime, timedelta, timezone

# 💡 실시간 터미널 로깅 설정
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("StockAppLogger")

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

# [신규, 2026-09] "보유자 대상 밴드 이탈(전일 종가 기준) 판정" 기능에서, 장중에 앱을 돌릴 경우
# 최신 일봉(latest)이 아직 미완성 종가일 수 있으므로 확정 전일 종가(prev)를 대신 써야 하는지
# 판단하기 위한 미국 정규장(9:30~16:00 ET, 월~금) 개장 여부 근사치.
# [한계] 추수감사절/성탄절 등 미국 증시 휴장일은 반영하지 않음 - 별도 휴일 캘린더 라이브러리
# 의존성을 추가하지 않기 위한 의도적 근사치이며, 휴장일을 '개장'으로 오판하더라도 실제 영향은
# 밴드 이탈 판정에 전일 종가 대신 그 하루 전 종가를 쓰는 정도로 미미하다.
def is_us_market_open_now():
    try:
        from zoneinfo import ZoneInfo
        now_et = datetime.now(ZoneInfo("America/New_York"))
    except Exception:
        # tzdata 미설치 등 zoneinfo 사용 불가 시, 미국 서머타임(2007년 이후 규칙: 3월 둘째 일요일
        # 02:00 ~ 11월 첫째 일요일 02:00, UTC 기준 근사)을 수동 계산해 ET 오프셋을 추정.
        now_utc = datetime.now(timezone.utc)
        year = now_utc.year
        def _nth_sunday(y, month, n):
            d = datetime(y, month, 1, tzinfo=timezone.utc)
            first_sunday_day = 1 + (6 - d.weekday()) % 7
            return datetime(y, month, first_sunday_day + 7 * (n - 1), 2, 0, tzinfo=timezone.utc)
        dst_start = _nth_sunday(year, 3, 2)
        dst_end = _nth_sunday(year, 11, 1)
        is_dst = dst_start <= now_utc < dst_end
        offset = timedelta(hours=-4) if is_dst else timedelta(hours=-5)
        now_et = now_utc.astimezone(timezone(offset))

    if now_et.weekday() >= 5:  # 토(5)/일(6) - 주말은 항상 폐장
        return False
    market_open = now_et.replace(hour=9, minute=30, second=0, microsecond=0)
    market_close = now_et.replace(hour=16, minute=0, second=0, microsecond=0)
    return market_open <= now_et <= market_close

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
    ticker_label = getattr(stock, 'ticker', '?')
    for attempt in range(retries):
        try:
            info = stock.info
            if isinstance(info, dict) and len(info) > 10 and any(k in info for k in ['marketCap', 'trailingPE', 'forwardPE', 'trailingEps', 'bookValue', 'currentPrice']):
                return info, "stock.info"
        except Exception as e:
            # [stock.info 진단용, 2026-09] 사용자가 "10회 이상 재시도해도 전부 N/A"라고 보고한 증상의
            # 원인을 확인하기 위한 로깅. yfinance 0.2.61+ 부터 커스텀 session(requests.Session) 전달 시
            # "Yahoo API requires curl_cffi session..." 예외를 던지도록 바뀌었는데(GitHub Issue #2496),
            # 이 except가 그 예외를 조용히 삼키고 있었다면 재시도를 아무리 늘려도 매번 동일하게 즉시
            # 실패해 지금 증상과 정확히 일치한다. 아래에서 session=GLOBAL_SESSION 제거로 이 원인 자체를
            # 선제 조치했으나, 혹시 남은 실패가 있다면 이 로그로 원인을 계속 구분할 수 있게 남겨둔다.
            logger.warning(f"[stock.info 조회 실패] {ticker_label} (시도 {attempt+1}/{retries}): {type(e).__name__}: {e}")
        if attempt < retries - 1:
            time.sleep(0.5 * (attempt + 1))

    try:
        fallback_info = stock.info or {}
        if fallback_info and len(fallback_info) > 5:
            return fallback_info, "stock.info"
    except Exception as e:
        logger.warning(f"[stock.info 최종 폴백 조회 실패] {ticker_label}: {type(e).__name__}: {e}")
    return {}, "stock.fast_info"


# [yahooquery 진단 코드 제거, 2026-09] yahooquery도 동일 티커/동일 시점에 yfinance와
# 완전히 같은 "Invalid Crumb" 에러로 실패하는 것이 실제 운영 로그로 확인되어(두 라이브러리
# 모두 Yahoo의 같은 crumb 기반 인증 레이어에 의존), 라이브러리 전환은 해결책이 아님이
# 판명됨에 따라 진단용 함수와 호출부를 정리함.

@st.cache_data(ttl=300, show_spinner=False)
def fetch_stock_technical_data(ticker: str):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    
    if df.empty:
        df = stock.history(period="6mo")
        
    if df.empty:
        return {}, "N/A", {}, "N/A", "N/A", pd.DataFrame(), {}
    
    df = df.dropna(subset=['Close', 'Volume'])
    
    if df.empty:
        return {}, "N/A", {}, "N/A", "N/A", pd.DataFrame(), {}
    
    # [이평선 기준 수정, 2026-09] 60일/120일선은 한국 주식시장 관습(분기/반기 단위)이며,
    # 미국 주식시장의 표준 기준은 50일/200일선(골든크로스/데드크로스의 기준)이다.
    # 이 앱은 미국 주식을 분석하므로 미국 시장 관습에 맞춰 50일/200일로 교체한다.
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    try:
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
        has_valid_atr = df['ATR'].notna().any()
    except Exception as e:
        # [item 19 진단용] ATR N/A의 원인이 (1)상장 초기 데이터 부족(무해, 예외 미발생)인지
        # (2)실제 계산 오류(이 except가 실제로 발동하는 경우)인지 구분하기 위한 로깅.
        # 데이터 행수(len(df))를 함께 남겨, 행수가 충분한데도 여기가 찍히면 (2)번임을 확인할 수 있음.
        logger.warning(f"[ATR 계산 실패] {ticker}: {type(e).__name__}: {e} - 데이터 행수: {len(df)}")
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
    # [신규, 2026-09] 보유자 대상 "밴드 이탈(확정 종가 기준) 판정" 기능에서, 장중 실행 시 아직
    # 미완성일 수 있는 최신 봉(latest) 대신 확정된 전일 종가를 쓸 수 있도록 별도 노출.
    prev_date = df.index[-2].strftime("%Y-%m-%d") if len(df) >= 2 else last_date
    
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
        "sma_50": round(float(latest['SMA_50']), 2) if pd.notnull(latest['SMA_50']) else "N/A",
        "sma_200": round(float(latest['SMA_200']), 2) if pd.notnull(latest['SMA_200']) else "N/A",
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
        "value_area_range_6m": volume_profile.get("value_area_range", "N/A"),
        "prev_close": round(float(prev['Close']), 2) if pd.notnull(prev['Close']) else "N/A",
        "prev_close_date": prev_date
    }
    return data, last_date, fibonacci_levels, high_52w_calc, low_52w_calc, df, volume_profile
    
# =============================================================================
# [BLOCK 05] 퀀트 전략 백테스팅 엔진 (V2 - 5년치 데이터 및 OOS 분할 검증 적용)
# =============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_backtest_data(ticker: str, period: str = "5y"):
    stock = yf.Ticker(ticker)
    df = stock.history(period=period)
    if df.empty:
        df = stock.history(period="1y")  # 5년치 조회 실패 시 폴백
    if df.empty:
        return None

    df = df.dropna(subset=['Close', 'Volume'])
    if df.empty:
        return None

    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    macd = ta.trend.MACD(df['Close'])
    df['MACD_Hist'] = macd.macd_diff()
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    try:
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    except Exception as e:
        # [item 19 진단용] fetch_stock_technical_data()와 동일한 이유로 로깅 - 이쪽은
        # 백테스트 손절가(run_single_strategy 등, entry_p*0.93 폴백)에 영향을 주는 별도 지점.
        logger.warning(f"[백테스트 ATR 계산 실패] {ticker}: {type(e).__name__}: {e} - 데이터 행수: {len(df)}")
        df['ATR'] = np.nan
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_Low'] = bb.bollinger_lband()
    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_Vol'] = df['Typical_Price'] * df['Volume']
    df['Cumulative_VWAP'] = df['TP_Vol'].cumsum() / df['Volume'].cumsum()
    df['Rolling_VWAP_20'] = df['TP_Vol'].rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

    return df.dropna(subset=['SMA_20', 'MACD_Hist', 'ATR', 'Cumulative_VWAP'])

TRADE_COST_PCT = 0.002  # 왕복 거래비용 가정치 (수수료+슬리피지 합산 0.2%)

def run_strategy_backtest_v2(df: pd.DataFrame, cost_pct: float = TRADE_COST_PCT):
    if df is None or len(df) < 60:
        return None

    b_df = df.copy().dropna(subset=['Close', 'SMA_20', 'MACD_Hist', 'ATR', 'Cumulative_VWAP'])
    if len(b_df) < 30:
        return None

    bh_return = (b_df['Close'].iloc[-1] - b_df['Close'].iloc[0]) / b_df['Close'].iloc[0] * 100

    def run_single_strategy(entry_fn, exit_fn):
        pos, entry_p, entry_date = 0, 0, None
        trades = []          # (entry_date, exit_date, net_ret)
        equity_daily = []    # (date, equity_multiplier)
        equity = 1.0

        for i in range(1, len(b_df)):
            cur, prev, date = b_df.iloc[i], b_df.iloc[i - 1], b_df.index[i]

            if pos == 1:
                daily_ret = (cur['Close'] - prev['Close']) / prev['Close']
                equity *= (1 + daily_ret)

            if pos == 1 and exit_fn(cur, entry_p):
                raw_ret = (cur['Close'] - entry_p) / entry_p
                net_ret = raw_ret - cost_pct
                trades.append((entry_date, date, net_ret))
                # [개선] equity_daily를 MDD 계산에도 쓰기 위해, 청산 시 거래비용을
                # trades와 동일하게 equity 곡선에도 반영(청산일에 반영되도록 append 이전에 적용).
                equity *= (1 - cost_pct)
                pos, entry_p, entry_date = 0, 0, None

            equity_daily.append((date, equity))

            if pos == 0 and entry_fn(cur, prev):
                pos, entry_p, entry_date = 1, cur['Close'], date

        if pos == 1:
            raw_ret = (b_df['Close'].iloc[-1] - entry_p) / entry_p
            trades.append((entry_date, b_df.index[-1], raw_ret - cost_pct))

        return trades, equity_daily

    def strat1_entry(cur, prev):
        cond_macd = (prev['MACD_Hist'] <= 0 and cur['MACD_Hist'] > 0)
        cond_trend = cur['Close'] > cur['SMA_20']
        cond_vwap = cur['Close'] > cur['Rolling_VWAP_20'] if pd.notnull(cur.get('Rolling_VWAP_20')) else True
        return cond_macd and cond_trend and cond_vwap

    def strat1_exit(cur, entry_p):
        stop_price = entry_p - (1.5 * cur['ATR']) if pd.notnull(cur['ATR']) else entry_p * 0.93
        return cur['Close'] < stop_price or cur['MACD_Hist'] < 0

    def strat2_entry(cur, prev):
        return (cur['Close'] < cur['Cumulative_VWAP']) and (cur['RSI'] < 42) and (cur['Close'] > cur['BB_Low'])

    def strat2_exit(cur, entry_p):
        return cur['Close'] >= cur['Cumulative_VWAP'] or cur['RSI'] >= 65 or cur['Close'] < (cur['BB_Low'] * 0.97)

    trades1, eq1 = run_single_strategy(strat1_entry, strat1_exit)
    trades2, eq2 = run_single_strategy(strat2_entry, strat2_exit)

    def calc_stats_v2(trades, equity_daily):
        rets = [t[2] for t in trades]
        n = len(rets)
        if n == 0:
            return {"total_ret": 0.0, "win_rate": 0.0, "trades_count": 0,
                     "profit_factor": 0.0, "profit_factor_note": "거래없음",
                     "mdd": 0.0,
                     "sharpe_ratio_annualized": 0.0, "sortino_ratio_annualized": 0.0,
                     "reliability": "⚠️ 표본 없음"}

        cum = 1.0
        wins = [r for r in rets if r > 0]
        losses = [r for r in rets if r < 0]
        for r in rets:
            cum *= (1 + r)

        # [개선] MDD는 거래 단위(rets, 청산 시점 수익률만) 대신 일별 equity_daily
        # 기준으로 계산. rets 기반은 보유 중 발생했다가 청산 전에 회복된 낙폭을
        # 놓치므로 실제 리스크를 과소평가할 수 있음 - 벤치마크(Buy&Hold) MDD를
        # 일별로 계산하는 것과 측정 해상도를 맞추기 위해 함께 변경 (2026-09).
        if equity_daily:
            peak_e, mdd = 1.0, 0.0
            for _, e in equity_daily:
                peak_e = max(peak_e, e)
                mdd = max(mdd, (peak_e - e) / peak_e) if peak_e > 0 else mdd
        else:
            mdd = 0.0

        tot_ret = (cum - 1) * 100
        win_rate = (len(wins) / n) * 100
        sum_win, sum_loss = sum(wins), abs(sum(losses))

        # [개선/item26-b, 2026-09] profit_factor는 항상 숫자로 반환한다(스코어링 함수가
        # 문자열 "N/A(...)"를 _num()으로 0.0 취급해, "손실거래 0건(사실상 완벽)"과
        # "진짜로 나쁨(PF=0)"을 구분 못 하던 문제를 근본적으로 제거). 표본 부족 등 신뢰도
        # 관련 주의는 새 profit_factor_note 필드로 분리하고, 샘플 크기 임계값은 새로
        # 만들지 않고 기존 reliability 3단계(n<10/n<30/n>=30) 판단에 맡긴다.
        pf_note = ""
        if sum_loss > 0:
            pf = round(sum_win / sum_loss, 2)
        elif sum_win > 0:
            pf = 99.9
            if n < 10:
                pf_note = "표본부족·손실거래0건(참고용, 확정적 근거로 사용 금지)"
        else:
            pf = 0.0

        # [신규/item24-2, 2026-09] 소르티노 비율 추가. Sharpe와 계산 구조는 같으나 분모를
        # "전체 변동성"이 아니라 "하방 변동성(마이너스 일별 수익률만의 표준편차)"으로 써서,
        # 상방으로 크게 튀는 변동성 때문에 Sharpe가 부당하게 깎이는 전략을 구분해내기 위함
        # (퀸트 실무 리서치 근거 - item 31 참고). 하락일이 아예 없는 경우(완벽한 상승 구간)엔
        # 0으로 나누기가 아니라 item 26-b의 profit_factor 99.9 캡 패턴과 동일하게, 평균 수익률이
        # 양수면 99.9로 캡핑(진짜 완벽한 케이스와 "데이터 부족으로 0.0" 케이스를 혼동하지 않도록).
        if len(equity_daily) > 2:
            eq_series = pd.Series([e for _, e in equity_daily])
            daily_rets = eq_series.pct_change().dropna()
            sharpe_ann = round((daily_rets.mean() / daily_rets.std()) * (252 ** 0.5), 2) if daily_rets.std() > 0 else 0.0
            downside_rets = daily_rets[daily_rets < 0]
            downside_std = downside_rets.std() if len(downside_rets) >= 2 else 0.0
            if not downside_std or pd.isna(downside_std) or downside_std <= 0:
                sortino_ann = 99.9 if daily_rets.mean() > 0 else 0.0
            else:
                sortino_ann = round((daily_rets.mean() / downside_std) * (252 ** 0.5), 2)
        else:
            sharpe_ann = 0.0
            sortino_ann = 0.0

        if n < 10:
            reliability = "⚠️ 표본 부족(참고용, 확정적 근거로 사용 금지)"
        elif n < 30:
            reliability = "🔶 표본 다소 부족(제한적 신뢰)"
        else:
            reliability = "✅ 표본 충분(통계적 신뢰 가능)"

        return {
            "total_ret": round(tot_ret, 2),
            "win_rate": round(win_rate, 1),
            "trades_count": n,
            "profit_factor": pf,
            "profit_factor_note": pf_note,
            "mdd": round(mdd * 100, 2),
            "sharpe_ratio_annualized": sharpe_ann,
            "sortino_ratio_annualized": sortino_ann,
            "reliability": reliability
        }

    def split_by_period(trades, equity_daily):
        if len(trades) < 4:
            return "표본 부족으로 구간 분할 생략"
        mid = len(trades) // 2
        first_half, second_half = trades[:mid], trades[mid:]

        def _period_stat(sub):
            rets = [t[2] for t in sub]
            if not rets:
                return "거래없음"
            wr = round((len([r for r in rets if r > 0]) / len(rets)) * 100, 1)
            cum = 1.0
            for r in rets:
                cum *= (1 + r)
            tot_pct = round((cum - 1) * 100, 2)

            # [개선] 전체기간 MDD 하나만으론 그 낙폭이 전반부/후반부 중 어느 국면에서 발생했는지
            # 알 수 없어("MDD가 노이즈인지 구조적 리스크인지" 확인 불가) 절반 구간별로도 계산.
            # 전체기간 MDD와 측정 해상도를 맞추기 위해 이 구간에 해당하는 equity_daily(일별)를
            # 슬라이스해서 계산하되, "구간 시작 시점(1.0)부터 새로 계산"하는 기존 의미를 유지하려고
            # 구간 시작일 equity 값을 기준(1.0)으로 재정규화한다 - 앞 구간의 절대적인 equity 레벨이
            # 이 구간의 국소 낙폭 측정에 영향을 주지 않도록 하기 위함 (앞 구간 낙폭 이월 방지).
            half_start, half_end = sub[0][0], sub[-1][1]
            eq_slice = [(d, e) for d, e in equity_daily if half_start <= d <= half_end]
            if eq_slice:
                base_e = eq_slice[0][1]
                peak_n, mdd = 1.0, 0.0
                for _, e in eq_slice:
                    norm_e = (e / base_e) if base_e > 0 else 1.0
                    peak_n = max(peak_n, norm_e)
                    mdd = max(mdd, (peak_n - norm_e) / peak_n) if peak_n > 0 else mdd
            else:
                mdd = 0.0
            return f"{len(rets)}건, 승률{wr}%, 수익{tot_pct:+.2f}%, 구간내MDD{round(mdd * 100, 2)}%"

        return (f"전반부({first_half[0][0].date()}~{first_half[-1][1].date()}): {_period_stat(first_half)} | "
                f"후반부({second_half[0][0].date()}~{second_half[-1][1].date()}): {_period_stat(second_half)}")

    # [신규] 벤치마크(Buy&Hold) 자체의 MDD. 일별 종가로 낙폭을 추적하며,
    # 전략 쪽 MDD를 equity_daily(일별) 기준으로 바꾼 것과 동일한 해상도를 사용해야
    # Calmar 비율(수익률÷MDD) 비교가 "동일 기준" 비교가 된다 (2026-09, 아이디어 A).
    def calc_benchmark_mdd(close_series):
        base = close_series.iloc[0]
        peak, mdd = 1.0, 0.0
        for p in close_series:
            norm = (p / base) if base > 0 else 1.0
            peak = max(peak, norm)
            mdd = max(mdd, (peak - norm) / peak) if peak > 0 else mdd
        return round(mdd * 100, 2)

    # [신규/item24, 2026-09] 벤치마크(Buy&Hold)의 샤프지수. classify_vs_benchmark()가
    # 기존의 "원시 수익률 우선 게이트"(수익률에서 밀리면 Calmar를 아예 확인하지 않고
    # underperform 확정) 대신, Sharpe·Calmar 2축을 독립적으로 비교하도록 바꾸면서
    # 벤치마크 쪽에도 Sharpe가 필요해져 추가. 전략 쪽 sharpe_ann과 동일한 방식(일별
    # 수익률의 평균/표준편차 * sqrt(252))으로 계산해 "동일 기준" 비교를 유지한다.
    def calc_benchmark_sharpe(close_series):
        daily_rets = close_series.pct_change().dropna()
        if len(daily_rets) > 2 and daily_rets.std() > 0:
            return round((daily_rets.mean() / daily_rets.std()) * (252 ** 0.5), 2)
        return 0.0

    # [신규/item31 후속, 2026-09] 벤치마크의 소르티노 비율. 전략 쪽 calc_stats_v2()에 추가한
    # 것과 동일한 방식(하방 변동성만 분모로 사용, 하락일 자체가 없으면 99.9로 캡)으로 계산해
    # classify_vs_benchmark()의 Sharpe·Calmar·Sortino 3축 비교에 사용한다.
    def calc_benchmark_sortino(close_series):
        daily_rets = close_series.pct_change().dropna()
        if len(daily_rets) <= 2:
            return 0.0
        downside_rets = daily_rets[daily_rets < 0]
        downside_std = downside_rets.std() if len(downside_rets) >= 2 else 0.0
        if not downside_std or pd.isna(downside_std) or downside_std <= 0:
            return 99.9 if daily_rets.mean() > 0 else 0.0
        return round((daily_rets.mean() / downside_std) * (252 ** 0.5), 2)

    return {
        "benchmark_buy_and_hold": round(bh_return, 2),
        "benchmark_mdd": calc_benchmark_mdd(b_df['Close']),
        "benchmark_sharpe": calc_benchmark_sharpe(b_df['Close']),
        "benchmark_sortino": calc_benchmark_sortino(b_df['Close']),
        "strategy_1_momentum_squeeze": calc_stats_v2(trades1, eq1),
        "strategy_1_period_split": split_by_period(trades1, eq1),
        "strategy_2_vwap_mean_reversion": calc_stats_v2(trades2, eq2),
        "strategy_2_period_split": split_by_period(trades2, eq2),
    }

# =============================================================================
# [BLOCK 06] 시장 수급 & 외부 데이터 수집기
# =============================================================================
@st.cache_data(ttl=300, show_spinner=False)
def fetch_nearest_options_data(ticker: str, retries: int = 2):
    for attempt in range(retries):
        try:
            stock = yf.Ticker(ticker)
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
            # [OI 폴백 수정, 2026-09] 기존엔 OI가 전 종목 0/null이라 "최다 OI"를 계산할 수 없을 때
            # calls.iloc[0]/puts.iloc[0](체인의 가장 낮은 행사가)로 조용히 대체했는데, 이 값은
            # 실제 OI와 무관한 임의의 행사가라 "상방 저항벽/하방 지지벽"이라는 라벨과 함께 보여주면
            # 마치 의미 있는 수급 신호인 것처럼 오인될 수 있음(TSLA 0DTE에서 발견 - 행사가 $100
            # 콜/$50 풋처럼 현재가와 동떨어진 값이 나옴). OI 데이터가 실제로 없는 경우엔 폴백 대신
            # None을 반환해 아래에서 "N/A"로 명시한다. Vol 기준(거래량)은 이번 사례에서 정상 작동을
            # 확인했고 범위 밖이라 기존 로직 유지.
            has_call_oi = calls['openInterest'].notnull().any() and calls['openInterest'].max() > 0
            has_put_oi = puts['openInterest'].notnull().any() and puts['openInterest'].max() > 0
            call_max_oi_row = calls.loc[calls['openInterest'].idxmax()] if has_call_oi else None
            call_max_vol_row = calls.loc[calls['volume'].idxmax()] if calls['volume'].notnull().any() and calls['volume'].max() > 0 else calls.iloc[0]
            put_max_oi_row = puts.loc[puts['openInterest'].idxmax()] if has_put_oi else None
            put_max_vol_row = puts.loc[puts['volume'].idxmax()] if puts['volume'].notnull().any() and puts['volume'].max() > 0 else puts.iloc[0]
            tot_call_vol = calls['volume'].sum() if calls['volume'].notnull().any() else 0
            tot_put_vol = puts['volume'].sum() if puts['volume'].notnull().any() else 0
            pc_ratio = round(tot_put_vol / tot_call_vol, 2) if tot_call_vol > 0 else "N/A"

            def _oi_entry(row):
                if row is None:
                    return {"strike": "N/A", "oi": "N/A", "price": "N/A"}
                return {"strike": row.get("strike", "N/A"), "oi": int(row.get("openInterest", 0)) if pd.notnull(row.get("openInterest")) else 0, "price": round(float(row.get("lastPrice", 0)), 2)}

            return {
                "expiration_date": nearest_exp,
                "pc_volume_ratio": pc_ratio,
                "call_max_oi": _oi_entry(call_max_oi_row),
                "call_max_vol": {"strike": call_max_vol_row.get("strike", "N/A"), "volume": int(call_max_vol_row.get("volume", 0)) if pd.notnull(call_max_vol_row.get("volume")) else 0, "price": round(float(call_max_vol_row.get("lastPrice", 0)), 2)},
                "put_max_oi": _oi_entry(put_max_oi_row),
                "put_max_vol": {"strike": put_max_vol_row.get("strike", "N/A"), "volume": int(put_max_vol_row.get("volume", 0)) if pd.notnull(put_max_vol_row.get("volume")) else 0, "price": round(float(put_max_vol_row.get("lastPrice", 0)), 2)}
            }
        except Exception:
            if attempt < retries - 1:
                time.sleep(0.5 * (attempt + 1))
                continue
            return None
    return None

@st.cache_data(ttl=300)
def fetch_macro_indicators():
    macro_data = {}
    try:
        fred_api_key = os.getenv("FRED_API_KEY") 
        if not fred_api_key:
            try: fred_api_key = st.secrets["FRED_API_KEY"]
            except: pass
        if fred_api_key:
            url = "https://api.stlouisfed.org/fred/series/observations"
            params = {"series_id": "DGS10", "api_key": fred_api_key, "file_type": "json", "sort_order": "desc", "limit": 5}
            response = GLOBAL_SESSION.get(url, params=params, timeout=15)
            response.raise_for_status()
            data = response.json().get("observations", [])
            valid_data_found = False
            for obs in data:
                if obs["value"] != ".":
                    macro_data["us_10y_yield"] = {"source": "FRED API", "value": f"{round(float(obs['value']), 2)}%", "date": obs["date"]}
                    valid_data_found = True
                    break
            if not valid_data_found: macro_data["us_10y_yield"] = {"source": "FRED API", "value": "N/A", "date": "N/A"}
        else: macro_data["us_10y_yield"] = {"source": "FRED API", "value": "N/A", "date": "N/A"}
    except Exception: macro_data["us_10y_yield"] = {"source": "FRED API", "value": "N/A", "date": "N/A"}
    
    asset_map = {"^VIX": ("vix", "CBOE Volatility Index"), "DX-Y.NYB": ("dollar_index", "ICE US Dollar Index"), "CL=F": ("wti_oil", "NYMEX WTI Crude Oil"), "GC=F": ("gold", "COMEX Gold Futures"), "BTC-USD": ("bitcoin", "Binance/Coinbase Crypto Market")}
    try:
        tickers = list(asset_map.keys())
        df = yf.download(tickers, period="5d", progress=False)['Close']
        for ticker, (name, src_name) in asset_map.items():
            try:
                hist = df[ticker].dropna()
                if not hist.empty: macro_data[name] = {"source": src_name, "value": round(float(hist.iloc[-1]), 2), "date": hist.index[-1].strftime("%Y-%m-%d")}
                else: macro_data[name] = {"source": src_name, "value": "N/A", "date": "N/A"}
            except Exception: macro_data[name] = {"source": src_name, "value": "N/A", "date": "N/A"}
    except Exception:
        for ticker, (name, src_name) in asset_map.items(): macro_data[name] = {"source": src_name, "value": "N/A", "date": "N/A"}
    return macro_data

def format_market_cap(market_cap):
    if not market_cap or market_cap == "N/A": return "N/A"
    try:
        mc = float(market_cap)
        if mc >= 1e12: return f"${mc / 1e12:.2f}T"
        elif mc >= 1e9: return f"${mc / 1e9:.2f}B"
        elif mc >= 1e6: return f"${mc / 1e6:.2f}M"
        return f"${mc:,.0f}"
    except Exception: return str(market_cap)

def fetch_hedge_funds_and_short_intel(stock, info):
    intel = {"top_holders": [], "short_intel": {}}
    
    # 💡 총 발행 주식수 확보 (직접 계산용)
    shares_out = info.get("sharesOutstanding", None) if isinstance(info, dict) else None

    try:
        inst_df = getattr(stock, 'institutional_holders', None)
        if inst_df is not None and isinstance(inst_df, pd.DataFrame) and not inst_df.empty:
            for _, row in inst_df.head(6).iterrows():
                holder_name = str(row.get("Holder", "N/A")).strip()
                shares_val = row.get("Shares", 0)
                pct_out_val = row.get("% Out", 0)
                val_val = row.get("Value", 0)
                
                # 만약 % Out이 문자열(예: '8.3%')로 들어올 경우를 대비한 파싱
                if isinstance(pct_out_val, str):
                    try: pct_out_val = float(pct_out_val.replace('%', '')) / 100
                    except Exception: pct_out_val = 0
                
                # 💡 [핵심 패치] yfinance가 0이나 NaN을 뱉으면, 우리가 직접 발행주식수로 계산
                if (pd.isna(pct_out_val) or pct_out_val == 0) and shares_out and shares_out > 0 and pd.notnull(shares_val):
                    pct_out_val = float(shares_val) / float(shares_out)
                    
                # 출력 포맷팅
                if pd.notnull(pct_out_val) and pct_out_val > 0:
                    pct_str = f"{pct_out_val * 100:.2f}%" if pct_out_val < 1.0 else f"{pct_out_val:.2f}%"
                else:
                    pct_str = "N/A"
                    
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
        if shares_short and shares_short_prior and shares_short_prior > 0: short_mom_pct = round(((shares_short - shares_short_prior) / shares_short_prior) * 100, 2)
        squeeze_risk = "해당없음 (원자재/코인/지수)" if not short_float_pct and not short_ratio_days else "🟢 안정 (Low Risk)"
        if short_float_pct is not None and short_ratio_days is not None:
            if short_float_pct >= 20.0 and short_ratio_days >= 5.0: squeeze_risk = "🚨 숏스퀴즈 고위험 (High Squeeze Potential)"
            elif short_float_pct >= 10.0 and short_ratio_days >= 3.0: squeeze_risk = "⚠️ 숏스퀴즈 주의 (Moderate Potential)"
            elif short_float_pct >= 5.0: squeeze_risk = "💡 모니터링 구간 (Low-Moderate)"
        elif short_float_pct is not None:
            if short_float_pct >= 20.0: squeeze_risk = "🚨 숏스퀴즈 고위험 (High Squeeze Potential)"
            elif short_float_pct >= 10.0: squeeze_risk = "⚠️ 숏스퀴즈 주의 (Moderate Potential)"
            elif short_float_pct >= 5.0: squeeze_risk = "💡 모니터링 구간 (Low-Moderate)"
        intel["short_intel"] = {"short_percent_of_float": f"{short_float_pct:.2f}%" if short_float_pct is not None else "N/A", "short_ratio_days": f"{short_ratio_days:.2f}일" if short_ratio_days is not None else "N/A", "shares_short_formatted": f"{shares_short:,.0f}주" if shares_short else "N/A", "short_mom_change": f"{short_mom_pct:+.2f}%" if short_mom_pct is not None else "N/A", "squeeze_risk_level": squeeze_risk}
    except Exception: intel["short_intel"] = {"short_percent_of_float": "N/A", "short_ratio_days": "N/A", "shares_short_formatted": "N/A", "short_mom_change": "N/A", "squeeze_risk_level": "N/A"}
    return intel

def fetch_ownership_and_shorts(stock, info):
    data = {"insider_own": "N/A", "insider_trans": "N/A", "inst_own": "N/A", "inst_trans": "N/A"}
    try:
        if isinstance(info, dict):
            ins_own_val = info.get("heldPercentInsiders", None)
            if ins_own_val is not None: data["insider_own"] = f"{ins_own_val * 100:.2f}%"
            inst_own_val = info.get("heldPercentInstitutions", None)
            if inst_own_val is not None: data["inst_own"] = f"{inst_own_val * 100:.2f}%"
    except Exception: pass

    try:
        ins_df = getattr(stock, 'insider_transactions', None)
        if ins_df is not None and isinstance(ins_df, pd.DataFrame) and not ins_df.empty and 'Shares' in ins_df.columns:
            recent_ins = ins_df.head(15)
            net_shares = recent_ins['Shares'].dropna().sum()
            shares_out = info.get("sharesOutstanding", None) if isinstance(info, dict) else None
            if shares_out and shares_out > 0: data["insider_trans"] = f"{(net_shares / shares_out) * 100:+.2f}%"
            else: data["insider_trans"] = f"{net_shares:+,.0f}주"
    except Exception: pass

    try:
        inst_df = getattr(stock, 'institutional_holders', None)
        if inst_df is not None and isinstance(inst_df, pd.DataFrame) and not inst_df.empty and '% Out' in inst_df.columns: data["inst_trans"] = f"{inst_df['% Out'].sum() * 100:.2f}% (Top10)"
        elif inst_df is not None and isinstance(inst_df, pd.DataFrame) and not inst_df.empty and 'Shares' in inst_df.columns: data["inst_trans"] = f"{inst_df['Shares'].sum():,.0f}주 (Top10)"
    except Exception: pass
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
                if days_diff >= 0: d_day_str = f"D-{days_diff}일"
                else: d_day_str = f"최근 발표완료 ({abs(days_diff)}일 전)"
        elif cal is not None and isinstance(cal, pd.DataFrame) and not cal.empty:
            if 'Earnings Date' in cal.index: earnings_date_str = str(cal.loc['Earnings Date'].iloc[0])[:10]
    except Exception: pass
    high_52w = (info.get("fiftyTwoWeekHigh", None) if isinstance(info, dict) else None) or high_52_calc
    low_52w = (info.get("fiftyTwoWeekLow", None) if isinstance(info, dict) else None) or low_52_calc
    return {"earnings_date": earnings_date_str, "d_day": d_day_str, "fiftyTwoWeekHigh": high_52w, "fiftyTwoWeekLow": low_52w}

# [밸류에이션 개선안 1안] 성장률 구간별 "적정 PSR 배수". fetch_fundamentals_and_valuation()의
# PSR 목표가(growth_models['psr_target']) 산출과, calculate_pre_scores()의 s_val(PSR 성장연동
# 채점) 양쪽에서 같은 기준을 공유하도록 모듈 레벨로 승격 (기존에는 fetch_fundamentals_and_valuation
# 내부에만 있던 함수라 스코어카드 쪽에서 재사용할 수 없었음).
def _get_psr_multiple(g): return 8.0 if g and g >= 30 else (5.0 if g and g >= 15 else (3.0 if g and g >= 5 else (1.5 if g else 3.0)))

@st.cache_data(ttl=300, show_spinner=False)
def fetch_fundamentals_and_valuation(ticker: str, curr_price: float, high_52_calc, low_52_calc):
    stock = yf.Ticker(ticker)
    info, info_source = get_stock_info_with_retry(stock, retries=2)

    # (item 30-2 fade out, 2026-09-03) stock.fast_info 백업 폴백 제거.
    # 기존엔 marketCap 결측 시(ETF/ADR/신규상장 등 stock.info 자체는 정상 응답한 경우 포함)
    # 매 분석마다 stock.fast_info 프로퍼티를 무조건 조회해 시가총액만 보정하려 했으나,
    # market_cap_fmt는 UI "시가총액" 메트릭 표시(단 한 곳)에만 쓰이고 스코어카드/프롬프트에는
    # 전혀 영향을 주지 않아 실익 대비 불필요한 호출로 판단, 결측 시 그냥 N/A로 표기하도록 단순화.
    market_cap = info.get("marketCap", None) if isinstance(info, dict) else None

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
            if tot_rev > 0 and pd.notnull(rnd_val): rnd_ratio_fmt = f"{(rnd_val / tot_rev) * 100:.2f}%"
    except Exception: pass

    fcf_raw = info.get("freeCashflow", None) if isinstance(info, dict) else None
    fcf_fmt = format_market_cap(fcf_raw) if fcf_raw else "N/A"
    de_ratio = info.get("debtToEquity", None) if isinstance(info, dict) else None
    de_fmt = f"{de_ratio:.2f}%" if isinstance(de_ratio, (int, float)) else "N/A"
    gross_margin = info.get("grossMargins", None) if isinstance(info, dict) else None
    gross_margin_fmt = f"{gross_margin * 100:.2f}%" if isinstance(gross_margin, (int, float)) else "N/A"
    op_margin = info.get("operatingMargins", None) if isinstance(info, dict) else None
    op_margin_fmt = f"{op_margin * 100:.2f}%" if isinstance(op_margin, (int, float)) else "N/A"

    quality_factors = {"free_cash_flow": fcf_fmt, "debt_to_equity": de_fmt, "gross_margin": gross_margin_fmt, "operating_margin": op_margin_fmt, "rnd_to_revenue": rnd_ratio_fmt}
    long_term_quality = {"3y_fcf_status": "N/A", "roic": "N/A", "shareholder_yield": "N/A", "shares_change_pct": "N/A"}
    try:
        cf = stock.cashflow
        if not cf.empty and "Free Cash Flow" in cf.index:
            fcf_data = cf.loc["Free Cash Flow"].dropna()
            if len(fcf_data) >= 3:
                fcf_3yr = fcf_data.iloc[:3]
                if all(val > 0 for val in fcf_3yr): long_term_quality["3y_fcf_status"] = "3년 연속 흑자 (+)"
                elif all(val < 0 for val in fcf_3yr): long_term_quality["3y_fcf_status"] = "3년 연속 적자 (-)"
                else: long_term_quality["3y_fcf_status"] = "흑자/적자 혼조"
                    
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
                if invested_capital > 0: long_term_quality["roic"] = f"{(nopat / invested_capital) * 100:.2f}%"
            except Exception: pass
        
        payout_ratio = info.get("payoutRatio", None) if isinstance(info, dict) else None
        if payout_ratio is not None: long_term_quality["shareholder_yield"] = f"배당성향 {payout_ratio * 100:.1f}%"
            
        if not inc.empty and "Basic Average Shares" in inc.index:
            shares = inc.loc["Basic Average Shares"].dropna()
            if len(shares) >= 2:
                recent_shares, prev_shares = shares.iloc[0], shares.iloc[1]
                if prev_shares > 0: long_term_quality["shares_change_pct"] = f"{((recent_shares - prev_shares) / prev_shares) * 100:+.2f}%"
    except Exception: pass

    ownership_and_shorts = fetch_ownership_and_shorts(stock, info)
    hedge_and_short_intel = fetch_hedge_funds_and_short_intel(stock, info)
    earnings_cal = fetch_earnings_calendar(stock, info, high_52_calc, low_52_calc)
    earnings_growth = info.get("earningsGrowth", None) if isinstance(info, dict) else None
    revenue_growth = info.get("revenueGrowth", None) if isinstance(info, dict) else None
    shares_outstanding = info.get("sharesOutstanding", None) if isinstance(info, dict) else None

    if earnings_growth is not None and earnings_growth > 0:
        est_growth, growth_source = min(earnings_growth * 100, 35.0), "실측 EPS 성장률(YoY)"
    elif revenue_growth is not None and revenue_growth > 0:
        est_growth, growth_source = min(revenue_growth * 100 * 0.5, 20.0), "매출 성장률 기반 보수적 추정치(이익 성장 데이터 부재/마이너스)"
    else: est_growth, growth_source = None, None

    used_growth_fallback = growth_source != "실측 EPS 성장률(YoY)"
    growth_factors = {"revenue_growth_yoy": f"{revenue_growth * 100:.2f}%" if revenue_growth is not None else "N/A", "earnings_growth_yoy": f"{earnings_growth * 100:.2f}%" if earnings_growth is not None else "N/A", "growth_model_input_used": f"{est_growth:.1f}%" if est_growth is not None else "N/A", "growth_model_input_source": growth_source or "해당없음 (성장 기반 모델 산출불가)"}

    def _value_model_sanity(value, label):
        try:
            if not isinstance(value, (int, float)): return "산출불가 (재무제표 미존재/해당없음)"
            if not isinstance(curr_price, (int, float)) or curr_price <= 0: return "산출불가"
            deviation = abs(value - curr_price) / curr_price
            if deviation > 0.6 and isinstance(trailing_pe, (int, float)) and trailing_pe >= 60.0: return f"산출불가 (고PER 성장주 - 자산가치 모델 부적합, PER {trailing_pe:.1f}배)"
            if deviation > 0.6: return f"산출불가 (모델 괴리율 과다: {deviation*100:.0f}%)"
            return value
        except Exception: return "산출불가 (해당없음)"

    value_models = {}
    try: value_models["graham"] = _value_model_sanity(round(math.sqrt(22.5 * float(eps) * float(bps)), 2), "graham") if eps and bps and eps > 0 and bps > 0 else "산출불가 (해당없음)"
    except Exception: value_models["graham"] = "산출불가 (해당없음)"
    try: value_models["peter_lynch"] = _value_model_sanity(round(float(eps) * min(float(roe_raw) * 100, 25.0), 2), "peter_lynch") if eps and eps > 0 and roe_raw and roe_raw > 0 else "산출불가 (해당없음)"
    except Exception: value_models["peter_lynch"] = "산출불가 (해당없음)"
    try: value_models["roe_pbr"] = _value_model_sanity(round(float(bps) * (float(roe_raw) / 0.10), 2), "roe_pbr") if bps and bps > 0 and roe_raw and roe_raw > 0 else "산출불가 (해당없음)"
    except Exception: value_models["roe_pbr"] = "산출불가 (해당없음)"

    def _sanity_capped(value, label):
        try:
            if not isinstance(value, (int, float)): return "산출불가 (해당없음)"
            if not isinstance(curr_price, (int, float)) or curr_price <= 0: return "산출불가"
            if abs(value - curr_price) / curr_price > 0.6: return f"산출불가 (모델 괴리율 과다: {abs(value - curr_price) / curr_price * 100:.0f}%)"
            return f"{value} (참고용·추정성장률 가정치)" if used_growth_fallback else value
        except Exception: return "산출불가 (해당없음)"

    def _get_fair_peg_multiple(g): return 1.8 if g and g >= 30 else (1.3 if g and g >= 15 else (1.0 if g else None))
    # _get_psr_multiple()는 이제 모듈 레벨 함수(위쪽)로 승격되어 있어 여기서 다시 정의하지 않음.

    growth_models = {}
    f_eps = forward_eps if forward_eps and forward_eps > 0 else eps
    peg_multiple = _get_fair_peg_multiple(est_growth)
    
    try:
        if f_eps and f_eps > 0 and est_growth is not None and peg_multiple: growth_models["forward_peg"] = _sanity_capped(round(float(f_eps) * (est_growth * peg_multiple), 2), "forward_peg")
        elif est_growth is None: growth_models["forward_peg"] = "산출불가 (신뢰 가능한 성장률 데이터 없음)"
        else: growth_models["forward_peg"] = "산출불가 (해당없음)"
    except Exception: growth_models["forward_peg"] = "산출불가 (해당없음)"

    psr_multiple = _get_psr_multiple(est_growth)
    try: growth_models["psr_target"] = _sanity_capped(round(float(revenue_per_share) * psr_multiple, 2), "psr_target") if revenue_per_share and revenue_per_share > 0 else "산출불가 (해당없음)"
    except Exception: growth_models["psr_target"] = "산출불가 (해당없음)"

    def _calc_real_fcf_dcf(stock_obj, growth_rate, shares_out):
        try:
            cf = stock_obj.cashflow
            if cf.empty or "Free Cash Flow" not in cf.index: return "산출불가 (FCF 데이터 없음)"
            fcf_series = cf.loc["Free Cash Flow"].dropna()
            if fcf_series.empty: return "산출불가 (FCF 데이터 없음)"
            if not shares_out or shares_out <= 0: return "산출불가 (발행주식수 데이터 없음)"
            latest_fcf = float(fcf_series.iloc[0])
            if latest_fcf <= 0: return "산출불가 (최근 FCF 마이너스 - 성장할인모델 부적합)"
            if growth_rate is None: return "산출불가 (신뢰 가능한 성장률 데이터 없음)"
            wacc, g_long, g_start, pv_sum, cur_fcf = 0.09, 0.025, growth_rate / 100, 0.0, latest_fcf
            for y in range(1, 6):
                cur_fcf *= (1 + (g_start - (g_start - g_long) * (y - 1) / 4))
                pv_sum += cur_fcf / ((1 + wacc) ** y)
            return round((pv_sum + ((cur_fcf * (1 + g_long)) / (wacc - g_long)) / ((1 + wacc) ** 5)) / shares_out, 2)
        except Exception: return "산출불가 (해당없음)"

    raw_dcf = _calc_real_fcf_dcf(stock, est_growth, shares_outstanding)
    growth_models["dcf_growth"] = _sanity_capped(raw_dcf, "dcf_growth") if isinstance(raw_dcf, (int, float)) else raw_dcf
    growth_models["_assumptions_used"] = {"growth_rate_used": f"{est_growth:.1f}%" if est_growth is not None else "N/A", "growth_rate_source": growth_source or "해당없음", "peg_multiple_used": peg_multiple, "psr_multiple_used": psr_multiple}

    return {"info_source": info_source, "market_cap_fmt": format_market_cap(market_cap), "trailing_pe": round(trailing_pe, 2) if isinstance(trailing_pe, (int, float)) else trailing_pe, "forward_pe": round(forward_pe, 2) if isinstance(forward_pe, (int, float)) else forward_pe, "pbr": round(pbr, 2) if isinstance(pbr, (int, float)) else pbr, "ps_ratio": round(ps_ratio, 2) if isinstance(ps_ratio, (int, float)) else ps_ratio, "roe": f"{roe_pct}%" if roe_pct != "N/A" else "N/A", "target_mean_price": target_mean_price, "quality_factors": quality_factors, "long_term_quality": long_term_quality, "growth_factors": growth_factors, "ownership_and_shorts": ownership_and_shorts, "hedge_and_short_intel": hedge_and_short_intel, "earnings_calendar": earnings_cal, "value_models": value_models, "growth_models": growth_models}

@st.cache_data(ttl=300)
def fetch_sector_performance():
    sector_etfs = {"XLK": "IT/기술 (Technology)", "XLC": "커뮤니케이션 (Communication Services)", "XLY": "임의소비재 (Consumer Discretionary)", "XLP": "필수소비재 (Consumer Staples)", "XLF": "금융 (Financials)", "XLV": "헬스케어 (Health Care)", "XLI": "산업재 (Industrials)", "XLE": "에너지 (Energy)", "XLB": "소재 (Materials)", "XLU": "유틸리티 (Utilities)", "XLRE": "부동산 (Real Estate)"}
    summary = {}
    try:
        df = yf.download(list(sector_etfs.keys()), period="1mo", progress=False)['Close']
        for etf, name in sector_etfs.items():
            try:
                hist = df[etf].dropna()
                if len(hist) >= 2: summary[etf] = {"sector_name": name, "return_5d": f"{((hist.iloc[-1] - hist.iloc[-5]) / hist.iloc[-5] * 100) if len(hist) >= 5 else ((hist.iloc[-1] - hist.iloc[0]) / hist.iloc[0] * 100):+.2f}%", "return_1m": f"{((hist.iloc[-1] - hist.iloc[0]) / hist.iloc[0] * 100):+.2f}%", "latest_close": round(float(hist.iloc[-1]), 2)}
                else: summary[etf] = {"sector_name": name, "return_5d": "N/A", "return_1m": "N/A", "latest_close": "N/A"}
            except Exception: summary[etf] = {"sector_name": name, "return_5d": "N/A", "return_1m": "N/A", "latest_close": "N/A"}
    except Exception:
        for etf, name in sector_etfs.items(): summary[etf] = {"sector_name": name, "return_5d": "N/A", "return_1m": "N/A", "latest_close": "N/A"}
    return summary

@st.cache_data(ttl=300)
def fetch_news(ticker: str, limit: int = 5):
    try:
        stock = yf.Ticker(ticker)
        raw_news = getattr(stock, 'news', None)
        if not raw_news: return []
        articles = []
        for n in raw_news[:limit]:
            content = n.get("content", {})
            if isinstance(content, dict) and content:
                title, summary, publisher = content.get("title", ""), content.get("summary", ""), content.get("provider", {}).get("displayName", "Yahoo Finance")
                click_url = content.get("clickThroughUrl", {})
                link = click_url.get("url", "") if isinstance(click_url, dict) else click_url
                if not link: link = content.get("canonicalUrl", {}).get("url", "")
                pub_date = str(content.get("pubDate", "최근"))[:10]
            else:
                title, summary, publisher, link = n.get("title", ""), "", n.get("publisher", "Yahoo Finance"), n.get("link", "")
                pub_time = n.get("providerPublishTime", None)
                pub_date = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d") if pub_time else "최근"
            if title: articles.append({"title": title, "summary": summary, "publisher": publisher, "date": pub_date, "link": link or f"https://finance.yahoo.com/quote/{ticker}"})
        return articles
    except Exception: return []

@st.cache_data(ttl=300)
def fetch_macro_news(limit: int = 4):
    macro_articles = []
    for sym in ["SPY", "TLT"]:
        try:
            stock = yf.Ticker(sym)
            raw = getattr(stock, 'news', None)
            if raw:
                for n in raw[:2]:
                    content = n.get("content", {})
                    if isinstance(content, dict) and content:
                        title, summary, publisher = content.get("title", ""), content.get("summary", ""), content.get("provider", {}).get("displayName", "MarketWatch")
                        click_url = content.get("clickThroughUrl", {})
                        link = click_url.get("url", "") if isinstance(click_url, dict) else click_url
                        if not link: link = content.get("canonicalUrl", {}).get("url", "")
                        pub_date = str(content.get("pubDate", "최근"))[:10]
                    else:
                        title, summary, publisher, link = n.get("title", ""), "", n.get("publisher", "MarketWatch"), n.get("link", "")
                        pub_time = n.get("providerPublishTime", None)
                        pub_date = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d") if pub_time else "최근"
                    if title and not any(a["title"] == title for a in macro_articles): macro_articles.append({"title": title, "summary": summary, "publisher": publisher, "date": pub_date, "link": link or f"https://finance.yahoo.com/quote/{sym}"})
        except Exception: pass
    return macro_articles[:limit]

def extract_clean_text(content):
    if isinstance(content, str): return content
    elif isinstance(content, list): return "\n".join([p["text"] if isinstance(p, dict) and "text" in p else str(p) for p in content])
    return str(content)
    
# =============================================================================
# [BLOCK 07] 증권사 투자의견 & LLM 응답 파서
# =============================================================================
def summarize_user_strategy(raw_text: str) -> str:
    if not raw_text or raw_text == "분석 리포트 참조": return "분석 리포트 참조"
    text = re.sub(r'\s+', ' ', raw_text.replace("\n", " ").strip())
    if len(text) > 300:
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 3]
        collected, cur_len = [], 0
        for s in sentences:
            collected.append(s)
            cur_len += len(s)
            if cur_len >= 220: break
        res = " ".join(collected)
        return res if res.endswith((".", "!", "?")) else res + "..."
    return text

def parse_full_trading_scenario(text):
    action, entry_grade, entry_rr, target_1, target_2, sell_target, buy_band, stop_loss, pyramiding, add_buy_note, user_strategy_raw, quality_badge = "홀딩", "분석 리포트 참조", "분석 리포트 참조", "분석 리포트 참조", "", "분석 리포트 참조", "분석 리포트 참조", "분석 리포트 참조", "", "", "", ""
    if "최상위 핵심 우량주" in text or "👑" in text: quality_badge = "👑 "
    elif "적격 우량주" in text or "🥇" in text: quality_badge = "🥇 "
    elif "조건부 종목" in text or "⚠️" in text: quality_badge = "⚠️ "
    elif "비우량주" in text or "🚨" in text: quality_badge = "🚨 "

    match_action = re.search(r"(?:최종\s*투자의견|최종투자\s*의견)[^:\n]*[:\-]?\s*([^\n\r]+)", text)
    if match_action:
        op_text = match_action.group(1).replace("*", "").replace("[", "").replace("]", "").strip()
        if "적극매수" in op_text or ("매수" in op_text and "관망" not in op_text and "보유" not in op_text and "홀딩" not in op_text): action = "매수"
        elif "매도" in op_text or "비중축소" in op_text or "차익실현" in op_text or "손절" in op_text: action = "매도"
        elif "홀딩" in op_text or "보유" in op_text or "관망" in op_text: action = "홀딩"

    match_entry_sec = re.search(r"\[신규\s*진입\s*적격성\s*평가\](.*?)(?=\[(?:정밀\s*매매\s*시나리오|최종\s*투자의견)|\Z)", text, re.DOTALL)
    if match_entry_sec:
        for line in match_entry_sec.group(1).split("\n"):
            line_clean = line.replace("*", "").replace("-", "").replace("•", "").strip()
            if ("신규 진입 등급" in line_clean or "진입 등급" in line_clean) and ":" in line_clean: entry_grade = ":".join(line_clean.split(":")[1:]).strip()
            elif ("예상 손익비" in line_clean or "손익비" in line_clean) and ":" in line_clean: entry_rr = ":".join(line_clean.split(":")[1:]).strip()

    match_scen_sec = re.search(r"\[정밀\s*매매\s*시나리오\](.*?)(?=\[(?:최종\s*투자의견)|\Z)", text, re.DOTALL)
    scenario_block = match_scen_sec.group(1) if match_scen_sec else text
    for line in scenario_block.split("\n"):
        line_clean = line.replace("*", "").replace("-", "").replace("•", "").strip()
        if ("1차 목표가" in line_clean or "1차목표가" in line_clean) and target_1 == "분석 리포트 참조" and ":" in line_clean: target_1 = ":".join(line_clean.split(":")[1:]).strip()
        elif ("2차 목표가" in line_clean or "2차목표가" in line_clean) and not target_2 and ":" in line_clean: target_2 = ":".join(line_clean.split(":")[1:]).strip()
        elif ("매도가 밴드" in line_clean or "비중축소" in line_clean or "매도가" in line_clean) and sell_target == "분석 리포트 참조" and ":" in line_clean: sell_target = ":".join(line_clean.split(":")[1:]).strip()
        elif ("분할 매수 밴드" in line_clean or "분할매수 밴드" in line_clean) and buy_band == "분석 리포트 참조" and ":" in line_clean: buy_band = ":".join(line_clean.split(":")[1:]).strip()
        elif ("손절" in line_clean or "Stop-loss" in line_clean) and stop_loss == "분석 리포트 참조" and ":" in line_clean: stop_loss = ":".join(line_clean.split(":")[1:]).strip()
        elif ("불타기 조건" in line_clean or "불타기" in line_clean) and not pyramiding and ":" in line_clean: pyramiding = ":".join(line_clean.split(":")[1:]).strip()
        # [버그수정] "비중 확대"만으로도 매칭되던 조건 제거 - 이 표현은 매크로/자산배분 코멘트 등
        # 다른 문장에도 흔히 등장해서, 실제 "추가 매수 여부:" 줄에 도달하기 전에 엉뚱한
        # 줄을 먼저 캡처해버리는 오탐이 발생했음(not add_buy_note 가드로 한번 잡히면 고정됨).
        # [물타기→추가 매수 여부 리네이밍, 2026-09] 레이블 문구를 "물타기"에서 "추가 매수"로
        # 바꾸면서 매칭 키워드도 함께 교체 - 레이블은 항상 "추가 매수"를 포함하므로 이 단어만
        # 매칭해도 충분함(과거 "물타기" 표기로 저장된 히스토리는 render 단계에서 구버전 키로
        # 별도 폴백 처리하며, 이 정규식은 앞으로 새로 생성되는 리포트만 대상으로 함).
        elif "추가 매수" in line_clean and not add_buy_note and ":" in line_clean: add_buy_note = ":".join(line_clean.split(":")[1:]).strip()

    match_opinion_sec = re.search(r"\[최종\s*투자의견\](.*?)(?=\Z|\[|\n\n#)", text, re.DOTALL)
    opinion_block = match_opinion_sec.group(1) if match_opinion_sec else text
    strategy_lines, collecting_strategy = [], False
    for line in opinion_block.split("\n"):
        line_clean = line.replace("*", "").replace("-", "").replace("•", "").replace("◦", "").strip()
        if "사용자 대응 전략" in line_clean or "사용자대응전략" in line_clean:
            content_after = ":".join(line_clean.split(":")[1:]).strip() if ":" in line_clean else ""
            if content_after: strategy_lines.append(content_after)
            collecting_strategy = True
            continue
        if collecting_strategy:
            if line.strip().startswith("[") or line.strip().startswith("#"): collecting_strategy = False
            elif line_clean: strategy_lines.append(line_clean)

    if strategy_lines: user_strategy_raw = " ".join(strategy_lines)
    return action, entry_grade, entry_rr, target_1, target_2, sell_target, buy_band, stop_loss, pyramiding, add_buy_note, summarize_user_strategy(user_strategy_raw), quality_badge

TIER_1_FIRMS = ["goldman", "morgan stanley", "jpmorgan", "jp morgan", "citi", "citigroup", "bank of america", "bofa", "merrill", "ubs", "barclays", "deutsche bank", "hsbc", "bernstein", "credit suisse", "bnp paribas"]
TIER_2_FIRMS = ["wells fargo", "rbc", "mizuho", "jefferies", "piper sandler", "wedbush", "baird", "oppenheimer", "bmo", "stifel", "td cowen", "cowen", "wolfe", "keybanc", "raymond james", "canaccord", "evercore", "truist", "guggenheim", "btig", "da davidson", "needham", "mmpm", "loop capital", "roth mkm", "bernstein"]

def classify_analyst_tier(firm_name: str):
    f_lower = firm_name.lower()
    if any(k in f_lower for k in TIER_1_FIRMS): return "🌟 Tier 1 (글로벌 탑티어)", 1
    elif any(k in f_lower for k in TIER_2_FIRMS): return "✨ Tier 2 (주요 전문리서치)", 2
    else: return "🔎 Tier 3 (독립/부티크)", 3

@st.cache_data(ttl=300, show_spinner=False)
def fetch_recent_upgrades_downgrades(ticker: str, months: int = 2):
    try:
        stock = yf.Ticker(ticker)
        upgrades = stock.upgrades_downgrades
        if upgrades is None or upgrades.empty: return []
        cutoff_date = datetime.now() - timedelta(days=months * 30)
        if isinstance(upgrades.index, pd.DatetimeIndex): filtered = upgrades[upgrades.index >= cutoff_date.strftime("%Y-%m-%d")]
        elif 'Date' in upgrades.columns:
            upgrades['Date'] = pd.to_datetime(upgrades['Date'])
            filtered = upgrades[upgrades['Date'] >= cutoff_date]
        else: filtered = upgrades.head(8)
            
        records = []
        for idx, row in filtered.head(8).iterrows():
            date_str = idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx)[:10]
            firm_name, from_g, to_g, action_raw = str(row.get("Firm", "N/A")).strip(), str(row.get("FromGrade", "")).strip(), str(row.get("ToGrade", "")).strip(), str(row.get("Action", "N/A")).strip()
            grade_str = f"{from_g} ➡️ {to_g}" if from_g and from_g.lower() != "nan" and from_g != to_g else (to_g if to_g and to_g.lower() != "nan" else "N/A")
            target_val = row.get("currentPriceTarget", None) or row.get("priceTarget", None) or row.get("TargetPrice", None)
            target_str = f"${float(target_val):.2f}" if pd.notnull(target_val) and target_val != "" else "-"
            tier_badge, tier_num = classify_analyst_tier(firm_name)
            records.append({"date": date_str, "firm": firm_name, "tier": tier_badge, "tier_num": tier_num, "action": action_raw, "grade_change": grade_str, "target_price": target_str})
        return records
    except Exception: return []
        
# =============================================================================
# [BLOCK 08] 사이드바 UI 컴포넌트 & 히스토리 카드 렌더링
# =============================================================================
with st.sidebar:
    st.markdown("### ⚡ **AI Stock Analyst Pro**")
    
    MODEL_OPTIONS = {
        "Gemini 3.5 Flash Lite": "gemini-3.5-flash-lite",
        "Gemini 3.1 Flash Lite": "gemini-3.1-flash-lite",
        "Gemini 3.6 Flash": "gemini-3.6-flash"
    }
    
    selected_model_label = st.selectbox("🤖 **AI 추론 모델 선택**", options=list(MODEL_OPTIONS.keys()), index=0)
    selected_model_id = MODEL_OPTIONS[selected_model_label]
    
    st.markdown("---")
    ticker_input = st.text_input("종목 티커 (Ticker)", value=st.session_state.selected_ticker).upper()
    is_holding = st.checkbox("💼 **현재 보유 중인 종목인가요?**", value=False)
    
    user_avg_price, user_shares = 0.0, 0.0
    if is_holding:
        u_col1, u_col2 = st.columns(2)
        with u_col1: user_avg_price = st.number_input("내 평단가 ($)", min_value=0.0, value=0.0, step=0.5, format="%.2f")
        with u_col2: user_shares = st.number_input("보유 수량 (주)", min_value=0.0, value=0.0, step=1.0, format="%.1f")
            
    st.write("")
    analyze_btn = st.button("🚀 분석 & 백테스팅 실행", type="primary", use_container_width=True)
    st.divider()

    st.markdown("#### 📌 **트레이딩 히스토리**")
    if st.session_state.history:
        tab_all, tab_buy, tab_sell, tab_hold = st.tabs(["전체", "🟢매수", "🔴매도", "🟡홀딩"])
        
        def render_history_card(t_code, data):
            action_badge = "🟢 매수" if data['action'] == "매수" else ("🔴 매도" if data['action'] == "매도" else "🟡 홀딩")
            q_badge = data.get('quality_badge', '')
            with st.expander(f"{q_badge}**{t_code}** (${data['price']}) | {action_badge}", expanded=False):
                st.markdown(f"- **현재가:** `${data['price']}`")
                entry_grade_val = data.get('entry_grade', '')
                if entry_grade_val and entry_grade_val != "분석 리포트 참조": st.markdown(f"- **🆕 신규 진입 판정:** `{entry_grade_val}`")
                if data.get('my_avg', 0) > 0: st.markdown(f"- **💼 내 평단:** `${data['my_avg']}` ({data.get('my_return', 'N/A')})")
                t1, t2 = data.get('target_1') or data.get('take_profit', '분석 리포트 참조'), data.get('target_2', '')
                st.markdown(f"- **🎯 1차 목표가:** `{t1}`")
                if t2: st.markdown(f"- **🎯 2차 목표가:** `{t2}`")
                st.markdown(f"- **📤 매도가 밴드:** `{data.get('sell_target', '분석 리포트 참조')}`")
                st.markdown(f"- **📥 분할매수 밴드:** `{data.get('buy_band', '분석 리포트 참조')}`")
                st.markdown(f"- **🛑 손절선:** `{data.get('stop_loss', '분석 리포트 참조')}`")
                if data.get('pyramiding'): st.markdown(f"- **🔥 불타기 조건:** `{data['pyramiding']}`")
                # [물타기→추가 매수 여부 리네이밍, 2026-09] 저장 키를 'additional_buy'로 바꿨지만,
                # 이미 저장돼 있던 과거 히스토리 항목은 여전히 구키 'averaging_down'으로 남아있으므로
                # 신키 우선 조회 + 구키 폴백으로 과거 데이터 내용은 손대지 않고 라벨만 새 용어로 통일.
                add_buy_display = data.get('additional_buy') or data.get('averaging_down')
                if add_buy_display: st.markdown(f"- **💧 추가 매수 여부:** `{add_buy_display}`")
                strat_text = data.get('user_strategy', '')
                st.markdown(f"- **💡 대응 전략:** `{strat_text if strat_text and strat_text != '분석 리포트 참조' else '분석 리포트 참조'}`")
                st.caption(f"분석 일시(KST): {data.get('time', 'N/A')}")

        with tab_all:
            for t_code, data in list(st.session_state.history.items())[::-1]: render_history_card(t_code, data)
        with tab_buy:
            buy_items = [item for item in list(st.session_state.history.items())[::-1] if item[1]['action'] == "매수"]
            if buy_items:
                for t_code, data in buy_items: render_history_card(t_code, data)
            else: st.caption("매수 판정 내역이 없습니다.")
        with tab_sell:
            sell_items = [item for item in list(st.session_state.history.items())[::-1] if item[1]['action'] == "매도"]
            if sell_items:
                for t_code, data in sell_items: render_history_card(t_code, data)
            else: st.caption("매도 판정 내역이 없습니다.")
        with tab_hold:
            hold_items = [item for item in list(st.session_state.history.items())[::-1] if item[1]['action'] == "홀딩"]
            if hold_items:
                for t_code, data in hold_items: render_history_card(t_code, data)
            else: st.caption("홀딩 판정 내역이 없습니다.")
                    
        st.write("")
        if st.button("🗑️ 히스토리 전체 삭제", use_container_width=True):
            st.session_state.history = {}
            if os.path.exists(HISTORY_FILE): os.remove(HISTORY_FILE)
            st.rerun()
    else: st.caption("분석 내역이 여기에 영구 저장됩니다.")
# =============================================================================
# [BLOCK 09] LLM RAG 추론 파이프라인 & 헬퍼 함수
# =============================================================================
# [패치 1] 목표가 산출 및 손익비
def grade_entry_by_rr(ratio):
    # [개선] 신규 진입 등급을 손익비(RR ratio) 기반으로 완전 사전계산 - LLM 자유판단으로 인한
    # "등급은 B인데 손익비는 0.07:1" 같은 내부 모순을 구조적으로 차단하기 위함.
    if not isinstance(ratio, (int, float)):
        return "판정불가 (손익비 계산 불가)"
    if ratio >= 2.0: return "A (매력적 진입 기회 - 손익비 우수)"
    elif ratio >= 1.0: return "B (준수한 진입 기회 - 손익비 양호)"
    elif ratio >= 0.5: return "C (조건부 관심 - 손익비 다소 불리, 신중한 분할 접근 권장)"
    else: return "D (진입 비추천 - 손익비 열위, 되돌림/조정 이후 재평가 권장)"

def calculate_targets_and_risk_reward(curr_price, fib, tech):
    fib_pairs = [
        ("피보나치 23.6%", fib.get('fib_23.6%')),
        ("피보나치 38.2%", fib.get('fib_38.2%')),
        ("피보나치 50.0%", fib.get('fib_50.0%')),
        ("피보나치 61.8%", fib.get('fib_61.8%')),
    ]
    valid = [(label, p) for label, p in fib_pairs if isinstance(p, (int, float))]
    valid.sort(key=lambda x: x[1])  # 가격 오름차순

    resistances = [(label, p) for label, p in valid if p > curr_price]

    # [개선] 피보나치 레벨 부재 시 폴백 목표가를 고정 비율(+10%/+20%) 대신 R-멀티플(R=ATR×2.0, 손절거리와 동일 단위)
    # 기반으로 산출 - 1차 목표가 1.5R(앱이 이미 쓰는 최소 손익비 기준 1.5와 동일), 2차 목표가 3.0R.
    # ATR 데이터가 없는 예외 상황에서는 기존 고정 비율 방식을 세이프티넷으로 유지.
    atr14 = tech.get('atr_14')
    has_atr = isinstance(atr14, (int, float)) and atr14 > 0

    if len(resistances) >= 2:
        t1_label, t1_price = resistances[0]
        t2_label, t2_price = resistances[1]
    elif len(resistances) == 1:
        t1_label, t1_price = resistances[0]
        if has_atr:
            t2_price = round(curr_price + atr14 * 6.0, 2)
            t2_label = "현재가 +3.0R(ATR×6.0배, 피보나치 2차 레벨 부재)"
        else:
            t2_label, t2_price = "현재가 +20%(피보나치 레벨 부재)", round(curr_price * 1.20, 2)
    else:
        if has_atr:
            t1_price = round(curr_price + atr14 * 3.0, 2)
            t1_label = "현재가 +1.5R(ATR×3.0배, 피보나치 레벨 부재)"
            t2_price = round(curr_price + atr14 * 6.0, 2)
            t2_label = "현재가 +3.0R(ATR×6.0배, 피보나치 레벨 부재)"
        else:
            t1_label, t1_price = "현재가 +10%(피보나치 레벨 부재)", round(curr_price * 1.10, 2)
            t2_label, t2_price = "현재가 +20%(피보나치 레벨 부재)", round(curr_price * 1.20, 2)

    stop_p = tech.get('atr_stop_2_0x')
    if not isinstance(stop_p, (int, float)) or stop_p >= curr_price:
        stop_p = round(curr_price * 0.90, 2)

    up_pct = (t1_price - curr_price) / curr_price * 100
    down_pct = (curr_price - stop_p) / curr_price * 100
    ratio = abs(up_pct / down_pct) if down_pct != 0 else 0

    targets_text = f"1차 목표가: {t1_label} (${t1_price}) | 2차 목표가: {t2_label} (${t2_price})"
    rr_text = f"(1차 목표가 {t1_label} ${t1_price} 기준) 기대수익 {up_pct:+.2f}% : 예상손실 {down_pct:+.2f}% (손익비 {ratio:.2f} : 1)"
    entry_grade_text = grade_entry_by_rr(ratio)
    # [패치7 연동] 손절가(stop_p)를 호출부(추가 매수 여부 2단계 게이트)에서도 쓸 수 있도록 함께 반환
    return targets_text, rr_text, t1_price, t2_price, t1_label, t2_label, stop_p, entry_grade_text

# [신규, 2026-09] 보유자 대상 "밴드 이탈(확정 종가 기준) → 손절 고려" 전용 판정 (사용자 요청:
# "기보유자 기준 전일 종가 기준으로 밴드를 이탈했으니 손절을 고려하는 로직은 없다" 논의 → 승인).
# 확정 기준 가격(basis_price - 장중이면 확정 전일 종가 prev_close, 마감 후면 최신 종가)을
# 이미 계산된 stop_price(=분할 매수 밴드 하단, atr_stop_2_0x)와 비교만 할 뿐, 지표/밴드 수치
# 자체는 전혀 재계산/조정하지 않는다 - 기존 스탠딩 제약("강제로 조정하는건 지표분석 결과를
# 훼손하는 행위")과 상충하지 않음. Section 7 전용 forced 필드([[F18]])로 주입되어 LLM이
# 문구를 완화/누락/왜곡할 수 없다.
def evaluate_band_breach_for_holder(is_holding, basis_price, stop_price, basis_label, basis_date):
    if not is_holding:
        return "해당없음 (미보유 - 신규 진입 검토 대상이라 보유 포지션 밴드 이탈 판정 비적용)"
    if not isinstance(basis_price, (int, float)) or not isinstance(stop_price, (int, float)):
        return "판정불가 (가격 또는 밴드 하단 데이터 부족)"
    if basis_price <= stop_price:
        return (f"🚨 이탈함 - {basis_label} ${basis_price}({basis_date} 기준)가 분할 매수 밴드 하단(${stop_price}) 이하로 마감. "
                f"보유 포지션에 대해 추가 매수 검토보다 전량 손절(리스크 관리)을 최우선으로 고려할 것.")
    else:
        return (f"이탈 안 함 - {basis_label} ${basis_price}({basis_date} 기준)는 분할 매수 밴드 하단(${stop_price}) 위에서 마감. "
                f"현재까지는 밴드 이탈에 따른 손절 트리거가 발동하지 않은 상태.")

# [패치 2] 사용자 대응 전략 프롬프트
# [개선, 2026-09] "ATR 기준선(-$X / -$Y) 이탈 시 리스크 관리" 처럼 1.5배/2.0배 두 ATR 선을
# 뭉뚱그려 인용하는 문제 수정. 이 앱의 손절가(stop_p)는 항상 atr_stop_2_0x이며(계산가/게이트
# 로직에서 실제로 쓰이는 값도 이것), [6. 정밀 매매 시나리오]의 [분할 매수 밴드] 하단도 이 선
# 위쪽/동일선상에서 서술되도록 유도되어 있음 - 즉 "밴드를 이탈했다"와 "2.0배 선을 이탈했다"는
# 사실상 같은 사건. 반면 1.5배 선은 밴드 '내부'에 위치하는 경우가 많아, 이를 밴드 이탈과 같은
# 급의 트리거로 서술하면 "밴드 안에서 분할 매수하라면서 그 안의 한 지점을 이탈이라 부르는"
# 자기모순이 생김. ⚠️ 이 함수는 지표/밴드 '수치'를 전혀 건드리지 않으며, LLM이 이미 계산된
# 두 ATR 값을 어떤 '말'로 구분해서 서술할지에 대한 지시문만 추가한다 (사용자 승인: "응 명시해줘").
def build_strategy_instruction(is_holding, user_avg_price, user_shares, curr_price, my_return_str, t1_price, t2_price, t1_label, t2_label, atr_1_5=None, atr_2_0=None, band_breach_text=None):
    atr_distinction_note = ""
    if isinstance(atr_1_5, (int, float)) and isinstance(atr_2_0, (int, float)):
        atr_distinction_note = f"""
⚠️[ATR 기준선 표현 규칙 - 절대 준수] 위 [6. 정밀 매매 시나리오]의 손절(Stop-loss) 기준선에는 1.5배 손절선(-${atr_1_5})과 2.0배 손절선(-${atr_2_0}) 두 값이 함께 제시되어 있으나, 성격이 다르므로 반드시 구분해서 서술하고 "ATR 기준선(-$A / -$B) 이탈 시"처럼 두 값을 묶어 인용하지 말 것:
- 2.0배 손절선(-${atr_2_0})은 [분할 매수 밴드]의 하단과 사실상 동일한 선이다. 종가 기준 이 가격 아래로 이탈하면 '밴드 이탈(밴드 자체가 무너진 것)'로 간주하여 리스크 관리(전량 손절 등)를 권고할 것 - 이때 반드시 "ATR 기준선 이탈"이 아니라 "분할 매수 밴드 하단(${atr_2_0}) 이탈"이라고 표현할 것.
- 1.5배 손절선(-${atr_1_5})은 밴드 내부에 위치한, 더 타이트한 개별 포지션 손절 옵션일 뿐이다. 이미 보유 중인 개별 포지션에 대해 좀 더 보수적으로 대응하고 싶은 사용자가 선택적으로 참고할 수 있는 라인이라고만 서술하고, 이 선을 밴드 이탈이나 시나리오 전체의 종료 신호와 동일시하지 말 것."""

    # [신규, 2026-09] 바로 위 [7. 최종 투자의견]에 강제 주입되는 [[F18]] "밴드 이탈 판정(보유자
    # 전용)"과 이 자유 서술 전략이 서로 모순되지 않도록 하는 정합성 지시문. F18가 이미 "이탈함"
    # 으로 확정했는데 이 문단이 "아직 밴드 안에 있다"는 식으로 서술하면 리포트 내부 모순이 생김.
    band_breach_note = ""
    if is_holding and isinstance(band_breach_text, str) and band_breach_text.startswith("🚨 이탈함"):
        band_breach_note = "\n⚠️[밴드 이탈 판정 반영 필수] 바로 위 [[F18]] 밴드 이탈 판정 결과, 확정 종가 기준으로 이미 분할 매수 밴드 하단을 이탈한 상태입니다. 이 사실을 부정하거나 회피하지 말고, 반드시 이를 최우선 근거로 전량 손절(리스크 관리)을 명확히 권고할 것."

    if is_holding and user_avg_price > 0:
        def pnl_label(target_price):
            if not isinstance(target_price, (int, float)): return "손익 판단 불가(가격 없음)"
            pnl_vs_avg = (target_price - user_avg_price) / user_avg_price * 100
            if pnl_vs_avg > 0: return f"익절 구간 (평단가 대비 {pnl_vs_avg:+.2f}%)"
            else: return f"손실 축소 매도 구간 (평단가 대비 {pnl_vs_avg:+.2f}%, 여전히 손실 상태)"

        t1_pnl_label = pnl_label(t1_price)
        t2_pnl_label = pnl_label(t2_price)

        return f"""* **사용자 대응 전략**: [현재 사용자가 평단가 ${user_avg_price:.2f}, 평가수익률 {my_return_str}로 주식을 보유 중인 상태입니다.
반드시 '보유자 관점'의 전략만 단독 작성할 것. 미보유자나 신규 진입 관련 문구는 일절 작성하지 말 것.
⚠️[매우 중요 - 절대 위반 금지] 1차 목표가({t1_label}, ${t1_price})는 사용자 평단가 대비 {t1_pnl_label}입니다.
2차 목표가({t2_label}, ${t2_price})는 평단가 대비 {t2_pnl_label}입니다.
목표가가 평단가보다 낮은 경우 절대 '익절'이라는 단어를 쓰지 말고 반드시 '손실 축소 매도' 또는 '비중 축소'라고 표현할 것.
목표가가 평단가보다 높은 경우에만 '익절'이라는 표현을 사용할 것.
손절선 이탈 시 전량 손절 계획을 명시할 것.{atr_distinction_note}{band_breach_note}]"""
    else:
        return f"""* **사용자 대응 전략**: [현재 사용자가 주식을 보유하지 않은 '미보유 상태'입니다. 반드시 '미보유자 신규 진입 관점'의 전략만 단독 작성할 것. 보유자 관련 문구는 일절 작성하지 말 것. 상단 [신규 진입 적격성 평가] 및 [분할 매수 밴드]와 100% 일치하는 진입 가격대와 진입 비중(예: 1차 30% 분할 진입 등)을 명확히 제시할 것.{atr_distinction_note}]"""

# [패치 3] 백테스트-기술적 신호 정합성 체크 (기간 동적 연동)
def build_backtest_consistency_note(backtest_results, bb_squeeze_status, bt_years):
    # [버그수정] 신규 상장 종목 등 데이터 부족으로 backtest_results가 None인 경우 AttributeError 방지
    squeeze_bt = (backtest_results or {}).get('strategy_1_momentum_squeeze', {})
    win_rate = squeeze_bt.get('win_rate')
    total_ret = squeeze_bt.get('total_ret')
    if "스퀴즈" in str(bb_squeeze_status) and isinstance(win_rate, (int, float)) and win_rate < 40:
        return f"⚠️ 현재 '{bb_squeeze_status}' 신호가 있으나, 동일 로직인 모멘텀 스퀴즈 전략은 최근 {bt_years}년 백테스트에서 승률 {win_rate:.1f}%, 총수익 {total_ret:+.2f}%로 부진했습니다. 이 신호를 매매 근거로 사용할 경우, 반드시 이 백테스트 성과를 함께 언급하고 신호의 신뢰도를 하향 평가하여 서술할 것."
    return "관련 백테스트 데이터와 현재 기술적 신호 간 특이 모순 없음."

# [패치 4] 백테스트 성과 평가 (기간 파싱 추가)
# [개선/2026-09, 아이디어 A] 원시 수익률(total_ret)만으로 벤치마크와 비교하면 "수익은 이겼지만
# 훨씬 큰 낙폭(MDD)을 감수한" 경우(예: TSLA VWAP 평균회귀 전략)를 "초과수익 확인"으로 잘못
# 판정하게 됨. Calmar 비율(수익률÷MDD, 리스크 조정 성과)을 함께 계산해 상대 비교하되, 절대
# 기준값(예: "Calmar 3 이상 우수")은 쓰지 않는다 - 이 백테스트는 연환산이 아니라서 업계 통상
# 절대 기준을 적용할 근거가 없기 때문. MDD가 0이거나 데이터가 없으면 Calmar 비교를 생략하고
# 기존처럼 원시 수익률만으로 판단한다(0 나눗셈 방지 및 무낙폭 구간이라는 극단적 예외 처리).
# [재설계/item24, 2026-09] 기존엔 "원시 수익률 우선 게이트"(수익률에서 밀리면 Calmar를
# 아예 확인하지 않고 underperform 확정) 방식이라, 전략이 원시 수익률로는 근소하게 밀려도
# 리스크 조정 성과(Calmar)로는 확실히 이기는 경우의 신호를 완전히 죽이는 결함이 있었다
# (퀀트 실무에서 Sharpe·Calmar는 이미 수익을 리스크로 나눈 위험조정 지표라 원시 수익률을
# 별도 1차 게이트로 두는 게 이론적 근거가 약함). Sharpe·Calmar 2축을 독립적으로
# 벤치마크와 비교해 결과를 5가지로 세분화한다: no_data / outperform_both(둘 다 승) /
# sharpe_only(샤프만 승) / calmar_only(Calmar만 승) / underperform(둘 다 패).
# PF·win_rate는 벤치마크(단일 Buy&Hold "거래")에는 구조적으로 정의되지 않아 축에서 제외.
# [재설계/item24-2, 2026-09] Sortino 비율 추가(item 31 논의에서 사용자가 "퀸트 전문가들은
# Sharpe/Sortino/Calmar/K-Ratio/SQN을 본다는데 지표를 더 늘려야 하지 않냐"고 질문 → Sortino는
# 벤치마크(연속 수익률만 있으면 계산 가능)에도 적용 가능하고 Sharpe와 다른 정보(하방 변동성만
# 봄)를 주는 것으로 판단해 채택. K-Ratio는 Calmar와 측정 정보가 상당히 겹치고, SQN은 거래별
# R-멀티플이 필요한데 트레이드 기록에 진입 시점 리스크가 없어 구조 변경 없인 계산 불가능해
# 이번엔 보류). 축이 2개(Sharpe/Calmar)에서 3개(+Sortino)로 늘면서, 기존의 "sharpe_only/
# calmar_only"처럼 축 조합별로 카테고리 이름을 하드코딩하는 방식은 축이 늘수록(예: 향후
# K-Ratio 재검토 시 4개) 조합이 기하급수적으로 늘어나 유지 불가능해짐 — 그래서 카테고리는
# no_data/outperform_all(전부 승)/mixed(엇갈림)/underperform_all(전부 패) 4가지로만 단순화하고,
# "어떤 지표가 이기고 졌는지"는 별도의 axes 딕셔너리로 반환해 호출부가 필요한 만큼만 조합해
# 문장을 만들도록 한다. 이 구조라면 지표가 늘어나도(K-Ratio 추가 등) 카테고리 개수는 그대로다.
AXIS_LABELS_VS_BENCHMARK = {"calmar": "Calmar 비율", "sharpe": "샤프지수", "sortino": "소르티노 비율"}

def classify_vs_benchmark(strat_ret, strat_mdd, strat_sharpe, strat_sortino, bench_ret, bench_mdd, bench_sharpe, bench_sortino):
    if not isinstance(strat_ret, (int, float)) or not isinstance(bench_ret, (int, float)) or bench_ret <= 0:
        return "no_data", {}

    def _pair_win(s_val, b_val):
        if isinstance(s_val, (int, float)) and isinstance(b_val, (int, float)):
            return s_val >= b_val
        return None

    calmar_ok = (isinstance(strat_mdd, (int, float)) and isinstance(bench_mdd, (int, float))
                 and strat_mdd > 0 and bench_mdd > 0)
    strat_calmar = round(strat_ret / strat_mdd, 2) if calmar_ok else None
    bench_calmar = round(bench_ret / bench_mdd, 2) if calmar_ok else None

    axes = {
        "calmar": (strat_calmar, bench_calmar, _pair_win(strat_calmar, bench_calmar) if calmar_ok else None),
        "sharpe": (strat_sharpe, bench_sharpe, _pair_win(strat_sharpe, bench_sharpe)),
        "sortino": (strat_sortino, bench_sortino, _pair_win(strat_sortino, bench_sortino)),
    }
    votes = [w for (_, _, w) in axes.values() if w is not None]
    if not votes:
        return "no_data", {}
    if all(votes):
        return "outperform_all", axes
    if not any(votes):
        return "underperform_all", axes
    return "mixed", axes

def _axes_win_lose_labels(axes):
    winning = [AXIS_LABELS_VS_BENCHMARK[k] for k, (_, _, w) in axes.items() if w is True]
    losing = [AXIS_LABELS_VS_BENCHMARK[k] for k, (_, _, w) in axes.items() if w is False]
    return winning, losing

# [item24-2 구현 중 자체 발견·수정] 지표 라벨을 "·"로 나열한 win_text/lose_text의 조사(은/는)를
# "은"/"는"으로 하드코딩했더니, 나열되는 지표 조합에 따라(예: "샤프지수"만 단독이면 모음
# 받침 없어 "는"이 맞는데 "은"이 붙거나, "Calmar 비율"처럼 받침 있는 단어에 "는"이 붙는 등)
# 문법적으로 틀린 문장이 생성될 수 있음을 이번 소르티노 추가에 대한 전수 분기 테스트 중
# 자체 발견. 마지막 글자의 받침 유무로 조사를 동적으로 결정하도록 수정.
def _eun_neun(text):
    if not text:
        return "는"
    code = ord(text[-1]) - 0xAC00
    if 0 <= code <= 11171:
        return "은" if code % 28 != 0 else "는"
    return "는"

# [개선] 백테스트 승자 전략이 벤치마크(단순 매수 후 보유) 대비 열위이거나, 위험조정 지표가
# 서로 엇갈릴 경우, Section 6 매매 시나리오 서술에 함께 반영할 주의사항 텍스트. Section 6에서
# "그대로 복사 출력"하는 데이터이므로 지시문("~할 것")이 아닌 서술형 문장으로만 구성한다
# (LLM이 지시문과 혼동하지 않도록).
def build_trading_style_caution(best_name, category, best_total_ret, benchmark, axes, bt_years):
    if category == "no_data":
        return "해당없음 (벤치마크 비교 데이터 부족)"
    if category == "outperform_all":
        winning, _ = _axes_win_lose_labels(axes)
        return f"해당없음 (선택된 전략이 벤치마크를 위험조정 지표({'·'.join(winning)}) 전부에서 상회하여 별도 주의 불필요)"
    if category == "mixed":
        winning, losing = _axes_win_lose_labels(axes)
        win_text = "·".join(winning) if winning else "없음"
        lose_text = "·".join(losing) if losing else "없음"
        return (f"⚠️ '{best_name}' 전략이 최근 {bt_years}년간 단순 매수 후 보유(Buy&Hold) 대비 {win_text}{_eun_neun(win_text)} 우위이지만, "
                f"{lose_text}로는 열위입니다. 위험조정 성과 지표들이 서로 엇갈리는 결과이므로, 공격적인 전액 매수보다 "
                f"지표별 성격(변동성 대비/낙폭 대비/하방리스크 대비)을 참고한 보수적 접근이 권장됩니다.")
    ratio = round(benchmark / best_total_ret, 1) if isinstance(best_total_ret, (int, float)) and best_total_ret > 0 else None
    ratio_text = f" (벤치마크가 약 {ratio}배 우수)" if ratio else ""
    return (f"⚠️ 두 트레이딩 전략 모두 최근 {bt_years}년간 단순 매수 후 보유(Buy&Hold, +{benchmark}%) 대비 열위{ratio_text}했습니다. "
            f"'{best_name}' 스타일로 접근하더라도 공격적인 전액 매수나 잦은 재진입보다 보수적인 분할 접근과 리스크 관리가 권장되며, "
            f"이 종목은 트레이딩보다 장기 보유가 유리했을 수 있습니다.")

# [개선] Section 7 [최종 투자의견]에 붙일 짧은 부기 문구. 매수/관망/홀딩 "결정" 자체에는 전혀
# 개입하지 않고, 이미 결정된 의견 뒤에 조건부로만 덧붙는 순수 표시용 텍스트 — 서술형으로만 구성.
# 벤치마크를 위험조정 지표 전부에서 상회하거나 비교 데이터가 없으면 빈 문자열("")을 반환한다.
def build_verdict_style_qualifier(category, axes):
    if category in ("no_data", "outperform_all"):
        return ""
    if category == "mixed":
        winning, losing = _axes_win_lose_labels(axes)
        win_text = "·".join(winning) if winning else "없음"
        lose_text = "·".join(losing) if losing else "없음"
        return f"단, 백테스트상 {win_text}{_eun_neun(win_text)} 벤치마크를 상회했으나 {lose_text}{_eun_neun(lose_text)} 벤치마크보다 낮아 위험조정 성과가 지표별로 엇갈려 보수적 접근이 권장됨"
    return "단, 백테스트상 액티브 트레이딩보다 매수 후 장기 보유가 유리했던 것으로 나타나 잦은 진입/청산보다 보유 전략이 권장됨"

def evaluate_strategy_backtest(bt):
    # [버그수정] SPCX 등 신규 상장 종목은 60거래일 미만이라 run_strategy_backtest_v2()가 None을
    # 반환하는데, 이걸 그대로 bt.get(...)에 넘기면 AttributeError: 'NoneType' object has no
    # attribute 'get' 로 앱 전체가 죽는다. 데이터 부족 시 안전하게 "판정불가" 처리.
    if not bt:
        return (
            "판정불가 (백테스트 데이터 부족)",
            "0/3",
            "⚠️ 신규 상장 등으로 백테스트에 필요한 최소 거래일(약 60거래일) 데이터가 확보되지 않아 전략 검증을 수행하지 못했으며, 이에 따라 백테스트 근거 없이 데이터 부족으로 신뢰도가 낮은 상태입니다.",
            "N/A",
            "해당없음 (백테스트 데이터 부족)",
            ""
        )
    s1 = bt.get('strategy_1_momentum_squeeze', {}) or {}
    s2 = bt.get('strategy_2_vwap_mean_reversion', {}) or {}
    benchmark = bt.get('benchmark_buy_and_hold', 0) or 0
    benchmark_mdd = bt.get('benchmark_mdd', 0) or 0
    benchmark_sharpe = bt.get('benchmark_sharpe', 0) or 0
    benchmark_sortino = bt.get('benchmark_sortino', 0) or 0

    # 백테스트 기간 동적 계산 (strategy_1_period_split 활용)
    bt_years = 5.0
    try:
        split_str = bt.get('strategy_1_period_split', '')
        dates = re.findall(r'\d{4}-\d{2}-\d{2}', split_str)
        if len(dates) >= 2:
            d1 = datetime.strptime(dates[0], "%Y-%m-%d")
            d2 = datetime.strptime(dates[-1], "%Y-%m-%d")
            bt_years = max(1.0, round((d2 - d1).days / 365.0, 1))
    except Exception: pass

    def _num(v, default=0.0): return float(v) if isinstance(v, (int, float)) else default

    # [재설계/item26, 2026-09] 기존엔 승률·PF·샤프·벤치마크승리보너스를 자의적인 절대 임계값
    # (예: (pf-1)*3배)으로 가산해 0~10점 단일 점수를 만들었는데, MDD는 추출만 하고 점수에
    # 전혀 반영하지 않는 결함이 있었다. 퀀트 실무의 "단일 블렌드 점수보다 지표별 상대 비교
    # (체크리스트) 후 다수결" 관행(리서치 근거: Sharpe/Sortino/Calmar/SQN 등은 서로 다른 축을
    # 측정하므로 하나로 뭉개면 정보 손실)을 반영해, 두 후보를 PF·샤프·Calmar·소르티노 4개
    # 지표에서 1:1로 맞대결시켜 다수결로 승자를 정하는 방식으로 교체한다(item 31 후속으로
    # 소르티노 추가 - 하방 변동성만 보는 지표라 Sharpe와 다른 정보를 줌). 원시 win_rate·
    # total_ret는 위험조정 비교에 부적합해 투표에서 제외(참고용으로만 verdict_text에 남김).
    # 표본 크기는 새 임계값을 만들지 않고 calc_stats_v2()의 기존 reliability 3단계 라벨을
    # 그대로 비-투표성 별도 주의문으로 재사용한다.
    def _calmar(s):
        ret, mdd = _num(s.get('total_ret')), _num(s.get('mdd'))
        return round(ret / mdd, 2) if mdd > 0 else None

    def _compare(v1, v2):
        # 1=s1 승, 2=s2 승, 0=무승부·비교불가(해당 지표는 투표에서 제외)
        if v1 is None or v2 is None or v1 == v2:
            return 0
        return 1 if v1 > v2 else 2

    pf1, pf2 = _num(s1.get('profit_factor')), _num(s2.get('profit_factor'))
    sh1, sh2 = _num(s1.get('sharpe_ratio_annualized')), _num(s2.get('sharpe_ratio_annualized'))
    cal1, cal2 = _calmar(s1), _calmar(s2)
    so1, so2 = _num(s1.get('sortino_ratio_annualized')), _num(s2.get('sortino_ratio_annualized'))

    metric_votes = [("손익비(PF)", _compare(pf1, pf2)), ("샤프지수", _compare(sh1, sh2)),
                     ("Calmar비율", _compare(cal1, cal2)), ("소르티노비율", _compare(so1, so2))]
    n_metrics = len(metric_votes)
    win1 = sum(1 for _, v in metric_votes if v == 1)
    win2 = sum(1 for _, v in metric_votes if v == 2)

    if win1 != win2:
        winner = 1 if win1 > win2 else 2
        tie_break_used = False
    else:
        # 동률(4개 지표라 2승2승, 1승1승+2무 등 다양하게 발생 가능) → 각 후보를 벤치마크와
        # 개별 비교한 등급(classify_vs_benchmark)으로 타이브레이크.
        rank = {"outperform_all": 3, "mixed": 2, "underperform_all": 1, "no_data": 0}
        cat1, _ = classify_vs_benchmark(_num(s1.get('total_ret')), s1.get('mdd'), s1.get('sharpe_ratio_annualized'), s1.get('sortino_ratio_annualized'), benchmark, benchmark_mdd, benchmark_sharpe, benchmark_sortino)
        cat2, _ = classify_vs_benchmark(_num(s2.get('total_ret')), s2.get('mdd'), s2.get('sharpe_ratio_annualized'), s2.get('sortino_ratio_annualized'), benchmark, benchmark_mdd, benchmark_sharpe, benchmark_sortino)
        winner = 1 if rank.get(cat1, 0) >= rank.get(cat2, 0) else 2
        tie_break_used = True

    if winner == 1:
        best_name, best_data, best_wins = "모멘텀 스퀴즈(추세추종형 돌파매매)", s1, win1
    else:
        best_name, best_data, best_wins = "VWAP 평균회귀(눌림목/되돌림 매매)", s2, win2

    vote_detail = ", ".join(f"{label}:{'모멘텀 스퀴즈' if v==1 else ('VWAP 평균회귀' if v==2 else '무승부')}" for label, v in metric_votes)
    vote_score_text = f"{best_wins}/{n_metrics} 지표 우위" + ("(동률 → 벤치마크 비교 등급으로 최종판정)" if tie_break_used else "")

    # [개선/2026-09, item24] 벤치마크(단순 매수 후 보유) 대비 성과 비교 - 샤프·Calmar·소르티노
    # 3개 위험조정 지표를 독립적으로 판정해 "일부 지표에서만 이긴" 경우를 구분한다.
    best_total_ret = _num(best_data.get('total_ret')) if isinstance(best_data.get('total_ret'), (int, float)) else None
    best_mdd = best_data.get('mdd')
    best_sharpe = best_data.get('sharpe_ratio_annualized')
    best_sortino = best_data.get('sortino_ratio_annualized')
    category, axes = classify_vs_benchmark(best_total_ret, best_mdd, best_sharpe, best_sortino, benchmark, benchmark_mdd, benchmark_sharpe, benchmark_sortino)

    if category == "no_data":
        benchmark_note = "벤치마크(단순 매수 후 보유) 데이터가 없어 비교 불가."
    elif category == "underperform_all":
        underperform_ratio = round(benchmark / best_total_ret, 1) if isinstance(best_total_ret, (int, float)) and best_total_ret > 0 else None
        ratio_text = f" (벤치마크가 전략 대비 약 {underperform_ratio}배)" if underperform_ratio else ""
        benchmark_note = f"⚠️ 동일 기간 단순 매수 후 보유(Buy&Hold) 대비 위험조정 지표(샤프지수·Calmar 비율·소르티노 비율) 전부 열위입니다{ratio_text} — 원시 수익률이 양호하더라도 위험조정 성과 기준으로는 벤치마크 대비 우위가 없으므로, 이 종목은 트레이딩보다 장기 보유가 유리했을 수 있습니다."
    elif category == "mixed":
        winning, losing = _axes_win_lose_labels(axes)
        win_text = "·".join(winning) if winning else "없음"
        lose_text = "·".join(losing) if losing else "없음"
        benchmark_note = (f"동일 기간 단순 매수 후 보유(Buy&Hold, +{benchmark}%) 대비 {win_text}{_eun_neun(win_text)} 우위이지만, "
                           f"{lose_text}로는 열위입니다 — 위험조정 성과 지표가 서로 엇갈리는 결과이며, 원시 수익률만으로 "
                           f"초과수익(알파) 확인이라 단정하기 어렵습니다.")
    else:  # outperform_all
        winning, _ = _axes_win_lose_labels(axes)
        benchmark_note = f"동일 기간 단순 매수 후 보유(Buy&Hold) 수익률(+{benchmark}%) 대비 위험조정 지표({'·'.join(winning)}) 전부 상회하여 액티브 전략으로서의 위험조정 초과성과가 확인됨."

    # [item26-b] profit_factor_note(표본부족 등 캐버트)가 있으면 표시에 괄호로 덧붙인다.
    pf_note = best_data.get('profit_factor_note') or ""
    pf_display = f"{best_data.get('profit_factor')}({pf_note})" if pf_note else f"{best_data.get('profit_factor')}"

    # [item26] 표본 크기 신뢰도는 투표에 넣지 않고(비-투표성 게이트) 신뢰도가 충분(✅)이
    # 아닐 때만 verdict_text에 별도 주의문으로 덧붙인다 - 새 임계값 없이 기존 reliability
    # 3단계 라벨을 그대로 재사용.
    reliability_label = str(best_data.get('reliability', ''))
    reliability_note = f" ⚠️ 다만 {reliability_label}." if reliability_label and "✅" not in reliability_label else ""

    verdict_text = (f"최근 {bt_years}년 백테스트 결과, '{best_name}' 전략이 위험조정 지표(손익비/샤프지수/Calmar비율/소르티노비율) "
                     f"상대비교에서 더 우수한 것으로 판정됩니다 ({vote_detail} → {vote_score_text}; 참고: 승률 {best_data.get('win_rate')}%, "
                     f"손익비(PF) {pf_display}, 샤프지수 {best_data.get('sharpe_ratio_annualized')}, 소르티노비율 {best_data.get('sortino_ratio_annualized')}, "
                     f"MDD {best_data.get('mdd')}%)."
                     f"{reliability_note} {benchmark_note}")
    trading_caution_text = build_trading_style_caution(best_name, category, best_total_ret, benchmark, axes, bt_years)
    verdict_qualifier_text = build_verdict_style_qualifier(category, axes)
    return best_name, vote_score_text, verdict_text, bt_years, trading_caution_text, verdict_qualifier_text

# [개선/2026-09, item 21 그룹 B] "그대로 복사 출력" 지시가 프롬프트 문구만으로는 강제되지 않는다는 문제
# (item 22/27/28에서 반복 확인, 특히 TSLA 재현 사례에서는 LLM이 스스로 오인식한 현재가를 기준으로
# 여러 "절대 임의 수정 금지" 필드를 동시에 재계산해버리는 것까지 확인됨)를 코드 레벨에서 원천 차단하기
# 위한 후처리 계층. 설계 논의 결론(사용자 승인):
#   1) 9개 "그대로 복사" 필드는 LLM에게 실제 문구 대신 플레이스홀더 토큰(`[[F##]]`)만 출력하게 하고,
#      응답을 받은 뒤 Python이 단순 문자열 치환으로 실제 값을 채워 넣는다 (토큰이 없으면 기존 방식대로
#      불릿 경계를 정규식으로 추정해 내용 전체를 강제 치환하는 폴백을 유지 - 이중 안전장치).
#   2) 24번(verdict_qualifier)은 애초에 "판정 근거로 사용 금지"라는 설계 원칙(item 15) 때문에 LLM에게
#      보여줄 필요 자체가 없으므로, 프롬프트에서 완전히 제거하고 Python이 최종 투자의견 뒤에 항상 직접
#      부기한다.
#   3) 위 두 가지로도 못 막는 잔여 리스크(자유 서술문 안에 "현재가($...)" 형태로 잘못된 현재가가
#      섞여 들어가는 경우, TSLA 사례의 발원지)는 리포트 전체를 대상으로 "현재가/현재 주가($...)"
#      패턴만 정규식으로 찾아 진짜 현재가로 강제 치환하는 별도 스윕으로 보완한다. 서술 자체(왜 그런
#      판단을 내렸는지)는 LLM 자유 서술을 그대로 유지 - 괄호 안 숫자 하나만 고정.
def force_copy_field(report_text, label, value, token=None, logger=None):
    """불릿 '* **{label}**: ...' 의 내용을 value로 강제 치환.
    1) token이 주어지고 응답 안에 그대로 있으면 단순 문자열 치환(가장 안전, 우선 시도).
    2) token이 없으면(LLM이 지시를 어기고 실제 내용을 써버린 경우) 불릿 경계를 정규식으로 추정해
       내용 전체를 강제 치환(기존 방식, 폴백).
    3) 그마저 실패하면(레이블 자체를 못 찾음 - LLM이 출력 포맷을 이탈) 원문을 그대로 두고 경고 로그만 남김
       (item 4의 "크래시 대신 안전 폴백" 철학과 동일)."""
    if token and token in report_text:
        return report_text.replace(token, value, 1)
    pattern = re.compile(
        r'(\*\s*\*\*' + re.escape(label) + r'\*\*\s*:\s*)(.*?)(?=\n\s*\*\s*\*\*|\n\s*\[\d+\.|\Z)',
        re.DOTALL
    )
    new_text, n = pattern.subn(lambda m: m.group(1) + value, report_text, count=1)
    if n == 0 and logger:
        logger.warning(f"[강제치환 실패] '{label}' 불릿을 리포트에서 찾지 못함 (토큰/폴백 모두 실패 - LLM이 출력 포맷을 이탈했을 가능성)")
    return new_text

def fix_current_price_mentions(report_text, curr_price):
    """리포트 전체에서 '현재가($X)' / '현재 주가($X)' (괄호 없는 '현재가 $X' 포함) 패턴을 찾아
    괄호/문장 구조는 그대로 두고 숫자만 실제 current_price로 강제 치환. 자유 서술문(진입 적합성
    분석 등) 안에서 LLM이 현재가를 스스로 오인식해 재기입하는 경우를 서술의 자유를 건드리지 않고
    발원지에서만 고정하기 위한 보완 조치."""
    if not isinstance(curr_price, (int, float)): return report_text
    pattern = re.compile(r'(현재\s*주?가[는가이]?\s*\(?\$)([\d,]+\.?\d*)(\)?)')
    return pattern.sub(lambda m: f"{m.group(1)}{curr_price}{m.group(3)}", report_text)

def apply_verdict_qualifier(report_text, verdict_qualifier_text, logger=None):
    """24번(verdict_qualifier)은 프롬프트에서 완전히 제외했으므로, LLM이 자유롭게 결정한
    매수/관망/홀딩 뒤에 Python이 항상 직접 부기(빈 문자열이면 아무것도 안 붙임)."""
    pattern = re.compile(r'(\*\s*\*\*최종\s*투자의견\*\*\s*:\s*)(매수|관망|홀딩)(\s*\([^)]*\))?')
    def repl(m):
        suffix = f" ({verdict_qualifier_text})" if verdict_qualifier_text else ""
        return m.group(1) + m.group(2) + suffix
    new_text, n = pattern.subn(repl, report_text, count=1)
    if n == 0 and logger:
        logger.warning("[부기 실패] '최종 투자의견' 불릿에서 매수/관망/홀딩 판정어를 찾지 못함")
    return new_text

def enforce_precalculated_fields(report_text, ctx, logger=None):
    """10개 '그대로 복사' 필드 강제 치환 + 24번 부기 + 현재가 오기재 스윕을 한 번에 적용."""
    field_map = [
        ("스코어카드 관점", ctx["precalc_scorecard"], "[[F15]]"),
        ("예상 손익비 (Risk/Reward)", ctx["rr_text"], "[[F16]]"),
        ("1차 목표가", f'{ctx["t1_label"]} (${ctx["t1_price"]})', "[[F17T1]]"),
        ("2차 목표가", f'{ctx["t2_label"]} (${ctx["t2_price"]})', "[[F17T2]]"),
        ("벤치마크 대비 평가", ctx["backtest_verdict"], "[[F19]]"),
        ("손절(Stop-loss) 기준선", ctx["atr_labels_text"], "[[F20]]"),
        ("추가 매수 여부", ctx["add_buy_text"], "[[F21]]"),
        ("신규 진입 등급", ctx["entry_grade_text"], "[[F22]]"),
        ("매매 접근 방식 주의사항", ctx["trading_caution_text"], "[[F23]]"),
        ("밴드 이탈 판정 (보유자 전용)", ctx["band_breach_text"], "[[F18]]"),
    ]
    for label, value, token in field_map:
        report_text = force_copy_field(report_text, label, value, token=token, logger=logger)
    report_text = apply_verdict_qualifier(report_text, ctx["verdict_qualifier_text"], logger=logger)
    report_text = fix_current_price_mentions(report_text, ctx["curr_p"])
    return report_text

# [패치 7] 추가 매수 여부(구: 물타기/Averaging Down) 2단계 게이트 (1단계 적격성 심사 하드게이트 -> 2단계 실효성/손익비 평가)
# [물타기→추가 매수 여부 리네이밍, 2026-09] item 36에서 논의한 취지가 "물타기 여부 판정"이
# 아니라 "추가 투자 유효성 판정"으로 기능의 정체성 자체를 바꾸는 것이었는데, 그동안 게이트
# 판정 기준(로직)만 바꾸고 사용자 노출 라벨/함수명은 "물타기"로 남아있던 불일치를 정리.
# 1단계는 "하나라도 걸리면 즉시 차단"하는 하드 게이트로, 통과한 종목에 한해서만 2단계(절대 금액 기준 손익비)를 계산한다.
def stage1_outlook_gate(curr_price, tech, s_moat, moat_detail, ret6_pct, backtest_results, short_intel, stop_price=None):
    reasons_block = []

    # (0) [필수 게이트 - 패치6에서 이어받음] 손절선 이미 이탈 여부는 최우선으로 즉시 차단
    #     stage2에서 (curr_price - stop_price)로 손실액을 계산하는데, 이미 손절선 아래인 경우
    #     이 값이 음수가 되어 "-$-500" 같은 이중 음수 표기가 나오는 것을 막기 위한 목적도 겸함.
    if isinstance(stop_price, (int, float)) and isinstance(curr_price, (int, float)) and curr_price <= stop_price:
        reasons_block.append(f"손절선 이미 이탈 (현재가 ${curr_price} ≤ 손절선 ${stop_price}) - 추가 매수 대신 리스크 관리(전량 손절) 우선")

    # (a) 기술적 하락 추세 훼손
    # [이평선 기준 수정, 2026-09] 60일/120일선(한국 시장 관습) 대신 미국 시장 표준인
    # 50일/200일선(골든크로스/데드크로스 기준)으로 교체.
    # [트렌드 체크 확장, 2026-09] 이평선 정배열 훼손에 더해 6개월 수익률 심각 부진(-15% 미만,
    # calculate_pre_scores의 s_mom3 최하단 구간과 동일 기준)도 OR로 추가 - 두 신호 중 하나만
    # 걸려도 보수적으로 차단하는 것이 리스크 관리 관점에서 안전하다고 판단.
    sma50, sma200 = tech.get('sma_50'), tech.get('sma_200')
    trend_reasons = []
    if isinstance(sma50, (int, float)) and isinstance(sma200, (int, float)):
        ma_dead_cross = sma50 < sma200  # 단/장기 완전 역배열 (데드크로스)
        price_below_support = curr_price < sma200  # 장기 지지선(SMA200) 붕괴
        if ma_dead_cross:
            trend_reasons.append(f"SMA50 ${sma50} < SMA200 ${sma200}(데드크로스)")
        if price_below_support:
            trend_reasons.append(f"현재가 ${curr_price}가 SMA200 ${sma200} 하회")
    if isinstance(ret6_pct, (int, float)) and ret6_pct < -15.0:
        trend_reasons.append(f"6개월 수익률 {ret6_pct:+.1f}% (-15% 미만)")
    if trend_reasons:
        reasons_block.append("기술적 추세 훼손 (" + ", ".join(trend_reasons) + ")")

    # (b) 해자(핵심 경쟁력) 훼손 여부
    # [게이트 재설계, 2026-09] 기존엔 total_score(밸류에이션 20% 포함) < 6.0, 또는 전통 가치평가
    # 모델(그레이엄/린치/ROE-PBR) 과반 고평가를 기준으로 삼았음. 논의 결과 이 방식은
    # (1) "추가 투자 유효성"이 아니라 "종목 자체가 매력적인가"라는 별개의 종목선택 질문에
    # 답하는 것이고, (2) 전통 가치평가 모델은 이 앱 스스로도 전통 제조업/청산가치 기준으로
    # 명시한 도구라 성장주에는 부적합함이 확인되어 폐기. 물타기든 분할매수든 "추가 투자
    # 유효성"을 가르는 핵심은 "이 사업의 본질적 경쟁력(해자)이 여전히 살아있는가"이므로
    # s_moat(item 18의 4대 구조적 지표 - ROIC/매출총이익률/R&D비중/3년FCF흑자, 균등가중) 자체를
    # 직접 기준으로 삼는다. 4개 중 과반(2개) 이상 미달 시 차단 - value_models 과반 판정에
    # 쓰던 것과 동일한 "과반" 기준을 재사용해 임의성을 줄임.
    if isinstance(s_moat, (int, float)) and s_moat < 5.0:
        failed_pillars = []
        for name, detail in (moat_detail or {}).items():
            val, ok, bar = detail
            if ok:
                continue
            if bar is not None:
                val_str = f"{val:.1f}%" if isinstance(val, (int, float)) else str(val)
                failed_pillars.append(f"{name} {val_str}(<{bar})")
            else:
                failed_pillars.append(f"{name} 미충족")
        pillar_text = ", ".join(failed_pillars) if failed_pillars else "상세 근거 없음"
        reasons_block.append(
            f"해자(핵심 경쟁력) 훼손 - 4대 구조지표 과반 미달 (해자점수 {s_moat:.1f} < 5.0): {pillar_text}"
        )

    # (c) 백테스트 및 수급 이탈
    s2 = (backtest_results or {}).get('strategy_2_vwap_mean_reversion', {}) or {}
    win_rate = s2.get('win_rate')
    trades_cnt = s2.get('trades_count')
    reliability = s2.get('reliability', 'N/A')
    # [최소 표본수 요건] 이 하드 게이트는 걸리면 즉시·영구 차단이라, calc_stats_v2()가 스스로
    # "✅ 표본 충분(통계적 신뢰 가능)"이라고 인정하는 30건 이상일 때만 승률로 판단한다.
    # 30건 미만(⚠️ 표본 부족 / 🔶 제한적 신뢰)인데도 승률만으로 차단하면, 실제로는 대부분의
    # 종목이 표본 부족을 이유로 차단당하는 결과가 되어 게이트의 취지(가짜 신호 배제)를 벗어난다.
    if isinstance(win_rate, (int, float)) and isinstance(trades_cnt, (int, float)) and trades_cnt >= 30 and win_rate < 50:
        reasons_block.append(f"평균회귀 전략 백테스트 승률 저조 ({win_rate}% < 50%, {trades_cnt}회 매매, {reliability})")

    squeeze_level = str((short_intel or {}).get('squeeze_risk_level', ''))
    if '🔴' in squeeze_level or '🟡' in squeeze_level:
        reasons_block.append(f"숏스퀴즈 리스크 상승 ({squeeze_level})")

    short_mom_raw = str((short_intel or {}).get('short_mom_change', '0'))
    try:
        short_mom_pct = float(short_mom_raw.replace('%', '').replace('+', ''))
    except ValueError:
        short_mom_pct = 0.0
    if short_mom_pct > 20:  # 숏 비중이 급격히 증가하는 중
        reasons_block.append(f"숏 비중 급증 ({short_mom_raw}) - 스마트머니 이탈 가능성")

    passed = len(reasons_block) == 0
    return {
        "1단계_통과여부": "✅ 통과 - 비중 확대 검토 대상" if passed else "🚫 차단 - 추가 매수 즉시 배제",
        "차단_사유": reasons_block if reasons_block else ["없음"],
    }


# 2단계: 비중 확대 실효성 및 손익비 평가 (1단계 통과 시에만 실행)
def stage2_profitability_test(curr_price, user_avg_price, user_shares, add_shares,
                                stop_price, target1_price):
    if not (add_shares and add_shares > 0):
        return {"평가": "add_shares 미지정으로 평가 생략"}

    new_shares = user_shares + add_shares
    new_avg_price = (user_avg_price * user_shares + curr_price * add_shares) / new_shares

    # (a) 기대 이익금 증가분 ($) - 새로 매수하는 물량이 1차 목표가 도달 시 벌어들이는 절대 수익금
    incremental_profit_usd = add_shares * (target1_price - curr_price)

    # (b) 손실 노출도 - 추가 매수분이 손절가에 도달했을 때의 손실액 + 전체 포지션 손절 시 총 손실액
    incremental_loss_usd = add_shares * (curr_price - stop_price)
    total_loss_if_stopped_usd = (new_avg_price - stop_price) * new_shares

    # (c) 갱신된 손익비 (새 평단가 기준)
    new_reward = target1_price - new_avg_price
    new_risk = new_avg_price - stop_price
    new_rr_ratio = round(new_reward / new_risk, 2) if new_risk > 0 else None

    rr_ok = isinstance(new_rr_ratio, (int, float)) and new_rr_ratio >= 1.5

    return {
        "매수후_평단가": f"${user_avg_price:.2f} → ${new_avg_price:.2f}",
        "① 기대 이익금 증가분": f"+${incremental_profit_usd:,.2f} (추가매수 {add_shares:.1f}주가 1차목표가(${target1_price}) 도달 시)",
        "② 손실 노출도": (
            f"추가매수분 손절 시 -${incremental_loss_usd:,.2f} / "
            f"전체 포지션({new_shares:.1f}주) 손절 시 총 -${total_loss_if_stopped_usd:,.2f}"
        ),
        "③ 갱신된 손익비(R/R)": (
            f"{new_rr_ratio} : 1 {'✅ 기준(1.5) 충족' if rr_ok else '⚠️ 기준(1.5) 미달 - 비중확대 매력도 낮음'}"
            if new_rr_ratio is not None else "계산불가"
        ),
        "2단계_최종판단": "🟢 비중 확대 실효성 있음" if rr_ok else "🟡 비중 확대는 가능하나 손익비 매력도 낮음",
    }


# 통합 함수 (patch6의 evaluate_averaging_down 대체, 2026-09 리네이밍: evaluate_averaging_down_v2 → evaluate_additional_buy_v2)
def evaluate_additional_buy_v2(curr_price, user_avg_price, user_shares, add_shares,
                                 tech, s_moat, moat_detail, ret6_pct, backtest_results,
                                 short_intel, stop_price, target1_price):
    stage1 = stage1_outlook_gate(curr_price, tech, s_moat, moat_detail, ret6_pct, backtest_results, short_intel, stop_price)

    if stage1["1단계_통과여부"].startswith("🚫"):
        return {
            "최종_판정": "🚫 추가 매수 부적합 (1단계 적격성 심사 탈락)",
            "1단계_결과": stage1,
            "2단계_결과": "1단계 탈락으로 평가 생략",
        }

    stage2 = stage2_profitability_test(curr_price, user_avg_price, user_shares, add_shares,
                                         stop_price, target1_price)

    final_verdict = stage2.get("2단계_최종판단", "🟡 조건부")
    return {
        "최종_판정": final_verdict,
        "1단계_결과": stage1,
        "2단계_결과": stage2,
    }


# Section 6 "추가 매수 여부" 프롬프트 출력용 텍스트 포맷터 (2026-09 리네이밍: format_averaging_down_text → format_additional_buy_text)
def format_additional_buy_text(add_buy_check, is_holding):
    if not is_holding:
        return "해당없음 (미보유 - 신규 진입 검토 대상이라 추가 매수 판정 비적용)"

    final_verdict = add_buy_check.get("최종_판정", "판정불가")

    if final_verdict.startswith("🚫"):
        stage1 = add_buy_check.get("1단계_결과", {}) or {}
        block_reasons = stage1.get("차단_사유", [])
        reasons_text = " / ".join(block_reasons) if block_reasons else "데이터 근거 부족"
        return f"{final_verdict} | 차단 사유: {reasons_text} | 추가 매수(평단 낮추기) 의견 제시는 부적절한 상황입니다. 손절선 이탈 시에는 추가 매수보다 전량 손절이 우선되는 상황입니다."

    stage2 = add_buy_check.get("2단계_결과", {}) or {}
    if not isinstance(stage2, dict):
        return f"{final_verdict} | 상세 평가 생략 ({stage2})"

    detail_parts = [f"{k}: {v}" for k, v in stage2.items() if k != "2단계_최종판단"]
    detail_text = " / ".join(detail_parts) if detail_parts else "상세 데이터 없음"
    return (
        f"{final_verdict} | 1단계 적격성 심사 통과. 현재가가 분할 매수 밴드 안일 경우에 한해 "
        f"'평단가를 낮추기 위한 소액 분할 재매수' 의견을 아래 실효성 근거(퍼센트 대신 반드시 달러 금액 명시)와 함께 제시 가능: "
        f"{detail_text}. 단, 손절선 이탈 시에는 추가 매수보다 전량 손절이 우선되는 상황입니다."
    )

# 스코어카드 연산 (기존 함수 유지)
def calculate_pre_scores(fund, tech, bt, curr_price, price_df=None):
    def parse_num(v):
        if isinstance(v, str):
            try: return float(re.sub(r'[^0-9.-]', '', v))
            except: return 0.0
        return float(v) if v else 0.0
    earn_g_str, rev_g_str = fund.get('growth_factors', {}).get('earnings_growth_yoy', 'N/A'), fund.get('growth_factors', {}).get('revenue_growth_yoy', 'N/A')
    if earn_g_str == 'N/A' and rev_g_str == 'N/A': s_growth = 5.5
    else:
        earn_g = parse_num(earn_g_str) if earn_g_str != 'N/A' else None
        rev_g = parse_num(rev_g_str) if rev_g_str != 'N/A' else None
        def get_g_score(val): return 9.5 if val and val >= 30.0 else (7.5 if val and val >= 15.0 else (5.5 if val and val >= 5.0 else (3.5 if val and val > 0.0 else 1.5))) if val is not None else None
        s_earn, s_rev = get_g_score(earn_g), get_g_score(rev_g)
        if s_earn is not None and s_rev is not None: s_growth = (s_earn + s_rev) / 2.0
        elif s_earn is not None: s_growth = s_earn
        elif s_rev is not None: s_growth = s_rev
        else: s_growth = 5.5
    opm = parse_num(fund.get('quality_factors', {}).get('operating_margin', 0))
    fcf_str = str(fund.get('quality_factors', {}).get('free_cash_flow', ''))
    roic = parse_num(fund.get('long_term_quality', {}).get('roic', 0))
    s_prof = 1.5
    if opm >= 20 and '-' not in fcf_str and fcf_str != 'N/A': s_prof = 9.5
    elif opm >= 15: s_prof = 7.5
    elif opm >= 8: s_prof = 5.5
    elif opm >= 0: s_prof = 3.5
    # [item 18 개선, 2026-09] ROIC 보너스는 해자(moat) 점수로 이전 - 동일 데이터 이중계산 방지
    # (기존엔 여기서 +1.0 보너스로 쓰였는데, 아래 s_moat에서 ROIC을 핵심 축으로 승격하면서 제거)
    s_moat = 0.0
    # [item 18 개선] 해자 재정의: 경쟁우위와 무관한 항목(기관보유율/부채비율/자사주매입) 제거.
    # ROE 대신 ROIC(자본비용 대비 초과수익 - Damodaran/Morningstar 경제적 해자 분석의 핵심 지표,
    # 기존 s_prof 보너스가 쓰던 15% 문턱을 그대로 재사용해 새 상수 도입 없음)을 편입하고,
    # 3년 연속 FCF 흑자(3y_fcf_status)로 "지속성(durability)" 신호를 신규 반영.
    # 4개 기준 전부 동일 가중치(Piotroski F-Score식 균등 이진 체크리스트) - 차등 가중치를
    # 뒷받침할 업계 표준 공식이 없어(Morningstar 해자 등급도 정성적 애널리스트 판단이지
    # 정량 공식이 아님), 임의의 차등 배점을 새로 만드는 대신 균등 배점으로 임의성을 최소화.
    gross_margin_val = parse_num(fund.get('quality_factors', {}).get('gross_margin', 0))
    rnd_pct_val = parse_num(fund.get('quality_factors', {}).get('rnd_to_revenue', 0))
    fcf_3y_ok = fund.get('long_term_quality', {}).get('3y_fcf_status', 'N/A') == "3년 연속 흑자 (+)"
    if roic >= 15.0: s_moat += 2.5
    if gross_margin_val >= 50.0: s_moat += 2.5
    if rnd_pct_val >= 5.0: s_moat += 2.5
    if fcf_3y_ok: s_moat += 2.5
    s_moat = min(10.0, max(1.5, s_moat))
    # [물타기 게이트 재설계, 2026-09] "추가 투자 유효성" 판정이 total_score/value_models 대신
    # 해자(s_moat)를 직접 참조하기로 함에 따라, stage1_outlook_gate가 "어떤 지표가 미달인지"를
    # 재계산 없이 그대로 인용할 수 있도록 4대 지표의 원값/충족여부/기준치를 패키징해 반환.
    moat_detail = {
        "ROIC": (roic, roic >= 15.0, "15%"),
        "매출총이익률": (gross_margin_val, gross_margin_val >= 50.0, "50%"),
        "R&D비중": (rnd_pct_val, rnd_pct_val >= 5.0, "5%"),
        "3년FCF흑자": (fcf_3y_ok, fcf_3y_ok, None),
    }
    pe, fpe, ps, pbr = fund.get('trailing_pe'), fund.get('forward_pe'), fund.get('ps_ratio'), fund.get('pbr')
    def score_v(val, b1, b2, b3, b4):
        if not isinstance(val, (int, float)) or val < 0: return 1.5
        if val <= b1: return 9.5
        elif val <= b2: return 7.5
        elif val <= b3: return 5.5
        elif val <= b4: return 3.5
        return 1.5
    # [밸류에이션 개선안 1안] PSR을 절대 문턱(3/6/10/15배)이 아니라 "성장률 구간별 적정 PSR 배수"
    # (_get_psr_multiple, growth_models['psr_target']과 동일 기준) 대비 몇 배인지로 채점.
    # 예: 고성장(30%+) 종목의 적정 PSR은 8배이므로, PSR 17배는 "절대 기준 초과"가 아니라
    # "적정가 대비 2.2배"로 평가됨 — 성장 없이 비싼 종목과 성장 때문에 비싼 종목을 구분.
    growth_input_str = fund.get('growth_factors', {}).get('growth_model_input_used', 'N/A')
    growth_for_val = parse_num(growth_input_str) if growth_input_str != 'N/A' else None
    def score_ps_growth_adjusted(val, growth):
        if not isinstance(val, (int, float)) or val < 0: return 1.5
        if growth is None or growth <= 0:
            return score_v(val, 3, 6, 10, 15)  # 성장률 데이터 없으면 기존 절대 문턱으로 폴백
        fair_ps = _get_psr_multiple(growth)
        ratio = val / fair_ps
        if ratio <= 0.7: return 9.5
        elif ratio <= 1.0: return 7.5
        elif ratio <= 1.3: return 5.5
        elif ratio <= 1.8: return 3.5
        return 1.5
    # [밸류에이션 개선안 2안] PBR도 절대 문턱(3/6/10/15배) 대신 "ROE 대비 적정 PBR"
    # (value_models['roe_pbr']와 동일 기준: 적정 PBR ≈ ROE% ÷ 10%) 대비 몇 배인지로 채점.
    # 예: ROE 100%대 초고수익 기업은 적정 PBR이 10배라, PBR 23배는 "절대 기준 초과"가 아니라
    # "ROE 대비 적정가의 2.3배"로 평가됨 — 고ROE라서 장부가 프리미엄이 정당한 종목을 구분.
    roe_for_val = parse_num(fund.get('roe', 0))
    def score_pbr_roe_adjusted(val, roe_pct):
        if not isinstance(val, (int, float)) or val < 0: return 1.5
        if not isinstance(roe_pct, (int, float)) or roe_pct <= 0:
            return score_v(val, 3, 6, 10, 15)  # ROE 데이터 없거나 마이너스면 기존 절대 문턱으로 폴백
        fair_pbr = roe_pct / 10.0
        ratio = val / fair_pbr
        if ratio <= 0.7: return 9.5
        elif ratio <= 1.0: return 7.5
        elif ratio <= 1.3: return 5.5
        elif ratio <= 1.8: return 3.5
        return 1.5
    s_val = (score_v(pe, 15, 25, 40, 60) + score_v(fpe, 12, 20, 28, 35) + score_ps_growth_adjusted(ps, growth_for_val) + score_pbr_roe_adjusted(pbr, roe_for_val)) / 4.0
    vwap_20d = parse_num(tech.get('vwap_20d'))
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
    def _trend_alignment_score(price, sma50, sma200):
        if not (isinstance(price, (int, float)) and isinstance(sma50, (int, float)) and isinstance(sma200, (int, float)) and price > 0): return None
        if price > sma50 > sma200: return 9.5
        elif price > sma50 and price > sma200: return 7.5
        elif price > sma200: return 5.5
        elif price > sma50: return 4.5
        else: return 2.5
    def _six_month_return_score(df):
        try:
            if df is None or df.empty: return None
            df6 = df.tail(126) if len(df) >= 126 else df
            if len(df6) < 2: return None
            start_p, end_p = float(df6['Close'].iloc[0]), float(df6['Close'].iloc[-1])
            if start_p <= 0: return None
            ret6 = (end_p - start_p) / start_p * 100
            if ret6 >= 30: return 9.5, ret6
            elif ret6 >= 15: return 7.5, ret6
            elif ret6 >= 0: return 5.5, ret6
            elif ret6 >= -15: return 3.5, ret6
            else: return 1.5, ret6
        except Exception: return None
    trend_score = _trend_alignment_score(curr_price, tech.get('sma_50'), tech.get('sma_200'))
    ret6_result = _six_month_return_score(price_df)
    ret6_score = ret6_result[0] if ret6_result else None
    ret6_pct = ret6_result[1] if ret6_result else None
    parts = [p for p in (trend_score, ret6_score) if p is not None]
    if parts:
        s_mom3 = sum(parts) / len(parts)
        note_bits = []
        if trend_score is not None: note_bits.append(f"이평선 정합성 {trend_score:.1f}")
        if ret6_score is not None: note_bits.append(f"6개월 수익률 {ret6_pct:+.1f}%({ret6_score:.1f}점)")
        s_mom3_note = "장기추세 " + ", ".join(note_bits) + " 기반"
    else: s_mom3, s_mom3_note = 5.5, "장기추세 데이터 부족 - 중립 처리"
    s_mom = (s_mom1 + s_mom2 + s_mom3) / 3.0
    total_score = (s_growth * 0.2) + (s_prof * 0.25) + (s_moat * 0.25) + (s_val * 0.2) + (s_mom * 0.1)
    if total_score >= 8.5: badge = "👑 최상위 핵심 우량주"
    elif total_score >= 7.5: badge = "🥇 적격 우량주"
    elif total_score >= 6.0: badge = "⚠️ 조건부 종목"
    else: badge = "🚨 비우량주"
    scorecard_text = f"성장성({s_growth:.1f}), 수익성({s_prof:.1f}), 밸류에이션({s_val:.1f}), 해자({s_moat:.1f}), 퀀트/모멘텀({s_mom:.1f}) | 종합 평점: {total_score:.2f} / 10 ({badge})"
    # [패치7] 물타기 2단계 게이트(stage1_outlook_gate)가 숫자 종합점수(total_score)를 필요로 해서 튜플로 반환
    # [물타기 게이트 재설계, 2026-09] "추가 투자 유효성" 판정용으로 s_moat/moat_detail/ret6_pct도 함께 반환
    # (total_score/s_val은 게이트에서는 더 이상 안 쓰지만 스코어카드 표시 등 다른 용도로 유지)
    return scorecard_text, total_score, s_val, s_moat, moat_detail, ret6_pct

# -------------------------------------------------------------
# 메인 분석 실행
# -------------------------------------------------------------
st.header(f"📊 {ticker_input} 종합 밸류에이션 & 정밀 트레이딩 리포트")

if analyze_btn:
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY")
            
    with st.spinner(f"🔍 [{ticker_input}] 정밀 지표 및 5Y 백테스팅 실행 중..."):
        from concurrent.futures import ThreadPoolExecutor

        # [Yahoo crumb 워밍업 + 동시성 완화, 2026-09] 아래 병렬 실행 블록은 fetch_stock_technical_data/
        # fetch_backtest_data/fetch_nearest_options_data/fetch_recent_upgrades_downgrades/fetch_news 등
        # 최소 5곳에서 같은 티커에 대해 각자 yf.Ticker(...)를 만들어 동시에 Yahoo에 요청을 쏜다.
        # 이 앱 프로세스에 아직 crumb가 캐싱되지 않은 "콜드 스타트" 시점(신규 티커 최초 조회 등)에는
        # 여러 스레드가 거의 동시에 crumb 발급을 시도하게 되고, 이 순간적인 요청 폭주(burst)가
        # 실제 운영 로그에서 확인된 "Crumb fetch rate-limited (HTTP 429), continuing without crumb"
        # 로 이어졌을 가능성이 있다(429 발생 시 yfinance가 crumb 없이 강행 -> 이후 모든 요청이
        # 결정론적으로 "Invalid Crumb" 401로 실패). 병렬 실행 전에 가벼운 요청을 한 번 먼저 보내
        # crumb 협상을 단일 스레드로 끝내두면, 이후 병렬 호출들은 이미 캐싱된 crumb를 재사용하게
        # 되어 이 폭주 빈도를 낮출 수 있다. 다만 이는 Yahoo 쪽 외부 rate-limit 자체를 없애는
        # 근본 해결책이 아니라 발생 빈도를 낮추는 완화책(mitigation)이며, 100% 재현/검증은
        # 사용자의 운영 환경(Streamlit Cloud)에서만 가능함을 명확히 해둔다.
        # fast_info는 quoteSummary(=crumb 필요) 엔드포인트를 타지 않고 1y 시세/메타데이터
        # 기반으로 동작해 crumb 협상을 유발하지 않으므로, 반드시 crumb가 필요한 .info로
        # 워밍업해야 실제로 의미가 있다.
        try:
            _ = yf.Ticker(ticker_input).info
        except Exception as e:
            logger.warning(f"[crumb 워밍업 실패] {ticker_input}: {type(e).__name__}: {e}")

        # max_workers를 6 -> 4로 낮춰 동시 순간에 Yahoo로 나가는 요청 수 자체도 함께 줄인다.
        with ThreadPoolExecutor(max_workers=4) as executor:
            future_tech = executor.submit(fetch_stock_technical_data, ticker_input)
            future_backtest = executor.submit(fetch_backtest_data, ticker_input, "5y")
            future_options = executor.submit(fetch_nearest_options_data, ticker_input, 2)
            future_macro = executor.submit(fetch_macro_indicators)
            future_sector = executor.submit(fetch_sector_performance)
            future_news = executor.submit(fetch_news, ticker_input, 5)
            future_macro_news = executor.submit(fetch_macro_news, 4)
            future_analyst = executor.submit(fetch_recent_upgrades_downgrades, ticker_input, 2)

            tech_data, stock_date, fib_levels, high_52_calc, low_52_calc, raw_df, vol_profile = future_tech.result()
            bt_df = future_backtest.result()
            options_data = future_options.result()
            macro_data = future_macro.result()
            sector_data = future_sector.result()
            news_data = future_news.result()
            macro_news_data = future_macro_news.result()
            analyst_data = future_analyst.result()

        # 패치 5 적용: 5년치 백테스트 호출
        backtest_results = run_strategy_backtest_v2(bt_df if bt_df is not None else raw_df)
        
        curr_p = tech_data.get('current_price', 0)
        fund_data = fetch_fundamentals_and_valuation(ticker_input, curr_p, high_52_calc, low_52_calc)
        
        info_source_flag = fund_data.get('info_source', 'stock.info')
        ownership = fund_data.get('ownership_and_shorts', {})
        hedge_short_intel = fund_data.get('hedge_and_short_intel', {})
        earnings_info = fund_data.get('earnings_calendar', {})
        
        # 패치 1 적용: 동적 목표가 및 손익비 (stop_p: 패치7 물타기 2단계 게이트에서 재사용)
        targets_text, rr_text, t1_price, t2_price, t1_label, t2_label, stop_p, entry_grade_text = calculate_targets_and_risk_reward(curr_p, fib_levels, tech_data)
        # 패치 7 적용: 숫자 종합점수(precalc_total_score)도 함께 반환받도록 변경
        # [물타기 게이트 재설계, 2026-09] s_moat/moat_detail/ret6_pct도 함께 반환받도록 확장
        precalc_scorecard, precalc_total_score, precalc_s_val, precalc_s_moat, precalc_moat_detail, precalc_ret6_pct = calculate_pre_scores(fund_data, tech_data, backtest_results, curr_p, price_df=raw_df)

        my_return_str = "N/A"
        if is_holding and user_avg_price > 0 and user_shares > 0 and isinstance(curr_p, (int, float)) and curr_p > 0:
            pnl_pct = ((curr_p - user_avg_price) / user_avg_price) * 100
            my_return_str = f"{pnl_pct:+.2f}%"
        user_position_text = f"사용자 보유 현황: 평단가 ${user_avg_price:.2f}, 보유수량 {user_shares:.1f}주, 평가수익률 {my_return_str}" if is_holding and user_avg_price > 0 else "사용자 미보유 종목 (신규 진입 검토 관점)"

        # 개선안 적용: ATR 배수 라벨 혼동 방지를 위한 고정 텍스트 생성
        # [순서 변경, 2026-09] 아래 build_strategy_instruction()이 "밴드 이탈 vs 개별 포지션
        # 손절 옵션" 구분 문구에 실제 ATR 달러 값을 함께 인용할 수 있도록, 이 계산을 해당 호출
        # 이전으로 옮김 (기존에는 이 아래에서 계산되어 strategy_instruction_text에 전달 불가했음).
        atr_1_5 = tech_data.get('atr_stop_1_5x', 'N/A')
        atr_2_0 = tech_data.get('atr_stop_2_0x', 'N/A')
        atr_labels_text = f"1.5배 손절선(-${atr_1_5}), 2.0배 손절선(-${atr_2_0})"

        # [신규, 2026-09] 보유자 대상 "밴드 이탈(확정 종가 기준) → 손절 고려" 판정용 기준 가격 선정.
        # 장중(정규장 개장 중)에 돌리면 최신 봉(curr_p)이 아직 미완성 종가일 수 있으므로, 그 경우엔
        # 확정된 전일 종가(prev_close)를 기준으로 삼는다 - 장 마감 후에는 최신 봉이 곧 확정 종가이므로
        # curr_p를 그대로 사용. (사용자 승인: "장중/마감 여부에 따라 동적으로 선택")
        market_currently_open = is_us_market_open_now()
        if market_currently_open:
            breach_basis_price = tech_data.get('prev_close', curr_p)
            breach_basis_date = tech_data.get('prev_close_date', stock_date)
            breach_basis_label = "전일 확정 종가"
        else:
            breach_basis_price = curr_p
            breach_basis_date = stock_date
            breach_basis_label = "최신 확정 종가"
        band_breach_text = evaluate_band_breach_for_holder(is_holding, breach_basis_price, stop_p, breach_basis_label, breach_basis_date)

        # 패치 2 적용: 전략 지시문 생성
        strategy_instruction_text = build_strategy_instruction(
            is_holding, user_avg_price, user_shares, curr_p, my_return_str,
            t1_price, t2_price, t1_label, t2_label, atr_1_5, atr_2_0, band_breach_text
        )

        # 패치 4 적용: 백테스트 비교 및 검증 점수 (동적 기간 반환 포함)
        best_name, backtest_score, backtest_verdict, bt_years, trading_caution_text, verdict_qualifier_text = evaluate_strategy_backtest(backtest_results)

        # 패치 7 적용: 추가 매수 여부 2단계 게이트 (1단계 적격성 하드게이트 -> 2단계 실효성/손익비 평가, 보유수량의 약 25% 소액 분할 할당)
        simulated_add_shares = round(user_shares * 0.25, 1) if user_shares > 0 else 1.0
        add_buy_check = evaluate_additional_buy_v2(
            curr_price=curr_p, user_avg_price=user_avg_price, user_shares=user_shares,
            add_shares=simulated_add_shares, tech=tech_data,
            s_moat=precalc_s_moat, moat_detail=precalc_moat_detail, ret6_pct=precalc_ret6_pct,
            backtest_results=backtest_results,
            short_intel=fund_data.get('hedge_and_short_intel', {}).get('short_intel', {}),
            stop_price=stop_p, target1_price=t1_price
        )
        # 추가 매수 여부 판정 결과는 Section 6 전용 데이터 항목(21번)으로 분리하여 프롬프트에 주입 (아래 add_buy_text)
        add_buy_text = format_additional_buy_text(add_buy_check, is_holding)

        # 패치 3 적용: 스퀴즈 백테스트 정합성 경고 (동적 기간 변수 주입)
        consistency_note = build_backtest_consistency_note(backtest_results, tech_data.get('bb_squeeze_status'), bt_years)

        full_rag_payload = {
            "meta": {"ticker": ticker_input, "data_source": info_source_flag, "model_used": selected_model_label, "analysis_requested_at": get_current_kst_time_str(), "stock_data_date": stock_date},
            "technical_vwap_and_squeeze": tech_data,
            "volume_profile_poc_6m": vol_profile,
            "multi_year_backtesting": backtest_results,
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
            "precalculated_risk_reward": rr_text,
            "precalculated_targets": targets_text,
            "precalculated_entry_grade": entry_grade_text,
            "additional_buy_check": add_buy_check,
            "band_breach_check_holder_only": {
                "market_open_at_analysis_time": market_currently_open,
                "basis_label": breach_basis_label,
                "basis_price": breach_basis_price,
                "basis_date": breach_basis_date,
                "band_lower_edge_stop_p": stop_p,
                "verdict_text": band_breach_text
            }
        }

        full_json_str = json.dumps(full_rag_payload, indent=2, ensure_ascii=False)
        compact_news = [{"title": i.get("title", ""), "publisher": i.get("publisher", ""), "date": i.get("date", "")} for i in news_data] if news_data else []
        compact_macro_news = [{"title": i.get("title", ""), "publisher": i.get("publisher", ""), "date": i.get("date", "")} for i in macro_news_data] if macro_news_data else []

        response_content = None
        if not api_key: response_content = "⚠️ GEMINI_API_KEY가 등록되지 않았습니다. 아래 [분석용 JSON 데이터 다운로드] 버튼으로 JSON을 내려받아 분석을 요청하세요."
        else:
            template = """
[RAG 심층 주입 데이터]
1. 기술적/수급, VWAP, 볼린저 밴드 스퀴즈 및 6개월 매물대 POC ({ticker}) (기준일: {stock_date}):
{tech_json}

2. 최근 6개월 최다 매물대(POC) 및 70% 핵심 매물대(Value Area):
{poc_json}

3. 최근 다년간(1~5년) 데이터 기반 듀얼 전략 백테스팅 결과:
{backtest_json}

4. 최근 6개월 피보나치 되돌림 밴드:
{fib_json}

5. 가장 빠른 만기 옵션 체인 수급 (콜/풋 Max OI & Volume):
{options_json}

6. 내부자/기관 지분율 및 주요 기관투자자 보유·공매도 세력 분석 (Short Squeeze Analysis) - ⚠️ top_holders는 대형 패시브 인덱스 운용사 위주 데이터이며 헤지펀드 데이터가 아님, "헤지펀드"로 지칭하지 말 것:
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

17. 파이썬 알고리즘 사전 연산 목표가 (절대 임의 수정 금지):
{targets_json}

18. 백테스트-기술적 신호 정합성 체크 (반드시 반영):
{consistency_note}

19. 백테스트 전략 비교 및 매매 스타일 판정 (절대 임의 수정 금지):
{backtest_verdict}

20. ATR 손절선 고정 라벨 (절대 임의 수정 금지):
{atr_labels_json}

21. 파이썬 알고리즘 사전 연산 추가 매수 여부 판정 (절대 임의 수정 금지):
{add_buy_json}

22. 파이썬 알고리즘 사전 연산 신규 진입 등급 (예상 손익비 기반, 절대 임의 수정 금지):
{entry_grade_json}

23. 파이썬 알고리즘 사전 연산 매매 접근방식 주의사항 (벤치마크 기반, 절대 임의 수정 금지):
{trading_caution_json}

---

[지시사항 - 분석 정합성, 11개 섹터 전수 분석 및 POC 매물벽 검증 규칙]
위 데이터를 바탕으로 최고 수준의 퀀트/금융 애널리스트 관점에서 정밀 리포트를 작성할 것:

0. [플레이스홀더 규칙 - 반드시 준수] 아래 [지시사항]과 [출력 포맷]에 `[[F15]]`, `[[F16]]`처럼 대괄호 두 개로 감싼 자리가 나오면, 그 자리는 시스템이 응답을 받은 뒤 실제 계산값으로 자동 치환하는 자리이다. 해당 자리에는 반드시 그 플레이스홀더 문자열만 대괄호 두 개를 포함해 정확히, 앞뒤에 다른 설명이나 숫자 없이 출력할 것 - 직접 값을 계산하거나 요약하거나 풀어쓰지 말 것.

1. 거시환경 및 시장 국면
- **[참고자료 및 기준일자]**: 분석에 활용된 핵심 매크로 지표의 **출처 및 수집 기준일자**를 명시할 것.
- **[자산배분 코멘트]**: 정량 배분표 대신 주입된 매크로 지표 기반의 정성적 방향성만 제시할 것.

2. 11개 전 섹터 전망 및 자금 순환매 심층 분석
- **11개 섹터 전수 리스트 작성**: 11개 모두 글머리 기호(*)로 작성.
- **자금 순환매 결론 분리 (필수)**: 리스트 후 빈 줄을 삽입하고 순환매 결론 작성.

3. 밸류에이션, 스마트머니 및 공매도 세력 분석
- **[장기 복리 체력]**: 3개년 FCF·ROIC·주주환원 분석.
- **주요 기관투자자 보유 현황(대형 패시브 인덱스 운용사 위주이므로 특정 종목에 대한 액티브 확신 매수 시그널로 과대 해석하지 말 것) / 공매도 세력 및 숏스퀴즈 리스크 / IB 투자의견 신뢰도 가중**.

4. 정밀 기술적 지표, VWAP, POC 매물대 및 백테스팅 평가 ({ticker})
- **스코어카드 산출**: ⚠️ 아래 [출력 포맷]의 해당 자리에 플레이스홀더 `[[F15]]`만 출력할 것.
- **[백테스트 신뢰도 표기 필수]**: JSON에 포함된 reliability(신뢰도 라벨)와 period_split(전반부/후반부 성과)을 반드시 언급하고, 표본 부족 시 참고 지표로만 서술할 것.
- **[벤치마크 대비 평가]**: ⚠️ 아래 [출력 포맷]의 해당 자리에 플레이스홀더 `[[F19]]`만 출력할 것.

5. [신규 진입 적격성 평가 (미보유자 관점 핵심 진단)]
- **신규 진입 등급**: ⚠️ 아래 [출력 포맷]의 해당 자리에 플레이스홀더 `[[F22]]`만 출력할 것.
- **진입 적합성 분석**: 22번 [사전 연산 신규 진입 등급]과 16번 [사전 연산 예상 손익비]의 근거를 기술적 지표(스퀴즈, 이평선, POC 매물대 등)와 함께 자유롭게 서술할 것. ⚠️[숫자 정확성] 이 문단에서 현재가를 언급할 경우 반드시 1번 데이터의 current_price 값을 그대로 사용할 것 - 반올림하거나 다른 값으로 재계산하지 말 것.
- **예상 손익비 (Risk/Reward)**: ⚠️ 아래 [출력 포맷]의 해당 자리에 플레이스홀더 `[[F16]]`만 출력할 것.

6. [정밀 매매 시나리오]
- **[매매 스타일 일치 규칙 (필수)]**: 반드시 [19. 백테스트 전략 비교]에서 판정된 전략 스타일에 맞춰 분할 매수 밴드/목표가/불타기 조건의 논리를 서술할 것.
- **[매수 밴드 및 진입 가격 상/하단 논리 일치 규칙]**: 하단 [사용자 대응 전략] 가격과 상단 [분할 매수 밴드] 가격 100% 일치시킬 것.
- **[가격 표기]**: 밴드는 오름차순(낮은 가격 ~ 높은 가격)으로 표기.
- **[플레이스홀더 대상]**: 매매 접근 방식 주의사항(`[[F23]]`), 1차 목표가(`[[F17T1]]`), 2차 목표가(`[[F17T2]]`), 손절(Stop-loss) 기준선(`[[F20]]`), 추가 매수 여부(`[[F21]]`)은 아래 [출력 포맷]의 해당 자리에 각 플레이스홀더만 출력할 것 - 분할 매수 밴드/매도가 밴드/불타기 조건 서술은 자유롭게 작성.

7. [최종 투자의견 규칙 (엄격 준수)]
- 관망이 유리하면 최종 투자의견을 절대 '매수'로 적지 말고 '관망' 또는 '홀딩'으로 명시.
- 매수/관망/홀딩 중 하나만 명확히 단독으로 출력할 것 (뒤에 괄호나 부기 문구를 붙이지 말 것 - 필요한 부기는 시스템이 사후에 자동으로 추가함).
- **[밴드 이탈 판정(보유자 전용)]**: ⚠️ 아래 [출력 포맷]의 해당 자리에 플레이스홀더 `[[F18]]`만 출력할 것 - 이는 확정 종가 기준으로 보유 포지션이 분할 매수 밴드 하단을 이미 이탈했는지 여부를 사전 판정한 결과이며, 바로 아래 [사용자 대응 전략]은 이 판정 결과와 모순되지 않게 서술할 것.

---

[리포트 최종 출력 포맷 - 반드시 아래 구조로 동일하게 작성]:

[1. 거시환경 및 시장 국면]
* **참고자료 및 기준일자**: [내용]
* **6대 유동성 자산 변동 및 자산배분 전략**: [내용]

[2. 11개 전 섹터 전망 및 자금 순환매 심층 분석]
* [11개 섹터 리스트 전수 나열]
* **자금 순환매 결론**: [내용]

[3. 밸류에이션, 스마트머니 및 공매도 세력/옵션 분석]
* **장기 복리 체력**: [내용]
* **주요 기관투자자 보유 현황**: [내용 - 패시브 인덱스 운용사 위주임을 감안해 과도한 긍정적 해석 지양]
* **공매도 세력 및 숏스퀴즈 리스크**: [내용]
* **IB 투자의견 신뢰도 가중**: [내용]

[4. 정밀 기술적 지표, VWAP, POC 매물대 및 백테스팅 평가]
* **스코어카드 관점**: [[F15]]
* **백테스트 신뢰도**: [내용 - reliability/period_split 자유 서술]
* **벤치마크 대비 평가**: [[F19]]

[5. 신규 진입 적격성 평가]
* **신규 진입 등급**: [[F22]]
* **진입 적합성 분석**: [내용]
* **예상 손익비 (Risk/Reward)**: [[F16]]

[6. 정밀 매매 시나리오]
* **매매 접근 방식 주의사항**: [[F23]]
* **분할 매수 밴드**: [내용 (오름차순)]
* **1차 목표가**: [[F17T1]]
* **2차 목표가**: [[F17T2]]
* **매도가 밴드**: [내용 (오름차순)]
* **손절(Stop-loss) 기준선**: [[F20]]
* **불타기 조건**: [스퀴즈 상방 돌파 등 (19번 전략 스타일에 맞춰 서술)]
* **추가 매수 여부**: [[F21]]

[7. 최종 투자의견]
* **최종 투자의견**: [매수/관망/홀딩 중 택1, 단독으로만 출력]
* **밴드 이탈 판정 (보유자 전용)**: [[F18]]
{strategy_guide}
"""
            prompt = PromptTemplate(
                input_variables=["ticker", "stock_date", "tech_json", "poc_json", "backtest_json", "fib_json", "options_json", "hedge_short_json", "earnings_json", "macro_json", "macro_news_json", "sector_json", "fund_json", "user_position", "strategy_guide", "news_json", "analyst_json", "score_json", "rr_json", "targets_json", "consistency_note", "backtest_verdict", "add_buy_json", "entry_grade_json", "trading_caution_json"],
                template=template
            )
            # [개선] SDK 자체 재시도/타임아웃을 명시적으로 짧게 제한해서, 아래 앱 자체 재시도 루프(0/5/10초 간격)가
            # 재시도 정책의 주도권을 갖도록 함. 기존엔 SDK 기본 정책(수 회, 최대 16초씩 늘어나는 지수 백오프)에 맡겨져
            # chain.invoke() 한 번이 2분 넘게 걸리는 경우가 관측됨.
            llm = ChatGoogleGenerativeAI(model=selected_model_id, google_api_key=api_key, max_retries=1, timeout=30)
            chain = prompt | llm
            
            payload = {
                "ticker": ticker_input, "stock_date": stock_date,
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
                "score_json": precalc_scorecard, "rr_json": rr_text,
                "targets_json": targets_text, "consistency_note": consistency_note,
                "backtest_verdict": backtest_verdict, "atr_labels_json": atr_labels_text,
                "add_buy_json": add_buy_text, "entry_grade_json": entry_grade_text,
                "trading_caution_json": trading_caution_text
            }
            
            # [개선] LLM 호출 소요시간 로깅 - timeout(30초) 값이 실제 정상 케이스 응답시간에 비해 적절한지
            # 나중에 실측 로그로 검증할 수 있도록 시도별 소요시간을 남김.
            for attempt_no, delay in enumerate([0, 5, 10], start=1):
                if delay > 0: time.sleep(delay)
                call_start = time.time()
                try:
                    res_llm = chain.invoke(payload)
                    elapsed = time.time() - call_start
                    response_content = extract_clean_text(res_llm.content)
                    logger.info(f"[LLM 호출] {ticker_input} 시도 {attempt_no}/3 성공 - 소요시간 {elapsed:.1f}초")
                    break
                except Exception as e:
                    elapsed = time.time() - call_start
                    logger.info(f"[LLM 호출] {ticker_input} 시도 {attempt_no}/3 실패 - 소요시간 {elapsed:.1f}초, 에러: {e}")
                    continue
            if not response_content: response_content = "⚠️ Gemini API 일시적 지연이 발생했습니다. [분석용 JSON 데이터 다운로드]를 통해 확인하세요."
            elif not response_content.startswith("⚠️"):
                # [개선/2026-09, item 21 그룹 B] LLM이 "그대로 복사" 지시를 어기고 스스로 재작성/재계산한
                # 경우를 코드 레벨에서 강제 교정 - 9개 사전연산 필드 플레이스홀더 치환 + 24번 부기 + 리포트
                # 전체의 "현재가($...)" 오기재 스윕. 예외 발생 시에도 원본 리포트는 보존(안전 폴백).
                try:
                    _enforce_ctx = {
                        "precalc_scorecard": precalc_scorecard, "rr_text": rr_text,
                        "t1_label": t1_label, "t1_price": t1_price, "t2_label": t2_label, "t2_price": t2_price,
                        "backtest_verdict": backtest_verdict, "atr_labels_text": atr_labels_text,
                        "add_buy_text": add_buy_text, "entry_grade_text": entry_grade_text,
                        "trading_caution_text": trading_caution_text, "verdict_qualifier_text": verdict_qualifier_text,
                        "curr_p": curr_p, "band_breach_text": band_breach_text,
                    }
                    response_content = enforce_precalculated_fields(response_content, _enforce_ctx, logger=logger)
                except Exception as e:
                    logger.warning(f"[강제치환 계층 오류] {ticker_input}: {e} - 원본 리포트 그대로 사용")

        if response_content and not response_content.startswith("⚠️"):
            act, ent_grade, ent_rr, t1, t2, sell_b, buy_b, sl_b, pyr, add_buy_note, u_strat_summary, q_badge = parse_full_trading_scenario(response_content)
            # [물타기→추가 매수 여부 리네이밍, 2026-09] 저장 키를 'averaging_down'에서 'additional_buy'로
            # 교체. 이미 저장된 과거 히스토리 항목의 내용(값)은 그대로 두고(재작성/마이그레이션 없음),
            # 앞으로 새로 저장되는 항목부터 새 키를 쓴다 - 조회 쪽(render_history_card)은 신키 우선 +
            # 구키 폴백으로 두 세대의 데이터를 모두 정상 표시한다.
            st.session_state.history[ticker_input] = {
                "action": act, "entry_grade": ent_grade, "entry_rr": ent_rr, "quality_badge": q_badge, "price": curr_p, "my_avg": user_avg_price if is_holding else 0, "my_return": my_return_str if is_holding else "미보유", "target_1": t1, "target_2": t2, "take_profit": t1, "sell_target": sell_b, "buy_band": buy_b, "stop_loss": sl_b, "pyramiding": pyr, "additional_buy": add_buy_note, "user_strategy": u_strat_summary, "time": get_current_kst_time_str()
            }
            save_history(st.session_state.history)

        st.session_state.last_analysis_result = {
            "ticker": ticker_input, "info_source": info_source_flag, "model_label": selected_model_label, "curr_p": curr_p, "is_holding": is_holding, "user_avg_price": user_avg_price, "user_shares": user_shares, "my_return_str": my_return_str, "tech_data": tech_data, "vol_profile": vol_profile, "backtest_results": backtest_results, "stock_date": stock_date, "fib_levels": fib_levels, "options_data": options_data, "macro_data": macro_data, "fund_data": fund_data, "sector_data": sector_data, "ownership": ownership, "hedge_short_intel": hedge_short_intel, "earnings_info": earnings_info, "response_content": response_content, "full_json_str": full_json_str, "macro_news_data": macro_news_data, "news_data": news_data, "analyst_data": analyst_data
        }
        st.rerun()

# =============================================================================
# [BLOCK 10] 메인 대시보드 시각화 렌더링
# =============================================================================
if st.session_state.last_analysis_result:
    res = st.session_state.last_analysis_result
    curr_p, ownership, hedge_short_intel, earnings_info, tech_data, vol_profile = res["curr_p"], res["ownership"], res.get("hedge_short_intel", {}), res["earnings_info"], res["tech_data"], res.get("vol_profile", {})
    backtest_results, macro_data, fib_levels, options_data, fund_data, sector_data, info_source = res.get("backtest_results", None), res["macro_data"], res["fib_levels"], res["options_data"], res["fund_data"], res.get("sector_data", {}), res.get("info_source", "stock.info")

    api_reply_time = st.session_state.history.get(res['ticker'], {}).get('time', get_current_kst_time_str())
    st.info(f"🔄 **API 데이터 회신 시간:** `{api_reply_time}` (KST)")

    fred_val = macro_data.get("us_10y_yield", {}).get("value", "N/A")
    fred_status = "🔴 실패 (N/A)" if fred_val == "N/A" else "🟢 정상"
    gemini_content = res.get("response_content", "")
    gemini_status = "🔴 실패 (API 에러)" if gemini_content.startswith("⚠️") else "🟢 정상"

    # [데이터 접근 실패 투명성 배너, 2026-09] Yahoo Finance의 "Invalid Crumb"/레이트리밋
    # 이슈는 이 앱이 통제할 수 없는 외부 요인이므로, 실패가 발생했을 때 조용히 N/A로만
    # 표시하지 않고 사용자에게 명시적으로 알린다. 가격/기술적 지표까지 아예 못 가져온
    # 경우(curr_p==0 또는 tech_data 비어있음)가 가장 심각하므로 st.error로, 펀더멘털/
    # 밸류에이션만 일부 실패한 경우(info_source가 stock.info가 아님)는 st.warning으로 구분.
    if curr_p == 0 or not tech_data:
        st.error("🚨 **가격/기술적 지표 데이터 조회 실패** — Yahoo Finance 접근 문제(레이트리밋 등 일시적인 외부 요인)로 이번 분석은 가격 데이터 자체를 가져오지 못했습니다. 이 리포트의 지표는 신뢰할 수 없으니 몇 분 후 다시 시도해 주세요.")
    elif info_source != "stock.info":
        st.warning("⚠️ **펀더멘털/밸류에이션 데이터 일부 조회 실패** — Yahoo Finance 접근 문제(Invalid Crumb/레이트리밋 등 외부 서비스 이슈)로 일부 펀더멘털 지표가 N/A로 표시되었을 수 있습니다. 가격/기술적 지표는 정상 반영되었습니다.")

    if info_source == "stock.info": st.markdown(f"📡 **데이터 소스:** `🟢 Yahoo Finance stock.info` ｜ **FRED API:** `{fred_status}` ｜ **Gemini AI:** `{gemini_status}`")
    else: st.markdown(f"📡 **데이터 소스:** `🟡 Yahoo Finance 조회 실패` (일부 항목 N/A) ｜ **FRED API:** `{fred_status}` ｜ **Gemini AI:** `{gemini_status}`")

    if res["is_holding"] and res["user_avg_price"] > 0 and res["user_shares"] > 0 and curr_p > 0:
        total_invested, total_current = res["user_avg_price"] * res["user_shares"], curr_p * res["user_shares"]
        pnl_dollar, pnl_pct = total_current - total_invested, ((curr_p - res["user_avg_price"]) / res["user_avg_price"]) * 100
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
        
        q_factors = fund_data.get('quality_factors', {})
        st.divider()
        st.markdown("**💎 펀더멘털 우량성 & 현금창출력 (Quality Factors)**")
        q1, q2, q3, q4 = st.columns(4)
        q1.metric("잉여현금흐름 (FCF)", str(q_factors.get('free_cash_flow', 'N/A')), "순수 현금창출")
        q2.metric("부채비율 (D/E)", str(q_factors.get('debt_to_equity', 'N/A')), "재무 건전성")
        q3.metric("매출총이익률 (GM)", str(q_factors.get('gross_margin', 'N/A')), "가격 결정력/해자")
        q4.metric("영업이익률 (OPM)", str(q_factors.get('operating_margin', 'N/A')), f"ROE: {fund_data.get('roe', 'N/A')}")

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
        own_c1.metric("Insider Own", str(ownership.get('insider_own', 'N/A')))
        own_c2.metric("Insider Trans", str(ownership.get('insider_trans', 'N/A')))
        own_c3.metric("Inst Own", str(ownership.get('inst_own', 'N/A')))
        own_c4.metric("Inst Trans", str(ownership.get('inst_trans', 'N/A')))
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
        s_c4.metric("MACD (Signal)", f"{tech_data.get('macd', 'N/A')} ({tech_data.get('macd_signal', 'N/A')})", f"Hist: {tech_data.get('macd_hist', 'N/A'):+}" if isinstance(tech_data.get('macd_hist'), (int, float)) else None)

    if sector_data:
        with st.expander("🧭 **S&P 500 11개 전 섹터 실시간 등락 및 순환매 현황 (11 Sectors Rotation) [클릭하여 펼치기]**", expanded=False):
            s_rows = [{"티커": etf, "섹터명": s_info.get("sector_name", ""), "5일 등락률": s_info.get("return_5d", "N/A"), "1개월 등락률": s_info.get("return_1m", "N/A"), "현재가 ($)": f"${s_info.get('latest_close', 'N/A')}"} for etf, s_info in sector_data.items()]
            st.dataframe(pd.DataFrame(s_rows), use_container_width=True, hide_index=True)

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
            st.markdown("##### 🔬 **다년간(1~5년) 퀀트 전략 백테스팅 시뮬레이션**")
            bh_ret = backtest_results.get("benchmark_buy_and_hold", 0.0)
            st.caption(f"📌 **벤치마크 (단순 보유 Buy & Hold 수익률):** `{bh_ret:+.2f}%`")
            
            bt_col1, bt_col2 = st.columns(2)
            with bt_col1:
                st.markdown("**🚀 전략 A: 모멘텀 스퀴즈 돌파 (Momentum Squeeze Breakout)**")
                s1 = backtest_results.get("strategy_1_momentum_squeeze", {})
                m1_1, m1_2, m1_3, m1_4 = st.columns(4)
                m1_1.metric("총 누적 수익률", f"{s1.get('total_ret', 0):+.2f}%", f"B&H 대비 {round(s1.get('total_ret', 0) - bh_ret, 2):+.2f}%p")
                m1_2.metric("승률 (Win Rate)", f"{s1.get('win_rate', 0)}%", f"총 {s1.get('trades_count', 0)}회 매매")
                m1_3.metric("Profit Factor / MDD", f"{s1.get('profit_factor', 0)}", f"MDD: -{s1.get('mdd', 0)}%")
                m1_4.metric("Sharpe Ratio", f"{s1.get('sharpe_ratio_annualized', 0)}", "위험조정수익 (연율화)")
                # [item26-b] profit_factor가 항상 숫자로 바뀌면서(예: 손실거래 0건 → 99.9)
                # "표본부족" 같은 캐버트가 숫자 뒤에 묻히지 않도록 신뢰도 캡션에 함께 노출.
                pf_note1 = s1.get('profit_factor_note') or ""
                st.caption(f"🛡️ **신뢰도:** `{s1.get('reliability', 'N/A')}`" + (f" | ⚠️ PF: {pf_note1}" if pf_note1 else "") + f" | 📅 **기간분할 검증:** `{backtest_results.get('strategy_1_period_split', 'N/A')}`")

            with bt_col2:
                st.markdown("**🔄 전략 B: VWAP + RSI 밸류 되돌림 (Mean Reversion)**")
                s2 = backtest_results.get("strategy_2_vwap_mean_reversion", {})
                m2_1, m2_2, m2_3, m2_4 = st.columns(4)
                m2_1.metric("총 누적 수익률", f"{s2.get('total_ret', 0):+.2f}%", f"B&H 대비 {round(s2.get('total_ret', 0) - bh_ret, 2):+.2f}%p")
                m2_2.metric("승률 (Win Rate)", f"{s2.get('win_rate', 0)}%", f"총 {s2.get('trades_count', 0)}회 매매")
                m2_3.metric("Profit Factor / MDD", f"{s2.get('profit_factor', 0)}", f"MDD: -{s2.get('mdd', 0)}%")
                m2_4.metric("Sharpe Ratio", f"{s2.get('sharpe_ratio_annualized', 0)}", "위험조정수익 (연율화)")
                pf_note2 = s2.get('profit_factor_note') or ""
                st.caption(f"🛡️ **신뢰도:** `{s2.get('reliability', 'N/A')}`" + (f" | ⚠️ PF: {pf_note2}" if pf_note2 else "") + f" | 📅 **기간분할 검증:** `{backtest_results.get('strategy_2_period_split', 'N/A')}`")

    with st.container(border=True):
        st.markdown(f"##### 📐 **최근 6개월 피보나치 되돌림 지지/저항 밴드** (최고: `${fib_levels.get('high_6m', 'N/A')}` / 최저: `${fib_levels.get('low_6m', 'N/A')}`)")
        fb1, fb2, fb3, fb4 = st.columns(4)
        f236, f382, f500, f618 = fib_levels.get('fib_23.6%', 'N/A'), fib_levels.get('fib_38.2%', 'N/A'), fib_levels.get('fib_50.0%', 'N/A'), fib_levels.get('fib_61.8%', 'N/A')
        fb1.metric("23.6% 되돌림 (단기 지지)", f"${f236}", f"{round(((f236-curr_p)/curr_p)*100, 1):+.1f}%" if isinstance(f236, (int, float)) and curr_p else None)
        fb2.metric("38.2% 되돌림 (1차 매수 지지)", f"${f382}", f"{round(((f382-curr_p)/curr_p)*100, 1):+.1f}%" if isinstance(f382, (int, float)) and curr_p else None)
        fb3.metric("50.0% 하프라인 (추세 기준선)", f"${f500}", f"{round(((f500-curr_p)/curr_p)*100, 1):+.1f}%" if isinstance(f500, (int, float)) and curr_p else None)
        fb4.metric("61.8% 되돌림 (강력한 2차 지지)", f"${f618}", f"{round(((f618-curr_p)/curr_p)*100, 1):+.1f}%" if isinstance(f618, (int, float)) and curr_p else None)

    with st.container(border=True):
        if options_data:
            exp_date, pc_rat = options_data['expiration_date'], options_data['pc_volume_ratio']
            st.markdown(f"##### 🎯 **가장 빠른 만기 옵션 체인 스마트머니 포지션** `만기일: {exp_date}` `P/C Ratio: {pc_rat}`")
            op_c1, op_c2, op_c3, op_c4 = st.columns(4)
            c_oi = options_data['call_max_oi']
            if isinstance(c_oi['strike'], (int, float)):
                diff_c_oi = round(((c_oi['strike'] - curr_p) / curr_p) * 100, 1) if curr_p else None
                op_c1.metric("콜옵션 Max OI (상방 저항벽)", f"${c_oi['strike']}", f"{diff_c_oi:+.1f}% (OI: {c_oi['oi']:,} / ${c_oi['price']})" if diff_c_oi is not None else f"OI: {c_oi['oi']:,}")
            else:
                op_c1.metric("콜옵션 Max OI (상방 저항벽)", "N/A", "OI 데이터 없음 (당일 미결제약정 미형성)")
            c_vol = options_data['call_max_vol']
            diff_c_vol = round(((c_vol['strike'] - curr_p) / curr_p) * 100, 1) if curr_p and isinstance(c_vol['strike'], (int, float)) else None
            op_c2.metric("콜옵션 Max Vol (당일 상방 수급)", f"${c_vol['strike']}", f"{diff_c_vol:+.1f}% (Vol: {c_vol['volume']:,} / ${c_vol['price']})" if diff_c_vol is not None else f"Vol: {c_vol['volume']:,}")
            p_oi = options_data['put_max_oi']
            if isinstance(p_oi['strike'], (int, float)):
                diff_p_oi = round(((p_oi['strike'] - curr_p) / curr_p) * 100, 1) if curr_p else None
                op_c3.metric("풋옵션 Max OI (하방 지지벽)", f"${p_oi['strike']}", f"{diff_p_oi:+.1f}% (OI: {p_oi['oi']:,} / ${p_oi['price']})" if diff_p_oi is not None else f"OI: {p_oi['oi']:,}")
            else:
                op_c3.metric("풋옵션 Max OI (하방 지지벽)", "N/A", "OI 데이터 없음 (당일 미결제약정 미형성)")
            p_vol = options_data['put_max_vol']
            diff_p_vol = round(((p_vol['strike'] - curr_p) / curr_p) * 100, 1) if curr_p and isinstance(p_vol['strike'], (int, float)) else None
            op_c4.metric("풋옵션 Max Vol (당일 하방 헤지)", f"${p_vol['strike']}", f"{diff_p_vol:+.1f}% (Vol: {p_vol['volume']:,} / ${p_vol['price']})" if diff_p_vol is not None else f"Vol: {p_vol['volume']:,}")
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
        with head_col1: st.markdown(f"### 📝 **{res['model_label']} 종합 분석 브리핑**")
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
                if m_item.get("summary"): st.caption(f"> {m_item['summary']}")
                st.caption(f"출처: {m_item['publisher']} | 게시일: {m_item['date']}\n")
        else: st.info("수집된 거시경제 기사가 없습니다.")
    st.write("")

    col_left, col_right = st.columns([0.9, 1.1])
    with col_left:
        with st.container(border=True):
            st.markdown(f"##### 📰 **{res['ticker']} 최신 주요 뉴스 및 기사 원문**")
            if res.get("news_data"):
                for item in res.get("news_data", []):
                    st.markdown(f"**[{item['title']}]({item['link']})**")
                    if item.get("summary"): st.markdown(f"> *{item['summary']}*")
                    st.caption(f"출처: {item['publisher']} | {item['date']}")
                    st.divider()
            else: st.info("수집된 최신 뉴스가 없습니다.")
                
    with col_right:
        with st.container(border=True):
            st.markdown(f"##### 🏛️ **{res['ticker']} 최근 2개월 증권가 투자의견 및 목표가 변동**")
            if res.get("analyst_data"):
                df_analyst = pd.DataFrame(res.get("analyst_data", []))
                display_cols = ["date", "firm", "tier", "action", "grade_change", "target_price"]
                df_analyst = df_analyst[[c for c in display_cols if c in df_analyst.columns]]
                df_analyst.columns = ["일자", "증권사", "기관 신뢰도 등급", "구분", "투자의견 변동", "제시 목표가"]
                st.dataframe(df_analyst, use_container_width=True, hide_index=True)
            else: st.info("최근 2개월간 등록된 투자의견 변동 내역이 없습니다.")