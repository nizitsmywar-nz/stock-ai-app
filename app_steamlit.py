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
    page_title="AI Multi-Asset Analyst Pro",
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
</style>
""", unsafe_allow_html=True)

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

if "last_news_analysis" not in st.session_state:
    st.session_state.last_news_analysis = None

def normalize_ticker(raw_input: str) -> tuple:
    clean = raw_input.strip().upper()
    if clean in ["GOLD", "금", "GC"]:
        return "GC=F", "금 (Gold Futures)"
    elif clean in ["BTC", "비트코인", "BITCOIN"]:
        return "BTC-USD", "비트코인 (Bitcoin)"
    else:
        return clean, f"주식 ({clean})"

def get_stock_info_with_retry(stock, retries=3):
    for attempt in range(retries):
        try:
            info = stock.info
            if isinstance(info, dict) and len(info) > 10 and any(k in info for k in ['marketCap', 'trailingPE', 'forwardPE', 'currentPrice']):
                return info, "stock.info"
        except Exception:
            pass
        if attempt < retries - 1:
            time.sleep(1.0 * (attempt + 1))
    return {}, "stock.fast_info"

# -------------------------------------------------------------
# 📌 코드 기반 사전 검증 로직 (신규성 및 거래량 스파이크 판정)
# -------------------------------------------------------------
def evaluate_market_action_for_news(df):
    if df is None or len(df) < 20:
        return {"status": "데이터 부족"}
    
    latest_vol = float(df['Volume'].iloc[-1])
    avg_vol = float(df['Volume'].tail(20).mean())
    vol_spike = latest_vol > (avg_vol * 1.5)
    
    price_change_3d = float((df['Close'].iloc[-1] - df['Close'].iloc[-3]) / df['Close'].iloc[-3] * 100) if len(df) >= 3 else 0.0
    price_change_1d = float((df['Close'].iloc[-1] - df['Close'].iloc[-2]) / df['Close'].iloc[-2] * 100) if len(df) >= 2 else 0.0
    
    # 선반영(Priced-in) 의심 판정: 거래량은 평소와 비슷한데 최근 3일간 주가가 4% 이상 선행 움직임이 있었을 경우
    pre_pricing_suspected = abs(price_change_3d) > 4.0 and not vol_spike
    
    return {
        "volume_spike_detected": vol_spike,
        "volume_ratio": round(latest_vol / avg_vol, 2) if avg_vol > 0 else 1.0,
        "price_change_1d_pct": round(price_change_1d, 2),
        "price_change_3d_pct": round(price_change_3d, 2),
        "pre_pricing_suspected": pre_pricing_suspected
    }

def fetch_stock_technical_data(ticker: str):
    stock = yf.Ticker(ticker)
    df = stock.history(period="1y")
    if df.empty:
        df = stock.history(period="6mo")
    if df.empty:
        return {}, "N/A", {}, "N/A", "N/A", pd.DataFrame(), {}
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    try:
        df['ATR'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14)
    except Exception:
        df['ATR'] = None

    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    df['BB_Mid'] = bb.bollinger_mavg()
    df['BB_Width'] = (df['BB_High'] - df['BB_Low']) / df['BB_Mid'] * 100

    df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
    df['TP_Vol'] = df['Typical_Price'] * df['Volume']
    df['Cumulative_VWAP'] = df['TP_Vol'].cumsum() / df['Volume'].cumsum()
    df['Rolling_VWAP_20'] = df['TP_Vol'].rolling(window=20).sum() / df['Volume'].rolling(window=20).sum()

    latest = df.iloc[-1]
    last_date = df.index[-1].strftime("%Y-%m-%d")
    
    high_6m = float(df.tail(126)['High'].max())
    low_6m = float(df.tail(126)['Low'].min())
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
        bin_indices = np.digitize(df.tail(126)['Close'], price_bins) - 1
        bin_indices = np.clip(bin_indices, 0, num_bins - 1)
        vol_by_bin = np.zeros(num_bins)
        for idx_val, vol_val in zip(bin_indices, df.tail(126)['Volume']):
            vol_by_bin[idx_val] += vol_val
        poc_idx = int(np.argmax(vol_by_bin))
        poc_price = round(float((price_bins[poc_idx] + price_bins[poc_idx + 1]) / 2), 2)
        volume_profile = {"poc_price": poc_price}
    except Exception:
        volume_profile = {"poc_price": "N/A"}

    data = {
        "current_price": round(float(latest['Close']), 2),
        "atr_14": round(float(latest['ATR']), 2) if pd.notnull(latest['ATR']) else "N/A",
        "rsi_14": round(float(latest['RSI']), 2) if pd.notnull(latest['RSI']) else "N/A",
        "vwap_1y": round(float(latest['Cumulative_VWAP']), 2) if pd.notnull(latest['Cumulative_VWAP']) else "N/A",
        "vwap_20d": round(float(latest['Rolling_VWAP_20']), 2) if pd.notnull(latest['Rolling_VWAP_20']) else "N/A",
        "poc_price_6m": volume_profile.get("poc_price", "N/A"),
        "bb_width_pct": round(float(latest['BB_Width']), 2) if pd.notnull(latest['BB_Width']) else "N/A"
    }
    return data, last_date, fibonacci_levels, float(df['High'].max()), float(df['Low'].min()), df, volume_profile

def run_strategy_backtest(df: pd.DataFrame):
    if df is None or len(df) < 60:
        return None
    b_df = df.copy().dropna(subset=['Close', 'SMA_20', 'MACD', 'MACD_Signal', 'ATR', 'Cumulative_VWAP'])
    if len(b_df) < 30:
        return None
    bh_return = (b_df['Close'].iloc[-1] - b_df['Close'].iloc[0]) / b_df['Close'].iloc[0] * 100
    return {"benchmark_buy_and_hold": round(bh_return, 2)}

def fetch_nearest_options_data(ticker: str):
    return None

def fetch_macro_indicators():
    return {"us_10y_yield": {"value": "4.2%"}, "dollar_index": {"value": "104.5"}, "vix": {"value": "15.2"}}

def fetch_fundamentals_and_valuation(ticker: str, curr_price: float):
    stock = yf.Ticker(ticker)
    info = stock.info
    return {
        "market_cap_fmt": f"${info.get('marketCap', 0)/1e9:.2f}B" if info.get('marketCap') else "N/A",
        "trailing_pe": info.get("trailingPE", "N/A"),
        "forward_pe": info.get("forwardPE", "N/A"),
        "pbr": info.get("priceToBook", "N/A"),
        "ps_ratio": info.get("priceToSalesTrailing12Months", "N/A")
    }

def fetch_news(ticker: str, limit: int = 5):
    try:
        stock = yf.Ticker(ticker)
        raw_news = stock.news
        if not raw_news:
            return []
        articles = []
        for n in raw_news[:limit]:
            content = n.get("content", {})
            title = content.get("title", "") if isinstance(content, dict) else n.get("title", "")
            if title:
                articles.append({"title": title, "publisher": "Yahoo Finance", "date": "최근"})
        return articles
    except Exception:
        return []

def extract_clean_text(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "\n".join([p["text"] if isinstance(p, dict) and "text" in p else str(p) for p in content])
    return str(content)

def parse_full_trading_scenario(text):
    action, t1, sell_b, buy_b, sl_b = "홀딩", "분석 리포트 참조", "분석 리포트 참조", "분석 리포트 참조", "분석 리포트 참조"
    match_action = re.search(r"\[최종\s*투자의견\s*[:\-]?\s*([^\]]+)\]", text)
    if match_action:
        op_text = match_action.group(1).strip()
        action = "매수" if "매수" in op_text and "관망" not in op_text else ("매도" if "매도" in op_text or "비중축소" in op_text else "홀딩")
    return action, "", "", t1, "", sell_b, buy_b, sl_b, "", "분석 리포트 참조"

# -------------------------------------------------------------
# 사이드바 UI
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ **AI Multi-Asset Analyst Pro**")
    selected_model_label = st.selectbox("🤖 **AI 모델 선택**", ["Gemini 3.1 Flash Lite", "Gemini 3.6 Flash Lite", "Gemini 3.6 Flash"])
    selected_model_id = {"Gemini 3.1 Flash Lite": "gemini-3.1-flash-lite", "Gemini 3.6 Flash Lite": "gemini-3.6-flash-lite", "Gemini 3.6 Flash": "gemini-3.6-flash"}[selected_model_label]
    
    st.markdown("---")
    raw_ticker_input = st.text_input("종목/자산 입력 (예: TSLA, GOLD, BTC)", value=st.session_state.selected_ticker).strip()
    ticker_input, asset_display_name = normalize_ticker(raw_ticker_input)
    st.caption(f"🎯 인식된 자산: **{asset_display_name}**")
    
    is_holding = st.checkbox("💼 **현재 보유 중인 자산인가요?**", value=False)
    user_avg_price, user_shares = 0.0, 0.0
    if is_holding:
        u1, u2 = st.columns(2)
        with u1: user_avg_price = st.number_input("평단가 ($)", min_value=0.0, value=0.0, step=0.5)
        with u2: user_shares = st.number_input("수량", min_value=0.0, value=0.0, step=1.0)
            
    st.write("")
    analyze_btn = st.button("🚀 분석 & 백테스팅 실행", type="primary", use_container_width=True)
    
    # 📌 신규 추가된 '뉴스/이슈 확인' 검증 버튼 (분석 완료 시에만 활성화)
    st.divider()
    news_check_btn = st.button(
        "📰 뉴스/이슈 정밀 검증 (16대 체크리스트)", 
        disabled=(st.session_state.last_analysis_result is None),
        use_container_width=True
    )

# -------------------------------------------------------------
# 메인 분석 실행
# -------------------------------------------------------------
st.header(f"📊 [{asset_display_name}] 종합 밸류에이션 & 정밀 트레이딩 리포트")
is_macro_asset = ticker_input in ["GC=F", "BTC-USD"]

if analyze_btn:
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    with st.spinner(f"🔍 [{asset_display_name}] 기술적 지표, POC 매물대 및 백테스팅 실행 중..."):
        tech_data, stock_date, fib_levels, _, _, raw_df, vol_profile = fetch_stock_technical_data(ticker_input)
        backtest_results = run_strategy_backtest(raw_df)
        macro_data = fetch_macro_indicators()
        curr_p = tech_data.get('current_price', 0)
        fund_data = fetch_fundamentals_and_valuation(ticker_input, curr_p)
        news_data = fetch_news(ticker_input, limit=5)
        
        user_position_text = f"평단가 ${user_avg_price:.2f}, 보유수량 {user_shares:.1f}" if is_holding and user_avg_price > 0 else "미보유"
        strategy_instruction_text = "보유자 관점 전략 작성" if is_holding else "미보유자 신규 진입 전략 작성"

        template = """
[심층 분석 데이터 ({asset_name})]
1. 기술적 지표 및 POC 매물대: {tech_json}
2. 피보나치 되돌림: {fib_json}
3. 백테스팅 성과: {backtest_json}
4. 펀더멘털/매크로: {fund_json}

[지시사항]
오름차순 가격 밴드 정렬 및 보수적 매도가 밴드 설정을 준수하여 정밀 리포트를 작성할 것.
[신규 진입 적격성 평가]
* **신규 진입 등급**: [...]
* **예상 손익비 (Risk/Reward)**: [...]
[정밀 매매 시나리오]
* **분할 매수 밴드**: [...]
* **1차 목표가**: [...]
* **매도가 밴드**: [...]
* **손절(Stop-loss) 기준선**: [...]
[최종 투자의견: 적극매수 | 분할매수 | 홀딩 | 비중축소 | 관망]
{strategy_guide}
"""
        prompt = PromptTemplate(input_variables=["asset_name", "tech_json", "fib_json", "backtest_json", "fund_json", "user_position", "strategy_guide"], template=template)
        llm = ChatGoogleGenerativeAI(model=selected_model_id, google_api_key=api_key)
        
        try:
            res_ai = (prompt | llm).invoke({
                "asset_name": asset_display_name,
                "tech_json": json.dumps(tech_data, indent=2, ensure_ascii=False),
                "fib_json": json.dumps(fib_levels, indent=2, ensure_ascii=False),
                "backtest_json": json.dumps(backtest_results, indent=2, ensure_ascii=False) if backtest_results else "데이터 부족",
                "fund_json": json.dumps(fund_data if not is_macro_asset else macro_data, indent=2, ensure_ascii=False),
                "user_position": user_position_text,
                "strategy_guide": strategy_instruction_text
            })
            response_content = extract_clean_text(res_ai.content)
        except Exception as e:
            response_content = f"⚠️ 분석 생성 오류: {str(e)}"

        st.session_state.last_analysis_result = {
            "ticker": ticker_input,
            "asset_name": asset_display_name,
            "curr_p": curr_p,
            "tech_data": tech_data,
            "fib_levels": fib_levels,
            "response_content": response_content,
            "raw_df": raw_df,
            "news_data": news_data
        }
        st.session_state.last_news_analysis = None # 새 분석 실행 시 초기화
        st.rerun()

# -------------------------------------------------------------
# 📌 뉴스/이슈 확인 버튼 클릭 시 16대 체크리스트 분석 수행
# -------------------------------------------------------------
if news_check_btn and st.session_state.last_analysis_result:
    api_key = os.getenv("GEMINI_API_KEY") or st.secrets.get("GEMINI_API_KEY", None)
    res_cache = st.session_state.last_analysis_result
    df_cache = res_cache.get("raw_df")
    news_cache = res_cache.get("news_data")
    
    # 1. 코드 기반 사전 검증 (Pre-processing) 수행
    market_action_metrics = evaluate_market_action_for_news(df_cache)
    
    with st.spinner("📰 16가지 체크리스트 및 신규성(Newness) 정밀 검증 중..."):
        news_template = """
[시장 반응 사전 검증 데이터 (코드 연산 결과)]
- 최근 거래량 스파이크 여부: {vol_spike} (평균 대비 {vol_ratio}배)
- 최근 1일/3일 주가 변동률: {p_1d}% / {p_3d}%
- 사전 선반영(Priced-in) 의심 여부: {pre_priced}

[최근 뉴스 기사 목록]
{news_list}

[지시사항]
위 코드 연산 지표와 뉴스 기사를 바탕으로 16가지 체크리스트 관점(컨센서스 서프라이즈, 정보의 신규성, 소스 신뢰도, 이벤트 카테고리, 거래량 급증 여부 등)을 적용하여 분석할 것:
1. **정보의 신규성 판단**: 이미 시장에 선반영(Priced-in)되었거나 단순 반복성 기사여서 신규성이 없는 경우, 첫 줄에 반드시 **"📰 최신 뉴스 신규성 검토: 시장 선반영 완료 (특이 변동 요인 없음)"**이라고 명시할 것.
2. 뚜렷한 신규성과 가격 변동 트리거가 있는 경우에만 **[🚨 실시간 주가 변동 핵심 트리거 요약]** 섹션 하단에 체크리스트 기반 요약을 작성할 것.
"""
        news_prompt = PromptTemplate(input_variables=["vol_spike", "vol_ratio", "p_1d", "p_3d", "pre_priced", "news_list"], template=news_template)
        llm_news = ChatGoogleGenerativeAI(model=selected_model_id, google_api_key=api_key)
        
        try:
            news_res = (news_prompt | llm_news).invoke({
                "vol_spike": market_action_metrics.get("volume_spike_detected"),
                "vol_ratio": market_action_metrics.get("volume_ratio"),
                "p_1d": market_action_metrics.get("price_change_1d_pct"),
                "p_3d": market_action_metrics.get("price_change_3d_pct"),
                "pre_priced": market_action_metrics.get("pre_pricing_suspicion"),
                "news_list": json.dumps(news_cache, indent=2, ensure_ascii=False)
            })
            st.session_state.last_news_analysis = extract_clean_text(news_res.content)
        except Exception as e:
            st.session_state.last_news_analysis = f"⚠️ 뉴스 검증 오류: {str(e)}"
    st.rerun()

# -------------------------------------------------------------
# 결과 렌더링
# -------------------------------------------------------------
if st.session_state.last_analysis_result:
    res = st.session_state.last_analysis_result
    curr_p = res["curr_p"]
    tech_data = res["tech_data"]
    fib_levels = res["fib_levels"]
    resp_text = res.get("response_content", "")
    news_analysis_text = st.session_state.get("last_news_analysis", None)

    # 🚨 신규성이 확인된 경우에만 알림 팝업(st.warning) 노출
    if news_analysis_text and "선반영 완료" not in news_analysis_text:
        st.warning(f"🚨 **[{res['asset_name']}] 실시간 주가 변동 핵심 트리거 포착 (신규성 검증 완료)**\n\n{news_analysis_text}")
    elif news_analysis_text and "선반영 완료" in news_analysis_text:
        st.info(f"ℹ️ **[{res['asset_name']}] 뉴스 신규성 검토 결과**: 시장에 이미 선반영되어 특이 변동 요인이 없습니다.")

    with st.container(border=True):
        st.markdown(f"**🪙 [{res['asset_name']}] 실시간 시세 및 핵심 지표**")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("현재 시세", f"${curr_p:,.2f}")
        c2.metric("6M 최다 매물대 (POC)", f"${tech_data.get('poc_price_6m', 'N/A')}")
        c3.metric("RSI (14)", str(tech_data.get('rsi_14', 'N/A')))
        c4.metric("볼린저 밴드폭", f"{tech_data.get('bb_width_pct', 'N/A')}%")

    with st.container(border=True):
        st.markdown(f"### 📝 **[{res['asset_name']}] AI 종합 분석 브리핑**")
        st.markdown(re.sub(r'(?<!\\)\$', r'\$', resp_text))