import os
import json
import warnings
from datetime import datetime, timedelta

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

# --- 페이지 기본 설정 ---
st.set_page_config(
    page_title="AI 주식 분석기",
    page_icon="📈",
    layout="wide"
)

# --- RAG 데이터 수집 함수 ---
def fetch_stock_technical_data(ticker: str):
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo")
    if df.empty:
        return {}
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    df['SMA_120'] = df['Close'].rolling(window=120).mean()
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    
    latest = df.iloc[-1]
    return {
        "ticker": ticker,
        "current_price": round(float(latest['Close']), 2),
        "sma_20": round(float(latest['SMA_20']), 2) if pd.notnull(latest['SMA_20']) else "N/A",
        "sma_60": round(float(latest['SMA_60']), 2) if pd.notnull(latest['SMA_60']) else "N/A",
        "sma_120": round(float(latest['SMA_120']), 2) if pd.notnull(latest['SMA_120']) else "N/A",
        "rsi_14": round(float(latest['RSI']), 2) if pd.notnull(latest['RSI']) else "N/A",
        "macd": round(float(latest['MACD']), 2) if pd.notnull(latest['MACD']) else "N/A",
        "macd_signal": round(float(latest['MACD_Signal']), 2) if pd.notnull(latest['MACD_Signal']) else "N/A",
        "bb_upper": round(float(latest['BB_High']), 2) if pd.notnull(latest['BB_High']) else "N/A",
        "bb_lower": round(float(latest['BB_Low']), 2) if pd.notnull(latest['BB_Low']) else "N/A",
        "recent_volume_trend": "상승" if latest['Volume'] > df['Volume'].tail(5).mean() else "하락"
    }

def fetch_macro_indicators():
    macro_data = {}
    try:
        end = datetime.now()
        start = end - timedelta(days=30)
        dgs10 = web.DataReader('DGS10', 'fred', start, end).dropna().iloc[-1, 0]
        macro_data["us_10y_yield"] = round(float(dgs10), 2)
    except Exception:
        macro_data["us_10y_yield"] = "N/A"
        
    for name, ticker in [("vix", "^VIX"), ("wti_oil", "CL=F"), ("dollar_index", "DX-Y.NYB")]:
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            macro_data[name] = round(float(hist['Close'].iloc[-1]), 2) if not hist.empty else "N/A"
        except Exception:
            macro_data[name] = "N/A"
    return macro_data

def fetch_fundamentals(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        return {
            "trailing_pe": info.get("trailingPE", "N/A"),
            "forward_pe": info.get("forwardPE", "N/A"),
            "price_to_book": info.get("priceToBook", "N/A"),
            "roe": info.get("returnOnEquity", "N/A"),
            "operating_margins": info.get("operatingMargins", "N/A"),
            "target_mean_price": info.get("targetMeanPrice", "N/A"),
            "recommendation_key": info.get("recommendationKey", "N/A")
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

def extract_clean_text(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "\n".join([p["text"] if isinstance(p, dict) and "text" in p else str(p) for p in content])
    return str(content)

# --- UI 레이아웃 ---
st.title("📊 AI 주식/시장 분석 대시보드")
st.caption("Gemini AI + RAG 실시간 시장 지표 기반 종합 투자 리포트")

# 사이드바
with st.sidebar:
    st.header("⚙️ 분석 설정")
    ticker_input = st.text_input("종목 티커 입력 (예: NVDA, AAPL, TSLA)", value="NVDA").upper()
    analyze_btn = st.button("분석 실행", type="primary", use_container_width=True)

# 메인 화면 실행
if analyze_btn:
    with st.spinner(f"[{ticker_input}] 실시간 시장 지표 수집 및 Gemini 분석 중..."):
        tech_data = fetch_stock_technical_data(ticker_input)
        macro_data = fetch_macro_indicators()
        fund_data = fetch_fundamentals(ticker_input)
        sector_data = fetch_sector_performance()
        
        # 상단 요약 카드 메트릭 렌더링
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("현재가", f"${tech_data.get('current_price', 'N/A')}")
        c2.metric("RSI(14)", tech_data.get('rsi_14', 'N/A'))
        c3.metric("미 국채 10년물", f"{macro_data.get('us_10y_yield', 'N/A')}%")
        c4.metric("VIX 변동성", macro_data.get('vix', 'N/A'))
        
        st.divider()
        
        # Gemini 분석 호출
        # Streamlit Secrets 또는 .env에서 키 읽기
        # .env 우선 탐색 후, 없을 경우 st.secrets 안전 조회
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            try:
                api_key = st.secrets["GEMINI_API_KEY"]
            except Exception:
                api_key = None
        
        if not api_key:
            st.error("GEMINI_API_KEY가 설정되지 않았습니다. Streamlit Secrets에 등록하세요.")
        else:
            template = """
[RAG 주입 데이터]
1. 기술적/수급 데이터 ({ticker}):
{tech_json}

2. 매크로/시장 지표:
{macro_json}

3. 주요 섹터 5일 등락률:
{sector_json}

4. 펀더멘털/밸류에이션:
{fund_json}

---

[지시사항]
위 [RAG 주입 데이터]를 기반으로 아래 항목을 서술어 없이 간결하게 분석할 것:

1. 거시환경 및 시장 국면
- 경기 국면 (회복 / 활황 / 둔화 / 침체 판정)
- 단기 변동성 촉발 요인
- 권장 자산 배분 비중 (주식 : 채권 : 현금)

2. 섹터 전망 및 순환매
- 상대적 강세 섹터 및 약세 섹터 요약
- 자금 순환매(Rotation) 방향

3. 종목 종합 평가 ({ticker})
- 기술적 분석: 이평선 배열, 지지/저항선, 과매수/과매도 여부
- 스코어카드 (각 10점 만점): 성장성, 수익성, 밸류에이션, 해자, 리스크
- 종합 평점 및 최종 투자 의견 (적극매수 / 분할매수 / 관망 / 비중축소)
- 매매 시나리오: 분할 매수 밴드, 목표가/익절 라인, 손절(Stop-loss) 기준선
"""
            prompt = PromptTemplate(
                input_variables=["ticker", "tech_json", "macro_json", "sector_json", "fund_json"],
                template=template
            )
            llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=api_key)
            chain = prompt | llm
            
            response = chain.invoke({
                "ticker": ticker_input,
                "tech_json": json.dumps(tech_data, indent=2, ensure_ascii=False),
                "macro_json": json.dumps(macro_data, indent=2, ensure_ascii=False),
                "sector_json": json.dumps(sector_data, indent=2, ensure_ascii=False),
                "fund_json": json.dumps(fund_data, indent=2, ensure_ascii=False)
            })
            
            st.markdown(extract_clean_text(response.content))