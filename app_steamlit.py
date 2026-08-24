import os
import json
import math
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

st.set_page_config(
    page_title="AI 주식 종합 분석기 Pro",
    page_icon="📈",
    layout="wide"
)

# -------------------------------------------------------------
# 1. RAG 데이터 수집 및 적정주가 계산 모듈
# -------------------------------------------------------------
def fetch_stock_technical_data(ticker: str):
    stock = yf.Ticker(ticker)
    df = stock.history(period="6mo")
    if df.empty:
        return {}, "N/A"
    
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_60'] = df['Close'].rolling(window=60).mean()
    df['SMA_120'] = df['Close'].rolling(window=120).mean()
    df['RSI'] = ta.momentum.rsi(df['Close'], window=14)
    macd = ta.trend.MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_Signal'] = macd.macd_signal()
    df['MACD_Hist'] = macd.macd_diff()
    
    try:
        df['MFI'] = ta.volume.money_flow_index(df['High'], df['Low'], df['Close'], df['Volume'], window=14)
    except Exception:
        df['MFI'] = None

    bb = ta.volatility.BollingerBands(df['Close'], window=20, window_dev=2)
    df['BB_High'] = bb.bollinger_hband()
    df['BB_Low'] = bb.bollinger_lband()
    
    latest = df.iloc[-1]
    last_date = df.index[-1].strftime("%Y-%m-%d")
    
    data = {
        "ticker": ticker,
        "data_date": last_date,
        "current_price": round(float(latest['Close']), 2),
        "sma_20": round(float(latest['SMA_20']), 2) if pd.notnull(latest['SMA_20']) else "N/A",
        "sma_60": round(float(latest['SMA_60']), 2) if pd.notnull(latest['SMA_60']) else "N/A",
        "sma_120": round(float(latest['SMA_120']), 2) if pd.notnull(latest['SMA_120']) else "N/A",
        "rsi_14": round(float(latest['RSI']), 2) if pd.notnull(latest['RSI']) else "N/A",
        "mfi_14": round(float(latest['MFI']), 2) if pd.notnull(latest['MFI']) else "N/A",
        "macd": round(float(latest['MACD']), 2) if pd.notnull(latest['MACD']) else "N/A",
        "macd_signal": round(float(latest['MACD_Signal']), 2) if pd.notnull(latest['MACD_Signal']) else "N/A",
        "macd_hist": round(float(latest['MACD_Hist']), 2) if pd.notnull(latest['MACD_Hist']) else "N/A",
        "bb_upper": round(float(latest['BB_High']), 2) if pd.notnull(latest['BB_High']) else "N/A",
        "bb_lower": round(float(latest['BB_Low']), 2) if pd.notnull(latest['BB_Low']) else "N/A",
        "recent_volume_trend": "상승" if latest['Volume'] > df['Volume'].tail(5).mean() else "하락"
    }
    return data, last_date

def fetch_macro_indicators():
    macro_data = {}
    try:
        end = datetime.now()
        start = end - timedelta(days=30)
        fred_res = web.DataReader('DGS10', 'fred', start, end).dropna()
        dgs10 = fred_res.iloc[-1, 0]
        macro_data["us_10y_yield"] = {
            "value": f"{round(float(dgs10), 2)}%",
            "date": fred_res.index[-1].strftime("%Y-%m-%d")
        }
    except Exception:
        macro_data["us_10y_yield"] = {"value": "N/A", "date": "N/A"}
        
    for name, ticker in [("vix", "^VIX"), ("wti_oil", "CL=F"), ("dollar_index", "DX-Y.NYB")]:
        try:
            hist = yf.Ticker(ticker).history(period="5d")
            if not hist.empty:
                macro_data[name] = {
                    "value": round(float(hist['Close'].iloc[-1]), 2),
                    "date": hist.index[-1].strftime("%Y-%m-%d")
                }
            else:
                macro_data[name] = {"value": "N/A", "date": "N/A"}
        except Exception:
            macro_data[name] = {"value": "N/A", "date": "N/A"}
    return macro_data

def format_market_cap(market_cap):
    if not market_cap or market_cap == "N/A":
        return "N/A"
    if market_cap >= 1e12:
        return f"${market_cap / 1e12:.2f}T (조 달러)"
    elif market_cap >= 1e9:
        return f"${market_cap / 1e9:.2f}B (십억 달러)"
    elif market_cap >= 1e6:
        return f"${market_cap / 1e6:.2f}M (백만 달러)"
    return f"${market_cap:,.0f}"

def fetch_fundamentals_and_valuation(ticker: str):
    try:
        stock = yf.Ticker(ticker)
        info = stock.info
        
        market_cap = info.get("marketCap", "N/A")
        trailing_pe = info.get("trailingPE", "N/A")
        forward_pe = info.get("forwardPE", "N/A")
        pbr = info.get("priceToBook", "N/A")
        roe_raw = info.get("returnOnEquity", None)
        roe_pct = round(roe_raw * 100, 2) if roe_raw is not None else "N/A"
        eps = info.get("trailingEps", None)
        bps = info.get("bookValue", None)
        target_mean_price = info.get("targetMeanPrice", "N/A")
        
        # --- 3대 적정주가 계산 ---
        fair_values = {}
        
        # 1. 벤저민 그레이엄 공식 (EPS > 0, BPS > 0 일 때)
        if eps and bps and eps > 0 and bps > 0:
            graham_num = math.sqrt(22.5 * eps * bps)
            fair_values["graham_number"] = round(graham_num, 2)
        else:
            fair_values["graham_number"] = "산출불가 (적자/자본잠식)"
            
        # 2. 피터 린치 적정주가 (EPS * min(ROE, 30))
        if eps and eps > 0 and roe_raw and roe_raw > 0:
            growth_rate = min(roe_raw * 100, 30.0)  # 상한선 30%
            lynch_val = eps * growth_rate
            fair_values["peter_lynch"] = round(lynch_val, 2)
        else:
            fair_values["peter_lynch"] = "산출불가"

        # 3. ROE-PBR 모델 (BPS * (ROE / 0.10))
        if bps and bps > 0 and roe_raw and roe_raw > 0:
            roe_pbr_val = bps * (roe_raw / 0.10)
            fair_values["roe_pbr_model"] = round(roe_pbr_val, 2)
        else:
            fair_values["roe_pbr_model"] = "산출불가"

        return {
            "market_cap_raw": market_cap,
            "market_cap_fmt": format_market_cap(market_cap),
            "trailing_pe": trailing_pe,
            "forward_pe": forward_pe,
            "pbr": pbr,
            "roe": f"{roe_pct}%" if roe_pct != "N/A" else "N/A",
            "roe_val": roe_pct,
            "eps": eps if eps else "N/A",
            "bps": bps if bps else "N/A",
            "operating_margins": info.get("operatingMargins", "N/A"),
            "target_mean_price": target_mean_price,
            "recommendation_key": info.get("recommendationKey", "N/A"),
            "fair_values": fair_values
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
                summary[etf] = {
                    "change_5d": f"{pct:+.2f}%",
                    "date": hist.index[-1].strftime("%Y-%m-%d")
                }
            else:
                summary[etf] = {"change_5d": "N/A", "date": "N/A"}
        except Exception:
            summary[etf] = {"change_5d": "N/A", "date": "N/A"}
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
                publisher = content.get("provider", {}).get("displayName", "Yahoo Finance")
                click_url = content.get("clickThroughUrl", {})
                link = click_url.get("url", "") if isinstance(click_url, dict) else click_url
                if not link:
                    link = content.get("canonicalUrl", {}).get("url", "")
                pub_date = content.get("pubDate", "최근")
                if "T" in str(pub_date):
                    pub_date = str(pub_date).split("T")[0]
            else:
                title = n.get("title", "")
                publisher = n.get("publisher", "Yahoo Finance")
                link = n.get("link", "")
                pub_time = n.get("providerPublishTime", None)
                pub_date = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d") if pub_time else "최근"
            
            if title:
                articles.append({
                    "title": title,
                    "publisher": publisher,
                    "date": pub_date,
                    "link": link or f"https://finance.yahoo.com/quote/{ticker}"
                })
        return articles
    except Exception:
        return []

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
            filtered = upgrades.head(7)
            
        records = []
        for idx, row in filtered.head(7).iterrows():
            date_str = idx.strftime("%Y-%m-%d") if isinstance(idx, pd.Timestamp) else str(idx)[:10]
            records.append({
                "date": date_str,
                "firm": row.get("Firm", "N/A"),
                "to_grade": row.get("ToGrade", "N/A"),
                "from_grade": row.get("FromGrade", "N/A"),
                "action": row.get("Action", "N/A")
            })
        return records
    except Exception:
        return []

def extract_clean_text(content):
    if isinstance(content, str):
        return content
    elif isinstance(content, list):
        return "\n".join([p["text"] if isinstance(p, dict) and "text" in p else str(p) for p in content])
    return str(content)

# -------------------------------------------------------------
# 2. UI 및 메인 실행
# -------------------------------------------------------------
st.title("📈 AI 주식 종합 분석 및 밸류에이션 대시보드")
st.caption("실시간 주가·기술적 지표·밸류에이션·적정주가 산출·뉴스·증권가 컨센서스 기반 Gemini 분석")

with st.sidebar:
    st.header("⚙️ 종목 검색")
    ticker_input = st.text_input("종목 티커 입력 (예: GOOGL, NVDA, TSLA, AAPL)", value="GOOGL").upper()
    analyze_btn = st.button("분석 실행", type="primary", use_container_width=True)

if analyze_btn:
    with st.spinner(f"[{ticker_input}] 펀더멘털 지표 수집 및 적정주가 계산 중..."):
        tech_data, stock_date = fetch_stock_technical_data(ticker_input)
        macro_data = fetch_macro_indicators()
        fund_data = fetch_fundamentals_and_valuation(ticker_input)
        sector_data = fetch_sector_performance()
        news_data = fetch_news(ticker_input, limit=5)
        analyst_data = fetch_recent_upgrades_downgrades(ticker_input, months=2)
        
        # 1. 상단 핵심 펀더멘털 및 기술 지표 카드 (시총, PER, PBR, ROE, 현재가)
        st.subheader(f"🏢 [{ticker_input}] 핵심 재무 및 기술 지표")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("현재가", f"${tech_data.get('current_price', 'N/A')}")
        c2.metric("시가총액", fund_data.get('market_cap_fmt', 'N/A'))
        c3.metric("PER (Trailing)", f"{fund_data.get('trailing_pe', 'N/A')}배")
        c4.metric("PBR", f"{fund_data.get('pbr', 'N/A')}배")
        c5.metric("ROE", fund_data.get('roe', 'N/A'))
        c6.metric("RSI(14) / MFI", f"{tech_data.get('rsi_14', 'N/A')} / {tech_data.get('mfi_14', 'N/A')}")
        
        # 2. 산출된 적정 주가 모델 비교 카드
        st.write("")
        st.subheader("🎯 모델별 적정주가 (Fair Value) 산출")
        f1, f2, f3, f4 = st.columns(4)
        
        curr_p = tech_data.get('current_price', 0)
        fair = fund_data.get('fair_values', {})
        
        # 증권가 평균 목표가
        target_p = fund_data.get('target_mean_price', 'N/A')
        diff_target = round(((target_p - curr_p) / curr_p) * 100, 1) if isinstance(target_p, (int, float)) and curr_p else None
        f1.metric("IB 목표주가 컨센서스", f"${target_p}", f"{diff_target:+.1f}%" if diff_target is not None else None)
        
        # 그레이엄 적정가
        graham_p = fair.get('graham_number', 'N/A')
        diff_graham = round(((graham_p - curr_p) / curr_p) * 100, 1) if isinstance(graham_p, (int, float)) and curr_p else None
        f2.metric("그레이엄 공식 (자산/수익)", f"${graham_p}" if isinstance(graham_p, (int, float)) else str(graham_p), f"{diff_graham:+.1f}%" if diff_graham is not None else None)
        
        # 피터 린치 적정가
        lynch_p = fair.get('peter_lynch', 'N/A')
        diff_lynch = round(((lynch_p - curr_p) / curr_p) * 100, 1) if isinstance(lynch_p, (int, float)) and curr_p else None
        f3.metric("피터 린치 모델 (성장가치)", f"${lynch_p}" if isinstance(lynch_p, (int, float)) else str(lynch_p), f"{diff_lynch:+.1f}%" if diff_lynch is not None else None)
        
        # ROE-PBR 모델
        roe_pbr_p = fair.get('roe_pbr_model', 'N/A')
        diff_roe_pbr = round(((roe_pbr_p - curr_p) / curr_p) * 100, 1) if isinstance(roe_pbr_p, (int, float)) and curr_p else None
        f4.metric("ROE-PBR 자본가치 모델", f"${roe_pbr_p}" if isinstance(roe_pbr_p, (int, float)) else str(roe_pbr_p), f"{diff_roe_pbr:+.1f}%" if diff_roe_pbr is not None else None)
        
        st.divider()
        
        # 3. 데이터 기준일 정보
        with st.expander("🕒 데이터 수집 기준일자", expanded=False):
            d1, d2, d3 = st.columns(3)
            d1.write(f"- **주가/재무제표 기준일:** {stock_date}")
            d2.write(f"- **미 국채 10년물 기준일:** {macro_data.get('us_10y_yield', {}).get('date', 'N/A')}")
            d3.write(f"- **VIX / 유가 기준일:** {macro_data.get('vix', {}).get('date', 'N/A')}")
        
        # 4. Gemini AI 분석
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
1. 기술적/수급 데이터 ({ticker}) (기준일: {stock_date}):
{tech_json}

2. 매크로/시장 지표:
{macro_json}

3. 주요 섹터 5일 등락률:
{sector_json}

4. 펀더멘털, 밸류에이션 및 산출된 모델별 적정주가:
{fund_json}

5. 최신 주요 기사 헤드라인:
{news_json}

6. 최근 2개월 증권가 투자의견 변동:
{analyst_json}

---

[지시사항]
위 [RAG 주입 데이터]를 기반으로 아래 항목을 정밀하게 분석할 것:

1. 거시환경 및 시장 국면
- 경기 국면 (회복 / 활황 / 둔화 / 침체 판정)
- 단기 변동성 촉발 요인
- 권장 자산 배분 비중 (주식 : 채권 : 현금)

2. 섹터 전망 및 순환매
- 상대적 강세 섹터 및 약세 섹터 요약
- 자금 순환매(Rotation) 방향

3. 밸류에이션 및 적정주가 평가 ({ticker})
- 현재 주가 대비 산출된 모델별(그레이엄, 린치, ROE-PBR) 적정주가 및 목표주가 괴리율 평가
- PER/PBR/ROE 관점에서의 고평가/저평가 종합 판정

4. 종목 종합 평가 ({ticker})
- 기술적 분석: 이평선 배열, MACD 모멘텀, MFI 수급 상태, 지지/저항선
- 스코어카드 (각 10점 만점): 성장성, 수익성, 밸류에이션, 해자, 리스크
- 종합 평점 및 최종 투자 의견 (적극매수 / 분할매수 / 관망 / 비중축소)
- 매매 시나리오: 분할 매수 밴드, 목표가/익절 라인, 손절(Stop-loss) 기준선
"""
            prompt = PromptTemplate(
                input_variables=["ticker", "stock_date", "tech_json", "macro_json", "sector_json", "fund_json", "news_json", "analyst_json"],
                template=template
            )
            llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=api_key)
            chain = prompt | llm
            
            response = chain.invoke({
                "ticker": ticker_input,
                "stock_date": stock_date,
                "tech_json": json.dumps(tech_data, indent=2, ensure_ascii=False),
                "macro_json": json.dumps(macro_data, indent=2, ensure_ascii=False),
                "sector_json": json.dumps(sector_data, indent=2, ensure_ascii=False),
                "fund_json": json.dumps(fund_data, indent=2, ensure_ascii=False),
                "news_json": json.dumps([n['title'] for n in news_data], indent=2, ensure_ascii=False),
                "analyst_json": json.dumps(analyst_data, indent=2, ensure_ascii=False)
            })
            
            st.markdown(extract_clean_text(response.content))
            
        st.divider()

        # 하단 뉴스 및 증권가 의견
        col_left, col_right = st.columns(2)
        
        with col_left:
            st.subheader("📰 최신 주요 뉴스 및 링크")
            if news_data:
                for item in news_data:
                    st.markdown(f"- [{item['title']}]({item['link']})  \n  <small style='color:gray;'>출처: {item['publisher']} | {item['date']}</small>", unsafe_allow_html=True)
            else:
                st.info("수집된 최신 뉴스가 없습니다.")
                
        with col_right:
            st.subheader("🏛️ 최근 2개월 증권사 투자의견 변동")
            if analyst_data:
                df_analyst = pd.DataFrame(analyst_data)
                df_analyst.columns = ["일자", "증권사", "투자의견", "이전의견", "액션"]
                st.dataframe(df_analyst, use_container_width=True, hide_index=True)
            else:
                st.info("최근 2개월간 등록된 투자의견 변동 내역이 없습니다.")