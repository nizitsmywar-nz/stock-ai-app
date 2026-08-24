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

# --- 페이지 설정 ---
st.set_page_config(
    page_title="AI Stock Valuation Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 커스텀 CSS ---
st.markdown("""
<style>
    @import url('https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css');
    * { font-family: 'Pretendard', -apple-system, sans-serif !important; }
    
    .block-container { padding-top: 1.8rem; padding-bottom: 3rem; }
    
    /* 상단 지표 카드 */
    .metric-card {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 14px 16px;
        margin-bottom: 10px;
        transition: transform 0.15s ease-in-out;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        border-color: rgba(99, 102, 241, 0.4);
    }
    .metric-title {
        font-size: 0.8rem;
        color: #94a3b8;
        font-weight: 500;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.35rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    /* 적정주가 카드 */
    .fair-card {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.4), rgba(15, 23, 42, 0.6));
        border: 1px solid rgba(148, 163, 184, 0.15);
        border-radius: 14px;
        padding: 16px 18px;
        text-align: left;
    }
    .fair-formula {
        font-size: 0.75rem;
        color: #64748b;
        margin-top: 4px;
    }
    .diff-badge-up {
        background: rgba(34, 197, 94, 0.15);
        color: #4ade80;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        float: right;
    }
    .diff-badge-down {
        background: rgba(239, 68, 68, 0.15);
        color: #f87171;
        padding: 3px 8px;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 600;
        float: right;
    }
    
    /* 뉴스 요약 카드 */
    .news-box {
        background: rgba(255, 255, 255, 0.03);
        border-left: 3px solid #6366f1;
        padding: 12px 16px;
        margin-bottom: 12px;
        border-radius: 0 10px 10px 0;
    }
    .news-summary {
        font-size: 0.9rem;
        color: #f1f5f9;
        font-weight: 500;
        line-height: 1.45;
        margin-bottom: 6px;
    }
    .news-title-link {
        text-decoration: none;
        color: #818cf8;
        font-weight: 600;
        font-size: 0.84rem;
        display: inline-block;
        margin-bottom: 3px;
    }
    .news-title-link:hover {
        text-decoration: underline;
        color: #a5b4fc;
    }
    .news-meta {
        font-size: 0.76rem;
        color: #94a3b8;
    }
    
    /* AI 리포트 컨테이너 */
    .report-container {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 24px;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# -------------------------------------------------------------
# 1. RAG 데이터 수집 모듈
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
        return f"${market_cap / 1e12:.2f}T"
    elif market_cap >= 1e9:
        return f"${market_cap / 1e9:.2f}B"
    elif market_cap >= 1e6:
        return f"${market_cap / 1e6:.2f}M"
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
        
        fair_values = {}
        if eps and bps and eps > 0 and bps > 0:
            graham_num = math.sqrt(22.5 * eps * bps)
            fair_values["graham_number"] = round(graham_num, 2)
        else:
            fair_values["graham_number"] = "산출불가"
            
        if eps and eps > 0 and roe_raw and roe_raw > 0:
            growth_rate = min(roe_raw * 100, 30.0)
            lynch_val = eps * growth_rate
            fair_values["peter_lynch"] = round(lynch_val, 2)
        else:
            fair_values["peter_lynch"] = "산출불가"

        if bps and bps > 0 and roe_raw and roe_raw > 0:
            roe_pbr_val = bps * (roe_raw / 0.10)
            fair_values["roe_pbr_model"] = round(roe_pbr_val, 2)
        else:
            fair_values["roe_pbr_model"] = "산출불가"

        return {
            "market_cap_fmt": format_market_cap(market_cap),
            "trailing_pe": round(trailing_pe, 2) if isinstance(trailing_pe, (int, float)) else trailing_pe,
            "forward_pe": round(forward_pe, 2) if isinstance(forward_pe, (int, float)) else forward_pe,
            "pbr": round(pbr, 2) if isinstance(pbr, (int, float)) else pbr,
            "roe": f"{roe_pct}%" if roe_pct != "N/A" else "N/A",
            "eps": round(eps, 2) if isinstance(eps, (int, float)) else "N/A",
            "bps": round(bps, 2) if isinstance(bps, (int, float)) else "N/A",
            "target_mean_price": target_mean_price,
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
                summary = content.get("summary", "")
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
                summary = ""
                publisher = n.get("publisher", "Yahoo Finance")
                link = n.get("link", "")
                pub_time = n.get("providerPublishTime", None)
                pub_date = datetime.fromtimestamp(pub_time).strftime("%Y-%m-%d") if pub_time else "최근"
            
            if title:
                articles.append({
                    "title": title,
                    "raw_summary": summary,
                    "publisher": publisher,
                    "date": pub_date,
                    "link": link or f"https://finance.yahoo.com/quote/{ticker}"
                })
        return articles
    except Exception:
        return []

def summarize_news_with_gemini(news_list, api_key):
    """뉴스 목록을 받아 Gemini를 통해 한 줄 한국어 요약 생성"""
    if not news_list or not api_key:
        return news_list
        
    try:
        titles_and_summaries = [
            f"[{i+1}] 제목: {n['title']} / 영문요약: {n.get('raw_summary', '')}"
            for i, n in enumerate(news_list)
        ]
        prompt_text = (
            "아래 영문 주식 기사 목록을 읽고, 각각의 핵심 내용을 투자자가 바로 이해할 수 있도록 "
            "자연스러운 한국어로 1~2문장으로 요약해 줘. 반드시 JSON 리스트 형식(예: [\"요약1\", \"요약2\", ...])으로만 응답해.\n\n"
            + "\n".join(titles_and_summaries)
        )
        
        llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash", google_api_key=api_key)
        res = llm.invoke(prompt_text)
        
        cleaned = extract_clean_text(res.content).strip()
        if "```json" in cleaned:
            cleaned = cleaned.split("```json")[1].split("```")[0].strip()
        elif "```" in cleaned:
            cleaned = cleaned.split("```")[1].split("```")[0].strip()
            
        summaries = json.loads(cleaned)
        for i, s in enumerate(summaries):
            if i < len(news_list):
                news_list[i]["ko_summary"] = s
    except Exception:
        for n in news_list:
            n["ko_summary"] = n["title"]  # 에러 시 영문 제목 대체
            
    return news_list

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
# 2. UI 레이아웃 및 렌더링
# -------------------------------------------------------------
with st.sidebar:
    st.markdown("### ⚡ **AI Stock Analyst**")
    ticker_input = st.text_input("종목 티커 (Ticker)", value="GOOGL").upper()
    analyze_btn = st.button("🚀 분석 실행", type="primary", use_container_width=True)
    st.markdown("---")
    st.caption("• **RAG 파이프라인:** Yahoo Finance, FRED  \n• **추론 모델:** Gemini 3.6 Flash  \n• **실시간 지표:** MACD, MFI, RSI, 3대 밸류에이션")

st.markdown(f"## 📊 **{ticker_input} 종합 밸류에이션 & 전략 리포트**")

if analyze_btn:
    # API 키 확인
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        try:
            api_key = st.secrets["GEMINI_API_KEY"]
        except Exception:
            api_key = None
            
    if not api_key:
        st.error("GEMINI_API_KEY가 설정되지 않았습니다. Streamlit Secrets에 등록하세요.")
    else:
        with st.spinner(f"🔍 [{ticker_input}] 실시간 RAG 데이터 수집 및 뉴스 AI 한국어 요약 중..."):
            tech_data, stock_date = fetch_stock_technical_data(ticker_input)
            macro_data = fetch_macro_indicators()
            fund_data = fetch_fundamentals_and_valuation(ticker_input)
            sector_data = fetch_sector_performance()
            raw_news_data = fetch_news(ticker_input, limit=5)
            news_data = summarize_news_with_gemini(raw_news_data, api_key)
            analyst_data = fetch_recent_upgrades_downgrades(ticker_input, months=2)
            
            curr_p = tech_data.get('current_price', 0)
            
            # 1. 상단 핵심 지표 그리드
            m1, m2, m3, m4, m5, m6 = st.columns(6)
            with m1:
                st.markdown(f"""<div class="metric-card"><div class="metric-title">현재 주가</div><div class="metric-value">${curr_p}</div></div>""", unsafe_allow_html=True)
            with m2:
                st.markdown(f"""<div class="metric-card"><div class="metric-title">시가총액</div><div class="metric-value">{fund_data.get('market_cap_fmt', 'N/A')}</div></div>""", unsafe_allow_html=True)
            with m3:
                st.markdown(f"""<div class="metric-card"><div class="metric-title">PER / PBR</div><div class="metric-value">{fund_data.get('trailing_pe', 'N/A')} / {fund_data.get('pbr', 'N/A')}</div></div>""", unsafe_allow_html=True)
            with m4:
                st.markdown(f"""<div class="metric-card"><div class="metric-title">ROE</div><div class="metric-value">{fund_data.get('roe', 'N/A')}</div></div>""", unsafe_allow_html=True)
            with m5:
                st.markdown(f"""<div class="metric-card"><div class="metric-title">RSI / MFI</div><div class="metric-value">{tech_data.get('rsi_14', 'N/A')} / {tech_data.get('mfi_14', 'N/A')}</div></div>""", unsafe_allow_html=True)
            with m6:
                st.markdown(f"""<div class="metric-card"><div class="metric-title">미 10년물 금리</div><div class="metric-value">{macro_data.get('us_10y_yield', {}).get('value', 'N/A')}</div></div>""", unsafe_allow_html=True)
                
            st.write("")
            
            # 2. 적정주가 모델 비교 카드
            st.markdown("##### 🎯 **모델별 적정주가 (Fair Value) 및 목표주가 괴리율**")
            f1, f2, f3, f4 = st.columns(4)
            fair = fund_data.get('fair_values', {})
            
            def render_fair_card(title, value, formula):
                badge = ""
                if isinstance(value, (int, float)) and curr_p:
                    diff = round(((value - curr_p) / curr_p) * 100, 1)
                    badge_class = "diff-badge-up" if diff >= 0 else "diff-badge-down"
                    badge = f'<span class="{badge_class}">{diff:+.1f}%</span>'
                    val_str = f"${value}"
                else:
                    val_str = str(value)
                
                return f"""
                <div class="fair-card">
                    {badge}
                    <div class="metric-title">{title}</div>
                    <div class="metric-value" style="margin-top: 6px;">{val_str}</div>
                    <div class="fair-formula">{formula}</div>
                </div>
                """
                
            with f1:
                target_p = fund_data.get('target_mean_price', 'N/A')
                st.markdown(render_fair_card("IB 컨센서스 목표가", target_p, "증권사 평균 목표주가"), unsafe_allow_html=True)
            with f2:
                st.markdown(render_fair_card("그레이엄 청산가치", fair.get('graham_number', 'N/A'), "√(22.5 × EPS × BPS)"), unsafe_allow_html=True)
            with f3:
                st.markdown(render_fair_card("피터 린치 성장가치", fair.get('peter_lynch', 'N/A'), "EPS × min(ROE, 30)"), unsafe_allow_html=True)
            with f4:
                st.markdown(render_fair_card("ROE-PBR 자본가치", fair.get('roe_pbr_model', 'N/A'), "BPS × (ROE / 10%)"), unsafe_allow_html=True)
                
            st.markdown(f"<div style='text-align: right; color: #64748b; font-size: 0.75rem; margin-top: 8px;'>기준일: 주가/재무 ({stock_date}) | FRED 금리 ({macro_data.get('us_10y_yield', {}).get('date', 'N/A')})</div>", unsafe_allow_html=True)
            st.divider()

            # 3. Gemini AI 분석 리포트
            template = """
[RAG 주입 데이터]
1. 기술적/수급 데이터 ({ticker}) (기준일: {stock_date}):
{tech_json}

2. 매크로/시장 지표:
{macro_json}

3. 주요 섹터 5일 등락률:
{sector_json}

4. 펀더멘털, 밸류에이션 및 모델별 적정주가:
{fund_json}

5. 최신 주요 기사 (한국어 요약):
{news_json}

6. 최근 2개월 증권가 투자의견 변동:
{analyst_json}

---

[지시사항]
위 데이터를 바탕으로 객관적이고 예리한 분석을 수행할 것:

1. 거시환경 및 시장 국면
- 경기 국면 (회복 / 활황 / 둔화 / 침체 판정)
- 단기 변동성 촉발 요인
- 권장 자산 배분 비중 (주식 : 채권 : 현금)

2. 섹터 전망 및 순환매
- 상대적 강세 섹터 및 약세 섹터 요약
- 자금 순환매(Rotation) 방향

3. 밸류에이션 및 적정주가 종합 평가 ({ticker})
- 현재 주가 대비 산출된 모델별(그레이엄, 린치, ROE-PBR) 적정가 및 컨센서스 괴리율 분석
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
                "news_json": json.dumps([n.get('ko_summary', n['title']) for n in news_data], indent=2, ensure_ascii=False),
                "analyst_json": json.dumps(analyst_data, indent=2, ensure_ascii=False)
            })
            
            st.markdown(f'<div class="report-container">{extract_clean_text(response.content)}</div>', unsafe_allow_html=True)
            
            st.write("")
            st.divider()

            # 4. 하단 뉴스 (한국어 요약 + 원문 링크) 및 증권가 의견
            col_left, col_right = st.columns([1.1, 0.9])
            
            with col_left:
                st.markdown("##### 📰 **최신 주요 뉴스 (AI 한국어 요약 & 원문 링크)**")
                if news_data:
                    for item in news_data:
                        summary_text = item.get("ko_summary", item["title"])
                        st.markdown(f"""
                        <div class="news-box">
                            <div class="news-summary">💡 {summary_text}</div>
                            <div>
                                <a class="news-title-link" href="{item['link']}" target="_blank">🔗 원문 기사: {item['title']}</a>
                            </div>
                            <div class="news-meta">출처: {item['publisher']} &nbsp;|&nbsp; {item['date']}</div>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.info("수집된 최신 뉴스가 없습니다.")
                    
            with col_right:
                st.markdown("##### 🏛️ **최근 2개월 증권사 투자의견 변동**")
                if analyst_data:
                    df_analyst = pd.DataFrame(analyst_data)
                    df_analyst.columns = ["일자", "증권사", "투자의견", "이전의견", "액션"]
                    st.dataframe(df_analyst, use_container_width=True, hide_index=True)
                else:
                    st.info("최근 2개월간 등록된 투자의견 변동 내역이 없습니다.")