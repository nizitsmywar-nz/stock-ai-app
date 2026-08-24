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
    page_title="AI Stock Valuation Dashboard Pro",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- 커스텀 CSS ---
st.markdown("""
<style>
    @import url('[https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css](https://cdn.jsdelivr.net/gh/orioncactus/pretendard/dist/web/static/pretendard.css)');
    * { font-family: 'Pretendard', -apple-system, sans-serif !important; }
    
    .block-container { padding-top: 1.8rem; padding-bottom: 3rem; }
    
    .metric-card {
        background: rgba(255, 255, 255, 0.04);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 12px 14px;
        margin-bottom: 8px;
    }
    .metric-title {
        font-size: 0.78rem;
        color: #94a3b8;
        font-weight: 500;
        margin-bottom: 3px;
    }
    .metric-value {
        font-size: 1.25rem;
        font-weight: 700;
        color: #f8fafc;
    }
    
    .fair-card-value {
        background: linear-gradient(145deg, rgba(30, 41, 59, 0.4), rgba(15, 23, 42, 0.6));
        border: 1px solid rgba(148, 163, 184, 0.2);
        border-radius: 12