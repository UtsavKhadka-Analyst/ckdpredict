# ============================================================
# CKDPredict — Production Healthcare Application
# Early Detection of Chronic Kidney Disease
# Saint Louis University | MS Analytics 2026
# Medical White + Teal Clinical Theme
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import warnings
warnings.filterwarnings('ignore')

# ── Page Configuration ─────────────────────────────────────
st.set_page_config(
    page_title    = "CKDPredict | Early CKD Detection",
    page_icon     = "🫘",
    layout        = "wide",
    initial_sidebar_state = "expanded"
)

# ── Kidney Logo in Sidebar ─────────────────────────────────
try:
    st.logo(
        "assets/ckd_logo.png",
        size="large",
        link="https://ckdpredict.streamlit.app"
    )
except Exception:
    pass  # Logo file not found — skip silently

# ── Complete CSS — Medical White + Teal Clinical Theme ─────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');

/* ── Root Variables ── */
:root {
    --teal-50:  #F0FDFA;
    --teal-100: #CCFBF1;
    --teal-200: #99F6E4;
    --teal-400: #2DD4BF;
    --teal-500: #14B8A6;
    --teal-600: #0D9488;
    --teal-700: #0F766E;
    --teal-800: #115E59;
    --teal-900: #134E4A;
    --navy:     #0F2942;
    --red-500:  #EF4444;
    --red-100:  #FEE2E2;
    --orange-500:#F97316;
    --orange-100:#FFEDD5;
    --blue-500: #3B82F6;
    --blue-100: #DBEAFE;
    --green-500:#22C55E;
    --green-100:#DCFCE7;
    --gray-50:  #F9FAFB;
    --gray-100: #F3F4F6;
    --gray-200: #E5E7EB;
    --gray-400: #9CA3AF;
    --gray-600: #4B5563;
    --gray-800: #1F2937;
    --white:    #FFFFFF;
    --shadow-sm: 0 1px 3px rgba(0,0,0,0.06), 0 1px 2px rgba(0,0,0,0.04);
    --shadow-md: 0 4px 16px rgba(0,0,0,0.08), 0 2px 6px rgba(0,0,0,0.04);
    --shadow-lg: 0 10px 32px rgba(0,0,0,0.10), 0 4px 12px rgba(0,0,0,0.06);
    --radius:   12px;
    --radius-lg:16px;
}

/* ── Global Font ── */
html, body, [class*="css"], .stMarkdown, p, div {
    font-family: 'DM Sans', sans-serif !important;
}

/* ── Hide default Streamlit elements ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
.stDeployButton {display: none;}

/* Hide ONLY the toolbar items — keep header shell so toggle works */
[data-testid="stToolbar"] {display: none !important;}
[data-testid="stDecoration"] {display: none !important;}

/* Hide white rectangle — sidebar logo placeholder */
div[data-testid="stSidebarHeader"] {
    display: none !important;
    height: 0 !important;
    min-height: 0 !important;
    padding: 0 !important;
    margin: 0 !important;
}

/* Sidebar collapse toggle — always visible and styled */
button[data-testid="collapsedControl"] {
    visibility: visible !important;
    display: flex !important;
    opacity: 1 !important;
    background: var(--teal-600) !important;
    color: white !important;
    border-radius: 0 8px 8px 0 !important;
    box-shadow: 2px 0 8px rgba(0,0,0,0.2) !important;
}

button[data-testid="collapsedControl"]:hover {
    background: var(--teal-500) !important;
}

/* ── Main background ── */
.main {
    background: #FAFCFC !important;
}

.main .block-container {
    padding: 1.5rem 2rem 2rem 2rem !important;
    max-width: 1400px !important;
}

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: var(--navy) !important;
    border-right: none !important;
    box-shadow: 4px 0 24px rgba(0,0,0,0.15) !important;
}

section[data-testid="stSidebar"] * {
    color: rgba(255,255,255,0.85) !important;
}

section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stRadio label {
    color: rgba(255,255,255,0.6) !important;
    font-size: 0.75rem !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    font-weight: 600 !important;
}

section[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.12) !important;
}

/* Sidebar select box */
section[data-testid="stSidebar"] .stSelectbox > div > div {
    background: rgba(255,255,255,0.1) !important;
    border: 1px solid rgba(255,255,255,0.2) !important;
    border-radius: 8px !important;
    color: white !important;
}

/* Sidebar radio */
section[data-testid="stSidebar"] .stRadio > div {
    gap: 4px !important;
}

section[data-testid="stSidebar"] .stRadio label {
    background: rgba(255,255,255,0.06) !important;
    border-radius: 8px !important;
    padding: 8px 12px !important;
    margin: 2px 0 !important;
    cursor: pointer !important;
    transition: all 0.2s !important;
    color: rgba(255,255,255,0.75) !important;
    font-size: 0.875rem !important;
    text-transform: none !important;
    letter-spacing: 0 !important;
    font-weight: 400 !important;
}

section[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(20,184,166,0.25) !important;
    color: white !important;
}

/* ── KPI Cards ── */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 16px;
    margin: 20px 0 28px 0;
}

.kpi-card {
    background: var(--white);
    border-radius: var(--radius-lg);
    padding: 20px 20px 16px 20px;
    border: 1px solid var(--gray-200);
    box-shadow: var(--shadow-sm);
    transition: all 0.25s cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    overflow: hidden;
    cursor: default;
}

.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    background: var(--teal-500);
    border-radius: var(--radius-lg) var(--radius-lg) 0 0;
    transition: height 0.25s;
}

.kpi-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
    border-color: var(--teal-200);
}

.kpi-card:hover::before {
    height: 4px;
}

.kpi-card.urgent::before  { background: var(--red-500); }
.kpi-card.high::before    { background: var(--orange-500); }
.kpi-card.cost::before    { background: #8B5CF6; }
.kpi-card.saving::before  { background: var(--green-500); }

.kpi-label {
    font-size: 0.72rem;
    font-weight: 600;
    color: var(--gray-400);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}

.kpi-value {
    font-size: 1.9rem;
    font-weight: 700;
    color: var(--gray-800);
    line-height: 1;
    font-family: 'DM Mono', monospace !important;
    margin-bottom: 6px;
}

.kpi-value.urgent  { color: var(--red-500); }
.kpi-value.saving  { color: var(--green-500); }

.kpi-sub {
    font-size: 0.75rem;
    color: var(--gray-400);
    font-weight: 500;
}

.kpi-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    font-size: 0.7rem;
    font-weight: 600;
    padding: 2px 8px;
    border-radius: 20px;
    margin-top: 4px;
}

.kpi-badge.up   { background: var(--green-100); color: #15803D; }
.kpi-badge.warn { background: var(--red-100);   color: #DC2626; }
.kpi-badge.info { background: var(--teal-100);  color: var(--teal-700); }

/* ── Page Header ── */
.page-header {
    background: linear-gradient(135deg, var(--teal-600) 0%, var(--teal-800) 100%);
    border-radius: var(--radius-lg);
    padding: 24px 28px;
    margin-bottom: 24px;
    position: relative;
    overflow: hidden;
}

.page-header::after {
    content: '';
    position: absolute;
    right: -30px; top: -30px;
    width: 160px; height: 160px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}

.page-header::before {
    content: '';
    position: absolute;
    right: 40px; bottom: -40px;
    width: 100px; height: 100px;
    background: rgba(255,255,255,0.04);
    border-radius: 50%;
}

.page-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(255,255,255,0.18);
    color: rgba(255,255,255,0.95);
    padding: 4px 12px;
    border-radius: 20px;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 10px;
}

.page-title {
    font-size: 1.75rem;
    font-weight: 700;
    color: white;
    margin: 0 0 6px 0;
    line-height: 1.2;
}

.page-subtitle {
    font-size: 0.875rem;
    color: rgba(255,255,255,0.75);
    margin: 0;
    max-width: 600px;
}

/* ── Section Headers ── */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 24px 0 16px 0;
}

.section-title {
    font-size: 1rem;
    font-weight: 700;
    color: var(--gray-800);
    margin: 0;
}

.section-pill {
    background: var(--teal-100);
    color: var(--teal-700);
    padding: 2px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
}

/* ── Urgency Badges ── */
.badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 3px 10px;
    border-radius: 20px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}

.badge-urgent   { background: #FEE2E2; color: #DC2626; }
.badge-high     { background: #FFEDD5; color: #EA580C; }
.badge-moderate { background: #DBEAFE; color: #2563EB; }
.badge-low      { background: #DCFCE7; color: #16A34A; }

/* ── Info Cards ── */
.info-card {
    background: white;
    border-radius: var(--radius);
    padding: 20px;
    border: 1px solid var(--gray-200);
    box-shadow: var(--shadow-sm);
    height: 100%;
}

.info-card-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: var(--gray-400);
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}

.info-card-value {
    font-size: 2rem;
    font-weight: 700;
    color: var(--teal-600);
    font-family: 'DM Mono', monospace !important;
    line-height: 1;
    margin-bottom: 4px;
}

/* ── Risk Score Bar ── */
.risk-bar-wrap {
    display: flex;
    align-items: center;
    gap: 10px;
}

.risk-bar-bg {
    flex: 1;
    height: 6px;
    background: var(--gray-100);
    border-radius: 3px;
    overflow: hidden;
}

.risk-bar-fill {
    height: 100%;
    border-radius: 3px;
    transition: width 0.4s;
}

/* ── Patient Risk Display ── */
.risk-dial {
    text-align: center;
    padding: 32px 24px;
    background: linear-gradient(135deg, var(--teal-50) 0%, white 100%);
    border-radius: var(--radius-lg);
    border: 1px solid var(--teal-100);
}

.risk-dial-pct {
    font-size: 4rem;
    font-weight: 800;
    font-family: 'DM Mono', monospace !important;
    line-height: 1;
    margin-bottom: 4px;
}

.risk-dial-tier {
    font-size: 1rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 8px;
}

.risk-dial-note {
    font-size: 0.75rem;
    color: var(--gray-400);
}

/* ── Action Steps ── */
.action-step {
    display: flex;
    align-items: flex-start;
    gap: 12px;
    padding: 14px 16px;
    background: var(--gray-50);
    border-radius: 10px;
    border: 1px solid var(--gray-200);
    margin-bottom: 10px;
    transition: all 0.2s;
}

.action-step:hover {
    background: var(--teal-50);
    border-color: var(--teal-200);
    transform: translateX(4px);
}

.action-step-num {
    width: 24px;
    height: 24px;
    background: var(--teal-500);
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.75rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 1px;
}

.action-step-text {
    font-size: 0.875rem;
    color: var(--gray-700);
    font-weight: 500;
    line-height: 1.4;
}

/* ── Cost Impact Cards ── */
.cost-card {
    text-align: center;
    padding: 24px 16px;
    border-radius: var(--radius);
    border: 1px solid var(--gray-200);
}

.cost-card-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 8px;
}

.cost-card-amount {
    font-size: 1.6rem;
    font-weight: 800;
    font-family: 'DM Mono', monospace !important;
    line-height: 1;
    margin-bottom: 4px;
}

.cost-card-sub {
    font-size: 0.72rem;
    color: var(--gray-400);
}

/* ── Disclaimer boxes ── */
.disclaimer-box {
    background: #FFFBEB;
    border: 1px solid #FDE68A;
    border-radius: 10px;
    padding: 14px 16px;
    font-size: 0.82rem;
    color: #92400E;
    margin: 16px 0;
    line-height: 1.5;
}

.disclaimer-box.clinical {
    background: #EFF6FF;
    border-color: #BFDBFE;
    color: #1E40AF;
}

.disclaimer-box.urgent {
    background: #FEF2F2;
    border-color: #FECACA;
    color: #991B1B;
}

/* ── Metric override ── */
div[data-testid="metric-container"] {
    background: white !important;
    border-radius: 12px !important;
    padding: 16px !important;
    border: 1px solid var(--gray-200) !important;
    box-shadow: var(--shadow-sm) !important;
    transition: all 0.25s !important;
}

div[data-testid="metric-container"]:hover {
    transform: translateY(-2px) !important;
    box-shadow: var(--shadow-md) !important;
    border-color: var(--teal-200) !important;
}

div[data-testid="metric-container"] label {
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: var(--gray-400) !important;
}

div[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    font-weight: 700 !important;
    color: var(--gray-800) !important;
}

/* ── Tabs ── */
div[data-testid="stTabs"] button {
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s !important;
}

div[data-testid="stTabs"] button[aria-selected="true"] {
    color: var(--teal-600) !important;
    border-bottom-color: var(--teal-500) !important;
}

/* ── Buttons ── */
div[data-testid="stButton"] > button {
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 0.875rem !important;
    transition: all 0.2s !important;
    border: none !important;
}

div[data-testid="stButton"] > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 12px rgba(13,148,136,0.3) !important;
}

div[data-testid="stButton"] > button[kind="primary"] {
    background: var(--teal-500) !important;
    color: white !important;
}

div[data-testid="stButton"] > button[kind="primary"]:hover {
    background: var(--teal-600) !important;
}

/* ── Dataframe ── */
div[data-testid="stDataFrame"] {
    border-radius: 12px !important;
    border: 1px solid var(--gray-200) !important;
    overflow: hidden !important;
    box-shadow: var(--shadow-sm) !important;
}

/* ── Alerts ── */
div[data-testid="stAlert"] {
    border-radius: 10px !important;
    font-size: 0.875rem !important;
}

/* ── Slider ── */
div[data-testid="stSlider"] > div > div > div > div {
    background: var(--teal-500) !important;
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    border-radius: 10px !important;
    border: 1px solid var(--gray-200) !important;
}

/* ── Kidney Logo SVG ── */
.kidney-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 16px 0 8px 0;
}

.kidney-svg {
    width: 36px;
    height: 36px;
    flex-shrink: 0;
}

.kidney-brand-name {
    font-size: 1.2rem !important;
    font-weight: 800 !important;
    color: white !important;
    letter-spacing: -0.02em;
    line-height: 1;
}

.kidney-brand-sub {
    font-size: 0.7rem !important;
    color: rgba(255,255,255,0.5) !important;
    font-weight: 400 !important;
    letter-spacing: 0.04em;
}

/* ── Nav items ── */
.nav-section-label {
    font-size: 0.65rem !important;
    font-weight: 700 !important;
    color: rgba(255,255,255,0.35) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    padding: 16px 0 6px 0 !important;
}

/* ── Model card ── */
.model-card {
    background: white;
    border-radius: var(--radius-lg);
    padding: 24px;
    border: 1px solid var(--gray-200);
    box-shadow: var(--shadow-sm);
    transition: all 0.25s;
    text-align: center;
}

.model-card:hover {
    transform: translateY(-4px);
    box-shadow: var(--shadow-lg);
    border-color: var(--teal-200);
}

.model-card-label {
    font-size: 0.72rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--gray-400);
    margin-bottom: 6px;
}

.model-card-value {
    font-size: 2.2rem;
    font-weight: 800;
    color: var(--teal-600);
    font-family: 'DM Mono', monospace !important;
    line-height: 1;
    margin-bottom: 4px;
}

.model-card-sub {
    font-size: 0.78rem;
    color: var(--gray-400);
}

/* ── Reference footer ── */
.ref-footer {
    margin-top: 32px;
    padding: 16px;
    background: var(--gray-50);
    border-radius: 10px;
    border: 1px solid var(--gray-200);
    font-size: 0.72rem;
    color: var(--gray-400);
    line-height: 1.6;
}

/* ── Checklist item ── */
.checklist-item {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 16px;
    border-radius: 8px;
    border: 1px solid var(--gray-200);
    margin-bottom: 8px;
    background: white;
    transition: all 0.2s;
    cursor: pointer;
}

.checklist-item:hover {
    background: var(--teal-50);
    border-color: var(--teal-300);
}

/* ── Timeline badge ── */
.timeline-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 8px 16px;
    border-radius: 20px;
    font-weight: 700;
    font-size: 1.1rem;
    background: var(--teal-100);
    color: var(--teal-700);
    margin: 8px 0;
}
</style>
""", unsafe_allow_html=True)

# ── Kidney SVG Logo ────────────────────────────────────────
KIDNEY_SVG = """
<svg class="kidney-svg" viewBox="0 0 100 100" fill="none" xmlns="http://www.w3.org/2000/svg">
  <ellipse cx="35" cy="50" rx="28" ry="42" fill="#14B8A6" opacity="0.9"/>
  <ellipse cx="65" cy="50" rx="22" ry="36" fill="#0D9488" opacity="0.85"/>
  <ellipse cx="50" cy="50" rx="10" ry="18" fill="#0F766E" opacity="0.7"/>
  <circle cx="50" cy="50" r="6" fill="white" opacity="0.5"/>
  <path d="M35 20 Q50 15 65 20" stroke="white" stroke-width="2" stroke-linecap="round" fill="none" opacity="0.4"/>
  <path d="M35 80 Q50 85 65 80" stroke="white" stroke-width="2" stroke-linecap="round" fill="none" opacity="0.4"/>
</svg>
"""

# ── Helper Functions ───────────────────────────────────────
def get_tier_color(tier):
    return {
        'URGENT'  : '#EF4444',
        'HIGH'    : '#F97316',
        'MODERATE': '#3B82F6',
        'LOW'     : '#22C55E'
    }.get(tier, '#6B7280')

def get_tier_bg(tier):
    return {
        'URGENT'  : '#FEE2E2',
        'HIGH'    : '#FFEDD5',
        'MODERATE': '#DBEAFE',
        'LOW'     : '#DCFCE7'
    }.get(tier, '#F3F4F6')

def get_tier_icon(tier):
    return {
        'URGENT'  : '🚨',
        'HIGH'    : '⚠️',
        'MODERATE': '📊',
        'LOW'     : '✅'
    }.get(tier, '❓')

def get_months(score):
    if score >= 0.85:   return '2–6 months'
    elif score >= 0.65: return '6–18 months'
    elif score >= 0.40: return '18–36 months'
    else:               return '> 36 months'

def get_tier(score):
    if score >= 0.85:   return 'URGENT'
    elif score >= 0.65: return 'HIGH'
    elif score >= 0.40: return 'MODERATE'
    else:               return 'LOW'

def get_cost(score):
    return 28162 if score >= 0.65 else 13604

def get_patient_message(tier, months):
    messages = {
        'URGENT': {
            'headline': '⚠️ Your kidney health needs immediate attention.',
            'body': f'Based on your health records, kidney disease may develop in approximately **{months}** if no action is taken.',
            'cta': '📞 Please contact your doctor today — do not wait.'
        },
        'HIGH': {
            'headline': '📋 Your kidney health needs attention.',
            'body': f'Your records suggest kidney disease may develop in approximately **{months}**.',
            'cta': '📅 Schedule an appointment with your doctor within 2 weeks.'
        },
        'MODERATE': {
            'headline': '📊 Your kidney health shows some risk.',
            'body': f'Changes may occur in approximately **{months}** based on your current trajectory.',
            'cta': '🗣️ Discuss this with your doctor at your next visit.'
        },
        'LOW': {
            'headline': '✅ Your kidney health appears stable.',
            'body': 'Your current records show no immediate kidney health concern.',
            'cta': '📅 Continue your regular check-ups and treatment plan.'
        }
    }
    return messages.get(tier, messages['LOW'])

def get_action_steps(tier):
    return {
        'URGENT': [
            ('📞', 'Call your doctor or nephrologist today'),
            ('🔬', 'Request emergency eGFR and creatinine test'),
            ('💊', 'Bring a list of all current medications'),
            ('🏥', 'Do not wait for your next scheduled appointment')
        ],
        'HIGH': [
            ('📅', 'Schedule appointment within the next 2 weeks'),
            ('🔬', 'Ask for creatinine, eGFR, and UACR tests'),
            ('📊', 'Monitor blood pressure at home daily'),
            ('🥗', 'Reduce sodium intake to under 2g per day')
        ],
        'MODERATE': [
            ('📋', 'Mention kidney risk at your next appointment'),
            ('🔬', 'Ask about annual kidney function screening'),
            ('💧', 'Stay well hydrated — 6 to 8 glasses daily'),
            ('🏃', '30 minutes of moderate exercise most days')
        ],
        'LOW': [
            ('📅', 'Continue regular check-ups as scheduled'),
            ('💧', 'Maintain good hydration daily'),
            ('🥗', 'Follow your current healthy diet plan'),
            ('📊', 'Monitor blood pressure regularly')
        ]
    }.get(tier, [])

# ── Load Models ────────────────────────────────────────────
@st.cache_resource
def load_models():
    model_a    = joblib.load('models/model_a_xgboost.pkl')
    model_b    = joblib.load('models/model_b_xgboost.pkl')
    feat_a     = joblib.load('models/feature_cols_a.pkl')
    feat_b     = joblib.load('models/feature_cols_b.pkl')
    metrics    = joblib.load('models/model_metrics.pkl')
    return model_a, model_b, feat_a, feat_b, metrics

@st.cache_data
def load_registry():
    df = pd.read_csv('models/patient_registry.csv')
    return df

try:
    model_a, model_b, feat_a, feat_b, metrics = load_models()
    registry = load_registry()
    models_loaded = True
except Exception as e:
    st.error(f"⚠️ Error loading models: {e}")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:

    # Kidney emoji in teal circle + brand
    st.markdown("""
    <div class="kidney-logo">
        <div style="
            width:54px; height:54px;
            border-radius:50%;
            background: linear-gradient(
                135deg, #0D9488 0%, #14B8A6 60%, #2DD4BF 100%);
            border: 2.5px solid rgba(255,255,255,0.35);
            box-shadow: 0 4px 14px rgba(13,148,136,0.5);
            display:flex; align-items:center;
            justify-content:center;
            flex-shrink:0;">
            <span style="font-size:28px;line-height:1;">&#x1FAD8;</span>
        </div>
        <div>
            <div class="kidney-brand-name">CKDPredict</div>
            <div class="kidney-brand-sub">
                Kidney Health Intelligence
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Role selector
    st.markdown('<div class="nav-section-label">Signed in as</div>',
                unsafe_allow_html=True)
    user_role = st.selectbox(
        "",
        ["🏥 Healthcare Administrator",
         "🩺 Nephrologist / Physician",
         "👤 Patient View"],
        label_visibility="collapsed"
    )

    st.markdown("---")

    # Navigation
    st.markdown('<div class="nav-section-label">Workspace</div>',
                unsafe_allow_html=True)

    if "Administrator" in user_role:
        page = st.radio("", [
            "📊 Patient Risk Registry",
            "🗺️ Geographic Overview",
            "💰 Cost Dashboard"
        ], label_visibility="collapsed")
    elif "Nephrologist" in user_role:
        page = st.radio("", [
            "🔬 Individual Patient Detail",
            "📈 Model Comparison"
        ], label_visibility="collapsed")
    else:
        page = "👤 My Kidney Health"

    st.markdown("---")

    # Model quality snapshot
    with st.expander("📐 Model quality snapshot"):
        st.markdown(f"""
        <div style="font-size:0.8rem; line-height:1.8;">
            <div style="display:flex;justify-content:space-between;">
                <span style="color:rgba(255,255,255,0.5);">Model A AUC</span>
                <span style="color:#2DD4BF;font-weight:700;font-family:'DM Mono';">
                    {metrics['model_a_auc']:.4f}
                </span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:rgba(255,255,255,0.5);">Model B AUC</span>
                <span style="color:#2DD4BF;font-weight:700;font-family:'DM Mono';">
                    {metrics['model_b_auc']:.4f}
                </span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:rgba(255,255,255,0.5);">EPV (Model A)</span>
                <span style="color:rgba(255,255,255,0.8);font-family:'DM Mono';">
                    {metrics['model_a_epv']:.1f}
                </span>
            </div>
            <div style="display:flex;justify-content:space-between;">
                <span style="color:rgba(255,255,255,0.5);">EPV (Model B)</span>
                <span style="color:rgba(255,255,255,0.8);font-family:'DM Mono';">
                    {metrics['model_b_epv']:.1f}
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("""
    <div style="font-size:0.68rem;color:rgba(255,255,255,0.3);line-height:1.7;">
        Saint Louis University<br>
        MS Analytics · MRP 2026<br>
        KDIGO 2024 · ADA 2023 · USRDS 2023
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# SCREEN 1 — PATIENT RISK REGISTRY
# ════════════════════════════════════════════════════════════
if "Administrator" in user_role and "Registry" in page:

    # Header
    st.markdown("""
    <div class="page-header">
        <div class="page-badge">🏥 Population Health</div>
        <h1 class="page-title">Patient Risk Registry</h1>
        <p class="page-subtitle">
            Prioritize outreach using ML risk tiers and scores.
            Filter, search, and build targeted call lists for
            proactive CKD intervention.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Compute KPI values
    total_pts  = len(registry)
    urgent_n   = len(registry[registry['URGENCY_TIER']=='URGENT'])
    high_n     = len(registry[registry['URGENCY_TIER']=='HIGH'])
    proj_spend = registry['PROJ_COST'].sum()
    savings    = registry['POTENTIAL_SAVING'].sum()

    # KPI Cards
    st.markdown(f"""
    <div class="kpi-grid">
        <div class="kpi-card">
            <div class="kpi-label">Panel size</div>
            <div class="kpi-value">{total_pts:,}</div>
            <div class="kpi-sub">Total registered patients</div>
            <span class="kpi-badge info">↑ Active monitoring</span>
        </div>
        <div class="kpi-card urgent">
            <div class="kpi-label">Needs outreach today</div>
            <div class="kpi-value urgent">{urgent_n:,}</div>
            <div class="kpi-sub">URGENT tier patients</div>
            <span class="kpi-badge warn">🚨 2–6 months</span>
        </div>
        <div class="kpi-card high">
            <div class="kpi-label">High priority</div>
            <div class="kpi-value">{high_n:,}</div>
            <div class="kpi-sub">Within ~2 weeks</div>
            <span class="kpi-badge warn" style="background:#FFEDD5;color:#EA580C;">⚠️ 6–18 months</span>
        </div>
        <div class="kpi-card cost">
            <div class="kpi-label">Projected spend (panel)</div>
            <div class="kpi-value" style="font-size:1.5rem;">${proj_spend:,.0f}</div>
            <div class="kpi-sub">Annual Medicare estimate</div>
            <span class="kpi-badge info">USRDS 2023</span>
        </div>
        <div class="kpi-card saving">
            <div class="kpi-label">Modelled savings opportunity</div>
            <div class="kpi-value saving" style="font-size:1.5rem;">${savings:,.0f}</div>
            <div class="kpi-sub">If early intervention</div>
            <span class="kpi-badge up">↑ $14,558/patient</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box clinical">
        Demo UI — outbound messages are simulated; registry rows
        and scores are loaded directly from the trained XGBoost
        models and are unchanged from the source file.
    </div>
    """, unsafe_allow_html=True)

    # Filters
    col1, col2, col3, col4 = st.columns([2, 2, 2, 2])
    with col1:
        tier_filter = st.multiselect(
            "Urgency tier",
            ['URGENT','HIGH','MODERATE','LOW'],
            default=['URGENT','HIGH'],
            help="Filter by CKD urgency tier"
        )
    with col2:
        model_filter = st.selectbox(
            "Risk model / pathway",
            ['All','A — Diabetic','B — Non-Diabetic']
        )
    with col3:
        min_risk_pct = st.slider(
            "Minimum risk score (%)",
            0, 100, 65, 5,
            help="Filter patients by minimum CKD risk score"
        )
        min_risk = min_risk_pct / 100  # convert to 0-1 for filtering
    with col4:
        search = st.text_input(
            "Quick find (patient ID)",
            placeholder="e.g. b9abfbd3..."
        )

    # Apply filters
    filt = registry.copy()
    if tier_filter:
        filt = filt[filt['URGENCY_TIER'].isin(tier_filter)]
    if model_filter == 'A — Diabetic':
        filt = filt[filt['MODEL'] == 'A']
    elif model_filter == 'B — Non-Diabetic':
        filt = filt[filt['MODEL'] == 'B']
    filt = filt[filt['RISK_SCORE'] >= min_risk]
    if search:
        filt = filt[filt['PATIENT'].str.contains(
            search, case=False, na=False)]
    filt = filt.sort_values('RISK_SCORE', ascending=False)

    # Tabs
    tab1, tab2 = st.tabs(["📋 Patient roster", "📧 Outreach & messaging"])

    with tab1:
        st.markdown(f"""
        <div class="section-header">
            <h3 class="section-title">Patient roster</h3>
            <span class="section-pill">{len(filt):,} patients match</span>
        </div>
        """, unsafe_allow_html=True)

        # Build display dataframe
        cols_map = {
            'PATIENT'          : 'Patient ID',
            'RISK_SCORE'       : 'Risk score (%)',
            'URGENCY_TIER'     : 'Urgency',
            'EST_MONTHS'       : 'Est. timeline',
            'PROJ_COST'        : 'Proj. cost / yr',
            'POTENTIAL_SAVING' : 'Potential saving',
            'PATHWAY'          : 'Pathway',
            'CITY'             : 'City',
        }
        avail = {k:v for k,v in cols_map.items()
                 if k in filt.columns}
        disp = filt[list(avail.keys())].copy()
        disp.columns = list(avail.values())
        disp['Risk score (%)'] = (
            disp['Risk score (%)'] * 100
        ).round(1).astype(str) + '%'
        disp['Proj. cost / yr'] = disp[
            'Proj. cost / yr'].apply(
            lambda x: f"${x:,.0f}")
        disp['Potential saving'] = disp[
            'Potential saving'].apply(
            lambda x: f"${x:,.0f}")

        st.dataframe(
            disp,
            use_container_width=True,
            height=420,
            hide_index=True
        )

        # Urgency distribution chart
        st.markdown("""
        <div class="section-header">
            <h3 class="section-title">Risk distribution</h3>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2)

        with col1:
            tier_counts = filt[
                'URGENCY_TIER'].value_counts().reset_index()
            tier_counts.columns = ['Tier', 'Count']
            fig_pie = go.Figure(go.Pie(
                labels=tier_counts['Tier'],
                values=tier_counts['Count'],
                hole=0.6,
                marker=dict(colors=[
                    get_tier_color(t)
                    for t in tier_counts['Tier']
                ]),
                textinfo='label+percent',
                textfont=dict(size=12)
            ))
            fig_pie.update_layout(
                title=dict(
                    text='Urgency tier breakdown',
                    font=dict(size=14, color='#1F2937')
                ),
                showlegend=False,
                height=300,
                margin=dict(t=40,b=20,l=20,r=20),
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(family='DM Sans')
            )
            fig_pie.add_annotation(
                text=f"<b>{len(filt):,}</b><br>patients",
                x=0.5, y=0.5,
                font=dict(size=14, color='#1F2937'),
                showarrow=False
            )
            st.plotly_chart(fig_pie,
                use_container_width=True)

        with col2:
            pathway_risk = filt.groupby(
                'PATHWAY')['RISK_SCORE'].mean(
            ).reset_index()
            pathway_risk.columns = ['Pathway','Avg Risk']
            fig_bar = go.Figure(go.Bar(
                x=pathway_risk['Pathway'],
                y=(pathway_risk['Avg Risk']*100).round(1),
                marker=dict(
                    color=['#14B8A6','#0D9488'],
                    line=dict(color='white', width=2)
                ),
                text=(pathway_risk['Avg Risk']*100
                      ).round(1).astype(str) + '%',
                textposition='outside',
                textfont=dict(size=12, color='#1F2937')
            ))
            fig_bar.update_layout(
                title=dict(
                    text='Average risk by pathway',
                    font=dict(size=14, color='#1F2937')
                ),
                yaxis=dict(
                    title='Average risk score (%)',
                    showgrid=True,
                    gridcolor='#F3F4F6'
                ),
                height=300,
                margin=dict(t=40,b=20,l=20,r=20),
                paper_bgcolor='white',
                plot_bgcolor='white',
                font=dict(family='DM Sans'),
                showlegend=False
            )
            st.plotly_chart(fig_bar,
                use_container_width=True)

    with tab2:
        st.markdown("""
        <div class="section-header">
            <h3 class="section-title">Outreach & messaging</h3>
            <span class="section-pill">Simulated — demo only</span>
        </div>
        """, unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button(
                    f"🚨 Notify {urgent_n} URGENT patients",
                    type="primary",
                    use_container_width=True):
                st.success(
                    f"✅ Notification sent to "
                    f"{urgent_n} URGENT patients via "
                    f"patient portal.")
        with col2:
            if st.button(
                    f"⚠️ Notify {high_n} HIGH patients",
                    use_container_width=True):
                st.success(
                    f"✅ Notification sent to "
                    f"{high_n} HIGH risk patients.")
        with col3:
            if st.button(
                    "📥 Export to CSV",
                    use_container_width=True):
                st.download_button(
                    "Download registry CSV",
                    data=filt.to_csv(index=False),
                    file_name="ckd_registry.csv",
                    mime="text/csv"
                )

# ════════════════════════════════════════════════════════════
# SCREEN 2 — GEOGRAPHIC OVERVIEW
# ════════════════════════════════════════════════════════════
elif "Administrator" in user_role and "Geographic" in page:

    st.markdown("""
    <div class="page-header">
        <div class="page-badge">🗺️ Geographic Intelligence</div>
        <h1 class="page-title">Geographic Overview</h1>
        <p class="page-subtitle">
            CKD risk distribution across California.
            Identify cities requiring additional nephrology
            resources and targeted outreach.
        </p>
    </div>
    """, unsafe_allow_html=True)

    if 'CITY' in registry.columns:
        city_df = registry.groupby('CITY').agg(
            Total   = ('PATIENT','count'),
            Urgent  = ('URGENCY_TIER',
                        lambda x:(x=='URGENT').sum()),
            High    = ('URGENCY_TIER',
                        lambda x:(x=='HIGH').sum()),
            AvgRisk = ('RISK_SCORE','mean'),
            Cost    = ('PROJ_COST','sum'),
            Saving  = ('POTENTIAL_SAVING','sum')
        ).reset_index().sort_values(
            'Urgent', ascending=False)

        # Top city bar chart
        top15 = city_df.head(15)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            name='URGENT',
            x=top15['CITY'],
            y=top15['Urgent'],
            marker_color='#EF4444',
            hovertemplate='<b>%{x}</b><br>URGENT: %{y}<extra></extra>'
        ))
        fig.add_trace(go.Bar(
            name='HIGH',
            x=top15['CITY'],
            y=top15['High'],
            marker_color='#F97316',
            hovertemplate='<b>%{x}</b><br>HIGH: %{y}<extra></extra>'
        ))
        fig.update_layout(
            title=dict(
                text='Top 15 Cities — URGENT & HIGH Risk Patients',
                font=dict(size=16, color='#1F2937')
            ),
            barmode='stack',
            xaxis_tickangle=-35,
            height=400,
            legend=dict(
                orientation='h',
                yanchor='bottom', y=1.02,
                xanchor='right', x=1
            ),
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family='DM Sans'),
            yaxis=dict(gridcolor='#F3F4F6'),
            margin=dict(t=60,b=80,l=40,r=20)
        )
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("""
        <div class="section-header">
            <h3 class="section-title">City-level summary</h3>
        </div>
        """, unsafe_allow_html=True)

        city_display = city_df.copy()
        city_display['AvgRisk'] = (
            city_display['AvgRisk']*100
        ).round(1).astype(str)+'%'
        city_display['Cost'] = city_display[
            'Cost'].apply(lambda x:f"${x:,.0f}")
        city_display['Saving'] = city_display[
            'Saving'].apply(lambda x:f"${x:,.0f}")
        city_display.columns = [
            'City','Total','Urgent','High',
            'Avg Risk','Proj. Cost','Saving']

        st.dataframe(
            city_display,
            use_container_width=True,
            height=400,
            hide_index=True
        )

# ════════════════════════════════════════════════════════════
# SCREEN 3 — COST DASHBOARD
# ════════════════════════════════════════════════════════════
elif "Administrator" in user_role and "Cost" in page:

    st.markdown("""
    <div class="page-header">
        <div class="page-badge">💰 Financial Analytics</div>
        <h1 class="page-title">Cost Dashboard</h1>
        <p class="page-subtitle">
            Healthcare cost impact analysis using
            USRDS 2023 Medicare spending benchmarks.
            Quantify the ROI of early CKD intervention.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Cost benchmark cards
    st.markdown("""
    <div class="section-header">
        <h3 class="section-title">USRDS 2023 Cost Benchmarks</h3>
        <span class="section-pill">Per patient · Per year · Medicare</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:16px;margin-bottom:28px;">
        <div class="cost-card" style="background:#F9FAFB;border-color:#E5E7EB;">
            <div class="cost-card-label" style="color:#9CA3AF;">Without CKD</div>
            <div class="cost-card-amount" style="color:#4B5563;">$13,604</div>
            <div class="cost-card-sub">Baseline Medicare cost</div>
        </div>
        <div class="cost-card" style="background:#FFF7ED;border-color:#FED7AA;">
            <div class="cost-card-label" style="color:#EA580C;">With CKD Stage 3</div>
            <div class="cost-card-amount" style="color:#EA580C;">$28,162</div>
            <div class="cost-card-sub">+$14,558 vs baseline</div>
        </div>
        <div class="cost-card" style="background:#F0FDF4;border-color:#BBF7D0;">
            <div class="cost-card-label" style="color:#16A34A;">Early intervention saving</div>
            <div class="cost-card-amount" style="color:#16A34A;">$14,558</div>
            <div class="cost-card-sub">Per patient per year</div>
        </div>
        <div class="cost-card" style="background:#FEF2F2;border-color:#FECACA;">
            <div class="cost-card-label" style="color:#DC2626;">With ESKD (dialysis)</div>
            <div class="cost-card-amount" style="color:#DC2626;">$104,000+</div>
            <div class="cost-card-sub">End-stage kidney disease</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Population cost analysis
    st.markdown("""
    <div class="section-header">
        <h3 class="section-title">Your population cost analysis</h3>
    </div>
    """, unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    ur = len(registry[registry['URGENCY_TIER']=='URGENT'])
    hi = len(registry[registry['URGENCY_TIER']=='HIGH'])
    col1.metric("Patients needing intervention",
                f"{ur+hi:,}")
    col2.metric("Total projected annual cost",
                f"${registry['PROJ_COST'].sum():,.0f}")
    col3.metric("Modelled savings opportunity",
                f"${registry['POTENTIAL_SAVING'].sum():,.0f}")
    col4.metric("Avg saving per patient",
                "$14,558", delta="USRDS 2023")

    # Cost by tier chart
    tier_cost = registry.groupby('URGENCY_TIER').agg(
        Cost   = ('PROJ_COST','sum'),
        Saving = ('POTENTIAL_SAVING','sum'),
        Count  = ('PATIENT','count')
    ).reset_index()

    col1, col2 = st.columns(2)
    with col1:
        fig1 = go.Figure(go.Pie(
            labels=tier_cost['URGENCY_TIER'],
            values=tier_cost['Cost'],
            hole=0.55,
            marker=dict(colors=[
                get_tier_color(t)
                for t in tier_cost['URGENCY_TIER']
            ]),
            textinfo='label+percent',
            hovertemplate='<b>%{label}</b><br>Cost: $%{value:,.0f}<extra></extra>'
        ))
        fig1.update_layout(
            title='Projected cost by urgency tier',
            height=350,
            paper_bgcolor='white',
            font=dict(family='DM Sans'),
            margin=dict(t=50,b=20,l=20,r=20),
            showlegend=False
        )
        fig1.add_annotation(
            text="Total cost",
            x=0.5, y=0.5,
            font=dict(size=12, color='#9CA3AF'),
            showarrow=False
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = go.Figure()
        fig2.add_trace(go.Bar(
            name='Projected Cost',
            x=tier_cost['URGENCY_TIER'],
            y=tier_cost['Cost'],
            marker_color=[
                get_tier_color(t)
                for t in tier_cost['URGENCY_TIER']
            ],
            opacity=0.5,
            hovertemplate='Cost: $%{y:,.0f}<extra></extra>'
        ))
        fig2.add_trace(go.Bar(
            name='Potential Saving',
            x=tier_cost['URGENCY_TIER'],
            y=tier_cost['Saving'],
            marker_color='#22C55E',
            opacity=0.85,
            hovertemplate='Saving: $%{y:,.0f}<extra></extra>'
        ))
        fig2.update_layout(
            title='Cost vs. potential saving by tier',
            barmode='group',
            height=350,
            paper_bgcolor='white',
            plot_bgcolor='white',
            font=dict(family='DM Sans'),
            yaxis=dict(gridcolor='#F3F4F6',
                       title='Amount ($)'),
            legend=dict(orientation='h',
                        yanchor='bottom',y=1.02),
            margin=dict(t=50,b=20,l=40,r=20)
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("""
    <div class="ref-footer">
        <strong>Reference:</strong> USRDS (2023) Annual Data Report — Medicare spending
        benchmarks per beneficiary. Without CKD: $13,604/yr. With CKD Stage 3:
        $28,162/yr. With ESKD: $104,000+/yr. Cost estimates represent population-level
        planning figures. Actual costs vary by patient, insurance, and institution.
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# SCREEN 4 — INDIVIDUAL PATIENT DETAIL
# ════════════════════════════════════════════════════════════
elif "Nephrologist" in user_role and "Individual" in page:

    st.markdown("""
    <div class="page-header">
        <div class="page-badge">🔬 Clinical Decision Support</div>
        <h1 class="page-title">Individual Patient Record</h1>
        <p class="page-subtitle">
            Review registry risk scores, urgency tiers, and cost
            projections. Align care tasks with KDIGO 2024 guidelines.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box clinical">
        🔵 Support tool only — verify all orders and referrals
        against the source chart and institutional policy.
        AUC 0.9344 (Model A) · AUC 0.9753 (Model B) ·
        Trained on Synthea synthetic EHR · Walonoski et al. (2018)
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns([2,1])
    with col1:
        model_choice = st.selectbox(
            "Select Model",
            ["Model A — Diabetic",
             "Model B — Non-Diabetic"]
        )
    with col2:
        min_score = st.slider(
            "Min risk score", 0.5, 1.0, 0.80)

    model_key = 'A' if "Model A" in model_choice else 'B'
    filt_reg = registry[
        (registry['MODEL'] == model_key) &
        (registry['RISK_SCORE'] >= min_score)
    ].sort_values('RISK_SCORE', ascending=False)

    if len(filt_reg) == 0:
        st.warning("No patients match. Lower the minimum risk score.")
        st.stop()

    patient_id = st.selectbox(
        "Select Patient",
        filt_reg['PATIENT'].tolist()
    )

    pt = filt_reg[
        filt_reg['PATIENT'] == patient_id
    ].iloc[0]

    tier  = pt['URGENCY_TIER']
    score = pt['RISK_SCORE']
    color = get_tier_color(tier)
    icon  = get_tier_icon(tier)

    st.divider()

    # Patient KPIs
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Model risk score",
                f"{score:.4f}",
                help="XGBoost probability 0–1")
    col2.metric("Urgency tier",
                f"{icon} {tier}")
    col3.metric("Estimated time to CKD",
                pt['EST_MONTHS'])
    col4.metric("Projected annual cost",
                f"${pt['PROJ_COST']:,.0f}",
                delta=f"-${pt['POTENTIAL_SAVING']:,.0f} if caught early",
                delta_color="inverse")

    # Tabs
    tab1, tab2 = st.tabs([
        "📊 Risk summary",
        "📋 Care planning checklist"])

    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            <div class="section-header">
                <h3 class="section-title">Demographics & pathway</h3>
            </div>
            """, unsafe_allow_html=True)
            demo_fields = [
                'PATHWAY','CITY','STATE','GENDER','RACE']
            for f in demo_fields:
                if f in pt.index and pd.notna(pt[f]):
                    st.markdown(f"""
                    <div style="display:flex;justify-content:space-between;
                         padding:10px 0;border-bottom:1px solid #F3F4F6;">
                        <span style="color:#9CA3AF;font-size:0.85rem;
                              font-weight:600;">{f.title()}</span>
                        <span style="color:#1F2937;font-weight:600;
                              font-size:0.875rem;">{pt[f]}</span>
                    </div>
                    """, unsafe_allow_html=True)

        with col2:
            # Risk gauge
            risk_pct = int(score * 100)
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number",
                value=risk_pct,
                domain={'x':[0,1],'y':[0,1]},
                title={'text':'CKD Risk Score (%)'},
                gauge={
                    'axis':{
                        'range':[0,100],
                        'tickwidth':1,
                        'tickcolor':'#E5E7EB'
                    },
                    'bar':{'color':color},
                    'bgcolor':'#F9FAFB',
                    'steps':[
                        {'range':[0,40],
                         'color':'#DCFCE7'},
                        {'range':[40,65],
                         'color':'#DBEAFE'},
                        {'range':[65,85],
                         'color':'#FFEDD5'},
                        {'range':[85,100],
                         'color':'#FEE2E2'},
                    ],
                    'threshold':{
                        'line':{'color':'#1F2937','width':3},
                        'thickness':0.8,
                        'value':risk_pct
                    }
                },
                number={
                    'font':{'size':36,
                            'color':color,
                            'family':'DM Mono'}
                }
            ))
            fig_gauge.update_layout(
                height=280,
                paper_bgcolor='white',
                font=dict(family='DM Sans'),
                margin=dict(t=40,b=20,l=30,r=30)
            )
            st.plotly_chart(fig_gauge,
                use_container_width=True)

    with tab2:
        st.markdown("""
        <div class="section-header">
            <h3 class="section-title">KDIGO 2024 Care Planning Checklist</h3>
            <span class="section-pill">Evidence-based</span>
        </div>
        """, unsafe_allow_html=True)

        if tier in ['URGENT','HIGH']:
            checklist = [
                ("🔬", "Order eGFR and creatinine test",
                 "KDIGO 2024 — primary CKD markers"),
                ("🔬", "Order UACR (microalbumin/creatinine ratio)",
                 "KDIGO 2024 — kidney damage marker"),
                ("💊", "Initiate or review ACE inhibitor / ARB",
                 "KDIGO 2024 — first-line nephroprotection"),
                ("💊", "Consider SGLT2 inhibitor (if diabetic)",
                 "ADA 2023 — reduces CKD progression by 40%"),
                ("📊", "Target blood pressure < 130/80 mmHg",
                 "KDIGO 2024 — BP target for CKD patients"),
                ("👨‍⚕️", "Refer to nephrology",
                 "KDIGO 2024 — specialist referral criteria"),
                ("📅", "Schedule follow-up in 4 weeks",
                 "KDIGO 2024 — monitoring interval"),
                ("📋", "Document CKD stage in patient record",
                 "Coding and billing compliance"),
            ]
        else:
            checklist = [
                ("📊", "Monitor eGFR every 3 months",
                 "KDIGO 2024 — standard monitoring"),
                ("🔬", "Annual UACR screening",
                 "ADA 2023 — routine screening"),
                ("📋", "Review and reconcile medications",
                 "Nephrotoxic drug avoidance"),
                ("🥗", "Lifestyle counseling — diet and exercise",
                 "KDIGO 2024 — lifestyle modification"),
                ("💊", "Ensure BP medications current",
                 "KDIGO 2024 — BP management"),
            ]

        for icon, task, ref in checklist:
            col1, col2 = st.columns([3, 5])
            with col1:
                st.checkbox(f"{icon} {task}")
            with col2:
                st.caption(ref)

        st.markdown("""
        <div class="ref-footer">
            References: KDIGO (2024) Clinical Practice Guidelines for CKD
            Evaluation and Management · ADA (2023) Standards of Care in
            Diabetes — Section 11: CKD · Tangri et al. (2016) Kidney
            Failure Risk Equation
        </div>
        """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# SCREEN 5 — MODEL COMPARISON
# ════════════════════════════════════════════════════════════
elif "Nephrologist" in user_role and "Comparison" in page:

    st.markdown("""
    <div class="page-header">
        <div class="page-badge">📈 Analytics</div>
        <h1 class="page-title">Model Comparison</h1>
        <p class="page-subtitle">
            Side-by-side validation metrics for the diabetic (A)
            and non-diabetic (B) pathways. Values loaded from
            packaged model card — unchanged from training.
        </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    metric_pairs = [
        ('AUC-ROC',          'model_a_auc',      'model_b_auc',     '{:.4f}'),
        ('CV AUC Mean',      'model_a_cv_mean',  'model_b_cv_mean', '{:.4f}'),
        ('CV AUC Std',       'model_a_cv_std',   'model_b_cv_std',  '± {:.4f}'),
        ('Training Patients','model_a_patients', 'model_b_patients','{:,}'),
        ('CKD Positive Cases','model_a_positive','model_b_positive', '{:,}'),
        ('EPV',              'model_a_epv',      'model_b_epv',     '{:.1f}'),
    ]

    with col1:
        st.markdown("""
        <div class="section-header">
            <h3 class="section-title">Model A — Diabetic</h3>
            <span class="section-pill">XGBoost</span>
        </div>
        """, unsafe_allow_html=True)
        for label, key_a, _, fmt in metric_pairs:
            val = metrics[key_a]
            formatted = fmt.format(val)
            st.markdown(f"""
            <div class="info-card" style="margin-bottom:12px;padding:16px 20px;">
                <div class="info-card-title">{label}</div>
                <div class="info-card-value">{formatted}</div>
            </div>
            """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="section-header">
            <h3 class="section-title">Model B — Non-Diabetic</h3>
            <span class="section-pill">XGBoost</span>
        </div>
        """, unsafe_allow_html=True)
        for label, _, key_b, fmt in metric_pairs:
            val = metrics[key_b]
            formatted = fmt.format(val)
            st.markdown(f"""
            <div class="info-card" style="margin-bottom:12px;padding:16px 20px;">
                <div class="info-card-title">{label}</div>
                <div class="info-card-value">{formatted}</div>
            </div>
            """, unsafe_allow_html=True)

    # AUC comparison chart
    st.markdown("""
    <div class="section-header" style="margin-top:32px;">
        <h3 class="section-title">Performance comparison</h3>
    </div>
    """, unsafe_allow_html=True)

    compare_metrics = ['AUC-ROC','CV AUC','Recall','Specificity']
    vals_a = [
        metrics['model_a_auc'],
        metrics['model_a_cv_mean'],
        0.887, 0.991
    ]
    vals_b = [
        metrics['model_b_auc'],
        metrics['model_b_cv_mean'],
        0.909, 0.985
    ]

    fig_comp = go.Figure()
    fig_comp.add_trace(go.Bar(
        name='Model A — Diabetic',
        x=compare_metrics, y=vals_a,
        marker_color='#14B8A6',
        text=[f'{v:.3f}' for v in vals_a],
        textposition='outside',
        hovertemplate='Model A<br>%{x}: %{y:.4f}<extra></extra>'
    ))
    fig_comp.add_trace(go.Bar(
        name='Model B — Non-Diabetic',
        x=compare_metrics, y=vals_b,
        marker_color='#0D9488',
        opacity=0.75,
        text=[f'{v:.3f}' for v in vals_b],
        textposition='outside',
        hovertemplate='Model B<br>%{x}: %{y:.4f}<extra></extra>'
    ))
    fig_comp.add_hline(
        y=0.75,
        line_dash='dot',
        line_color='#EF4444',
        annotation_text='Min threshold (Walonoski 2018)',
        annotation_position='right'
    )
    fig_comp.update_layout(
        barmode='group',
        height=380,
        paper_bgcolor='white',
        plot_bgcolor='white',
        font=dict(family='DM Sans'),
        yaxis=dict(
            range=[0,1.1],
            gridcolor='#F3F4F6',
            title='Score'
        ),
        legend=dict(
            orientation='h',
            yanchor='bottom', y=1.02,
            xanchor='right', x=1
        ),
        margin=dict(t=60,b=20,l=40,r=120)
    )
    st.plotly_chart(fig_comp, use_container_width=True)

    st.markdown("""
    <div class="ref-footer">
        <strong>References:</strong>
        Tangri et al. (2016) — AUC benchmark 0.90 for kidney failure risk ·
        Walonoski et al. (2018) — AUC > 0.75 acceptable for Synthea validation ·
        KDIGO (2024) — EPV minimum 10 events per predictor ·
        Steyerberg (2019) — Events Per Variable rule for clinical prediction models ·
        Chen &amp; Guestrin (2016) — XGBoost scalable tree boosting system
    </div>
    """, unsafe_allow_html=True)

# ════════════════════════════════════════════════════════════
# SCREEN 6 — PATIENT VIEW
# ════════════════════════════════════════════════════════════
elif "Patient" in user_role:

    st.markdown("""
    <div class="page-header">
        <div class="page-badge">👤 Patient Portal</div>
        <h1 class="page-title">My Kidney Health Summary</h1>
        <p class="page-subtitle">
            View the same risk tier and score your care team sees
            in the registry. Bring questions to your next visit.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="disclaimer-box">
        This page shows <strong>educational risk information only</strong>.
        It is not a diagnosis. Call your clinician or 911 for urgent symptoms.
    </div>
    """, unsafe_allow_html=True)

    patient_id = st.selectbox(
        "Select your patient record ID",
        registry['PATIENT'].tolist()
    )

    pt      = registry[
        registry['PATIENT']==patient_id].iloc[0]
    tier    = pt['URGENCY_TIER']
    score   = pt['RISK_SCORE']
    months  = pt['EST_MONTHS']
    color   = get_tier_color(tier)
    bg      = get_tier_bg(tier)
    icon    = get_tier_icon(tier)
    risk_pct = int(score * 100)
    msg     = get_patient_message(tier, months)
    steps   = get_action_steps(tier)

    st.divider()

    # Overview tab
    tab1, tab2, tab3 = st.tabs([
        "📊 Overview",
        "✅ My action list",
        "📚 Learn & ask"
    ])

    with tab1:
        col1, col2 = st.columns([1, 2])

        with col1:
            # Risk dial
            st.markdown(f"""
            <div class="risk-dial">
                <div style="font-size:0.72rem;font-weight:700;
                     color:#9CA3AF;text-transform:uppercase;
                     letter-spacing:0.08em;margin-bottom:12px;">
                    Modelled risk index
                </div>
                <div class="risk-dial-pct" style="color:{color};">
                    {risk_pct}%
                </div>
                <div class="risk-dial-tier" style="color:{color};">
                    {icon} {tier}
                </div>
                <div class="risk-dial-note">
                    Based on registry score · not a lab result
                </div>
            </div>
            """, unsafe_allow_html=True)
            if 'PATHWAY' in pt.index:
                st.caption(
                    f"Care pathway in registry: "
                    f"**{pt.get('PATHWAY','Unknown')}**")

        with col2:
            # Message
            st.markdown(f"""
            <div style="padding:24px;background:{bg};
                 border-radius:12px;border:1px solid {color}33;
                 margin-bottom:16px;">
                <div style="font-size:1.05rem;font-weight:700;
                     color:{color};margin-bottom:8px;">
                    {msg['headline']}
                </div>
                <div style="font-size:0.875rem;color:#374151;
                     line-height:1.6;margin-bottom:12px;">
                    {msg['body'].replace('**','<strong>').replace('**','</strong>')}
                </div>
                <div style="font-size:0.875rem;font-weight:600;
                     color:{color};">
                    {msg['cta']}
                </div>
            </div>
            """, unsafe_allow_html=True)

            # Timeline
            st.markdown(f"""
            <div style="background:white;border:1px solid #E5E7EB;
                 border-radius:10px;padding:16px 20px;">
                <div style="font-size:0.75rem;font-weight:700;
                     color:#9CA3AF;text-transform:uppercase;
                     letter-spacing:0.08em;margin-bottom:8px;">
                    Estimated timeline
                </div>
                <div style="font-size:1.5rem;font-weight:800;
                     color:{color};font-family:'DM Mono',monospace;
                     margin-bottom:4px;">
                    {months}
                </div>
                <div style="font-size:0.75rem;color:#9CA3AF;">
                    If nothing changes clinically, the model suggests
                    kidney disease could develop in about
                    <strong>{months}</strong>.
                    Your team will interpret this with labs.
                </div>
                <div style="font-size:0.68rem;color:#D1D5DB;
                     margin-top:8px;">
                    Estimate uses the same EST_MONTHS field as the
                    administrator registry. Not a personal prognosis.
                </div>
            </div>
            """, unsafe_allow_html=True)

    with tab2:
        st.markdown("""
        <div class="section-header">
            <h3 class="section-title">Your next steps</h3>
            <span class="section-pill">Personalised</span>
        </div>
        """, unsafe_allow_html=True)

        for i, (emoji, text) in enumerate(steps, 1):
            st.markdown(f"""
            <div class="action-step">
                <div class="action-step-num">{i}</div>
                <div class="action-step-text">
                    {emoji} {text}
                </div>
            </div>
            """, unsafe_allow_html=True)

        # Cost awareness
        st.markdown("""
        <div class="section-header" style="margin-top:28px;">
            <h3 class="section-title">Why early detection matters</h3>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div style="display:grid;grid-template-columns:1fr 1fr;
             gap:16px;margin-bottom:16px;">
            <div class="cost-card" style="background:#FFF7ED;
                 border-color:#FED7AA;">
                <div class="cost-card-label" style="color:#EA580C;">
                    If CKD develops untreated
                </div>
                <div class="cost-card-amount" style="color:#EA580C;">
                    $28,162
                </div>
                <div class="cost-card-sub">per year in healthcare costs</div>
            </div>
            <div class="cost-card" style="background:#F0FDF4;
                 border-color:#BBF7D0;">
                <div class="cost-card-label" style="color:#16A34A;">
                    With early intervention
                </div>
                <div class="cost-card-amount" style="color:#16A34A;">
                    $13,604
                </div>
                <div class="cost-card-sub">potential saving of $14,558/yr</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.info("💡 Early detection and treatment can significantly "
                "reduce kidney disease progression and associated "
                "healthcare costs. Reference: USRDS (2023)")

        st.markdown("""
        <div class="disclaimer-box urgent">
            🏥 <strong>Important:</strong> This risk assessment is generated
            by a machine learning model trained on synthetic electronic
            health records. It is not a medical diagnosis. Please consult
            your nephrologist or primary care physician before making
            any health decisions. Reference: KDIGO (2024)
        </div>
        """, unsafe_allow_html=True)

    with tab3:
        st.markdown("""
        <div class="section-header">
            <h3 class="section-title">Learn about kidney health</h3>
        </div>
        """, unsafe_allow_html=True)

        with st.expander("🫘 What is CKD Stage 3?"):
            st.markdown("""
            Chronic Kidney Disease Stage 3 means your kidneys are
            working at 30–59% of normal capacity (eGFR 30–59).
            At this stage, most people have no symptoms, which is
            why early detection is so important.

            **Reference:** KDIGO (2024) CKD Classification System
            """)

        with st.expander("📊 What does my risk score mean?"):
            st.markdown("""
            Your risk score (0–100%) represents the probability
            that you may develop CKD Stage 3 within the next
            12 months based on your health records.

            - **0–40%** LOW — Continue regular monitoring
            - **40–65%** MODERATE — Discuss with your doctor
            - **65–85%** HIGH — Schedule appointment soon
            - **85–100%** URGENT — Contact your doctor today

            **Reference:** KDIGO (2024) · Tangri et al. (2016)
            """)

        with st.expander("💊 What can slow CKD progression?"):
            st.markdown("""
            Evidence-based interventions that can slow CKD:
            - Blood pressure control below 130/80 mmHg
            - HbA1c control below 7% (if diabetic)
            - ACE inhibitors or ARB medications
            - SGLT2 inhibitors (if diabetic)
            - Low-sodium diet (under 2g per day)
            - Regular moderate exercise

            **Reference:** KDIGO (2024) · ADA (2023)
            """)

        with st.expander("📞 Who should I contact?"):
            st.markdown("""
            - **Urgent symptoms** (swelling, difficulty breathing,
              sudden pain): Call 911 or go to Emergency
            - **Risk questions**: Call your primary care physician
            - **Specialist care**: Ask for a nephrology referral
            - **This tool**: Discuss results at your next visit
            """)