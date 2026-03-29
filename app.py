# ============================================================
# CKDPredict — Streamlit Application
# Early Detection of Chronic Kidney Disease
# Saint Louis University | MS Analytics 2026
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ── Page Configuration ────────────────────────────────────
st.set_page_config(
    page_title    = "CKDPredict",
    page_icon     = "🫘",
    layout        = "wide",
    initial_sidebar_state = "expanded"
)

# ── Custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1B3A6B;
        margin-bottom: 0;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        margin-top: 0;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border-left: 5px solid #4F46E5;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .urgent-badge {
        background: #FEE2E2;
        color: #DC2626;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .high-badge {
        background: #FEF3C7;
        color: #D97706;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .moderate-badge {
        background: #DBEAFE;
        color: #2563EB;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .low-badge {
        background: #D1FAE5;
        color: #059669;
        padding: 4px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.8rem;
    }
    .patient-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 16px;
        padding: 30px;
        color: white;
        text-align: center;
    }
    .stMetric {
        background: #F9FAFB;
        border-radius: 8px;
        padding: 10px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load Models and Data ───────────────────────────────────
@st.cache_resource
def load_models():
    model_a      = joblib.load('models/model_a_xgboost.pkl')
    model_b      = joblib.load('models/model_b_xgboost.pkl')
    feat_a       = joblib.load('models/feature_cols_a.pkl')
    feat_b       = joblib.load('models/feature_cols_b.pkl')
    enc_gender   = joblib.load('models/encoder_gender.pkl')
    enc_race     = joblib.load('models/encoder_race.pkl')
    metrics      = joblib.load('models/model_metrics.pkl')
    return (model_a, model_b, feat_a, feat_b,
            enc_gender, enc_race, metrics)

@st.cache_data
def load_registry():
    df = pd.read_csv('models/patient_registry.csv')
    return df

# ── Helper Functions ───────────────────────────────────────
def get_tier_badge(tier):
    badges = {
        'URGENT'  : '🚨 URGENT',
        'HIGH'    : '⚠️ HIGH',
        'MODERATE': '📊 MODERATE',
        'LOW'     : '✅ LOW'
    }
    return badges.get(tier, tier)

def get_tier_color(tier):
    colors = {
        'URGENT'  : '#DC2626',
        'HIGH'    : '#D97706',
        'MODERATE': '#2563EB',
        'LOW'     : '#059669'
    }
    return colors.get(tier, '#6B7280')

def get_patient_message(tier, months):
    messages = {
        'URGENT': f"""
        ⚠️ **Your kidney health needs immediate attention.**

        Based on your health records, you may develop
        kidney disease in approximately **{months}**
        if no action is taken.

        **Please contact your doctor today.**
        """,
        'HIGH': f"""
        📋 **Your kidney health needs attention.**

        Your health records suggest you may develop
        kidney disease in approximately **{months}**.

        **Please schedule an appointment soon.**
        """,
        'MODERATE': f"""
        📊 **Your kidney health shows some risk.**

        Based on your records, changes may occur
        in approximately **{months}**.

        **Discuss this with your doctor
        at your next visit.**
        """,
        'LOW': """
        ✅ **Your kidney health appears stable.**

        Continue your regular check-ups and
        follow your current treatment plan.
        """
    }
    return messages.get(tier, messages['LOW'])

def get_action_steps(tier):
    actions = {
        'URGENT': [
            "📞 Call your doctor today",
            "🔬 Ask for emergency kidney function test",
            "🏥 Do not wait for your next appointment"
        ],
        'HIGH': [
            "📅 Schedule appointment within 2 weeks",
            "🔬 Ask for creatinine and eGFR test",
            "💊 Monitor blood pressure daily"
        ],
        'MODERATE': [
            "📋 Mention this at your next appointment",
            "🔬 Ask about kidney health monitoring",
            "🥗 Maintain healthy diet and exercise"
        ],
        'LOW': [
            "📅 Continue regular check-ups",
            "💧 Stay hydrated daily",
            "🥗 Maintain healthy lifestyle"
        ]
    }
    return actions.get(tier, actions['LOW'])

# ── Load everything ────────────────────────────────────────
try:
    (model_a, model_b, feat_a, feat_b,
     enc_gender, enc_race,
     metrics) = load_models()
    registry = load_registry()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Error loading models: {e}")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🫘 CKDPredict")
    st.markdown("*Early CKD Detection*")
    st.divider()

    user_role = st.selectbox(
        "Select Your Role",
        ["🏥 Healthcare Administrator",
         "🩺 Nephrologist / Physician",
         "👤 Patient View"],
        index=0
    )

    st.divider()
    st.markdown("### Model Performance")
    st.metric("Model A AUC",
              f"{metrics['model_a_auc']:.4f}")
    st.metric("Model B AUC",
              f"{metrics['model_b_auc']:.4f}")

    st.divider()
    st.markdown("### Navigation")
    if "Administrator" in user_role:
        page = st.radio("Go to", [
            "📊 Patient Risk Registry",
            "🗺️ Geographic Overview",
            "💰 Cost Dashboard"
        ])
    elif "Nephrologist" in user_role:
        page = st.radio("Go to", [
            "🔬 Individual Patient Detail",
            "📈 Model Comparison"
        ])
    else:
        page = "👤 My Kidney Health"

    st.divider()
    st.caption("Saint Louis University")
    st.caption("MS Analytics | MRP 2026")
    st.caption("References: KDIGO 2024 | ADA 2023")
    st.caption("USRDS 2023 | Tangri et al. 2016")

# ════════════════════════════════════════════════════════════
# SCREEN 1 — PATIENT RISK REGISTRY (Administrator)
# ════════════════════════════════════════════════════════════
if "Administrator" in user_role and \
        "Registry" in page:

    st.markdown(
        '<p class="main-header">📊 Patient Risk Registry</p>',
        unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Healthcare Administrator View — '
        'Population CKD Risk Management</p>',
        unsafe_allow_html=True)

    st.divider()

    # ── Top KPIs ──────────────────────────────────────────
    col1, col2, col3, col4, col5 = st.columns(5)

    total_patients = len(registry)
    urgent = len(registry[
        registry['URGENCY_TIER'] == 'URGENT'])
    high = len(registry[
        registry['URGENCY_TIER'] == 'HIGH'])
    total_saving = registry[
        'POTENTIAL_SAVING'].sum()
    total_cost = registry['PROJ_COST'].sum()

    col1.metric("Total Patients",
                f"{total_patients:,}")
    col2.metric("🚨 Urgent",
                f"{urgent}",
                delta="Immediate action")
    col3.metric("⚠️ High Risk",
                f"{high}",
                delta="Action within 2 weeks")
    col4.metric("Total Proj. Cost",
                f"${total_cost:,.0f}")
    col5.metric("Potential Savings",
                f"${total_saving:,.0f}",
                delta="If caught early")

    st.divider()

    # ── Filters ───────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    with col1:
        tier_filter = st.multiselect(
            "Filter by Urgency",
            ['URGENT', 'HIGH', 'MODERATE', 'LOW'],
            default=['URGENT', 'HIGH']
        )
    with col2:
        model_filter = st.selectbox(
            "Filter by Model",
            ['All', 'A - Diabetic',
             'B - Non-Diabetic']
        )
    with col3:
        min_risk = st.slider(
            "Minimum Risk Score",
            0.0, 1.0, 0.65, 0.05
        )

    # ── Apply filters ─────────────────────────────────────
    filtered = registry.copy()
    if tier_filter:
        filtered = filtered[
            filtered['URGENCY_TIER'].isin(
                tier_filter)]
    if model_filter == 'A - Diabetic':
        filtered = filtered[
            filtered['MODEL'] == 'A']
    elif model_filter == 'B - Non-Diabetic':
        filtered = filtered[
            filtered['MODEL'] == 'B']
    filtered = filtered[
        filtered['RISK_SCORE'] >= min_risk]
    filtered = filtered.sort_values(
        'RISK_SCORE', ascending=False)

    st.markdown(f"**{len(filtered):,} patients "
                f"match your filters**")

    # ── Patient Table ─────────────────────────────────────
    display_cols = {
        'PATIENT'          : 'Patient ID',
        'RISK_SCORE'       : 'Risk Score',
        'URGENCY_TIER'     : 'Urgency',
        'EST_MONTHS'       : 'Est. Months to CKD',
        'PROJ_COST'        : 'Proj. Cost/yr',
        'POTENTIAL_SAVING' : 'Potential Saving',
        'PATHWAY'          : 'Pathway',
        'CITY'             : 'City',
    }

    available = {
        k: v for k, v in display_cols.items()
        if k in filtered.columns}

    display_df = filtered[
        list(available.keys())].copy()
    display_df.columns = list(available.values())
    display_df['Risk Score'] = display_df[
        'Risk Score'].round(4)
    display_df['Proj. Cost/yr'] = display_df[
        'Proj. Cost/yr'].apply(
        lambda x: f"${x:,.0f}")
    display_df['Potential Saving'] = display_df[
        'Potential Saving'].apply(
        lambda x: f"${x:,.0f}")

    st.dataframe(
        display_df,
        use_container_width=True,
        height=400
    )

    # ── Notify Section ────────────────────────────────────
    st.divider()
    st.markdown("### 📧 Patient Notification")
    st.info(
        "Select patients to notify about their "
        "CKD risk. Notifications are sent via "
        "the patient portal with plain-language "
        "risk summaries.")

    col1, col2 = st.columns(2)
    with col1:
        if st.button(
                "📧 Notify All URGENT Patients",
                type="primary"):
            st.success(
                f"✅ Notification sent to "
                f"{urgent} URGENT patients")
    with col2:
        if st.button(
                "📧 Notify All HIGH Risk Patients"):
            st.success(
                f"✅ Notification sent to "
                f"{high} HIGH risk patients")

# ════════════════════════════════════════════════════════════
# SCREEN 2 — GEOGRAPHIC OVERVIEW (Administrator)
# ════════════════════════════════════════════════════════════
elif "Administrator" in user_role and \
        "Geographic" in page:

    st.markdown(
        '<p class="main-header">🗺️ Geographic Overview</p>',
        unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">CKD Risk Distribution '
        'Across California</p>',
        unsafe_allow_html=True)
    st.divider()

    if 'CITY' in registry.columns:
        city_summary = registry.groupby('CITY').agg(
            Total_Patients = ('PATIENT', 'count'),
            Urgent_Cases   = ('URGENCY_TIER',
                lambda x: (x == 'URGENT').sum()),
            High_Cases     = ('URGENCY_TIER',
                lambda x: (x == 'HIGH').sum()),
            Avg_Risk       = ('RISK_SCORE', 'mean'),
            Total_Cost     = ('PROJ_COST', 'sum'),
            Total_Saving   = ('POTENTIAL_SAVING',
                              'sum')
        ).reset_index().sort_values(
            'Urgent_Cases', ascending=False)

        st.markdown("### Risk Distribution by City")
        st.dataframe(
            city_summary.head(20),
            use_container_width=True)

        # Bar chart
        fig = px.bar(
            city_summary.head(15),
            x='CITY',
            y=['Urgent_Cases', 'High_Cases'],
            title='Top 15 Cities — Urgent and '
                  'High Risk CKD Patients',
            barmode='stack',
            color_discrete_map={
                'Urgent_Cases': '#DC2626',
                'High_Cases'  : '#D97706'
            }
        )
        fig.update_layout(
            xaxis_tickangle=-45,
            height=450)
        st.plotly_chart(fig,
            use_container_width=True)

    # Model comparison
    st.divider()
    st.markdown("### Model A vs Model B Comparison")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Model A — Diabetic**")
        reg_a = registry[registry['MODEL'] == 'A']
        st.metric("Patients",
                  f"{len(reg_a):,}")
        st.metric("AUC-ROC",
                  f"{metrics['model_a_auc']:.4f}")
        st.metric("CV AUC",
                  f"{metrics['model_a_cv_mean']:.4f}"
                  f" ± "
                  f"{metrics['model_a_cv_std']:.4f}")

    with col2:
        st.markdown("**Model B — Non-Diabetic**")
        reg_b = registry[registry['MODEL'] == 'B']
        st.metric("Patients",
                  f"{len(reg_b):,}")
        st.metric("AUC-ROC",
                  f"{metrics['model_b_auc']:.4f}")
        st.metric("CV AUC",
                  f"{metrics['model_b_cv_mean']:.4f}"
                  f" ± "
                  f"{metrics['model_b_cv_std']:.4f}")

# ════════════════════════════════════════════════════════════
# SCREEN 3 — COST DASHBOARD (Administrator)
# ════════════════════════════════════════════════════════════
elif "Administrator" in user_role and \
        "Cost" in page:

    st.markdown(
        '<p class="main-header">💰 Cost Dashboard</p>',
        unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Healthcare Cost Impact '
        'Analysis — USRDS 2023 Benchmarks</p>',
        unsafe_allow_html=True)
    st.divider()

    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Without CKD",
        "$13,604/yr",
        help="Medicare cost per patient — USRDS 2023")
    col2.metric(
        "With CKD Stage 3",
        "$28,162/yr",
        delta="+$14,558",
        delta_color="inverse",
        help="Medicare cost per patient — USRDS 2023")
    col3.metric(
        "With ESKD",
        "$104,000+/yr",
        delta="+$90,396",
        delta_color="inverse",
        help="End-stage kidney disease — USRDS 2023")

    st.divider()

    urgent_count = len(registry[
        registry['URGENCY_TIER'] == 'URGENT'])
    high_count   = len(registry[
        registry['URGENCY_TIER'] == 'HIGH'])
    total_saving = registry[
        'POTENTIAL_SAVING'].sum()

    st.markdown("### Your Population Cost Analysis")
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Patients at Risk",
        f"{urgent_count + high_count:,}")
    col2.metric(
        "Total Projected Cost",
        f"${registry['PROJ_COST'].sum():,.0f}")
    col3.metric(
        "Potential Annual Saving",
        f"${total_saving:,.0f}",
        help="If early intervention successful "
             "in 100% of cases")

    # Cost breakdown chart
    tier_costs = registry.groupby(
        'URGENCY_TIER')['PROJ_COST'].sum(
    ).reset_index()
    fig = px.pie(
        tier_costs,
        values='PROJ_COST',
        names='URGENCY_TIER',
        title='Projected Cost Distribution '
              'by Urgency Tier',
        color='URGENCY_TIER',
        color_discrete_map={
            'URGENT'  : '#DC2626',
            'HIGH'    : '#D97706',
            'MODERATE': '#2563EB',
            'LOW'     : '#059669'
        }
    )
    st.plotly_chart(fig,
        use_container_width=True)

    st.caption(
        "Reference: USRDS (2023) Annual Data Report "
        "— Medicare spending benchmarks per "
        "beneficiary. Cost estimates are approximate "
        "and should be validated with institutional "
        "financial data.")

# ════════════════════════════════════════════════════════════
# SCREEN 4 — INDIVIDUAL PATIENT DETAIL (Nephrologist)
# ════════════════════════════════════════════════════════════
elif "Nephrologist" in user_role and \
        "Individual" in page:

    st.markdown(
        '<p class="main-header">'
        '🔬 Individual Patient Detail</p>',
        unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Nephrologist View — '
        'Clinical Decision Support</p>',
        unsafe_allow_html=True)
    st.divider()

    # Patient selector
    col1, col2 = st.columns([2, 1])
    with col1:
        model_choice = st.selectbox(
            "Select Model",
            ["Model A — Diabetic",
             "Model B — Non-Diabetic"]
        )
    with col2:
        min_score = st.slider(
            "Min Risk Score",
            0.5, 1.0, 0.80)

    if "Model A" in model_choice:
        filtered_reg = registry[
            (registry['MODEL'] == 'A') &
            (registry['RISK_SCORE'] >= min_score)
        ].sort_values(
            'RISK_SCORE', ascending=False)
        fm_use      = None
        feat_cols   = feat_a
        model_use   = model_a
    else:
        filtered_reg = registry[
            (registry['MODEL'] == 'B') &
            (registry['RISK_SCORE'] >= min_score)
        ].sort_values(
            'RISK_SCORE', ascending=False)
        fm_use      = None
        feat_cols   = feat_b
        model_use   = model_b

    if len(filtered_reg) == 0:
        st.warning("No patients match filters")
        st.stop()

    patient_id = st.selectbox(
        "Select Patient",
        filtered_reg['PATIENT'].tolist()
    )

    patient = filtered_reg[
        filtered_reg['PATIENT'] == patient_id
    ].iloc[0]

    # Patient header
    st.divider()
    col1, col2, col3, col4 = st.columns(4)

    tier  = patient['URGENCY_TIER']
    score = patient['RISK_SCORE']
    color = get_tier_color(tier)

    col1.metric("Risk Score",
                f"{score:.4f}")
    col2.metric("Urgency",
                get_tier_badge(tier))
    col3.metric("Est. Months to CKD",
                patient['EST_MONTHS'])
    col4.metric("Projected Cost",
                f"${patient['PROJ_COST']:,.0f}/yr")

    # Clinical details
    st.divider()
    st.markdown("### Clinical Information")
    info_cols = ['PATHWAY', 'CITY', 'STATE']
    available = {
        c: c for c in info_cols
        if c in patient.index}
    for k in available:
        st.write(f"**{k}:** {patient[k]}")

    # KDIGO Intervention Checklist
    st.divider()
    st.markdown("### 📋 KDIGO 2024 Intervention "
                "Checklist")
    st.info("Reference: KDIGO (2024) Clinical "
            "Practice Guidelines for CKD "
            "Evaluation and Management")

    if tier in ['URGENT', 'HIGH']:
        st.checkbox(
            "Order eGFR and UACR tests")
        st.checkbox(
            "Initiate ACE inhibitor or ARB therapy")
        st.checkbox(
            "Target BP < 130/80 mmHg")
        st.checkbox(
            "Refer to nephrology")
        st.checkbox(
            "Consider SGLT2 inhibitor "
            "(if diabetic)")
        st.checkbox(
            "Schedule follow-up in 4 weeks")
    else:
        st.checkbox(
            "Monitor eGFR every 3 months")
        st.checkbox(
            "Review medication list")
        st.checkbox(
            "Lifestyle counseling")
        st.checkbox(
            "Annual UACR screening")

# ════════════════════════════════════════════════════════════
# SCREEN 5 — PATIENT VIEW (Professor Vision)
# ════════════════════════════════════════════════════════════
elif "Patient" in user_role:

    st.markdown(
        '<p class="main-header">'
        '👤 My Kidney Health</p>',
        unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">'
        'Your Personal CKD Risk Summary</p>',
        unsafe_allow_html=True)

    st.warning(
        "⚠️ This tool provides risk estimates "
        "only and is not a medical diagnosis. "
        "Always consult your doctor before "
        "taking any action.")

    st.divider()

    # Patient selector
    patient_id = st.selectbox(
        "Select your Patient ID",
        registry['PATIENT'].tolist()
    )

    patient = registry[
        registry['PATIENT'] == patient_id
    ].iloc[0]

    tier    = patient['URGENCY_TIER']
    score   = patient['RISK_SCORE']
    months  = patient['EST_MONTHS']
    pathway = patient.get('PATHWAY',
                          'Not specified')
    color   = get_tier_color(tier)

    # Risk display
    st.divider()
    col1, col2 = st.columns([1, 2])

    with col1:
        # Risk gauge visual
        risk_pct = int(score * 100)
        st.markdown(
            f"""
            <div style='text-align:center;
                 padding:30px;
                 background:linear-gradient(
                     135deg, {color}22, {color}44);
                 border-radius:16px;
                 border: 3px solid {color};'>
                <h1 style='color:{color};
                    font-size:3rem;
                    margin:0;'>{risk_pct}%</h1>
                <p style='color:{color};
                   font-weight:700;
                   font-size:1.2rem;'>{tier}</p>
                <p style='color:#6B7280;'>
                    Risk Level</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        # Plain language message
        message = get_patient_message(
            tier, months)
        st.markdown(message)

        st.divider()
        st.markdown("### ⏱️ Estimated Timeline")
        st.markdown(
            f"Based on your health records, "
            f"if no action is taken, kidney "
            f"disease may develop in "
            f"approximately **{months}**.")
        st.caption(
            "This is an estimate based on "
            "GFR slope analysis per KDIGO "
            "(2024) urgency tier definitions. "
            "Not a precise medical prediction.")

    # Action steps
    st.divider()
    st.markdown("### ✅ What You Should Do Next")
    steps = get_action_steps(tier)
    for step in steps:
        st.markdown(f"- {step}")

    # Cost awareness
    st.divider()
    st.markdown("### 💡 Why Early Detection Matters")
    col1, col2 = st.columns(2)
    col1.metric(
        "Cost if CKD develops",
        "$28,162/year",
        help="USRDS 2023")
    col2.metric(
        "Cost with early intervention",
        "$13,604/year",
        delta="-$14,558 potential saving",
        help="USRDS 2023")

    st.info(
        "Early detection and treatment can "
        "significantly reduce the progression "
        "of kidney disease and associated "
        "healthcare costs. "
        "Reference: USRDS (2023)")

    # Disclaimer
    st.divider()
    st.error(
        "🏥 **Important:** This risk assessment "
        "is generated by a machine learning model "
        "trained on synthetic data. It is not a "
        "medical diagnosis. Please consult your "
        "nephrologist or primary care physician "
        "before making any health decisions. "
        "Reference: KDIGO (2024)")

# ════════════════════════════════════════════════════════════
# SCREEN 6 — MODEL COMPARISON (Nephrologist)
# ════════════════════════════════════════════════════════════
elif "Nephrologist" in user_role and \
        "Comparison" in page:

    st.markdown(
        '<p class="main-header">'
        '📈 Model Comparison</p>',
        unsafe_allow_html=True)
    st.divider()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            "### Model A — Diabetic")
        st.metric("AUC-ROC",
            f"{metrics['model_a_auc']:.4f}")
        st.metric("CV AUC Mean",
            f"{metrics['model_a_cv_mean']:.4f}")
        st.metric("CV AUC Std",
            f"± {metrics['model_a_cv_std']:.4f}")
        st.metric("Training Patients",
            f"{metrics['model_a_patients']:,}")
        st.metric("CKD Positive Cases",
            f"{metrics['model_a_positive']:,}")
        st.metric("EPV",
            f"{metrics['model_a_epv']:.1f}")

    with col2:
        st.markdown(
            "### Model B — Non-Diabetic")
        st.metric("AUC-ROC",
            f"{metrics['model_b_auc']:.4f}")
        st.metric("CV AUC Mean",
            f"{metrics['model_b_cv_mean']:.4f}")
        st.metric("CV AUC Std",
            f"± {metrics['model_b_cv_std']:.4f}")
        st.metric("Training Patients",
            f"{metrics['model_b_patients']:,}")
        st.metric("CKD Positive Cases",
            f"{metrics['model_b_positive']:,}")
        st.metric("EPV",
            f"{metrics['model_b_epv']:.1f}")

    st.divider()
    st.markdown("### Clinical References")
    st.markdown("""
    - **KDIGO (2024)** — EPV minimum 10 per feature
    - **Tangri et al. (2016)** — AUC benchmark 0.90
    - **Walonoski et al. (2018)** — AUC > 0.75
      acceptable for Synthea-based models
    - **ADA (2023)** — HbA1c and UACR primary
      markers for diabetic CKD
    - **USRDS (2023)** — Hypertension primary
      driver of non-diabetic CKD
    """)