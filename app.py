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

# ── Custom CSS (clinical dashboard styling) ───────────────
st.markdown("""
<style>
    .block-container { padding-top: 1.25rem; }
    .main-header {
        font-size: 2.15rem;
        font-weight: 750;
        color: #0f766e;
        margin-bottom: 0;
        letter-spacing: -0.02em;
    }
    .sub-header {
        font-size: 1.02rem;
        color: #64748b;
        margin-top: 0.35rem;
        line-height: 1.5;
        max-width: 52rem;
    }
    .hero-strip {
        background: linear-gradient(120deg, #f0fdfa 0%, #ecfeff 45%, #f8fafc 100%);
        border: 1px solid #ccfbf1;
        border-radius: 14px;
        padding: 1.1rem 1.35rem;
        margin-bottom: 1rem;
    }
    .role-pill {
        display: inline-block;
        background: #0d9488;
        color: white;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        padding: 0.35rem 0.65rem;
        border-radius: 999px;
        margin-bottom: 0.5rem;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        border-left: 5px solid #0d9488;
        box-shadow: 0 1px 3px rgba(15,118,110,0.12);
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
        background: linear-gradient(135deg, #0d9488 0%, #115e59 100%);
        border-radius: 16px;
        padding: 30px;
        color: white;
        text-align: center;
    }
    div[data-testid="stMetric"] {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 0.65rem 0.85rem;
    }
    .hint-text { font-size: 0.88rem; color: #64748b; }
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

def page_hero(role_label, title_html, subtitle_html):
    """Top-of-page strip for a consistent clinical-app header."""
    st.markdown(
        f'<div class="hero-strip">'
        f'<span class="role-pill">{role_label}</span><br/>'
        f'{title_html}{subtitle_html}</div>',
        unsafe_allow_html=True
    )

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
    st.markdown("## CKDPredict")
    st.caption("Kidney health risk intelligence · demo environment")
    st.divider()

    user_role = st.selectbox(
        "I am signed in as",
        ["🏥 Healthcare Administrator",
         "🩺 Nephrologist / Physician",
         "👤 Patient View"],
        index=0,
        help="Switches workflows only — same underlying registry and models."
    )

    st.divider()
    with st.expander("Model quality snapshot", expanded=False):
        st.metric("Model A (diabetes) AUC",
                  f"{metrics['model_a_auc']:.4f}")
        st.metric("Model B (non-diabetes) AUC",
                  f"{metrics['model_b_auc']:.4f}")
        st.caption("Held-out validation metrics from bundled model cards.")

    st.divider()
    st.markdown("**Workspace**")
    if "Administrator" in user_role:
        page = st.radio(
            "Screen",
            [
                "📊 Patient Risk Registry",
                "🗺️ Geographic Overview",
                "💰 Cost Dashboard"
            ],
            label_visibility="collapsed"
        )
    elif "Nephrologist" in user_role:
        page = st.radio(
            "Screen",
            [
                "🔬 Individual Patient Detail",
                "📈 Model Comparison"
            ],
            label_visibility="collapsed"
        )
    else:
        page = "👤 My Kidney Health"

    st.divider()
    with st.expander("About this build"):
        st.caption("Saint Louis University · MS Analytics | MRP 2026")
        st.caption(
            "Guidelines cited: KDIGO 2024 · ADA 2023 · "
            "USRDS 2023 · Tangri et al. 2016"
        )

# ════════════════════════════════════════════════════════════
# SCREEN 1 — PATIENT RISK REGISTRY (Administrator)
# ════════════════════════════════════════════════════════════
if "Administrator" in user_role and \
        "Registry" in page:

    page_hero(
        "Population health",
        '<p class="main-header">Patient Risk Registry</p>',
        '<p class="sub-header">Prioritize outreach using the same risk tiers and '
        'scores as your live analytics pipeline. Use filters and quick search to '
        'build call lists.</p>'
    )
    st.caption(
        "Demo UI — outbound messages are simulated; registry rows and scores are "
        "unchanged from the source file."
    )

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

    col1.metric(
        "Panel size",
        f"{total_patients:,}",
        help="Distinct patients in the loaded registry."
    )
    col2.metric(
        "Needs outreach today",
        f"{urgent}",
        delta="Urgent tier",
        help="Count of patients flagged URGENT in the registry."
    )
    col3.metric(
        "High priority",
        f"{high}",
        delta="Within ~2 weeks",
        help="Count of patients flagged HIGH."
    )
    col4.metric(
        "Projected spend (panel)",
        f"${total_cost:,.0f}",
        help="Sum of PROJ_COST from the registry."
    )
    col5.metric(
        "Modeled savings opportunity",
        f"${total_saving:,.0f}",
        delta="If early intervention",
        help="Sum of POTENTIAL_SAVING from the registry."
    )

    st.divider()

    # ── Filters ───────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        tier_filter = st.multiselect(
            "Urgency tier",
            ['URGENT', 'HIGH', 'MODERATE', 'LOW'],
            default=['URGENT', 'HIGH'],
            help="Include only selected KDIGO-style urgency bands."
        )
    with col2:
        model_filter = st.selectbox(
            "Risk model / pathway",
            ['All', 'A - Diabetic',
             'B - Non-Diabetic'],
            help="Model A = diabetes cohort features; Model B = non-diabetic."
        )
    with col3:
        min_risk = st.slider(
            "Minimum risk score",
            0.0, 1.0, 0.65, 0.05,
            help="Filter to rows with RISK_SCORE ≥ this value."
        )
    with col4:
        search_q = st.text_input(
            "Quick find (patient ID)",
            "",
            placeholder="e.g. P001…",
            help="Substring match on PATIENT; case-insensitive."
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
    if search_q.strip():
        filtered = filtered[
            filtered['PATIENT'].astype(str).str.contains(
                search_q.strip(), case=False, na=False
            )
        ]
    filtered = filtered.sort_values(
        'RISK_SCORE', ascending=False)

    roster_tab, outreach_tab = st.tabs(
        ["Patient roster", "Outreach & messaging"]
    )

    with roster_tab:
        st.markdown(
            f"**{len(filtered):,}** patients match the current criteria — "
            "sort by risk score (highest first)."
        )

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
        # RISK_SCORE is stored 0–1; ProgressColumn format applies to the cell
        # value, so %.0f%% on 0.99 showed "1%". Scale to 0–100 for label + bar.
        display_df['Risk Score'] = (
            display_df['Risk Score'].astype(float) * 100.0
        ).round(1)

        dc = {
            "Patient ID": st.column_config.TextColumn(
                "Patient ID", width="medium"
            ),
            "Risk Score": st.column_config.ProgressColumn(
                "Risk score (%)",
                help="Registry RISK_SCORE × 100 (stored probability is 0–1).",
                format="%.1f%%",
                min_value=0.0,
                max_value=100.0,
            ),
            "Urgency": st.column_config.TextColumn("Urgency"),
            "Est. Months to CKD": st.column_config.TextColumn(
                "Est. timeline", width="medium"
            ),
            "Proj. Cost/yr": st.column_config.NumberColumn(
                "Proj. cost / yr",
                format="$%d",
                help="From registry PROJ_COST.",
            ),
            "Potential Saving": st.column_config.NumberColumn(
                "Potential saving",
                format="$%d",
                help="From registry POTENTIAL_SAVING.",
            ),
            "Pathway": st.column_config.TextColumn("Pathway", width="large"),
            "City": st.column_config.TextColumn("City", width="small"),
        }
        use_cfg = {
            k: dc[k] for k in display_df.columns if k in dc
        }

        st.dataframe(
            display_df,
            column_config=use_cfg,
            use_container_width=True,
            height=440,
            hide_index=True,
        )

    with outreach_tab:
        st.markdown("### Care-management outreach")
        st.info(
            "Queue templated patient-portal messages based on urgency. Buttons "
            "below **simulate** sending — your underlying registry is not modified."
        )

        col1, col2 = st.columns(2)
        with col1:
            if st.button(
                    "Notify all URGENT patients",
                    type="primary",
                    use_container_width=True):
                if hasattr(st, "toast"):
                    st.toast(
                        f"Queued secure message for {urgent} URGENT patients",
                        icon="📧",
                    )
                st.success(
                    f"Demo: notification workflow triggered for **{urgent}** "
                    "patients in the URGENT tier."
                )
        with col2:
            if st.button(
                    "Notify all HIGH-risk patients",
                    use_container_width=True):
                if hasattr(st, "toast"):
                    st.toast(
                        f"Queued secure message for {high} HIGH-risk patients",
                        icon="📧",
                    )
                st.success(
                    f"Demo: notification workflow triggered for **{high}** "
                    "patients in the HIGH tier."
                )

# ════════════════════════════════════════════════════════════
# SCREEN 2 — GEOGRAPHIC OVERVIEW (Administrator)
# ════════════════════════════════════════════════════════════
elif "Administrator" in user_role and \
        "Geographic" in page:

    page_hero(
        "Population health",
        '<p class="main-header">Geographic risk overview</p>',
        '<p class="sub-header">Compare hotspots of urgent and high-risk patients '
        'by city. Narrow the map to specific communities without altering any '
        'source data.</p>'
    )

    if 'CITY' in registry.columns:
        all_cities = sorted(
            registry['CITY'].dropna().astype(str).unique()
        )
        pick = st.multiselect(
            "Cities to include",
            all_cities,
            default=[],
            placeholder="Leave empty to include all cities",
            help="Subset the registry before aggregation; empty means full panel."
        )
        reg_geo = registry[
            registry['CITY'].isin(pick)
        ] if pick else registry

        city_summary = reg_geo.groupby('CITY').agg(
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

        map_tab, table_tab = st.tabs(["Hotspot chart", "City data table"])

        with map_tab:
            top_n = st.slider(
                "Show top N cities by urgent count",
                5, 25, 15,
                help="Chart uses cities with the most URGENT cases after your filter."
            )
            chart_df = city_summary.head(top_n)
            if chart_df.empty:
                st.info(
                    "No city rows for this filter. Clear the city filter or pick "
                    "different cities."
                )
            else:
                # graph_objects stacked bars — avoids Plotly Express + pandas
                # groupby bugs (KeyError on color categories) in some versions.
                fig = go.Figure(
                    data=[
                        go.Bar(
                            name='Urgent',
                            x=chart_df['CITY'].astype(str),
                            y=chart_df['Urgent_Cases'],
                            marker_color='#DC2626',
                        ),
                        go.Bar(
                            name='High risk',
                            x=chart_df['CITY'].astype(str),
                            y=chart_df['High_Cases'],
                            marker_color='#D97706',
                        ),
                    ]
                )
                fig.update_layout(
                    title='Urgent vs high-risk CKD patients by city',
                    barmode='stack',
                    xaxis_tickangle=-45,
                    height=480,
                    legend_title_text='',
                    xaxis_title='City',
                    yaxis_title='Patients',
                )
                st.plotly_chart(fig, use_container_width=True)

        with table_tab:
            st.caption(
                "Sorted by urgent case count. Columns are aggregates of the "
                "filtered registry only."
            )
            st.dataframe(
                city_summary.head(40),
                use_container_width=True,
                hide_index=True,
            )

    st.divider()
    st.markdown("### Pathway volume: Model A vs Model B")

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

    page_hero(
        "Finance & value",
        '<p class="main-header">Cost & utilization dashboard</p>',
        '<p class="sub-header">Benchmark against USRDS 2023 spend ranges, then '
        'reconcile with your registry-level projections. Numbers below reuse the '
        'same PROJ_COST and POTENTIAL_SAVING fields.</p>'
    )

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

    st.markdown("### Registry-linked financial snapshot")
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

    pie_tab, meth_tab = st.tabs(
        ["Spend mix by tier", "Methodology notes"]
    )

    with pie_tab:
        tier_costs = registry.groupby(
            'URGENCY_TIER')['PROJ_COST'].sum(
        ).reset_index()
        fig = px.pie(
            tier_costs,
            values='PROJ_COST',
            names='URGENCY_TIER',
            title='Projected registry cost share by urgency tier',
            color='URGENCY_TIER',
            color_discrete_map={
                'URGENT'  : '#DC2626',
                'HIGH'    : '#D97706',
                'MODERATE': '#2563EB',
                'LOW'     : '#059669'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with meth_tab:
        st.markdown(
            "- **USRDS (2023)** benchmarks drive the per-patient reference "
            "metrics in the row above.\n"
            "- **Panel metrics** sum fields already stored in "
            "`patient_registry.csv` — no recomputation of model outputs.\n"
            "- Replace placeholder **potential savings** assumptions with your "
            "health system’s finance models before operational use."
        )

    st.caption(
        "Reference: USRDS (2023) Annual Data Report — Medicare spending "
        "benchmarks per beneficiary. Cost estimates are approximate and should "
        "be validated with institutional financial data."
    )

# ════════════════════════════════════════════════════════════
# SCREEN 4 — INDIVIDUAL PATIENT DETAIL (Nephrologist)
# ════════════════════════════════════════════════════════════
elif "Nephrologist" in user_role and \
        "Individual" in page:

    page_hero(
        "Clinical decision support",
        '<p class="main-header">Individual patient record</p>',
        '<p class="sub-header">Review registry risk scores, urgency, and cost '
        'projections; align charting tasks with KDIGO-aligned reminders.</p>'
    )
    st.info(
        "Support tool only — verify all orders and referrals against the "
        "source chart and institutional policy."
    )

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

    tier  = patient['URGENCY_TIER']
    score = patient['RISK_SCORE']
    color = get_tier_color(tier)
    pid = str(patient['PATIENT'])

    sum_tab, plan_tab = st.tabs(["Risk summary", "Care planning checklist"])

    with sum_tab:
        col1, col2, col3, col4 = st.columns(4)

        col1.metric(
            "Model risk score",
            f"{score:.4f}",
            help="Registry RISK_SCORE (same value used in analytics export).",
        )
        col2.metric(
            "Urgency tier",
            get_tier_badge(tier),
            help="KDIGO-aligned urgency label from registry.",
        )
        col3.metric(
            "Estimated time to CKD",
            patient['EST_MONTHS'],
            help="EST_MONTHS from registry for care prioritization.",
        )
        col4.metric(
            "Projected annual cost",
            f"${patient['PROJ_COST']:,.0f}",
            help="USRDS-calibrated projection stored in registry.",
        )

        st.markdown("#### Demographics & pathway")
        info_cols = ['PATHWAY', 'CITY', 'STATE']
        ic = [c for c in info_cols if c in patient.index]
        if ic:
            info_df = pd.DataFrame(
                {"Field": ic,
                 "Value": [patient[c] for c in ic]}
            )
            st.dataframe(info_df, hide_index=True, use_container_width=True)

    with plan_tab:
        st.markdown("### KDIGO 2024 — reminder checklist")
        st.caption(
            "Interactive tick boxes for workflow only; they are not saved to "
            "the registry or EHR."
        )
        st.info(
            "Reference: KDIGO (2024) Clinical Practice Guidelines for CKD "
            "Evaluation and Management."
        )

        if tier in ['URGENT', 'HIGH']:
            st.checkbox(
                "Order eGFR and UACR tests",
                key=f"egfr_{pid}")
            st.checkbox(
                "Initiate ACE inhibitor or ARB therapy (if appropriate)",
                key=f"ace_{pid}")
            st.checkbox(
                "Target BP < 130/80 mmHg",
                key=f"bp_{pid}")
            st.checkbox(
                "Refer to nephrology",
                key=f"neph_{pid}")
            st.checkbox(
                "Consider SGLT2 inhibitor (if diabetic)",
                key=f"sglt2_{pid}")
            st.checkbox(
                "Schedule follow-up in ~4 weeks",
                key=f"fu_{pid}")
        else:
            st.checkbox(
                "Monitor eGFR every 3 months",
                key=f"m3_{pid}")
            st.checkbox(
                "Review medication list",
                key=f"med_{pid}")
            st.checkbox(
                "Lifestyle counseling",
                key=f"life_{pid}")
            st.checkbox(
                "Annual UACR screening",
                key=f"uacr_{pid}")

# ════════════════════════════════════════════════════════════
# SCREEN 5 — PATIENT VIEW (Professor Vision)
# ════════════════════════════════════════════════════════════
elif "Patient" in user_role:

    page_hero(
        "Patient portal (demo)",
        '<p class="main-header">My kidney health summary</p>',
        '<p class="sub-header">View the same risk tier and score your care team '
        'sees in the registry. Bring questions to your next visit.</p>'
    )

    st.warning(
        "This page shows **educational risk information only**. It is not a "
        "diagnosis. Call your clinician or 911 for urgent symptoms."
    )

    patient_id = st.selectbox(
        "Select your patient record ID",
        registry['PATIENT'].tolist(),
        help="Matches an ID in the demonstration registry file."
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

    over_tab, act_tab, learn_tab = st.tabs(
        ["Overview", "My action list", "Learn & ask"]
    )

    with over_tab:
        col1, col2 = st.columns([1, 2])

        with col1:
            risk_pct = int(score * 100)
            st.markdown(
                f"""
                <div style='text-align:center;
                     padding:28px 20px;
                     background:linear-gradient(
                         145deg, {color}18, {color}38);
                     border-radius:16px;
                     border: 2px solid {color};
                     box-shadow: 0 4px 14px rgba(15,118,110,0.12);'>
                    <div style='font-size:0.85rem;color:#475569;
                        font-weight:600;'>Modeled risk index</div>
                    <h1 style='color:{color};
                        font-size:2.75rem;
                        margin:0.25rem 0;'>{risk_pct}%</h1>
                    <p style='color:{color};
                       font-weight:700;
                       font-size:1.1rem;margin:0;'>{tier}</p>
                    <p style='color:#64748b;font-size:0.9rem;margin-top:0.5rem;'>
                        Based on registry score · not a lab result</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            st.caption(f"Care pathway in registry: **{pathway}**")

        with col2:
            message = get_patient_message(tier, months)
            st.markdown(message)

            st.markdown("#### Estimated timeline")
            st.markdown(
                f"If nothing changes clinically, the model suggests kidney "
                f"disease could develop in about **{months}**. Your team "
                f"will interpret this with labs."
            )
            st.caption(
                "Estimate uses the same EST_MONTHS field as the administrator "
                "registry. Not a personal prognosis."
            )

    with act_tab:
        st.markdown("### Suggested next steps")
        st.caption("Use this as a conversation starter with your doctor.")
        steps = get_action_steps(tier)
        for i, step in enumerate(steps, 1):
            st.markdown(f"{i}. {step}")

        st.divider()
        st.markdown("**Questions you might ask**")
        st.markdown(
            "- What do my latest creatinine and eGFR show compared to this score?\n"
            "- Should I start or adjust blood-pressure medicine?\n"
            "- When should I repeat urine albumin testing?"
        )

    with learn_tab:
        st.markdown("### Why timing matters")
        c1, c2 = st.columns(2)
        c1.metric(
            "Illustrative cost with advanced CKD",
            "$28,162/yr",
            help="USRDS 2023 benchmark — education only.",
        )
        c2.metric(
            "Illustrative cost without CKD",
            "$13,604/yr",
            delta="Lower spend band",
            help="USRDS 2023 benchmark — not your personal bill.",
        )

        st.info(
            "Earlier treatment often slows kidney disease. Figures are national "
            "averages from USRDS (2023), not your charges."
        )

        st.error(
            "**Important:** This summary comes from a machine-learning model on "
            "demo data. It does not replace your nephrologist or primary care "
            "clinician. See KDIGO (2024) for guideline context."
        )

# ════════════════════════════════════════════════════════════
# SCREEN 6 — MODEL COMPARISON (Nephrologist)
# ════════════════════════════════════════════════════════════
elif "Nephrologist" in user_role and \
        "Comparison" in page:

    page_hero(
        "Analytics",
        '<p class="main-header">Model comparison</p>',
        '<p class="sub-header">Side-by-side validation metrics for the diabetic '
        '(A) and non-diabetic (B) pathways. Values are loaded from the packaged '
        'model card pickle — unchanged.</p>'
    )

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
        st.metric(
            "EPV",
            f"{metrics['model_a_epv']:.1f}",
            help="From model_metrics.pkl: positive cases ÷ features used at "
                 "training export. Differs slightly if you recount features "
                 "in a notebook.",
        )

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
        st.metric(
            "EPV",
            f"{metrics['model_b_epv']:.1f}",
            help="From model_metrics.pkl: positive cases ÷ features used at "
                 "training export. Differs slightly if you recount features "
                 "in a notebook.",
        )

    st.caption(
        "EPV values above match the packaged model card snapshot, not a live "
        "recompute from your current feature list."
    )

    with st.expander("Clinical & methods references"):
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