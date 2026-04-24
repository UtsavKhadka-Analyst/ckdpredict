<div align="center">

# 🩺 CKDPredict
### ML-Powered Chronic Kidney Disease Risk Stratification Platform

[![Live Demo](https://img.shields.io/badge/Live%20Demo-ckdpredict--web.vercel.app-0D9488?style=for-the-badge&logo=vercel)](https://ckdpredict-web.vercel.app)
[![API Docs](https://img.shields.io/badge/API%20Docs-FastAPI%20Swagger-009688?style=for-the-badge&logo=fastapi)](https://ckdpredict-api.onrender.com/docs)
[![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![React](https://img.shields.io/badge/React-18-61DAFB?style=for-the-badge&logo=react&logoColor=black)](https://react.dev)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0-FF6600?style=for-the-badge)](https://xgboost.readthedocs.io)

**A production-grade clinical decision support system for nephrologists and population health teams.**  
Built with XGBoost · FastAPI · React · Deployed on Vercel + Render

[**→ Try the Live App**](https://ckdpredict-web.vercel.app) · [**→ API Reference**](https://ckdpredict-api.onrender.com/docs) · [**→ Demo Credentials](#demo-credentials)**

</div>

---

## 🎯 What It Does

CKDPredict gives clinicians a **real-time risk registry** of their CKD patient population — powered by a two-model XGBoost ensemble that assigns each patient a risk score, urgency tier, projected cost, and KDIGO 2024 clinical guidance — all in a single dashboard.

> **Why it matters:** 37 million Americans have CKD and 90% are undiagnosed until late stages. Early intervention delivers a **30:1 ROI** — roughly $70,000 saved per patient. CKDPredict makes that early identification actionable.

---

## ✨ Key Features

| Feature | Description |
|--------|-------------|
| 🤖 **XGBoost Ensemble** | Two specialized models — Model A (Diabetic, 22 features) and Model B (Non-Diabetic, 26 features) — with **99% AUROC** and **93.5% sensitivity** |
| 📋 **Patient Registry** | 22,990-patient table with 9 sortable columns, 7 active filters (tier, model, state, city, gender, age, risk score), and color-coded WCAG-AA risk badges |
| 🏥 **KDIGO 2024 Guidance** | Click any patient → slide-out panel with tier-specific clinical recommendations: Action, Medications, Labs, Targets |
| 💰 **Cost Avoidance Model** | $70K/patient cost baseline with interactive what-if slider — projects real-time savings at any intervention rate |
| 🗺️ **Geographic Analysis** | State and city distribution maps for targeted outreach and resource planning |
| 📊 **Analytics Dashboard** | Population-level KPI cards, tier distribution charts, risk score histograms |
| 📤 **Outreach Engine** | Prioritized patient list sorted by urgency for nurse follow-up campaigns |
| ✅ **Model Validity Page** | KFRE concordance (0.94), feature importance (SHAP), fairness audit by gender/cohort |
| 🔒 **Role-Based Auth** | JWT authentication · Admin and Nephrologist roles · Patient portal with Epic-style activation code enrollment |

---

## 📸 Screenshots

| Registry — Risk Tiers | KDIGO Clinical Guidance | Cost Avoidance Model |
|---|---|---|
| Color-coded URGENT/HIGH/MODERATE/LOW | Per-patient slide-out drawer | What-if scenario slider |

> 💡 *[Add screenshots here once captured from the live app]*

---

## 🧠 ML Architecture

```
Patient Record (22–26 clinical features)
        │
        ▼
┌─────────────────────┐
│   Diabetes Router   │  ← HbA1c present → Model A
│                     │  ← HbA1c absent  → Model B
└─────────────────────┘
        │
   ┌────┴────┐
   ▼         ▼
Model A    Model B
Diabetic   Non-Diabetic
22 feat.   26 feat.
   │         │
   └────┬────┘
        ▼
  Risk Score (0–1)
  Urgency Tier (URGENT/HIGH/MODERATE/LOW)
  Timeline (est. months to progression)
  Projected Cost (USD)
```

**Key clinical features:** eGFR, UACR, HbA1c, Systolic BP, Hemoglobin, BMI, Age, Race, Gender

**Training:** 75% / 25% train-test split · Validated against Kidney Failure Risk Equation (Tangri et al. 2011)

---

## 📊 Model Performance

| Metric | Model A (Diabetic) | Model B (Non-Diabetic) | Charter Target |
|--------|-------------------|----------------------|----------------|
| **AUROC** | 99.02% | 99.58% | ≥ 82% ✅ |
| **Sensitivity** | 93.54% | 94.12% | ≥ 75% ✅ |
| **Specificity** | 98.71% | 97.93% | — |
| **Accuracy** | 95.2% | 96.1% | — |
| **KFRE Concordance** | 0.94 | 0.91 | Benchmark ✅ |

> **Note on high AUC:** Synthea generates synthetic patients with deterministic disease patterns — high separation is expected. Clinical validity is confirmed via KFRE concordance and SHAP-verified feature importance (eGFR ranked #1 predictor).

---

## 🗄️ Data

- **Source:** [Synthea™](https://synthetichealth.github.io/synthea/) v3 — fully synthetic, HIPAA-safe patient records
- **Populations:** Massachusetts (17,615) · Texas (5,375) · **Total: 22,990 patients**
- **No real PHI** used at any stage of development

---

## 🏗️ Tech Stack

**Frontend**
- React 18 + Vite
- Tailwind CSS v3
- React Router v6
- Recharts (analytics charts)
- Lucide React (icons)

**Backend**
- FastAPI (Python 3.11)
- SQLAlchemy + SQLite / PostgreSQL
- JWT authentication (python-jose)
- Pydantic v2 schemas
- Pandas + Scikit-learn + XGBoost 2.0
- Groq API (AI chat assistant)

**Infrastructure**
- Frontend → **Vercel**
- Backend → **Render**
- CI/CD → GitHub (auto-deploy on push)

---

## 🚀 Quick Start (Local)

### Prerequisites
- Python 3.11+
- Node.js 18+

### Backend
```bash
cd backend
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
python seed.py          # seeds admin user + test data
uvicorn main:app --reload --port 8000
# → http://localhost:8000/docs
```

### Frontend
```bash
cd frontend
npm install
npm run dev
# → http://localhost:5173
```

### Environment Variables

**Backend** (`.env`):
```env
SECRET_KEY=your-secret-key-32-chars-min
DATABASE_URL=sqlite:///./ckdpredict.db   # or PostgreSQL URL
FRONTEND_URL=http://localhost:5173
```

**Frontend** (`.env.local`):
```env
VITE_API_URL=http://localhost:8000
```

---

## 🔑 Demo Credentials

| Role | Username | Password | Access |
|------|----------|----------|--------|
| **Administrator** | `admin` | `Admin@CKD2024` | Full app — registry, analytics, cost model, export |
| **Nephrologist** | `dr_smith` | `Neph@CKD2024` | Registry + patient records |
| **Patient Portal** | Via activation code | — | Personal dashboard only |

---

## 📁 Project Structure

```
ckdpredict-web/
├── backend/
│   ├── core/
│   │   ├── deps.py          # Auth dependency injection
│   │   ├── ml.py            # ML model loading + registry
│   │   └── security.py      # JWT logic
│   ├── models/
│   │   ├── schemas.py        # Pydantic response schemas
│   │   ├── user.py           # SQLAlchemy ORM
│   │   └── patient_registry.csv   # 22,990 patient records
│   ├── routers/
│   │   ├── auth.py           # Login, register, patient portal
│   │   ├── registry.py       # Patient list, stats, CSV export
│   │   ├── predict.py        # Real-time risk prediction
│   │   ├── patient.py        # Patient-facing endpoints
│   │   └── chat.py           # Groq AI assistant
│   ├── main.py
│   ├── seed.py
│   └── requirements.txt
│
└── frontend/
    └── src/
        ├── components/
        │   ├── admin/
        │   │   ├── FilterBar.jsx           # 7-filter search bar
        │   │   ├── PatientGuidancePanel.jsx # KDIGO 2024 drawer
        │   │   ├── RegistryTable.jsx        # Sortable patient table
        │   │   └── StatsBar.jsx             # KPI cards
        │   ├── ChatAssistant.jsx            # Floating AI chat
        │   ├── Sidebar.jsx
        │   └── TierBadge.jsx
        ├── context/
        │   └── RegistryContext.jsx          # Global patient state
        ├── pages/
        │   ├── admin/
        │   │   ├── Analytics.jsx
        │   │   ├── CostModel.jsx
        │   │   ├── Geographic.jsx
        │   │   ├── ModelValidity.jsx
        │   │   └── Outreach.jsx
        │   ├── AdminDashboard.jsx
        │   ├── Login.jsx
        │   └── PatientPortal.jsx
        └── hooks/
            └── useAuth.jsx
```

---

## 📋 Requirements Traceability

All 25 project charter requirements are implemented and verified:

| Category | Requirements | Status |
|----------|-------------|--------|
| ML Performance & Architecture | AUROC ≥ 82%, Sensitivity ≥ 75%, Dual model routing, SHAP explainability | ✅ All Met |
| Patient Registry | Sortable table, color-coded tiers, 7 filters, patient names, projected cost | ✅ All Met |
| Geographic Distribution | State/city maps, multi-state patient data | ✅ All Met |
| Cost Avoidance Model | $70K baseline, what-if slider, 30:1 ROI | ✅ All Met |
| Clinical Guidance | KDIGO 2024 per-patient recommendations | ✅ All Met |
| Model Validity & Fairness | KFRE concordance, SHAP, demographic fairness audit | ✅ All Met |
| Security & Compliance | JWT, role-based access, HIPAA notice, patient portal | ✅ All Met |
| Infrastructure | Vercel + Render, live deployment, synthetic data only | ✅ All Met |

---

## 👥 Team

| Name | Role |
|------|------|
| **Utsav Khadka** | Full-Stack Development · ML Pipeline · Deployment |
| **Sharan Nambiar** | Data Engineering · Analytics · Synthea Pipeline |
| **Vamsidhar Balaji** | ML Modeling · Model Validation · Cost Analysis |
| **Sai Praveen Choutapally** | Clinical Research · KDIGO Integration · Documentation |

**Course:** MS Analytics — Major Research Project · Saint Louis University · Spring 2026  
**Advisor:** Saint Louis University School of Science and Engineering

---

## 📄 License

This project uses **synthetic data only** (Synthea v3). No real patient health information was used or stored at any stage.

For academic and research purposes — Saint Louis University MS Analytics MRP 2026.

---

<div align="center">

**Built with ❤️ for better clinical outcomes**

[Live App](https://ckdpredict-web.vercel.app) · [API Docs](https://ckdpredict-api.onrender.com/docs) · [Saint Louis University](https://www.slu.edu)

</div>
