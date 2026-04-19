"""
app.py — CreditIQ Kenya v2
Model  : Tuned Gradient Boosting (fair_gbm_kenya_v2.pkl)
Data   : FinAccess 2021 (CBK/KNBS/FSD Kenya) + real Kenyan financials
Features: 64 — ratio-only financial features, all 47 counties, M-Pesa signal
"""

import os
import pickle
import warnings

import joblib
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="CreditIQ Kenya",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display&family=DM+Sans:wght@300;400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'DM Serif Display', serif; }
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0f1923 0%, #1a2f45 100%);
    border-right: 1px solid #2a4060;
}
[data-testid="stSidebar"] * { color: #cbd5e0 !important; }
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 { color: #f0f4f8 !important; }
.main { background: #f7f9fc; }
.block-container { padding-top: 2rem; padding-bottom: 3rem; }
.metric-card {
    background: white; border-radius: 16px; padding: 1.5rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.06);
    border: 1px solid #e8eef4; margin-bottom: 1rem;
}
.result-high {
    background: linear-gradient(135deg, #fff5f5, #ffe8e8);
    border: 2px solid #fc8181; border-radius: 20px;
    padding: 2rem; text-align: center;
}
.result-low {
    background: linear-gradient(135deg, #f0fff4, #dcfce7);
    border: 2px solid #68d391; border-radius: 20px;
    padding: 2rem; text-align: center;
}
.section-header {
    font-family: 'DM Serif Display', serif; font-size: 1.3rem;
    color: #1a2f45; border-bottom: 2px solid #e2e8f0;
    padding-bottom: 0.4rem; margin-bottom: 1.2rem;
}
.stButton > button {
    background: linear-gradient(135deg, #1a2f45, #2d5282) !important;
    color: white !important; border: none !important;
    border-radius: 12px !important; padding: 0.75rem 2rem !important;
    font-size: 1rem !important; font-weight: 600 !important; width: 100%;
}
[data-testid="metric-container"] {
    background: white; border-radius: 12px;
    padding: 1rem 1.2rem; border: 1px solid #e2e8f0;
}
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Exact features from model.feature_names_in_ (ratio-only model v2) ────────
# INCOME_log, SAVINGS_log, DEBT_log intentionally excluded —
# absolute values caused bias for high/low income borrowers
EXPECTED_FEATURES = [
    "age",
    "DEBT_to_INCOME_log", "SAVINGS_to_INCOME_log", "DEBT_to_SAVINGS_log",
    "education_Primary", "education_Secondary", "education_Tertiary",
    "marital_status_Divorced/separated",
    "marital_status_Married/Living with partner",
    "marital_status_Single/Never Married",
    "marital_status_Widowed",
    "county_Baringo", "county_Bomet", "county_Bungoma", "county_Busia",
    "county_Elgeyo-Marakwet", "county_Embu", "county_Garissa",
    "county_Homabay", "county_Isiolo", "county_Kajiado", "county_Kakamega",
    "county_Kericho", "county_Kiambu", "county_Kilifi", "county_Kirinyaga",
    "county_Kisii", "county_Kisumu", "county_Kitui", "county_Kwale",
    "county_Laikipia", "county_Lamu", "county_Machakos", "county_Makueni",
    "county_Mandera", "county_Marsabit", "county_Meru", "county_Migori",
    "county_Mombasa", "county_Murang'a", "county_Nairobi City",
    "county_Nakuru", "county_Nandi", "county_Narok", "county_Nyamira",
    "county_Nyandarua", "county_Nyeri", "county_Samburu", "county_Siaya",
    "county_Taita-Taveta", "county_Tana River", "county_Tharaka-Nithi",
    "county_Trans Nzoia", "county_Turkana", "county_Uasin Gishu",
    "county_Vihiga", "county_Wajir", "county_West Pokot",
    "mpesa_status_Currently have", "mpesa_status_Never had",
    "mpesa_status_Used to have",
    "loan_status_Currently have", "loan_status_Never had",
    "loan_status_Used to have",
]

# ── Kenyan dropdown options ───────────────────────────────────────────────────
EDUCATION_OPTIONS = ["Primary", "Secondary", "Tertiary"]

MARITAL_OPTIONS = [
    "Married/Living with partner",
    "Single/Never Married",
    "Divorced/separated",
    "Widowed",
]

COUNTY_OPTIONS = sorted([
    "Nairobi City", "Mombasa", "Kisumu", "Nakuru", "Uasin Gishu",
    "Baringo", "Bomet", "Bungoma", "Busia", "Elgeyo-Marakwet",
    "Embu", "Garissa", "Homabay", "Isiolo", "Kajiado",
    "Kakamega", "Kericho", "Kiambu", "Kilifi", "Kirinyaga",
    "Kisii", "Kitui", "Kwale", "Laikipia", "Lamu",
    "Machakos", "Makueni", "Mandera", "Marsabit", "Meru",
    "Migori", "Murang'a", "Nandi", "Narok", "Nyamira",
    "Nyandarua", "Nyeri", "Samburu", "Siaya", "Taita-Taveta",
    "Tana River", "Tharaka-Nithi", "Trans Nzoia", "Turkana",
    "Vihiga", "Wajir", "West Pokot",
])

MPESA_OPTIONS = ["Currently have", "Never had", "Used to have"]
LOAN_OPTIONS  = ["Currently have", "Never had", "Used to have"]


# ── Feature engineering ───────────────────────────────────────────────────────
def build_features(income, savings, debt, age,
                   education, marital, county, mpesa, loan):
    """Replicate training pipeline exactly — ratio-only features (v2)."""
    eps = 1e-9
    dti = debt    / (income  + eps)
    sti = savings / (income  + eps)
    dts = debt    / (savings + eps)

    # Clip ratios at same threshold used in training (99th percentile ~4.85)
    CLIP = 4.85
    row = {
        "age":                   age,
        "DEBT_to_INCOME_log":    min(np.log1p(dti), CLIP),
        "SAVINGS_to_INCOME_log": min(np.log1p(sti), CLIP),
        "DEBT_to_SAVINGS_log":   min(np.log1p(dts), CLIP),
        f"education_{education}":    1,
        f"marital_status_{marital}": 1,
        f"county_{county}":          1,
        f"mpesa_status_{mpesa}":     1,
        f"loan_status_{loan}":       1,
    }
    return pd.DataFrame([{f: row.get(f, 0) for f in EXPECTED_FEATURES}])


def align_to_model(df, model):
    """model.feature_names_in_ is the single source of truth."""
    try:
        cols = list(model.feature_names_in_)
        for c in cols:
            if c not in df.columns:
                df[c] = 0
        return df[cols]
    except AttributeError:
        return df


def risk_band(prob):
    if prob < 0.30: return "Very Low",  "#2f855a", "🟢"
    if prob < 0.50: return "Low",       "#276749", "🟡"
    if prob < 0.65: return "Moderate",  "#c05621", "🟠"
    if prob < 0.80: return "High",      "#c53030", "🔴"
    return               "Very High",   "#742a2a", "🔴"


def gauge_chart(prob):
    fig, ax = plt.subplots(figsize=(5, 3), subplot_kw=dict(aspect="equal"))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    for lo, hi, col in [
        (0.00, 0.30, "#68d391"), (0.30, 0.50, "#f6e05e"),
        (0.50, 0.65, "#f6ad55"), (0.65, 0.80, "#fc8181"),
        (0.80, 1.00, "#9b2335"),
    ]:
        ax.add_patch(mpatches.Wedge((0, 0), 1.0, 180-hi*180, 180-lo*180,
            width=0.35, facecolor=col, edgecolor="white", linewidth=1.5))
    a = np.radians(180 - prob * 180)
    ax.annotate("", xy=(0.7*np.cos(a), 0.7*np.sin(a)), xytext=(0, 0),
        arrowprops=dict(arrowstyle="-|>", color="#1a2f45",
                        lw=2.5, mutation_scale=15))
    ax.plot(0, 0, "o", color="#1a2f45", markersize=8, zorder=5)
    ax.text(0, -0.38, f"{prob:.0%}", ha="center", fontsize=22,
            fontweight="bold", color="#1a2f45", fontfamily="serif")
    ax.text(0, -0.62, "Default Probability", ha="center",
            fontsize=8, color="#718096")
    ax.set_xlim(-1.2, 1.2); ax.set_ylim(-0.8, 1.15); ax.axis("off")
    plt.tight_layout(pad=0)
    return fig


def _label(s):
    return (s.replace("marital_status_", "marital: ")
             .replace("education_", "edu: ")
             .replace("county_", "county: ")
             .replace("mpesa_status_", "M-Pesa: ")
             .replace("loan_status_", "loan: ")
             .replace("_log", " (log)").replace("_", " "))


def importance_chart(model, feat_names, top_n=12):
    imps = model.feature_importances_
    idx  = np.argsort(imps)[::-1][:top_n]
    vals = imps[idx]
    labels = [_label(feat_names[i]) for i in idx]
    colors = ["#2d5282" if v == vals.max() else "#a0aec0" for v in vals]
    fig, ax = plt.subplots(figsize=(6, 5))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    ax.barh(range(top_n), vals[::-1], color=colors[::-1],
            edgecolor="none", height=0.65)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(labels[::-1], fontsize=9, color="#4a5568")
    ax.set_xlabel("Importance", fontsize=9, color="#718096")
    ax.tick_params(axis="x", colors="#a0aec0", labelsize=8)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.spines["bottom"].set_color("#e2e8f0")
    ax.set_title("Global Feature Importances", fontsize=11,
                 fontweight="bold", color="#1a2f45", pad=10)
    plt.tight_layout(); return fig


def contributions_chart(model, input_df, feat_names, top_n=10):
    """
    Show top features by importance for THIS borrower — 
    only features that are active (non-zero) for this prediction.
    """
    imps  = pd.Series(model.feature_importances_, index=feat_names)
    vals  = input_df.iloc[0]

    # Only show features that are non-zero for this borrower
    active = {f: imps.get(f, 0) for f in feat_names if abs(vals.get(f, 0)) > 0}
    if not active:
        active = {f: imps.get(f, 0) for f in feat_names}

    top = pd.Series(active).sort_values(ascending=False).head(top_n)
    labels = [_label(k) for k in top.index]

    # Financial ratio features increase risk when high — colour by feature type
    ratio_features = ["DEBT_to_SAVINGS_log", "DEBT_to_INCOME_log",
                      "DEBT_log", "SAVINGS_to_INCOME_log"]
    colors = []
    for k in top.index:
        if any(r in k for r in ["DEBT_to", "DEBT_log"]):
            colors.append("#fc8181")   # red — debt features increase risk
        else:
            colors.append("#68d391")   # green — savings/income features decrease risk

    fig, ax = plt.subplots(figsize=(6, 4))
    fig.patch.set_alpha(0); ax.set_facecolor("none")
    ax.barh(range(len(top)), top.values[::-1],
            color=colors[::-1], edgecolor="none", height=0.65)
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels[::-1], fontsize=9, color="#4a5568")
    ax.set_xlabel("Feature importance", fontsize=9, color="#718096")
    ax.tick_params(axis="x", colors="#a0aec0", labelsize=8)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.spines["bottom"].set_color("#e2e8f0")
    ax.set_title("Key Factors for This Borrower", fontsize=11,
                 fontweight="bold", color="#1a2f45", pad=10)
    ax.legend(
        handles=[mpatches.Patch(color="#fc8181", label="Debt factors"),
                 mpatches.Patch(color="#68d391", label="Income/savings factors")],
        fontsize=8, loc="lower right",
        framealpha=0.6, edgecolor="#e2e8f0")
    plt.tight_layout(); return fig


@st.cache_resource
def load_model(path="fair_gbm_kenya_v2.pkl"):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        with open(path, "rb") as f:
            return pickle.load(f)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🏦 CreditIQ Kenya")
    st.markdown("---")
    st.markdown("""
**About**

Credit risk scoring built for Kenyan microfinance institutions and digital lenders.

**Data Sources**
- 🇰🇪 FinAccess 2021 (CBK/KNBS/FSD Kenya)
- 📊 Real Kenyan financial records

**Model**
- 🌲 Gradient Boosting (tuned)
- ⚖️ Gender-fair (0.71pp gap)
- 🗺️ All 47 Kenyan counties
- 📱 M-Pesa usage signal

**Performance**

| Metric | Score |
|--------|-------|
| AUC-ROC | 76.7% |
| Accuracy | 69.4% |
| Recall | 61.4% |
| Gender gap | 0.71pp |
""")
    st.markdown("---")
    st.caption("MSc Data Science · Strathmore · 2025")
    st.caption("Regina Wanjiru Gathimba")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="background:linear-gradient(135deg,#0f1923,#1a3a5c);
            border-radius:20px;padding:2.5rem;margin-bottom:2rem;
            border:1px solid #2a4060">
  <h1 style="color:white;margin:0;font-size:2rem">CreditIQ Kenya</h1>
  <p style="color:#90cdf4;margin:.4rem 0 0;font-weight:300">
    AI-Powered Credit Risk Assessment · Built on Real Kenyan Data
  </p>
</div>
""", unsafe_allow_html=True)

model = load_model()
if model is None:
    st.warning("""
**Model not found.** Place `fair_gbm_kenya_v2.pkl` in the same folder as `app.py`.

Download it from Google Drive after running the training notebook.
    """)
    st.stop()

# ── Inputs ────────────────────────────────────────────────────────────────────
st.markdown("<div class='section-header'>📋 Borrower Information</div>",
            unsafe_allow_html=True)

col_l, col_r = st.columns(2, gap="large")

with col_l:
    st.markdown("**Financial Profile**")
    income  = st.number_input("Monthly Income (KES)",
                min_value=0, max_value=5_000_000, value=50_000, step=5_000)
    savings = st.number_input("Total Savings (KES)",
                min_value=0, max_value=5_000_000, value=80_000, step=5_000)
    debt    = st.number_input("Total Outstanding Debt (KES)",
                min_value=0, max_value=5_000_000, value=40_000, step=5_000)

    eps = 1e-9
    dti = debt / (income + eps)
    sti = savings / (income + eps)
    st.markdown("**Live Ratios**")
    r1, r2 = st.columns(2)
    r1.metric("Debt / Income",    f"{dti:.2f}",
              help="Above 0.5 = elevated risk")
    r2.metric("Savings / Income", f"{sti:.2f}",
              help="Higher = stronger buffer")

    st.markdown("**Financial Behaviour**")
    mpesa = st.selectbox("M-Pesa Status",  MPESA_OPTIONS,
                         help="Current M-Pesa account status")
    loan  = st.selectbox("Loan History",   LOAN_OPTIONS,
                         help="Current or past loan experience")

with col_r:
    st.markdown("**Demographics**")
    age       = st.slider("Age", 18, 75, 32)
    education = st.selectbox("Education Level", EDUCATION_OPTIONS)
    marital   = st.selectbox("Marital Status",  MARITAL_OPTIONS)
    county    = st.selectbox("County",          COUNTY_OPTIONS,
                             index=COUNTY_OPTIONS.index("Nairobi City"))

st.markdown("<br>", unsafe_allow_html=True)
run = st.button("🔍  Assess Credit Risk", use_container_width=True)

# ── Prediction ────────────────────────────────────────────────────────────────
if run:
    raw_df   = build_features(income, savings, debt, age,
                               education, marital, county, mpesa, loan)
    input_df = align_to_model(raw_df, model)
    feat_names = list(input_df.columns)

    pred = int(model.predict(input_df)[0])
    try:
        prob = float(model.predict_proba(input_df)[0][1])
    except Exception:
        prob = float(pred)

    band, color, emoji = risk_band(prob)

    st.markdown("---")
    st.markdown("<div class='section-header'>📊 Assessment Result</div>",
                unsafe_allow_html=True)

    res_col, gauge_col = st.columns(2, gap="large")

    with res_col:
        is_high  = pred == 1
        card_cls = "result-high" if is_high else "result-low"
        headline = "High Credit Risk"  if is_high else "Low Credit Risk"
        subtext  = ("Borrower is likely to default"
                    if is_high else "Borrower is likely to repay")
        h_col = "#c53030" if is_high else "#2f855a"
        s_col = "#9b2335" if is_high else "#276749"

        st.markdown(f"""
        <div class='{card_cls}'>
          <div style='font-size:3rem;margin-bottom:.5rem'>
            {'⚠️' if is_high else '✅'}
          </div>
          <h2 style='color:{h_col};margin:0'>{headline}</h2>
          <p style='color:{s_col};margin:.5rem 0 0'>{subtext}</p>
          <div style='margin-top:1rem'>
            <span style='background:{"#fff5f5" if is_high else "#f0fff4"};
                         color:{color};border:1px solid {color};
                         padding:.4rem 1.2rem;border-radius:999px;
                         font-size:.88rem;font-weight:600'>
              {emoji} {band} Risk
            </span>
          </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        m1, m2, m3 = st.columns(3)
        m1.metric("Default Prob.", f"{prob:.1%}")
        m2.metric("Risk Band",     band)
        m3.metric("Debt/Income",   f"{dti:.2f}")

    with gauge_col:
        st.pyplot(gauge_chart(prob), use_container_width=True)

    # Explainability
    st.markdown("---")
    st.markdown("<div class='section-header'>🔎 Explainability</div>",
                unsafe_allow_html=True)
    e1, e2 = st.columns(2, gap="large")
    with e1:
        st.pyplot(importance_chart(model, feat_names),
                  use_container_width=True)
    with e2:
        st.pyplot(contributions_chart(model, input_df, feat_names),
                  use_container_width=True)

    # Recommendation
    st.markdown("---")
    st.markdown("<div class='section-header'>💡 Recommendation</div>",
                unsafe_allow_html=True)

    if is_high:
        st.markdown(f"""
        <div class='metric-card'>
          <h4 style='color:#c53030;margin-top:0'>🚦 Proceed with Caution</h4>
          <ul style='color:#4a5568;line-height:1.9'>
            <li>Debt-to-Income of <strong>{dti:.2f}</strong> —
                {'exceeds safe 0.5 threshold' if dti > 0.5
                 else 'approaching limit'}</li>
            <li>Consider requiring collateral or a co-guarantor</li>
            <li>Start with smaller disbursement and graduated repayment</li>
            <li>Recommend financial literacy programme before full approval</li>
          </ul>
        </div>""", unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class='metric-card'>
          <h4 style='color:#2f855a;margin-top:0'>✅ Eligible for Loan</h4>
          <ul style='color:#4a5568;line-height:1.9'>
            <li>Savings buffer of <strong>{sti:.2f}x</strong> monthly income —
                {'strong' if sti > 0.5 else 'adequate'}</li>
            <li>Standard loan terms applicable</li>
            <li>Suitable for microfinance or SME loan product</li>
            <li>Recommend quarterly repayment review</li>
          </ul>
        </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div style='background:#fffbeb;border:1px solid #f6e05e;
                border-radius:10px;padding:.8rem 1.2rem;
                margin-top:.8rem;font-size:.82rem;color:#744210'>
      ⚠️ <strong>Disclaimer:</strong> Research prototype —
      Strathmore University MSc 2025. All predictions must be
      reviewed by a qualified credit officer.
    </div>""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align:center;color:#a0aec0;font-size:.82rem;
            border-top:1px solid #e2e8f0;padding-top:1.5rem;margin-top:3rem'>
  CreditIQ Kenya · MSc Data Science · Strathmore University · 2025
  · Regina Wanjiru Gathimba
</div>""", unsafe_allow_html=True)
