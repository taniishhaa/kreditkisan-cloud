"""
KreditKisan — Dynamic Microloan Pricing Engine
Streamlit UI | Clean, Presentation-Ready
Run with: streamlit run kreditkisan_app.py
"""

import streamlit as st
import pandas as pd
import joblib

# ──────────────────────────────────────────────────────────
# PAGE CONFIG — must be first Streamlit call
# ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="KreditKisan · Pricing Engine",
    page_icon="🌾",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────
# GLOBAL CSS — refined earthy fintech palette
# ──────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* ── Google Fonts ── */
  @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=DM+Sans:wght@400;500;600&display=swap');

  /* ── Root tokens ── */
  :root {
    --cream:   #faf6f0;
    --brown:   #5c3d1e;
    --gold:    #c9973a;
    --green:   #2d6a4f;
    --orange:  #e07b39;
    --red:     #b5341f;
    --slate:   #3d3d3d;
    --muted:   #8a7e72;
    --border:  #e8ddd0;
    --card-bg: #ffffff;
  }

  /* ── App shell ── */
  .stApp { background: var(--cream); }
  .block-container { max-width: 780px; padding: 2.5rem 2rem 4rem; }

  /* ── Hide Streamlit chrome ── */
  #MainMenu, footer, header { visibility: hidden; }

  /* ── Global typography ── */
  html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: var(--slate);
  }

  /* ── Hero banner ── */
  .hero {
    background: linear-gradient(135deg, #5c3d1e 0%, #3a2510 100%);
    border-radius: 18px;
    padding: 2.6rem 2.4rem 2.2rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
  }
  .hero::before {
    content: "🌾";
    position: absolute;
    right: 2rem;
    top: 50%;
    transform: translateY(-50%);
    font-size: 5rem;
    opacity: 0.15;
  }
  .hero h1 {
    font-family: 'Playfair Display', serif;
    color: #fff;
    font-size: 2.1rem;
    margin: 0 0 0.4rem;
    line-height: 1.2;
  }
  .hero p {
    color: #d4bfa0;
    font-size: 0.97rem;
    margin: 0;
  }
  .hero .badge {
    display: inline-block;
    background: rgba(201,151,58,0.25);
    border: 1px solid rgba(201,151,58,0.5);
    color: #f0c97a;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    border-radius: 20px;
    padding: 3px 12px;
    margin-bottom: 0.8rem;
  }

  /* ── Section header ── */
  .section-label {
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: var(--muted);
    margin: 1.8rem 0 0.8rem;
  }

  /* ── Input card ── */
  .input-card {
    background: var(--card-bg);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 1.6rem 1.8rem;
    margin-bottom: 0.5rem;
    box-shadow: 0 1px 6px rgba(92,61,30,0.06);
  }

  /* ── Streamlit widget labels ── */
  label { font-weight: 500 !important; font-size: 0.88rem !important; color: #000000 !important; }

  /* ── Primary button ── */
  .stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #5c3d1e, #8b5e3c) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    padding: 0.75rem 1.5rem !important;
    letter-spacing: 0.02em !important;
    cursor: pointer !important;
    transition: opacity 0.2s !important;
    margin-top: 0.8rem;
  }
  .stButton > button:hover { opacity: 0.88 !important; }

  /* ── Result card ── */
  .result-card {
    border-radius: 16px;
    padding: 2rem 2.2rem;
    margin-top: 1.8rem;
    border: 2px solid transparent;
  }
  .result-card.low      { background: #f0faf4; border-color: #2d6a4f; }
  .result-card.medium   { background: #fffbf0; border-color: #c9973a; }
  .result-card.high     { background: #fff7f0; border-color: #e07b39; }
  .result-card.veryhigh { background: #fff1f0; border-color: #b5341f; }

  .result-rate {
    font-family: 'Playfair Display', serif;
    font-size: 3.2rem;
    font-weight: 700;
    line-height: 1;
    margin: 0;
  }
  .result-card.low      .result-rate { color: #2d6a4f; }
  .result-card.medium   .result-rate { color: #c9973a; }
  .result-card.high     .result-rate { color: #e07b39; }
  .result-card.veryhigh .result-rate { color: #b5341f; }

  .result-label {
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    opacity: 0.65;
    margin-bottom: 0.3rem;
  }

  .tier-pill {
    display: inline-block;
    border-radius: 20px;
    padding: 4px 16px;
    font-size: 0.82rem;
    font-weight: 700;
    letter-spacing: 0.05em;
    margin-top: 0.6rem;
  }
  .result-card.low      .tier-pill { background:#2d6a4f; color:#fff; }
  .result-card.medium   .tier-pill { background:#c9973a; color:#fff; }
  .result-card.high     .tier-pill { background:#e07b39; color:#fff; }
  .result-card.veryhigh .tier-pill { background:#b5341f; color:#fff; }

  .result-msg {
    margin-top: 1rem;
    padding-top: 1rem;
    border-top: 1px solid rgba(0,0,0,0.07);
    font-size: 0.92rem;
    color: var(--slate);
    line-height: 1.6;
  }

  /* ── Comparison row ── */
  .compare-row {
    display: flex;
    gap: 1rem;
    margin-top: 1rem;
  }
  .compare-box {
    flex: 1;
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
    background: #f5f0e8;
    border: 1px solid var(--border);
  }
  .compare-box .cb-label { font-size: 0.72rem; font-weight: 600; text-transform: uppercase;
                           letter-spacing: 0.08em; color: var(--muted); }
  .compare-box .cb-val   { font-size: 1.5rem; font-weight: 700; color: var(--brown);
                           font-family: 'Playfair Display', serif; }
  .compare-box .cb-sub   { font-size: 0.75rem; color: var(--muted); }

  /* ── Tip strip ── */
  .tip-strip {
    background: #f5f0e8;
    border-left: 3px solid var(--gold);
    border-radius: 0 10px 10px 0;
    padding: 0.9rem 1.1rem;
    font-size: 0.88rem;
    color: var(--brown);
    margin-top: 1.4rem;
    line-height: 1.6;
  }

  /* ── Footer ── */
  .footer {
    text-align: center;
    color: var(--muted);
    font-size: 0.78rem;
    margin-top: 3rem;
    padding-top: 1.5rem;
    border-top: 1px solid var(--border);
  }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────
# LOAD MODEL
# ──────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    return joblib.load("kreditkisan_model.pkl")

try:
    model = load_model()
    model_loaded = True
except Exception as e:
    model_loaded = False
    model_error = str(e)


# ──────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────
FLAT_RATE = 18.0

TIER_CONFIG = {
    "Low": {
        "css": "low",
        "icon": "🟢",
        "msg": (
            "This applicant presents a <strong>strong risk profile</strong>. "
            "A competitive rate improves financial inclusion and builds long-term "
            "customer loyalty for KreditKisan."
        ),
    },
    "Medium": {
        "css": "medium",
        "icon": "🟡",
        "msg": (
            "This applicant shows a <strong>balanced risk profile</strong>. "
            "The rate reflects moderate exposure — standard documentation and "
            "regular repayment monitoring is recommended."
        ),
    },
    "High": {
        "css": "high",
        "icon": "🟠",
        "msg": (
            "This applicant carries <strong>elevated risk</strong>. "
            "Consider requesting additional collateral or a co-signer. "
            "The higher rate compensates for the increased probability of default."
        ),
    },
    "Very High": {
        "css": "veryhigh",
        "icon": "🔴",
        "msg": (
            "This applicant presents <strong>significant credit risk</strong>. "
            "The premium rate reflects this exposure. A detailed business assessment "
            "and enhanced monitoring schedule is strongly advised."
        ),
    },
}

def assign_risk_tier(rate: float) -> str:
    if rate < 13.5:   return "Low"
    elif rate < 16.55: return "Medium"
    elif rate < 19.3:  return "High"
    else:              return "Very High"

def savings_vs_flat(predicted: float, loan_amount: float) -> tuple:
    """Returns (model_annual_interest, flat_annual_interest, delta)"""
    model_int = predicted / 100 * loan_amount
    flat_int  = FLAT_RATE  / 100 * loan_amount
    return model_int, flat_int, flat_int - model_int  # positive = borrower saves


# ──────────────────────────────────────────────────────────
# HERO
# ──────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <div class="badge">Powered by Gradient Boosting · ML-Driven Pricing</div>
  <h1>KreditKisan Pricing Engine</h1>
  <p>Dynamic, risk-based interest rates for microfinance applicants — moving beyond flat 18%.</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error(f"⚠️ Could not load model: `{model_error}`\n\nEnsure `kreditkisan_model.pkl` is in the same directory.")
    st.stop()


# ──────────────────────────────────────────────────────────
# INPUT FORM
# ──────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Applicant Profile</div>', unsafe_allow_html=True)
st.markdown('<div class="input-card">', unsafe_allow_html=True)

col1, col2 = st.columns(2)
with col1:
    age           = st.slider("Applicant Age", 18, 65, 32, help="Age of the loan applicant")
    business_type = st.selectbox("Business Type",
                                 ["Agriculture", "Retail", "Manufacturing", "Services", "Handicraft"])
    revenue       = st.number_input("Monthly Revenue (₹)", min_value=1_000, max_value=500_000,
                                    value=35_000, step=1_000,
                                    help="Average monthly business revenue in INR")

with col2:
    loan_amount      = st.number_input("Loan Amount Requested (₹)", min_value=5_000,
                                       max_value=500_000, value=1_50_000, step=5_000)
    collateral       = st.selectbox("Collateral Available", ["Yes", "No"])
    credit_score     = st.slider("Credit Score", 300, 900, 620,
                                 help="CIBIL score (300=poor, 900=excellent)")
    repayment_score  = st.slider("Repayment History Score", 0.0, 10.0, 5.5, step=0.1,
                                 help="Internal score: 0 = poor history, 10 = perfect")

st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# CALCULATE BUTTON
# ──────────────────────────────────────────────────────────
calculate = st.button("📊 Calculate Personalised Rate")

# ──────────────────────────────────────────────────────────
# OUTPUT
# ──────────────────────────────────────────────────────────
if calculate:
    input_df = pd.DataFrame([{
        "Applicant_Age"              : age,
        "Business_Type"              : business_type,
        "Monthly_Revenue_INR"        : revenue,
        "Loan_Amount_Requested_INR"  : loan_amount,
        "Collateral_Available"       : collateral,
        "Credit_Score"               : credit_score,
        "Repayment_History_Score"    : repayment_score,
    }])

    predicted_rate = round(float(model.predict(input_df)[0]), 2)
    risk_tier      = assign_risk_tier(predicted_rate)
    cfg            = TIER_CONFIG[risk_tier]
    model_int, flat_int, delta = savings_vs_flat(predicted_rate, loan_amount)

    css_class = cfg["css"]

    # ── Result card ──
    tier_display = risk_tier.replace(" ", "&nbsp;")
    st.markdown(f"""
    <div class="result-card {css_class}">
      <div style="display:flex; align-items:flex-start; justify-content:space-between; flex-wrap:wrap; gap:1rem;">
        <div>
          <div class="result-label">Recommended Interest Rate</div>
          <div class="result-rate">{predicted_rate}%</div>
          <span class="tier-pill">{cfg['icon']} {tier_display} Risk</span>
        </div>
        <div style="text-align:right;">
          <div class="result-label">Loan Amount</div>
          <div style="font-size:1.2rem;font-weight:700;color:var(--brown);">₹{loan_amount:,.0f}</div>
          <div style="font-size:0.8rem;color:var(--muted);margin-top:4px;">
            Est. annual interest<br>
            <strong style="color:var(--slate);">₹{model_int:,.0f}</strong>
          </div>
        </div>
      </div>
      <div class="result-msg">{cfg['msg']}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Comparison vs Flat 18% ──
    st.markdown('<div class="section-label" style="margin-top:1.6rem;">How this compares to flat 18% pricing</div>',
                unsafe_allow_html=True)

    delta_label = f"₹{abs(delta):,.0f} {'saved by borrower' if delta > 0 else 'extra cost'}"
    delta_color = "#2d6a4f" if delta > 0 else "#b5341f"

    st.markdown(f"""
    <div class="compare-row">
      <div class="compare-box">
        <div class="cb-label">This Rate</div>
        <div class="cb-val">{predicted_rate}%</div>
        <div class="cb-sub">ML model</div>
      </div>
      <div class="compare-box">
        <div class="cb-label">Flat Rate</div>
        <div class="cb-val">{FLAT_RATE}%</div>
        <div class="cb-sub">Old system</div>
      </div>
      <div class="compare-box">
        <div class="cb-label">Annual Δ</div>
        <div class="cb-val" style="color:{delta_color};font-size:1.2rem;">₹{abs(delta):,.0f}</div>
        <div class="cb-sub">{'↓ borrower saves' if delta > 0 else '↑ borrower pays more'}</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Contextual tip ──
    tips = {
        "Low":       "💡 <strong>Tip:</strong> Consider offering this applicant a loyalty rate or a higher loan ceiling to grow the relationship.",
        "Medium":    "💡 <strong>Tip:</strong> A 6-month review clause can help adjust the rate if repayment performance improves.",
        "High":      "💡 <strong>Tip:</strong> Suggest the applicant improve their repayment score over the next 6 months for a better rate at renewal.",
        "Very High": "💡 <strong>Tip:</strong> If loan approval is required, consider a smaller disbursal in two tranches with performance conditions.",
    }
    st.markdown(f'<div class="tip-strip">{tips[risk_tier]}</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  KreditKisan · Dynamic Pricing Engine &nbsp;·&nbsp;
  Gradient Boosting Regressor (R² = 0.83) &nbsp;·&nbsp;
  700-record training dataset &nbsp;·&nbsp;
  For internal use only
</div>
""", unsafe_allow_html=True)