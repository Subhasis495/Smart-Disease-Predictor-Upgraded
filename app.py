import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Disease Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Load models ───────────────────────────────────────────────────────────────
BASE = Path(__file__).parent / "models"

@st.cache_resource
def load_models():
    rf  = joblib.load(BASE / "rf_model.pkl")
    svm = joblib.load(BASE / "svm_model.pkl")
    le  = joblib.load(BASE / "label_encoder.pkl")
    with open(BASE / "metadata.json") as f:
        meta = json.load(f)
    with open(BASE / "disease_info.json") as f:
        disease_info = json.load(f)
    return rf, svm, le, meta, disease_info

rf_model, svm_model, label_encoder, metadata, disease_info = load_models()
SYMPTOMS = metadata["feature_cols"]
DISEASES  = metadata["diseases"]

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ---------- global ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* gradient hero banner */
.hero {
    background: linear-gradient(135deg, #0f4c81 0%, #1a8fe3 50%, #00c6ff 100%);
    border-radius: 16px;
    padding: 2.2rem 2.5rem 2rem;
    margin-bottom: 1.5rem;
    color: white;
    box-shadow: 0 8px 32px rgba(26,143,227,.30);
}
.hero h1 { font-size: 2.4rem; font-weight: 700; margin:0 0 .3rem; }
.hero p  { font-size: 1.05rem; opacity:.88; margin:0; }

/* metric cards */
.metric-card {
    background: white;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    text-align: center;
    box-shadow: 0 2px 12px rgba(0,0,0,.07);
    border-top: 4px solid #1a8fe3;
}
.metric-card .value { font-size:2rem; font-weight:700; color:#0f4c81; }
.metric-card .label { font-size:.85rem; color:#666; margin-top:.2rem; }

/* result card */
.result-card {
    background: linear-gradient(135deg, #e8f5e9, #f1f8e9);
    border-left: 5px solid #43a047;
    border-radius: 10px;
    padding: 1.4rem 1.6rem;
    margin: 1rem 0;
    color: #ff4b4b;
}
.result-card.warning {
    background: linear-gradient(135deg, #fff3e0, #fbe9e7);
    border-left-color: #ef6c00;
}
.result-card.danger {
    background: linear-gradient(135deg, #fce4ec, #f3e5f5);
    border-left-color: #c62828;
}
.result-card h2 { margin:0 0 .4rem; color:#ff4b4b; font-size:1.5rem; }
.result-card.warning h2 { color:#ff4b4b; }
.result-card.danger  h2 { color:#ff4b4b; }

/* symptom tags */
.symptom-tag {
    display:inline-block;
    background:#e3f2fd;
    color:#1565c0;
    border-radius:20px;
    padding:.25rem .8rem;
    margin:.25rem .2rem;
    font-size:.82rem;
    font-weight:500;
}

/* precaution pill */
.precaution {
    background: #f0f4ff;
    border-left: 3px solid #3f51b5;
    border-radius: 6px;
    padding: .55rem 1rem;
    margin: .4rem 0;
    font-size:.9rem;
    color:#283593;
}

/* confidence bar */
.conf-bar-bg {
    background:#e0e0e0; border-radius:8px; height:12px; overflow:hidden; margin-top:.3rem;
}
.conf-bar { height:100%; border-radius:8px; }

/* section headers */
.section-header {
    font-size:1.1rem; font-weight:600; color:#0f4c81;
    border-bottom:2px solid #e3f2fd; padding-bottom:.4rem; margin:1.2rem 0 .8rem;
}

/* sidebar style */
[data-testid="stSidebar"] { background: #0c1361; }
</style>
""", unsafe_allow_html=True)

# ── Hero ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1>🏥 Smart Disease Predictor</h1>
    <p>AI-powered symptom analysis using Random Forest & SVM — 41 diseases · 132 symptoms · 98%+ accuracy</p>
</div>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Settings")

    model_choice = st.selectbox(
        "Select ML Model",
        ["Random Forest", "SVM", "Ensemble (Both)"],
        index=2,
        help="Random Forest: robust & interpretable\nSVM: high accuracy on clean data\nEnsemble: combines both"
    )

    st.markdown("---")
    st.markdown("### 📊 Model Performance")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Random Forest", f"{metadata['rf_accuracy']*100:.1f}%", "Accuracy")
    with col2:
        st.metric("SVM", f"{metadata['svm_accuracy']*100:.1f}%", "Accuracy")

    st.markdown("---")
    st.markdown("### ℹ️ About")
    st.markdown("""
    This tool uses **Machine Learning** trained on a clinical symptom dataset with:
    - 🔬 **4920** training samples
    - 🦠 **41** diseases
    - 🩺 **132** symptoms
    
    > ⚠️ *For educational purposes only. Always consult a qualified doctor.*
    """)

# ── Main tabs ─────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🩺 Predict Disease", "📋 Disease Explorer", "📈 Model Insights"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Predictor
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.markdown('<div class="section-header">Select Your Symptoms</div>', unsafe_allow_html=True)
    st.markdown("Search and select all symptoms you are currently experiencing:")

    # Group symptoms alphabetically for better UX
    clean_symptoms = [s.strip().replace("_", " ").title() for s in SYMPTOMS]
    symptom_map = {clean: raw for clean, raw in zip(clean_symptoms, SYMPTOMS)}

    selected_clean = st.multiselect(
        "🔍 Type to search symptoms...",
        options=sorted(clean_symptoms),
        placeholder="e.g. Fever, Headache, Fatigue...",
        help="Select at least 3 symptoms for best accuracy"
    )

    # Show selected as pills
    if selected_clean:
        pills_html = "".join(f'<span class="symptom-tag">✓ {s}</span>' for s in selected_clean)
        st.markdown(f"**Selected ({len(selected_clean)}):** {pills_html}", unsafe_allow_html=True)

    col_btn1, col_btn2, _ = st.columns([1, 1, 4])
    with col_btn1:
        predict_btn = st.button("🔮 Predict Disease", type="primary", use_container_width=True)
    with col_btn2:
        clear_btn = st.button("🗑️ Clear All", use_container_width=True)

    # ── Prediction ────────────────────────────────────────────────────────────
    if predict_btn:
        if len(selected_clean) < 2:
            st.warning("⚠️ Please select at least **2 symptoms** for a meaningful prediction.")
        else:
            # Build feature vector
            feature_vec = np.zeros(len(SYMPTOMS))
            for s_clean in selected_clean:
                raw = symptom_map[s_clean]
                idx = SYMPTOMS.index(raw)
                feature_vec[idx] = 1

            X = feature_vec.reshape(1, -1)

            # Predict
            with st.spinner("Analyzing symptoms..."):
                rf_proba  = rf_model.predict_proba(X)[0]
                svm_proba = svm_model.predict_proba(X)[0]

                if model_choice == "Random Forest":
                    final_proba = rf_proba
                elif model_choice == "SVM":
                    final_proba = svm_proba
                else:  # Ensemble
                    final_proba = (rf_proba + svm_proba) / 2

                top_indices   = np.argsort(final_proba)[::-1][:5]
                top_diseases  = [label_encoder.classes_[i] for i in top_indices]
                top_probs     = [final_proba[i] for i in top_indices]

            pred_disease = top_diseases[0]
            confidence   = top_probs[0] * 100
            info = disease_info.get(pred_disease, {})
            severity = info.get("severity", "Unknown")

            # Severity → card class
            card_class = ""
            if severity in ("Serious", "Critical"):
                card_class = " danger"
            elif severity == "Moderate":
                card_class = " warning"

            sev_emoji = {"Mild": "🟢", "Moderate": "🟡", "Serious": "🟠", "Critical": "🔴"}.get(severity, "⚪")

            st.markdown(f"""
            <div class="result-card{card_class}">
                <h2>🦠 {pred_disease}</h2>
                <p style="margin:.2rem 0; font-size:.95rem;">{info.get('description','')}</p>
                <p style="margin:.5rem 0 0; font-weight:600;">
                    {sev_emoji} Severity: {severity} &nbsp;|&nbsp;
                    🎯 Confidence: {confidence:.1f}%
                </p>
            </div>
            """, unsafe_allow_html=True)

            # Precautions
            if info.get("precautions"):
                st.markdown('<div class="section-header">🛡️ Recommended Precautions</div>', unsafe_allow_html=True)
                for p in info["precautions"]:
                    st.markdown(f'<div class="precaution">✅ {p}</div>', unsafe_allow_html=True)

            # Top 5
            st.markdown('<div class="section-header">📊 Top 5 Possible Diagnoses</div>', unsafe_allow_html=True)
            for disease, prob in zip(top_diseases, top_probs):
                pct = prob * 100
                bar_color = "#43a047" if disease == pred_disease else "#1a8fe3"
                info_d = disease_info.get(disease, {})
                sev    = info_d.get("severity", "")
                sev_e  = {"Mild": "🟢", "Moderate": "🟡", "Serious": "🟠", "Critical": "🔴"}.get(sev, "")
                st.markdown(f"""
                <div style="margin-bottom:.8rem;">
                    <div style="display:flex;justify-content:space-between;font-size:.9rem;margin-bottom:.2rem;">
                        <span><b>{disease}</b> {sev_e}</span>
                        <span style="color:#555;">{pct:.1f}%</span>
                    </div>
                    <div class="conf-bar-bg">
                        <div class="conf-bar" style="width:{pct}%;background:{bar_color};"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            if severity in ("Serious", "Critical"):
                st.error("🚨 **High-risk condition detected!** Please seek immediate medical attention.")
            elif severity == "Moderate":
                st.warning("⚠️ **Please consult a healthcare provider** for proper diagnosis and treatment.")
            else:
                st.info("💡 These symptoms may indicate a mild condition. Monitor your health and rest well.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Disease Explorer
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown('<div class="section-header">🔍 Explore Disease Database</div>', unsafe_allow_html=True)

    col_search, col_filter = st.columns([2, 1])
    with col_search:
        search_term = st.text_input("Search disease...", placeholder="e.g. Diabetes, Malaria...")
    with col_filter:
        sev_filter = st.selectbox("Filter by Severity", ["All", "Mild", "Moderate", "Serious", "Critical"])

    # Filter diseases
    filtered = {}
    for name, info in disease_info.items():
        if search_term and search_term.lower() not in name.lower():
            continue
        if sev_filter != "All" and info.get("severity") != sev_filter:
            continue
        filtered[name] = info

    st.markdown(f"**Showing {len(filtered)} of {len(disease_info)} diseases**")

    for name, info in filtered.items():
        sev   = info.get("severity", "Unknown")
        sev_e = {"Mild": "🟢", "Moderate": "🟡", "Serious": "🟠", "Critical": "🔴"}.get(sev, "⚪")
        with st.expander(f"{sev_e} **{name}** — {sev}"):
            st.markdown(f"**Description:** {info.get('description','')}")
            if info.get("precautions"):
                st.markdown("**Precautions:**")
                for p in info["precautions"]:
                    st.markdown(f"- {p}")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Insights
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.markdown('<div class="section-header">📈 Model & Dataset Insights</div>', unsafe_allow_html=True)

    # Metric cards
    c1, c2, c3, c4 = st.columns(4)
    metrics = [
        ("4,920", "Training Samples"),
        ("41",    "Diseases Covered"),
        ("132",   "Symptom Features"),
        ("98%+",  "SVM Accuracy"),
    ]
    for col, (val, label) in zip([c1,c2,c3,c4], metrics):
        with col:
            st.markdown(f"""
            <div class="metric-card">
                <div class="value">{val}</div>
                <div class="label">{label}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("")

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 🌲 Random Forest vs SVM")
        comparison = pd.DataFrame({
            "Model": ["Random Forest", "SVM"],
            "Accuracy": [metadata['rf_accuracy']*100, metadata['svm_accuracy']*100],
            "Type": ["Ensemble", "Kernel-based"]
        })
        st.dataframe(comparison, use_container_width=True, hide_index=True)

        st.markdown("#### 🦠 Severity Distribution")
        sev_counts = {}
        for info in disease_info.values():
            sev = info.get("severity", "Unknown")
            sev_counts[sev] = sev_counts.get(sev, 0) + 1
        sev_df = pd.DataFrame(list(sev_counts.items()), columns=["Severity", "Count"])
        st.bar_chart(sev_df.set_index("Severity"))

    with col_r:
        st.markdown("#### 📋 Algorithm Details")
        st.markdown("""
        **Random Forest Classifier**
        - 200 decision trees (n_estimators=200)
        - Feature importance via Gini impurity
        - Bootstrap aggregation (bagging)
        - Out-of-bag error estimation

        **Support Vector Machine (SVM)**
        - RBF (Radial Basis Function) kernel
        - Probability calibration enabled
        - Excellent generalization on structured data
        - Finds optimal hyperplane between classes

        **Ensemble Method**
        - Averages probability outputs from both
        - Reduces variance & bias
        - More robust predictions overall
        """)

        st.markdown("#### 🩺 Top Symptom Features")
        importances = rf_model.feature_importances_
        top_idx = np.argsort(importances)[::-1][:10]
        feat_df = pd.DataFrame({
            "Symptom": [SYMPTOMS[i].replace("_", " ").title() for i in top_idx],
            "Importance": [round(importances[i]*100, 2) for i in top_idx]
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#888; font-size:.82rem;'>"
    "⚠️ <i>For educational purposes only — not a substitute for professional medical advice</i>"
    "</p>",
    unsafe_allow_html=True
)
