import streamlit as st
import pandas as pd
import numpy as np
import re
import joblib
import json
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import io
import base64
from pathlib import Path

# ─── PAGE CONFIG ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MindGuard — Mental Health Risk Detector",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── COLOR PALETTE ──────────────────────────────────────────────────────────
PINK       = "#F7CBCA"   # soft pink  → high risk
MAUVE      = "#C9B8C8"   # muted mauve → secondary
WHITE_BG   = "#F1F7F7"   # near white → main background
PALE_TEAL  = "#D5E6E5"   # pale teal  → cards
MINT       = "#B0D7D6"   # soft mint  → low risk / positive
DARK_TEAL  = "#5D6B6B"   # deep teal-grey → sidebar, headers, primary text
MID_TEAL   = "#7A9E9F"   # mid accent

# ─── GLOBAL CSS ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
/* ── Base ── */
html, body, [class*="css"] {{
    font-family: 'Segoe UI', 'Inter', sans-serif;
    color: {DARK_TEAL};
}}
.stApp {{
    background-color: {WHITE_BG};
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] {{
    background: linear-gradient(160deg, {DARK_TEAL} 0%, #3e5454 100%) !important;
}}
[data-testid="stSidebar"] * {{
    color: {WHITE_BG} !important;
}}
[data-testid="stSidebar"] .stRadio label {{
    color: {WHITE_BG} !important;
    font-size: 15px;
    font-weight: 500;
}}
[data-testid="stSidebar"] hr {{
    border-color: {MINT}44;
}}

/* ── Buttons ── */
.stButton > button {{
    background: linear-gradient(135deg, {DARK_TEAL}, {MID_TEAL});
    color: white !important;
    border: none;
    border-radius: 25px;
    padding: 10px 28px;
    font-size: 15px;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s ease;
    box-shadow: 0 4px 15px {DARK_TEAL}55;
}}
.stButton > button:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 20px {DARK_TEAL}88;
}}

/* ── Text areas & inputs ── */
.stTextArea textarea {{
    background-color: white !important;
    border: 2px solid {PALE_TEAL} !important;
    border-radius: 12px !important;
    color: {DARK_TEAL} !important;
    font-size: 15px;
}}
.stTextArea textarea:focus {{
    border-color: {MID_TEAL} !important;
    box-shadow: 0 0 0 3px {MINT}55 !important;
}}

/* ── File uploader ── */
[data-testid="stFileUploader"] {{
    background: white;
    border: 2px dashed {PALE_TEAL};
    border-radius: 12px;
    padding: 20px;
}}

/* ── Metric boxes ── */
[data-testid="stMetric"] {{
    background: white;
    border-radius: 14px;
    padding: 16px;
    box-shadow: 0 2px 12px {DARK_TEAL}18;
    border-left: 4px solid {MINT};
}}

/* ── Divider ── */
hr {{
    border: none;
    border-top: 2px solid {PALE_TEAL};
    margin: 24px 0;
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 6px; }}
::-webkit-scrollbar-track {{ background: {WHITE_BG}; }}
::-webkit-scrollbar-thumb {{ background: {MINT}; border-radius: 10px; }}
</style>
""", unsafe_allow_html=True)


# ─── LOAD MODEL ─────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

@st.cache_data
def load_metrics():
    with open("metrics.json") as f:
        return json.load(f)

model, vectorizer = load_model()
metrics = load_metrics()


# ─── CORE FUNCTIONS ─────────────────────────────────────────────────────────
def preprocess_text(text: str) -> str:
    """Clean and normalize input text for model inference."""
    text = str(text).lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    stop_words = set(['the','a','an','and','or','but','in','on','at','to','for',
                      'of','with','by','from','is','was','are','were','be','been',
                      'being','have','has','had','do','does','did','will','would',
                      'could','should','may','might','i','me','my','we','you',
                      'your','he','him','his','she','her','they','them','it',
                      'this','that','not','no'])
    words = [w for w in text.split() if w not in stop_words and len(w) > 2]
    return ' '.join(words)


def predict_risk(text: str) -> dict:
    """Predict depression risk and return label + confidence."""

    processed = preprocess_text(text)

    # Pass RAW TEXT to pipeline model
    pred = model.predict([processed])[0]
    proba = model.predict_proba([processed])[0]

    confidence = proba[pred]

    return {
        "label": int(pred),
        "confidence": float(confidence),
        "risk_text": "HIGH RISK — Depression Detected" if pred == 1 else "LOW RISK — No Signs Detected",
        "color": PINK if pred == 1 else MINT,
        "emoji": "🔴" if pred == 1 else "🟢"
    }


def plot_confusion_matrix() -> plt.Figure:
    """Render a styled confusion matrix heatmap."""
    cm = np.array(metrics['confusion_matrix'])
    fig, ax = plt.subplots(figsize=(5, 4))
    fig.patch.set_facecolor(WHITE_BG)
    ax.set_facecolor(WHITE_BG)
    cmap = sns.color_palette([WHITE_BG, PALE_TEAL, MINT, DARK_TEAL], as_cmap=True)
    sns.heatmap(cm, annot=True, fmt='d', cmap='YlGnBu',
                xticklabels=['Non-Depression', 'Depression'],
                yticklabels=['Non-Depression', 'Depression'],
                linewidths=2, linecolor=WHITE_BG, ax=ax,
                annot_kws={"size": 14, "weight": "bold", "color": DARK_TEAL})
    ax.set_xlabel('Predicted', fontsize=12, color=DARK_TEAL, labelpad=10)
    ax.set_ylabel('Actual', fontsize=12, color=DARK_TEAL, labelpad=10)
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold', color=DARK_TEAL, pad=14)
    ax.tick_params(colors=DARK_TEAL)
    plt.tight_layout()
    return fig


def batch_predict(df_input: pd.DataFrame, text_col: str) -> pd.DataFrame:
    """Run predictions on a dataframe text column and return enriched results."""
    results = []
    for text in df_input[text_col]:
        r = predict_risk(str(text))
        results.append({
            "text": text[:120] + "..." if len(str(text)) > 120 else text,
            "prediction": "Depression" if r["label"] == 1 else "Non-Depression",
            "confidence": f"{r['confidence']*100:.1f}%",
            "risk_level": "HIGH" if r["label"] == 1 else "LOW"
        })
    return pd.DataFrame(results)


# ─── SIDEBAR NAVIGATION ─────────────────────────────────────────────────────
st.sidebar.markdown(f"""
<div style='text-align:center; padding: 20px 0 10px;'>
  <div style='font-size:40px;'>🧠</div>
  <div style='font-size:20px; font-weight:700; letter-spacing:1px; color:{WHITE_BG};'>MindGuard</div>
  <div style='font-size:11px; color:{MINT}; margin-top:4px; letter-spacing:2px;'>MENTAL HEALTH AI</div>
</div>
""", unsafe_allow_html=True)
st.sidebar.markdown("---")

pages = {
    "🏠  Home": "home",
    "🔍  Text Risk Analyzer": "analyzer",
    "📊  Model Insights": "insights",
    "📖  About Mental Health": "about",
    "📁  Batch Analyzer": "batch"
}
page_label = st.sidebar.radio("Navigate", list(pages.keys()), label_visibility="collapsed")
page = pages[page_label]

st.sidebar.markdown("---")
st.sidebar.markdown(f"""
<div style='font-size:11px; color:{MINT}; text-align:center; padding:10px; line-height:1.8;'>
  Mental Health Risk Detection<br>
  Accuracy: <b>{metrics['accuracy']*100:.1f}%</b>
</div>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE 1 — HOME
# ════════════════════════════════════════════════════════════════
if page == "home":
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {PALE_TEAL} 0%, {MINT}88 100%);
                border-radius: 24px; padding: 56px 40px; text-align: center; margin-bottom: 36px;
                box-shadow: 0 8px 32px {DARK_TEAL}22;'>
        <div style='font-size:64px; margin-bottom:12px;'>🧠</div>
        <h1 style='color:{DARK_TEAL}; font-size:42px; font-weight:800; margin:0; letter-spacing:-1px;'>
            MindGuard
        </h1>
        <p style='color:{DARK_TEAL}; font-size:18px; font-weight:500; margin:12px 0 0;
                  opacity:0.85; letter-spacing:0.5px;'>
            AI-Powered Mental Health Text Risk Detector
        </p>
        <div style='margin-top:16px; display:inline-block; background:{DARK_TEAL}18;
                    padding:8px 20px; border-radius:50px;'>
            <span style='color:{DARK_TEAL}; font-size:13px; font-weight:600;'>
            ✨ 95.9% Accuracy &nbsp;|&nbsp; 7,731 Reddit Posts Trained &nbsp;|&nbsp; Real-Time Analysis
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Mission
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown(f"""
        <div style='background:white; border-radius:18px; padding:32px;
                    box-shadow:0 4px 20px {DARK_TEAL}15; border-left:5px solid {MINT};'>
            <h2 style='color:{DARK_TEAL}; margin:0 0 12px;'>Our Mission</h2>
            <p style='color:{DARK_TEAL}; font-size:15px; line-height:1.8; margin:0; opacity:0.9;'>
                Mental health is one of the most underdiagnosed challenges of our time.
                <b>MindGuard</b> uses advanced Natural Language Processing to analyze text
                and detect early signs of depression — helping individuals, counselors, and
                researchers take timely, informed action.<br><br>
                Early detection saves lives. Our model, trained on real-world Reddit data,
                achieves <b>95.9% accuracy</b> in identifying depressive language patterns.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div style='background: linear-gradient(135deg, {PINK}88, {MAUVE}55);
                    border-radius:18px; padding:28px; text-align:center;
                    box-shadow:0 4px 20px {DARK_TEAL}15;'>
            <div style='font-size:36px;'>⚠️</div>
            <h3 style='color:{DARK_TEAL}; margin:10px 0 6px; font-size:16px;'>Disclaimer</h3>
            <p style='color:{DARK_TEAL}; font-size:12px; line-height:1.6; margin:0; opacity:0.85;'>
                This tool is for <b>educational & research</b> purposes only. It does not
                replace professional medical diagnosis or therapy.
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Feature cards
    st.markdown(f"<h2 style='color:{DARK_TEAL}; font-weight:700;'>What Can MindGuard Do?</h2>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    features = [
        ("🔍", "Text Analysis", "Paste any text and get instant risk prediction with confidence score."),
        ("📊", "Model Insights", "Explore accuracy, F1, confusion matrix and top keywords."),
        ("📖", "Mental Health Info", "Learn about depression signs and get helpline resources."),
        ("📁", "Batch Processing", "Upload CSV files to analyze multiple texts at once."),
    ]
    for col, (icon, title, desc) in zip([c1,c2,c3,c4], features):
        col.markdown(f"""
        <div style='background:white; border-radius:16px; padding:24px 18px; text-align:center;
                    box-shadow:0 3px 16px {DARK_TEAL}12; border-top:4px solid {PALE_TEAL};
                    transition:all 0.3s; height:100%;'>
            <div style='font-size:32px; margin-bottom:10px;'>{icon}</div>
            <h4 style='color:{DARK_TEAL}; margin:0 0 8px; font-size:15px;'>{title}</h4>
            <p style='color:{DARK_TEAL}; font-size:12px; line-height:1.6; margin:0; opacity:0.75;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {DARK_TEAL}, {MID_TEAL});
                border-radius:18px; padding:32px; text-align:center;'>
        <h3 style='color:white; margin:0 0 8px;'>Ready to Analyze?</h3>
        <p style='color:{MINT}; margin:0 0 20px; font-size:14px;'>
            Navigate to <b>Text Risk Analyzer</b> in the sidebar to get started.
        </p>
        <div style='font-size:28px;'>👈</div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE 2 — TEXT RISK ANALYZER
# ════════════════════════════════════════════════════════════════
elif page == "analyzer":
    st.markdown(f"<h1 style='color:{DARK_TEAL}; font-weight:800;'>🔍 Text Risk Analyzer</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{DARK_TEAL}; opacity:0.7; margin-top:-8px;'>Enter any text below to analyze for depression risk indicators.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col_input, col_result = st.columns([1, 1], gap="large")

    with col_input:
        st.markdown(f"<h3 style='color:{DARK_TEAL};'>📝 Enter Text</h3>", unsafe_allow_html=True)
        user_text = st.text_area(
            label="Input Text",
            placeholder="Type or paste a paragraph here — journal entry, social media post, message...",
            height=250,
            label_visibility="collapsed"
        )

        char_count = len(user_text)
        word_count = len(user_text.split()) if user_text.strip() else 0
        st.markdown(f"""
        <div style='display:flex; gap:16px; margin-top:8px;'>
            <span style='background:{PALE_TEAL}; color:{DARK_TEAL}; padding:4px 12px;
                         border-radius:20px; font-size:12px; font-weight:600;'>
                {char_count} characters
            </span>
            <span style='background:{PALE_TEAL}; color:{DARK_TEAL}; padding:4px 12px;
                         border-radius:20px; font-size:12px; font-weight:600;'>
                {word_count} words
            </span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        analyze_btn = st.button("🔍 Analyze Text", use_container_width=True)

        # Sample texts
        st.markdown(f"<p style='color:{DARK_TEAL}; font-size:12px; margin-top:16px; opacity:0.6;'>Try a sample:</p>", unsafe_allow_html=True)
        s1, s2 = st.columns(2)
        if s1.button("😔 Sample: At-risk", use_container_width=True):
            st.session_state['sample'] = "I feel so empty and hopeless every single day. Nothing brings me joy anymore. I don't see the point in anything and I just want to disappear."
        if s2.button("😊 Sample: Normal", use_container_width=True):
            st.session_state['sample'] = "Had a great day today! Went for a morning run, caught up with old friends and tried a new restaurant. Life feels really good right now."

        if 'sample' in st.session_state:
            st.info(f"Sample loaded ↑ — paste it in the text area above:\n\n*{st.session_state['sample']}*")

    with col_result:
        st.markdown(f"<h3 style='color:{DARK_TEAL};'>📋 Analysis Result</h3>", unsafe_allow_html=True)

        if analyze_btn and user_text.strip():
            if word_count < 5:
                st.warning("⚠️ Please enter at least 5 words for a meaningful analysis.")
            else:
                with st.spinner("Analyzing..."):
                    result = predict_risk(user_text)

                risk_bg = f"{PINK}55" if result["label"] == 1 else f"{MINT}55"
                border_c = PINK if result["label"] == 1 else MINT
                risk_icon = "🚨" if result["label"] == 1 else "✅"
                conf_pct = result["confidence"] * 100

                st.markdown(f"""
                <div style='background:{risk_bg}; border:2px solid {border_c};
                            border-radius:18px; padding:28px; text-align:center;
                            box-shadow: 0 6px 24px {border_c}44;'>
                    <div style='font-size:48px; margin-bottom:8px;'>{risk_icon}</div>
                    <h2 style='color:{DARK_TEAL}; margin:0; font-size:20px; font-weight:800;'>
                        {result["risk_text"]}
                    </h2>
                    <p style='color:{DARK_TEAL}; font-size:13px; margin:8px 0 0; opacity:0.7;'>
                        Model Confidence: <b>{conf_pct:.1f}%</b>
                    </p>
                </div>
                """, unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown(f"<p style='color:{DARK_TEAL}; font-weight:600; font-size:13px;'>Confidence Score</p>", unsafe_allow_html=True)
                st.progress(result["confidence"])

                st.markdown("<br>", unsafe_allow_html=True)
                if result["label"] == 1:
                    st.markdown(f"""
                    <div style='background:white; border-radius:14px; padding:20px;
                                border-left:4px solid {PINK};'>
                        <h4 style='color:{DARK_TEAL}; margin:0 0 8px;'>💙 Recommended Actions</h4>
                        <ul style='color:{DARK_TEAL}; font-size:13px; line-height:2; margin:0; opacity:0.85;'>
                            <li>Consider speaking with a mental health professional</li>
                            <li>Reach out to a trusted friend or family member</li>
                            <li>Call iCall India: <b>9152987821</b></li>
                            <li>Vandrevala Foundation: <b>1860-2662-345</b></li>
                        </ul>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div style='background:white; border-radius:14px; padding:20px;
                                border-left:4px solid {MINT};'>
                        <h4 style='color:{DARK_TEAL}; margin:0 0 8px;'>💚 Looks Good!</h4>
                        <p style='color:{DARK_TEAL}; font-size:13px; line-height:1.8; margin:0; opacity:0.85;'>
                            No significant depression indicators detected. Keep maintaining healthy
                            mental habits — exercise, sleep, social connections, and mindfulness all help.
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

        elif analyze_btn:
            st.warning("⚠️ Please enter some text first.")
        else:
            st.markdown(f"""
            <div style='background:{PALE_TEAL}44; border:2px dashed {PALE_TEAL};
                        border-radius:18px; padding:48px 24px; text-align:center;'>
                <div style='font-size:40px; margin-bottom:12px;'>💭</div>
                <p style='color:{DARK_TEAL}; opacity:0.6; font-size:14px; margin:0;'>
                    Your analysis result will appear here after you click <b>Analyze Text</b>.
                </p>
            </div>
            """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE 3 — MODEL INSIGHTS
# ════════════════════════════════════════════════════════════════
elif page == "insights":
    st.markdown(f"<h1 style='color:{DARK_TEAL}; font-weight:800;'>📊 Model Insights</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{DARK_TEAL}; opacity:0.7; margin-top:-8px;'>Performance metrics and interpretability of the trained Logistic Regression model.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # Metrics
    m1, m2, m3, m4 = st.columns(4)
    metric_data = [
        (m1, "🎯 Accuracy", f"{metrics['accuracy']*100:.1f}%", "Overall correctness"),
        (m2, "⚖️ F1 Score", f"{metrics['f1']*100:.1f}%", "Balance of precision & recall"),
        (m3, "🔬 Precision", f"{metrics['precision']*100:.1f}%", "True positive rate"),
        (m4, "📡 Recall", f"{metrics['recall']*100:.1f}%", "Depression catch rate"),
    ]
    for col, icon_label, val, sub in metric_data:
        col.markdown(f"""
        <div style='background:white; border-radius:16px; padding:22px; text-align:center;
                    box-shadow:0 3px 16px {DARK_TEAL}12; border-top:4px solid {MINT};'>
            <div style='font-size:13px; color:{DARK_TEAL}; opacity:0.6; margin-bottom:6px;'>{icon_label}</div>
            <div style='font-size:30px; font-weight:800; color:{DARK_TEAL};'>{val}</div>
            <div style='font-size:11px; color:{DARK_TEAL}; opacity:0.5; margin-top:4px;'>{sub}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col_cm, col_kw = st.columns([1, 1], gap="large")

    with col_cm:
        st.markdown(f"<h3 style='color:{DARK_TEAL};'>Confusion Matrix</h3>", unsafe_allow_html=True)
        fig = plot_confusion_matrix()
        st.pyplot(fig, use_container_width=True)

    with col_kw:
        st.markdown(f"<h3 style='color:{DARK_TEAL};'>Top Predictive Keywords</h3>", unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🔴 Depression Words", "🟢 Non-Depression Words"])

        with tab1:
            dep_words = metrics['top_depression_words'][:15]
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            fig2.patch.set_facecolor(WHITE_BG)
            ax2.set_facecolor(WHITE_BG)
            colors = [PINK if i < 5 else MAUVE if i < 10 else PALE_TEAL for i in range(len(dep_words))]
            bars = ax2.barh(dep_words[::-1], range(len(dep_words), 0, -1), color=colors[::-1], edgecolor='none', height=0.65)
            ax2.set_xlabel('Relative Importance', fontsize=10, color=DARK_TEAL)
            ax2.tick_params(colors=DARK_TEAL, labelsize=10)
            ax2.spines['top'].set_visible(False)
            ax2.spines['right'].set_visible(False)
            for spine in ['left','bottom']:
                ax2.spines[spine].set_color(PALE_TEAL)
            plt.tight_layout()
            st.pyplot(fig2, use_container_width=True)

        with tab2:
            non_words = metrics['top_non_depression_words'][:15]
            fig3, ax3 = plt.subplots(figsize=(6, 5))
            fig3.patch.set_facecolor(WHITE_BG)
            ax3.set_facecolor(WHITE_BG)
            colors2 = [MINT if i < 5 else PALE_TEAL if i < 10 else MAUVE for i in range(len(non_words))]
            ax3.barh(non_words[::-1], range(len(non_words), 0, -1), color=colors2[::-1], edgecolor='none', height=0.65)
            ax3.set_xlabel('Relative Importance', fontsize=10, color=DARK_TEAL)
            ax3.tick_params(colors=DARK_TEAL, labelsize=10)
            ax3.spines['top'].set_visible(False)
            ax3.spines['right'].set_visible(False)
            for spine in ['left','bottom']:
                ax3.spines[spine].set_color(PALE_TEAL)
            plt.tight_layout()
            st.pyplot(fig3, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div style='background:{PALE_TEAL}55; border-radius:16px; padding:24px;'>
        <h4 style='color:{DARK_TEAL}; margin:0 0 10px;'>🧬 Model Architecture</h4>
        <div style='display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px;'>
            <div><span style='color:{DARK_TEAL}; font-size:12px; opacity:0.6;'>Algorithm</span><br>
                 <b style='color:{DARK_TEAL};'>Logistic Regression</b></div>
            <div><span style='color:{DARK_TEAL}; font-size:12px; opacity:0.6;'>Vectorizer</span><br>
                 <b style='color:{DARK_TEAL};'>TF-IDF (10K features)</b></div>
            <div><span style='color:{DARK_TEAL}; font-size:12px; opacity:0.6;'>N-grams</span><br>
                 <b style='color:{DARK_TEAL};'>Unigrams + Bigrams</b></div>
            <div><span style='color:{DARK_TEAL}; font-size:12px; opacity:0.6;'>Training Set</span><br>
                 <b style='color:{DARK_TEAL};'>6,184 samples (80%)</b></div>
            <div><span style='color:{DARK_TEAL}; font-size:12px; opacity:0.6;'>Test Set</span><br>
                 <b style='color:{DARK_TEAL};'>1,547 samples (20%)</b></div>
            <div><span style='color:{DARK_TEAL}; font-size:12px; opacity:0.6;'>Dataset</span><br>
                 <b style='color:{DARK_TEAL};'>Reddit Depression Posts</b></div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE 4 — ABOUT MENTAL HEALTH
# ════════════════════════════════════════════════════════════════
elif page == "about":
    st.markdown(f"<h1 style='color:{DARK_TEAL}; font-weight:800;'>📖 About Mental Health</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{DARK_TEAL}; opacity:0.7; margin-top:-8px;'>Understanding depression and knowing when to seek help can make all the difference.</p>", unsafe_allow_html=True)
    st.markdown("---")

    # What is depression
    st.markdown(f"""
    <div style='background:white; border-radius:18px; padding:30px; margin-bottom:20px;
                box-shadow:0 4px 20px {DARK_TEAL}12; border-left:5px solid {PALE_TEAL};'>
        <h2 style='color:{DARK_TEAL}; margin:0 0 12px;'>🧠 What is Depression?</h2>
        <p style='color:{DARK_TEAL}; font-size:14px; line-height:1.9; margin:0; opacity:0.85;'>
            Depression (major depressive disorder) is a common and serious medical illness that
            negatively affects how you feel, the way you think, and how you act. It causes feelings
            of sadness and/or a loss of interest in activities once enjoyed. It can lead to a variety
            of emotional and physical problems and can decrease a person's ability to function at work
            and at home. Depression is <b>not a weakness</b> and you can't simply "snap out" of it —
            it requires treatment and care.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Warning signs
    st.markdown(f"<h3 style='color:{DARK_TEAL};'>⚠️ Warning Signs</h3>", unsafe_allow_html=True)
    signs = [
        ("😔", "Persistent sadness or hopelessness", "Feeling empty, worthless, or tearful most of the day, nearly every day."),
        ("😴", "Sleep disturbances", "Insomnia or sleeping too much, chronic fatigue and lack of energy."),
        ("🚫", "Loss of interest", "Reduced interest or pleasure in activities that were once enjoyable."),
        ("💭", "Difficulty concentrating", "Trouble thinking, focusing, remembering, or making decisions."),
        ("😤", "Irritability or restlessness", "Feeling unusually frustrated, agitated, or on edge."),
        ("🍽️", "Appetite changes", "Significant weight loss/gain or changes in appetite without dieting."),
    ]
    c1, c2, c3 = st.columns(3)
    for i, (icon, title, desc) in enumerate(signs):
        col = [c1, c2, c3][i % 3]
        col.markdown(f"""
        <div style='background:{PALE_TEAL}44; border-radius:14px; padding:20px; margin-bottom:16px;
                    border-top:3px solid {MINT};'>
            <div style='font-size:24px; margin-bottom:8px;'>{icon}</div>
            <h4 style='color:{DARK_TEAL}; margin:0 0 6px; font-size:14px;'>{title}</h4>
            <p style='color:{DARK_TEAL}; font-size:12px; line-height:1.6; margin:0; opacity:0.75;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    # When to seek help
    st.markdown(f"""
    <div style='background: linear-gradient(135deg, {MINT}44, {PALE_TEAL}88);
                border-radius:18px; padding:28px; margin:8px 0 20px;'>
        <h3 style='color:{DARK_TEAL}; margin:0 0 12px;'>🩺 When to Seek Help</h3>
        <p style='color:{DARK_TEAL}; font-size:14px; line-height:1.9; margin:0; opacity:0.85;'>
            If you experience <b>5 or more</b> of the above symptoms for <b>2 weeks or longer</b>,
            please consider seeking professional help. You should seek <b>immediate help</b> if you
            have thoughts of self-harm or suicide — you are not alone and support is available.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Helplines
    st.markdown(f"<h3 style='color:{DARK_TEAL};'>📞 Helplines & Resources (India)</h3>", unsafe_allow_html=True)
    h1, h2, h3 = st.columns(3)
    helplines = [
        (h1, "iCall", "9152987821", "TISS-backed professional counselling via phone, chat, and email.", PINK),
        (h2, "Vandrevala Foundation", "1860-2662-345", "Free 24/7 mental health helpline available across India.", MINT),
        (h3, "NIMHANS", "080-46110007", "National Institute of Mental Health and Neurosciences helpline.", PALE_TEAL),
    ]
    for col, name, number, desc, color in helplines:
        col.markdown(f"""
        <div style='background:white; border-radius:16px; padding:24px; text-align:center;
                    box-shadow:0 4px 16px {DARK_TEAL}12; border-bottom:4px solid {color};'>
            <div style='font-size:28px; margin-bottom:8px;'>📞</div>
            <h4 style='color:{DARK_TEAL}; margin:0 0 6px; font-size:15px;'>{name}</h4>
            <div style='background:{color}55; color:{DARK_TEAL}; padding:6px 14px;
                        border-radius:20px; font-size:15px; font-weight:700;
                        display:inline-block; margin-bottom:10px;'>{number}</div>
            <p style='color:{DARK_TEAL}; font-size:12px; line-height:1.6; margin:0; opacity:0.7;'>{desc}</p>
        </div>
        """, unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════
# PAGE 5 — BATCH ANALYZER
# ════════════════════════════════════════════════════════════════
elif page == "batch":
    st.markdown(f"<h1 style='color:{DARK_TEAL}; font-weight:800;'>📁 Batch Analyzer</h1>", unsafe_allow_html=True)
    st.markdown(f"<p style='color:{DARK_TEAL}; opacity:0.7; margin-top:-8px;'>Upload a CSV file with text data to analyze multiple entries at once.</p>", unsafe_allow_html=True)
    st.markdown("---")

    col_up, col_info = st.columns([2, 1])
    with col_up:
        uploaded_file = st.file_uploader("Upload CSV file", type=["csv"], label_visibility="collapsed")
        st.markdown(f"<p style='color:{DARK_TEAL}; font-size:12px; opacity:0.6; margin-top:8px;'>📌 Accepted format: .csv with a text column</p>", unsafe_allow_html=True)

    with col_info:
        st.markdown(f"""
        <div style='background:{PALE_TEAL}55; border-radius:14px; padding:20px;'>
            <h4 style='color:{DARK_TEAL}; margin:0 0 10px;'>📌 Instructions</h4>
            <ul style='color:{DARK_TEAL}; font-size:12px; line-height:2; margin:0; opacity:0.85;'>
                <li>Upload a <b>.csv</b> file</li>
                <li>Select the text column</li>
                <li>Click <b>Analyze All</b></li>
                <li>Download results as CSV</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    if uploaded_file:
        df_uploaded = pd.read_csv(uploaded_file)
        st.markdown(f"<p style='color:{DARK_TEAL}; font-size:13px;'>✅ Loaded <b>{len(df_uploaded)}</b> rows | Columns: {', '.join(df_uploaded.columns.tolist())}</p>", unsafe_allow_html=True)

        text_col = st.selectbox("Select the text column:", df_uploaded.columns.tolist())
        st.dataframe(df_uploaded[[text_col]].head(5), use_container_width=True)

        if st.button("🚀 Analyze All Texts", use_container_width=True):
            with st.spinner(f"Analyzing {len(df_uploaded)} texts..."):
                results_df = batch_predict(df_uploaded, text_col)

            st.markdown("<br>", unsafe_allow_html=True)
            high = (results_df['risk_level'] == 'HIGH').sum()
            low = (results_df['risk_level'] == 'LOW').sum()

            s1, s2, s3 = st.columns(3)
            s1.markdown(f"""<div style='background:white;border-radius:14px;padding:20px;text-align:center;
                box-shadow:0 3px 12px {DARK_TEAL}12;border-top:3px solid {PALE_TEAL};'>
                <div style='font-size:24px;font-weight:800;color:{DARK_TEAL};'>{len(results_df)}</div>
                <div style='font-size:12px;color:{DARK_TEAL};opacity:0.6;'>Total Analyzed</div></div>""", unsafe_allow_html=True)
            s2.markdown(f"""<div style='background:white;border-radius:14px;padding:20px;text-align:center;
                box-shadow:0 3px 12px {DARK_TEAL}12;border-top:3px solid {PINK};'>
                <div style='font-size:24px;font-weight:800;color:{DARK_TEAL};'>{high}</div>
                <div style='font-size:12px;color:{DARK_TEAL};opacity:0.6;'>High Risk</div></div>""", unsafe_allow_html=True)
            s3.markdown(f"""<div style='background:white;border-radius:14px;padding:20px;text-align:center;
                box-shadow:0 3px 12px {DARK_TEAL}12;border-top:3px solid {MINT};'>
                <div style='font-size:24px;font-weight:800;color:{DARK_TEAL};'>{low}</div>
                <div style='font-size:12px;color:{DARK_TEAL};opacity:0.6;'>Low Risk</div></div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            st.dataframe(results_df, use_container_width=True, height=320)

            csv_bytes = results_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="⬇️ Download Results as CSV",
                data=csv_bytes,
                file_name="mindguard_batch_results.csv",
                mime="text/csv",
                use_container_width=True
            )
    else:
        st.markdown(f"""
        <div style='background:{PALE_TEAL}33; border:2px dashed {PALE_TEAL};
                    border-radius:18px; padding:60px; text-align:center;'>
            <div style='font-size:48px; margin-bottom:12px;'>📂</div>
            <p style='color:{DARK_TEAL}; opacity:0.6; font-size:14px; margin:0;'>
                Upload a CSV file above to begin batch analysis.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ─── FOOTER ─────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown(f"""
<div style='text-align:center; padding:20px; border-top:2px solid {PALE_TEAL};'>
    <p style='color:{DARK_TEAL}; font-size:12px; opacity:0.5; margin:0;'>
        🧠 MindGuard &nbsp;|&nbsp; Built with Streamlit &nbsp;|&nbsp;
        For educational & research purposes only &nbsp;|&nbsp;
        Not a substitute for professional medical advice
    </p>
</div>
""", unsafe_allow_html=True)