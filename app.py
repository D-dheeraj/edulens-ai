import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

try:
    from openai import OpenAI
except Exception:
    OpenAI = None

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="EduLens AI | Student Success Intelligence",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- CSS ----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

:root{
    --bg:#020617;
    --panel:#0f172a;
    --panel2:#111827;
    --text:#e5e7eb;
    --muted:#94a3b8;
    --cyan:#38bdf8;
    --blue:#2563eb;
    --purple:#7c3aed;
    --green:#22c55e;
    --red:#fb7185;
    --amber:#f59e0b;
}

html, body, [class*="css"] { font-family:'Inter', sans-serif; }

.stApp {
    background:
      radial-gradient(circle at 15% 5%, rgba(59,130,246,.28), transparent 28%),
      radial-gradient(circle at 90% 15%, rgba(124,58,237,.22), transparent 30%),
      linear-gradient(135deg, #020617 0%, #0b1022 48%, #020617 100%);
    color: var(--text);
}

.block-container { padding-top: 2.4rem; max-width: 1320px; }

/* Hide ugly Streamlit top spacing/menu look */
header[data-testid="stHeader"] { background: transparent; }
#MainMenu, footer { visibility: hidden; }

[data-testid="stSidebar"] {
    background: rgba(2, 6, 23, .92);
    border-right: 1px solid rgba(148,163,184,.16);
}
[data-testid="stSidebar"] * { color:#e5e7eb !important; }
[data-testid="stSidebar"] .stCaption { color:#94a3b8 !important; }

/* Clean uploader */
[data-testid="stFileUploader"] {
    background: rgba(15, 23, 42, .95);
    border: 1px dashed rgba(56,189,248,.55);
    border-radius: 18px;
    padding: 1rem;
}
[data-testid="stFileUploaderDropzone"]{
    background: rgba(15,23,42,.9) !important;
    border: none !important;
    border-radius: 14px !important;
}
[data-testid="stFileUploaderDropzone"] button{
    background: rgba(56,189,248,.12) !important;
    border: 1px solid rgba(56,189,248,.35) !important;
    color: white !important;
    border-radius: 12px !important;
}

/* Inputs + cursor fix */
.stTextInput input, .stTextArea textarea {
    background: rgba(15,23,42,.95) !important;
    color: #f8fafc !important;
    caret-color: #38bdf8 !important;
    border: 1px solid rgba(56,189,248,.45) !important;
    border-radius: 14px !important;
    box-shadow: 0 0 0 1px rgba(56,189,248,.05);
}
.stTextInput input:focus, .stTextArea textarea:focus {
    border: 1px solid #38bdf8 !important;
    box-shadow: 0 0 0 3px rgba(56,189,248,.12) !important;
}
input::placeholder, textarea::placeholder { color:#94a3b8 !important; opacity:1; }

.stButton>button {
    width: 100%;
    border-radius: 14px;
    border: 1px solid rgba(255,255,255,.12);
    background: linear-gradient(90deg, #2563eb, #7c3aed);
    color: white;
    font-weight: 800;
    padding: .8rem 1rem;
    transition: .2s ease;
}
.stButton>button:hover { transform: translateY(-1px); filter: brightness(1.1); }

.hero {
    position: relative;
    overflow: hidden;
    padding: 2.2rem;
    border-radius: 32px;
    background: linear-gradient(135deg, rgba(15,23,42,.92), rgba(30,41,59,.55));
    border: 1px solid rgba(148,163,184,.22);
    box-shadow: 0 28px 90px rgba(0,0,0,.35);
    backdrop-filter: blur(16px);
}
.hero:before{
    content:""; position:absolute; inset:-2px;
    background: radial-gradient(circle at 20% 0%, rgba(56,189,248,.22), transparent 28%), radial-gradient(circle at 80% 20%, rgba(124,58,237,.22), transparent 28%);
    pointer-events:none;
}
.badge {
    display:inline-block; padding:.55rem .9rem; border-radius:999px;
    background:rgba(56,189,248,.12); color:#7dd3fc;
    border:1px solid rgba(56,189,248,.34); font-weight:800; font-size:.85rem;
}
.main-title {
    font-size: clamp(3rem, 6vw, 5.8rem);
    font-weight:900; letter-spacing:-.06em; margin:.5rem 0 .2rem;
    background: linear-gradient(90deg, #38bdf8, #818cf8, #c084fc);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
}
.subtitle { font-size:1.2rem; color:#cbd5e1; line-height:1.7; max-width:880px; }
.hero-mini {
    display:flex; gap:.7rem; flex-wrap:wrap; margin-top:1.2rem;
}
.pill { padding:.55rem .8rem; border-radius:999px; background:rgba(15,23,42,.82); border:1px solid rgba(148,163,184,.20); color:#dbeafe; font-weight:700; font-size:.86rem; }

.section-title { font-size:1.75rem; font-weight:900; color:#f8fafc; margin:2.2rem 0 .9rem; letter-spacing:-.03em; }
.panel, .metric-card, .ai-card {
    border-radius:24px;
    background: linear-gradient(145deg, rgba(15,23,42,.88), rgba(17,24,39,.78));
    border:1px solid rgba(148,163,184,.18);
    box-shadow:0 18px 55px rgba(0,0,0,.25);
    backdrop-filter: blur(12px);
}
.panel { padding:1.35rem; margin-bottom:1rem; }
.metric-card { padding:1.2rem; min-height:138px; }
.metric-label { color:#94a3b8; font-size:.9rem; font-weight:800; margin-bottom:.55rem; }
.metric-value { color:#f8fafc; font-size:2.25rem; font-weight:900; letter-spacing:-.04em; }
.metric-note { color:#38bdf8; font-size:.86rem; font-weight:700; margin-top:.25rem; }

.reason-step {
    padding: .9rem 1rem;
    border-radius: 16px;
    background: rgba(30,41,59,.82);
    border: 1px solid rgba(148,163,184,.16);
    border-left: 4px solid #38bdf8;
    margin-bottom: .75rem;
    color:#dbeafe;
}
.ai-answer {
    padding: 1rem;
    border-radius: 18px;
    background: rgba(2,6,23,.50);
    border:1px solid rgba(56,189,248,.18);
    line-height:1.65;
}
.microsoft-box {
    padding:1rem; border-radius:18px;
    background:linear-gradient(135deg, rgba(37,99,235,.20), rgba(124,58,237,.20));
    border:1px solid rgba(129,140,248,.32);
    line-height:1.9; font-weight:700;
}
.small-text { color:#94a3b8; line-height:1.65; font-size:.95rem; }
.warning-box { padding:1rem 1.1rem; border-radius:18px; background:rgba(245,158,11,.12); border:1px solid rgba(245,158,11,.25); color:#fde68a; }
.success-box { padding:1rem 1.1rem; border-radius:18px; background:rgba(34,197,94,.10); border:1px solid rgba(34,197,94,.25); color:#bbf7d0; }
hr { border-color: rgba(148,163,184,.18); }
</style>
""", unsafe_allow_html=True)

# ---------------- HELPERS ----------------
def get_secret(name: str, default: str = "") -> str:
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


def load_dataset(uploaded_file):
    if uploaded_file is None:
        return None
    try:
        if uploaded_file.name.lower().endswith(".csv"):
            return pd.read_csv(uploaded_file, sep=None, engine="python")
        return pd.read_excel(uploaded_file)
    except Exception as e:
        st.error(f"Dataset read nahi ho paaya: {e}")
        return None


def prepare_student_data(df):
    data = df.copy()
    if "G3" not in data.columns:
        st.error("G3 final grade column nahi mila. UCI student-mat.csv upload karo.")
        return None, None, None, None

    data["at_risk"] = (data["G3"] < 10).astype(int)
    numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
    feature_cols = [c for c in numeric_cols if c not in ["G3", "at_risk"]]
    if len(feature_cols) < 2:
        st.error("Enough numeric features nahi mile.")
        return None, None, None, None

    X = data[feature_cols].fillna(data[feature_cols].median())
    y = data["at_risk"]

    if y.nunique() < 2:
        data["risk_score"] = y * 100
        importance = pd.DataFrame({"Feature": feature_cols, "Importance": [0]*len(feature_cols)})
        return data, 1.0, importance, feature_cols

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=.25, random_state=42, stratify=y
    )
    model = RandomForestClassifier(n_estimators=180, random_state=42, max_depth=7)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    data["risk_score"] = model.predict_proba(X)[:, 1] * 100

    importance = pd.DataFrame({
        "Feature": feature_cols,
        "Importance": model.feature_importances_
    }).sort_values("Importance", ascending=False).head(10)

    return data, accuracy, importance, feature_cols


def dataset_context(data, risk_threshold, importance):
    total = len(data)
    at_risk_count = int((data["risk_score"] >= risk_threshold).sum())
    avg_g3 = data["G3"].mean() if "G3" in data.columns else 0
    avg_abs = data["absences"].mean() if "absences" in data.columns else 0
    cols = [c for c in ["G1", "G2", "G3", "absences", "studytime", "failures", "risk_score"] if c in data.columns]
    top = data.sort_values("risk_score", ascending=False)[cols].head(8).round(2)
    top_features = ", ".join(importance["Feature"].head(5).tolist()) if importance is not None and not importance.empty else "Not available"
    return f"""
Dataset Summary:
- Total students: {total}
- At-risk students using threshold {risk_threshold}%: {at_risk_count}
- Average final grade G3: {avg_g3:.2f}
- Average absences: {avg_abs:.2f}
- Top model risk drivers: {top_features}

Top high-risk student rows:
{top.to_string(index=False)}
"""


def rule_based_ai(question, data, risk_threshold, importance):
    ctx = dataset_context(data, risk_threshold, importance)
    top_ids = data.sort_values("risk_score", ascending=False).head(5).index.tolist()
    return f"""
<div class='ai-answer'>
<h4>🧠 AI Reasoning Trace</h4>
<div class='reason-step'><b>Step 1 — Dataset Scan</b><br>Student records, grades, absences, study indicators and risk scores were analysed.</div>
<div class='reason-step'><b>Step 2 — Risk Pattern Detection</b><br>The model prioritised students with weak grade progression, higher absence pattern and low academic indicators.</div>
<div class='reason-step'><b>Step 3 — Intervention Mapping</b><br>Highest-risk student indexes: <b>{', '.join(map(str, top_ids))}</b>.</div>
<h4>✅ Final Recommendation</h4>
Start with urgent mentoring for the highest-risk students, monitor attendance weekly, arrange parent/teacher follow-up, and review G1 → G2 → G3 progress every week.
<br><br><b>Question asked:</b> {question}
</div>
"""


def github_models_answer(question, data, risk_threshold, importance, token):
    if not token or OpenAI is None:
        return rule_based_ai(question, data, risk_threshold, importance), False

    prompt = f"""
You are EduLens AI, a student-success analytics assistant for teachers.
Answer clearly using this structure:
1. AI Reasoning Trace with Step 1, Step 2, Step 3
2. Key finding
3. Teacher action plan
Keep it concise, practical, and grounded only in the dataset context.

Teacher question: {question}

{dataset_context(data, risk_threshold, importance)}
"""
    try:
        client = OpenAI(
            base_url="https://models.github.ai/inference",
            api_key=token,
        )
        response = client.chat.completions.create(
            model="openai/gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a grounded education analytics assistant. Do not invent private student identities. Use only provided numeric context."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.25,
            max_tokens=650,
        )
        text = response.choices[0].message.content
        return f"<div class='ai-answer'>{text}</div>", True
    except Exception as e:
        fallback, _ = rule_based_ai(question, data, risk_threshold, importance), False
        return f"<div class='warning-box'>Live GitHub Models response failed, fallback reasoning shown.<br><b>Error:</b> {str(e)[:180]}</div><br>{fallback}", False


def metric_card(label, value, note, icon):
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{icon} {label}</div>
        <div class="metric-value">{value}</div>
        <div class="metric-note">{note}</div>
    </div>
    """, unsafe_allow_html=True)


def dark_layout(fig):
    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font_color="#e5e7eb",
        margin=dict(l=20, r=20, t=55, b=20),
        legend=dict(bgcolor="rgba(0,0,0,0)")
    )
    return fig

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.markdown("## 🎓 EduLens AI")
    st.caption("Student Success Intelligence Platform")
    st.markdown("---")

    uploaded_file = st.file_uploader("Upload student CSV/XLSX", type=["csv", "xlsx"])
    risk_threshold = st.slider("Risk threshold", 0, 100, 60)

    st.markdown("---")
    st.markdown("### 🔐 AI API Key")
    secret_token = get_secret("GITHUB_TOKEN", "")
    token_input = st.text_input(
        "GitHub Models token",
        value="" if secret_token else "",
        type="password",
        placeholder="github_pat_xxx...",
        help="Deployment ke liye Streamlit Secrets use karo. Local testing ke liye yahan paste kar sakte ho."
    )
    github_token = secret_token or token_input
    if github_token:
        st.markdown("<div class='success-box'>✅ Live AI mode ready</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='warning-box'>⚠️ Fallback AI mode: token nahi hai</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### ⚡ Microsoft AI Stack")
    st.markdown("""
    <div class='microsoft-box'>
    ✅ GitHub Models<br>
    ✅ GPT-4o Mini<br>
    ✅ Azure-compatible API<br>
    ✅ AI reasoning workflow
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    st.caption("Built for Microsoft League Hackathon 2026")

# ---------------- HERO ----------------
st.markdown("""
<div class="hero">
    <span class="badge">⚡ AI-Powered Education Analytics</span>
    <div class="main-title">EduLens AI</div>
    <div class="subtitle">
        Predict at-risk students early, explain academic risk factors, and generate teacher-ready intervention plans using machine learning, visual analytics, and Microsoft GitHub Models reasoning.
    </div>
    <div class="hero-mini">
        <span class="pill">🎯 Early Risk Prediction</span>
        <span class="pill">🧠 AI Reasoning Trace</span>
        <span class="pill">📊 Interactive BI Dashboard</span>
        <span class="pill">⚡ Microsoft AI Stack</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD ----------------
df = load_dataset(uploaded_file)
if df is None:
    st.markdown("<div class='section-title'>🚀 Start Here</div>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1: metric_card("Upload Dataset", "CSV/XLSX", "Use student-mat.csv", "📁")
    with c2: metric_card("Analyze Risk", "ML Model", "Random Forest scoring", "🧠")
    with c3: metric_card("Take Action", "AI Plan", "Teacher recommendations", "✅")
    st.markdown("""
    <div class='panel'>
    <h3>What this dashboard does</h3>
    <p class='small-text'>Upload the UCI student performance dataset. EduLens AI will clean the data, calculate risk scores, show business-intelligence charts, and generate an AI reasoning trace for teacher decisions.</p>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

processed, accuracy, importance, feature_cols = prepare_student_data(df)
if processed is None:
    st.stop()

# ---------------- KPI ----------------
total_students = len(processed)
at_risk_students = int((processed["risk_score"] >= risk_threshold).sum())
avg_grade = processed["G3"].mean()
avg_risk = processed["risk_score"].mean()
critical = int((processed["risk_score"] >= 75).sum())

st.markdown("<div class='section-title'>📌 Executive Intelligence Dashboard</div>", unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)
with col1: metric_card("Total Students", total_students, "Dataset loaded", "👨‍🎓")
with col2: metric_card("At-Risk", at_risk_students, f"Threshold {risk_threshold}%", "⚠️")
with col3: metric_card("Accuracy", f"{accuracy*100:.1f}%", "Random Forest", "🎯")
with col4: metric_card("Critical Cases", critical, "Risk score ≥ 75%", "🚨")

# ---------------- AI ASSISTANT ----------------
st.markdown("<div class='section-title'>🧠 EduLens AI Assistant</div>", unsafe_allow_html=True)
left, right = st.columns([1.25, .9])

with left:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("### Ask a teacher-style question")
    question = st.text_area(
        "",
        value="Which students are most at risk and what should the teacher do first?",
        height=95,
        placeholder="Ask EduLens AI about risk, attendance, grades, or intervention plan...",
        label_visibility="collapsed"
    )
    ask = st.button("🔍 Analyze with AI")
    if ask:
        with st.spinner("EduLens AI is analyzing dataset patterns..."):
            answer_html, live = github_models_answer(question, processed, risk_threshold, importance, github_token)
        st.markdown("<div class='success-box'>✅ Live GitHub Models AI response</div>" if live else "<div class='warning-box'>⚠️ Fallback reasoning response</div>", unsafe_allow_html=True)
        st.markdown(answer_html, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class='ai-answer'>
        <h4>Ready for AI Analysis</h4>
        <div class='reason-step'><b>Step 1</b><br>Upload data and review model-generated risk scores.</div>
        <div class='reason-step'><b>Step 2</b><br>Ask a question about student risk, grades, attendance, or intervention.</div>
        <div class='reason-step'><b>Step 3</b><br>EduLens AI returns a teacher-ready reasoning trace and action plan.</div>
        </div>
        """, unsafe_allow_html=True)
    st.markdown("</div>", unsafe_allow_html=True)

with right:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    st.markdown("### 🔥 Highest Priority Students")
    display_cols = [c for c in ["G1", "G2", "G3", "absences", "studytime", "failures", "risk_score"] if c in processed.columns]
    st.dataframe(
        processed.sort_values("risk_score", ascending=False)[display_cols].head(10).round(2),
        use_container_width=True,
        hide_index=True
    )
    st.caption("These rows are sorted by ML-generated risk score.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- CHARTS ----------------
st.markdown("<div class='section-title'>📈 Intelligence Visuals</div>", unsafe_allow_html=True)
chart1, chart2 = st.columns(2)

with chart1:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    fig = px.histogram(processed, x="risk_score", nbins=25, title="Risk Score Distribution", template="plotly_dark")
    st.plotly_chart(dark_layout(fig), use_container_width=True)
    st.caption("Insight: Right-side bars show students needing urgent intervention.")
    st.markdown("</div>", unsafe_allow_html=True)

with chart2:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    if all(c in processed.columns for c in ["G1", "G2", "G3"]):
        grade_avg = pd.DataFrame({"Exam": ["G1", "G2", "G3"], "Average Grade": [processed["G1"].mean(), processed["G2"].mean(), processed["G3"].mean()]})
        fig = px.line(grade_avg, x="Exam", y="Average Grade", markers=True, title="Average Grade Progression", template="plotly_dark")
        fig.update_traces(line=dict(width=4), marker=dict(size=12))
        st.plotly_chart(dark_layout(fig), use_container_width=True)
    st.caption("Insight: Falling grade trend signals early intervention need.")
    st.markdown("</div>", unsafe_allow_html=True)

chart3, chart4 = st.columns(2)
with chart3:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    fig = px.bar(importance, x="Importance", y="Feature", orientation="h", title="Top ML Risk Drivers", template="plotly_dark")
    fig.update_layout(yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(dark_layout(fig), use_container_width=True)
    st.caption("Insight: These features influenced the prediction model most.")
    st.markdown("</div>", unsafe_allow_html=True)

with chart4:
    st.markdown("<div class='panel'>", unsafe_allow_html=True)
    if "absences" in processed.columns:
        fig = px.scatter(processed, x="absences", y="G3", size="risk_score", color="risk_score", title="Absences vs Final Grade", template="plotly_dark")
        st.plotly_chart(dark_layout(fig), use_container_width=True)
    st.caption("Insight: Attendance patterns can reveal hidden academic risk.")
    st.markdown("</div>", unsafe_allow_html=True)

# ---------------- PLAN ----------------
st.markdown("<div class='section-title'>✅ Teacher Intervention Plan</div>", unsafe_allow_html=True)
st.markdown("<div class='panel'>", unsafe_allow_html=True)
intervention = processed.sort_values("risk_score", ascending=False).head(12).copy()
intervention["Recommended Action"] = np.where(
    intervention["risk_score"] >= 75,
    "Urgent mentoring + attendance follow-up",
    np.where(intervention["risk_score"] >= 60, "Weekly monitoring + academic support", "Normal monitoring")
)
cols = [c for c in ["G1", "G2", "G3", "absences", "studytime", "failures", "risk_score", "Recommended Action"] if c in intervention.columns]
st.dataframe(intervention[cols].round(2), use_container_width=True, hide_index=True)
st.markdown("</div>", unsafe_allow_html=True)

# ---------------- FOOTER ----------------
st.markdown("""
<div class='panel'>
<h3>🚀 Built for Microsoft League Hackathon 2026</h3>
<p class='small-text'>EduLens AI combines Python, Pandas, scikit-learn, Plotly, Streamlit, and Microsoft GitHub Models to turn student performance data into explainable early-intervention intelligence.</p>
</div>
""", unsafe_allow_html=True)
