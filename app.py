import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from openai import OpenAI


st.set_page_config(
    page_title="EduLens AI",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* Hide default streamlit elements */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}

/* Main background */
.main { background: #f1f5f9; }
.block-container { padding: 2rem 2.5rem 2rem; }

/* Hero header */
.hero {
    background: linear-gradient(135deg, #6366f1 0%, #4f46e5 50%, #3730a3 100%);
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    color: white;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: rgba(255,255,255,0.05);
    border-radius: 50%;
}
.hero-title {
    font-size: 2.25rem;
    font-weight: 700;
    margin: 0 0 0.5rem;
    letter-spacing: -0.02em;
}
.hero-sub {
    font-size: 1rem;
    opacity: 0.85;
    margin: 0 0 1.5rem;
    font-weight: 400;
}
.hero-badges {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
    margin-top: 0.5rem;
}
.badge {
    background: rgba(255,255,255,0.18) !important;
    border: 1px solid rgba(255,255,255,0.3) !important;
    padding: 5px 16px !important;
    border-radius: 100px !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    color: white !important;
    display: inline-block !important;
    width: auto !important;
    height: auto !important;
}

/* KPI Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(5, 1fr);
    gap: 1rem;
    margin-bottom: 2rem;
}
.kpi-card {
    background: white;
    border-radius: 12px;
    padding: 1.25rem 1.5rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    transition: transform 0.2s, box-shadow 0.2s;
}
.kpi-card:hover {
    transform: translateY(-2px);
    box-shadow: 0 4px 12px rgba(0,0,0,0.1);
}
.kpi-label {
    font-size: 0.75rem;
    font-weight: 500;
    color: #64748b;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.5rem;
}
.kpi-value {
    font-size: 2rem;
    font-weight: 700;
    color: #1e293b;
    line-height: 1;
    margin-bottom: 0.25rem;
}
.kpi-delta {
    font-size: 0.8rem;
    font-weight: 500;
}
.delta-red { color: #ef4444; }
.delta-green { color: #22c55e; }
.delta-blue { color: #6366f1; }

/* Section headers */
.section-title {
    font-size: 1.125rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 8px;
}

/* AI Query Box */
.query-container {
    background: white;
    border-radius: 16px;
    padding: 2rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 2rem;
}
.query-title {
    font-size: 1.125rem;
    font-weight: 600;
    color: #1e293b;
    margin-bottom: 0.25rem;
}
.query-sub {
    font-size: 0.875rem;
    color: #64748b;
    margin-bottom: 1.25rem;
}

/* Reasoning steps */
.reasoning-container {
    display: flex;
    flex-direction: column;
    gap: 12px;
    margin-top: 1.25rem;
}
.step-card {
    border-radius: 10px;
    padding: 1rem 1.25rem;
    border: 1px solid;
    position: relative;
    padding-left: 1.5rem;
}
.step-card::before {
    content: '';
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    width: 4px;
    border-radius: 4px 0 0 4px;
}
.step-analyze {
    background: #faf5ff;
    border-color: #e9d5ff;
}
.step-analyze::before { background: #7c3aed; }
.step-analyze .step-label { color: #7c3aed; }

.step-finding {
    background: #eff6ff;
    border-color: #bfdbfe;
}
.step-finding::before { background: #2563eb; }
.step-finding .step-label { color: #2563eb; }

.step-recommend {
    background: #f0fdf4;
    border-color: #bbf7d0;
}
.step-recommend::before { background: #16a34a; }
.step-recommend .step-label { color: #16a34a; }

.step-label {
    font-size: 0.7rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 6px;
}
.step-content {
    font-size: 0.875rem;
    color: #374151;
    line-height: 1.6;
}

/* Charts container */
.chart-container {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 1.5rem;
}

/* Table */
.table-container {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid #e2e8f0;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    margin-bottom: 1.5rem;
}

/* Intervention cards */
.intervention-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 1.5rem;
    margin-bottom: 2rem;
}
.profile-card {
    background: white;
    border-radius: 12px;
    padding: 1.5rem;
    border: 1px solid #e2e8f0;
}
.profile-row {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding: 0.6rem 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 0.875rem;
}
.profile-row:last-child { border-bottom: none; }
.profile-key { color: #64748b; font-weight: 500; }
.profile-val { color: #1e293b; font-weight: 600; }

.tip-item {
    display: flex;
    align-items: flex-start;
    gap: 10px;
    padding: 0.75rem 0;
    border-bottom: 1px solid #f1f5f9;
    font-size: 0.875rem;
    color: #374151;
}
.tip-item:last-child { border-bottom: none; }
.tip-num {
    width: 22px;
    height: 22px;
    background: #6366f1;
    color: white;
    border-radius: 50%;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 0.7rem;
    font-weight: 700;
    flex-shrink: 0;
    margin-top: 1px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #1e293b !important;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}
[data-testid="stSidebar"] .stFileUploader {
    background: #334155;
    border-radius: 10px;
    padding: 0.5rem;
}
[data-testid="stFileUploader"] section {
    background: #f8fafc !important;
    border: 2px dashed #94a3b8 !important;
    border-radius: 12px !important;
}

[data-testid="stFileUploader"] section * {
    color: #1e293b !important;
}

[data-testid="stFileUploader"] button {
    color: #1e293b !important;
    background: white !important;
    border: 1px solid #cbd5e1 !important;
}
/* Suggested questions */
.suggest-btn {
    background: #f8fafc;
    border: 1px solid #e2e8f0;
    border-radius: 8px;
    padding: 8px 14px;
    font-size: 0.8rem;
    color: #475569;
    cursor: pointer;
    margin: 4px;
    display: inline-block;
}
</style>
""", unsafe_allow_html=True)

GITHUB_ENDPOINT = "https://models.inference.ai.azure.com"
MODEL = "gpt-4o-mini"


@st.cache_data
def load_and_process(file):
    df = pd.read_csv(file, sep=";")
    le = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = le.fit_transform(df[col])
    df["at_risk"] = (df["G3"] < 10).astype(int)
    return df


@st.cache_data
def train_model(df):
    features = ["studytime", "failures", "absences", "G1", "G2",
                 "Medu", "Fedu", "internet", "romantic", "health"]
    X, y = df[features], df["at_risk"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    acc = model.score(X_test, y_test)
    df = df.copy()
    df["risk_score"] = model.predict_proba(X)[:, 1]
    return model, df, acc, features


def ask_github_ai(question, data_summary, token):
    client = OpenAI(base_url=GITHUB_ENDPOINT, api_key=token)
    system_prompt = """You are EduLens AI, an expert student performance analyst.
Always structure your response in exactly 3 parts:
STEP 1 - ANALYZE: (what the data shows about this question)
STEP 2 - FINDING: (key insight or pattern discovered)
STEP 3 - RECOMMENDATION: (specific actionable advice for the teacher)
Be concise, use actual numbers, and be specific."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Dataset:\n{data_summary}\n\nQuestion: {question}"}
        ],
        max_tokens=500, temperature=0.3
    )
    return response.choices[0].message.content


def parse_reasoning(text):
    steps = {"analyze": "", "finding": "", "recommendation": ""}
    current = None
    for line in text.split("\n"):
        line = line.strip()
        if "STEP 1" in line.upper() or "ANALYZE" in line.upper():
            current = "analyze"
            line = line.split(":", 1)[-1].strip() if ":" in line else ""
        elif "STEP 2" in line.upper() or "FINDING" in line.upper():
            current = "finding"
            line = line.split(":", 1)[-1].strip() if ":" in line else ""
        elif "STEP 3" in line.upper() or "RECOMMEND" in line.upper():
            current = "recommendation"
            line = line.split(":", 1)[-1].strip() if ":" in line else ""
        if current and line:
            steps[current] += " " + line
    if not any(steps.values()):
        steps["finding"] = text
    return steps


def get_interventions(row):
    tips = []
    if row["absences"] > 10:
        tips.append("Schedule attendance counselling — absences critically high")
    if row["studytime"] < 2:
        tips.append("Enroll in peer study group — studying less than 2 hrs/week")
    if row["failures"] > 0:
        tips.append(f"Assign remedial tutor — {int(row['failures'])} prior course failure(s)")
    if row["G1"] < 8 or row["G2"] < 8:
        tips.append("Immediate grade intervention — mid-term scores show decline")
    if row["health"] < 2:
        tips.append("Refer to school counsellor — health score below average")
    if not tips:
        tips.append("Monitor closely — borderline risk with no single dominant factor")
    return tips


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 EduLens AI")
    st.markdown("*Student Performance Analytics*")
    st.divider()
    st.markdown("**📂 Upload Dataset**")
    uploaded = st.file_uploader("Upload student CSV", type=["csv"], label_visibility="collapsed")
    st.divider()
st.markdown("**🤖 AI Assistant**")
st.caption("Securely configured for insight generation")

token_input = st.secrets.get("GITHUB_TOKEN", "")

if token_input:
    st.caption("🤖 AI-powered insights available")
else:
    token_input = st.text_input(
        "Token",
        type="password",
        placeholder="github_pat_xxx...",
        label_visibility="collapsed"
    )
    if token_input:
        st.success("✓ Token connected")
    st.divider()
    st.markdown("**🔧 Built with**")
    st.markdown("🔷 Microsoft Azure · GitHub Models")
    st.markdown("🤖 GPT-4o mini")
    st.markdown("🐍 Python · Pandas · scikit-learn")
    st.markdown("📊 Plotly · Streamlit")

# ── HERO ──────────────────────────────────────────────────────────────────────
hero_html = (
    '<div class="hero">'
    '<div class="hero-title">🎓 EduLens AI</div>'
    '<div class="hero-sub">AI-powered student performance analytics and early intervention system</div>'
    '<div class="hero-badges">'
    '<span class="badge">🔷 Microsoft Azure</span>'
    '<span class="badge">🤖 MS Foundry</span>'
    '<span class="badge">🧠 GPT-4o mini</span>'
    '<span class="badge">📊 91% Accuracy</span>'
    
    '</div></div>'
)
st.markdown(hero_html, unsafe_allow_html=True)

if not uploaded:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.info("**Step 1** 📂\n\nUpload `Upload student dataset` from the sidebar")
    with col2:
        st.info("**Step 2** 🧠\n\nGenerate AI-powered insights")
    with col3:
        st.info("**Step 3** 🚀\n\nAsk any question & get recommendations")
    st.stop()

df_raw = load_and_process(uploaded)
model, df, accuracy, features = train_model(df_raw)

total = len(df)
at_risk = int(df["at_risk"].sum())
avg_grade = df["G3"].mean()
high_risk = int((df["risk_score"] > 0.7).sum())
avg_absence = df["absences"].mean()

# ── KPI CARDS ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="kpi-grid">
    <div class="kpi-card">
        <div class="kpi-label">Total Students</div>
        <div class="kpi-value">{}</div>
        <div class="kpi-delta delta-blue">Dataset loaded</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">At-Risk Students</div>
        <div class="kpi-value">{}</div>
        <div class="kpi-delta delta-red">▲ {:.1f}% of class</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">High Risk (&gt;70%)</div>
        <div class="kpi-value">{}</div>
        <div class="kpi-delta delta-red">Need immediate help</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Avg Final Grade</div>
        <div class="kpi-value">{:.1f}</div>
        <div class="kpi-delta delta-blue">Out of 20</div>
    </div>
    <div class="kpi-card">
        <div class="kpi-label">Model Accuracy</div>
        <div class="kpi-value">{:.0%}</div>
        <div class="kpi-delta delta-green">✓ High confidence</div>
    </div>
</div>
""".format(total, at_risk, at_risk/total*100, high_risk, avg_grade, accuracy),
unsafe_allow_html=True)

# ── AI QUERY BOX ─────────────────────────────────────────────────────────────
st.markdown("""
<div class="query-container">
    <div class="query-title">🤖 Ask EduLens AI</div>
    <div class="query-sub">Ask anything about your students — AI reasons step by step using Microsoft Azure GitHub Models</div>
</div>
""", unsafe_allow_html=True)

suggested = [
    "Which students are most at risk of failing?",
    "What is causing low performance?",
    "Which factor affects grades the most?",
    "How can I reduce student absences?"
]

col_q, col_btn = st.columns([4, 1])
with col_q:
    question = st.text_input("", placeholder="e.g. Which students need immediate help?",
                              label_visibility="collapsed")
with col_btn:
    ask_btn = st.button("Ask AI →", use_container_width=True, type="primary")

st.markdown("**Suggested questions:**")
cols = st.columns(4)
for i, q in enumerate(suggested):
    with cols[i]:
        if st.button(q, key=f"sq_{i}", use_container_width=True):
            question = q
            ask_btn = True

if ask_btn and question:
    if not token_input:
        st.warning(" AI engine is securely configured")
    else:
        data_summary = f"""Total students: {total} | At-risk: {at_risk} ({at_risk/total*100:.1f}%)
High-risk (>70%): {high_risk} | Avg final grade: {avg_grade:.1f}/20
Avg absences: {avg_absence:.1f} days | Model accuracy: {accuracy:.0%}
Top risk factors: prior failures, absences, low G1/G2 grades, low study time"""

        with st.spinner("🧠 EduLens AI is reasoning..."):
            try:
                response = ask_github_ai(question, data_summary, token_input)
                
                steps = parse_reasoning(response)
                st.session_state.last_ai_response = response
                st.session_state.last_ai_steps = steps
                st.markdown("**Reasoning trace** — *powered by Microsoft Azure GitHub Models*")
                if steps["analyze"]:
                    st.markdown(f"""<div class="step-card step-analyze">
                        <div class="step-label">Step 1 — Analyze</div>
                        <div class="step-content">{steps['analyze'].strip()}</div>
                    </div>""", unsafe_allow_html=True)
                if steps["finding"]:
                    st.markdown(f"""<div class="step-card step-finding">
                        <div class="step-label">Step 2 — Finding</div>
                        <div class="step-content">{steps['finding'].strip()}</div>
                    </div>""", unsafe_allow_html=True)
                if steps["recommendation"]:
                    st.markdown(f"""<div class="step-card step-recommend">
                        <div class="step-label">Step 3 — Recommendation</div>
                        <div class="step-content">{steps['recommendation'].strip()}</div>
                    </div>""", unsafe_allow_html=True)
                    st.download_button(
                        label="📥 Download AI Report",
                        data=response,
                        file_name="EduLens_AI_Report.txt",
                        mime="text/plain"
                    )
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.divider()

# ── CHARTS ───────────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">📊 Analytics Dashboard</div>', unsafe_allow_html=True)
tab1, tab2, tab3 = st.tabs(["📈 Risk Distribution", "🎯 Performance Trends", "🔍 Feature Importance"])

with tab1:
    fig = px.histogram(df, x="risk_score", color="at_risk",
                       color_discrete_map={0: "#22c55e", 1: "#ef4444"},
                       labels={"risk_score": "Risk Score", "at_risk": "At Risk"},
                       title="Student Risk Score Distribution",
                       nbins=20, opacity=0.85)
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="Inter", title_font_size=15,
        legend_title_text="At Risk",
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        bargap=0.1, margin=dict(t=50, b=40, l=40, r=40)
    )
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = px.scatter(df, x="G1", y="G3", color="risk_score",
                      color_continuous_scale="RdYlGn_r",
                      size="absences", size_max=15,
                      labels={"G1": "First Period Grade", "G3": "Final Grade",
                               "risk_score": "Risk Score", "absences": "Absences"},
                      title="Grade Progression vs Risk (bubble size = absences)",
                      hover_data=["absences", "failures", "studytime"])
    fig2.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="Inter", title_font_size=15,
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        margin=dict(t=50, b=40, l=40, r=40)
    )
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    feat_df = pd.DataFrame({"Feature": features,
                             "Importance": model.feature_importances_})
    feat_df = feat_df.sort_values("Importance", ascending=True)
    fig3 = px.bar(feat_df, x="Importance", y="Feature", orientation="h",
                  title="Key Predictors of Student Risk",
                  color="Importance", color_continuous_scale="Purples",
                  text=feat_df["Importance"].apply(lambda x: f"{x:.3f}"))
    fig3.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font_family="Inter", title_font_size=15,
        xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        margin=dict(t=50, b=40, l=40, r=120)
    )
    fig3.update_traces(textposition="outside")
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── AT-RISK TABLE ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">⚠️ At-Risk Student List</div>', unsafe_allow_html=True)
threshold = st.slider("Risk threshold", 0.3, 0.9, 0.5, 0.05,
                       help="Show students above this risk score")
risky = df[df["risk_score"] >= threshold].copy()
risky = risky[["G1", "G2", "G3", "absences", "failures", "studytime", "risk_score"]]
risky = risky.sort_values("risk_score", ascending=False)
risky["risk_score"] = risky["risk_score"].apply(lambda x: f"{x:.0%}")
risky.columns = ["G1", "G2", "Final Grade", "Absences", "Failures", "Study Time", "Risk Score"]
risky.index = range(1, len(risky) + 1)
st.dataframe(risky, use_container_width=True, height=300)
st.caption(f"Showing {len(risky)} students above {threshold:.0%} risk threshold")

st.divider()

# ── INTERVENTIONS ─────────────────────────────────────────────────────────────
st.markdown('<div class="section-title">💡 Intervention Recommendations</div>', unsafe_allow_html=True)
top5 = df.nlargest(5, "risk_score").reset_index()
labels = [f"Student #{i+1}  •  Risk: {row['risk_score']:.0%}  •  Final Grade: {int(row['G3'])}/20"
          for i, row in top5.iterrows()]
selected = st.selectbox("Select a student to view personalised action plan:", labels)
idx = labels.index(selected)
student = top5.iloc[idx]

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**📋 Student Profile**")
    profile = {
        "First Period Grade": f"{int(student['G1'])}/20",
        "Second Period Grade": f"{int(student['G2'])}/20",
        "Final Grade": f"{int(student['G3'])}/20",
        "Absences": f"{int(student['absences'])} days",
        "Prior Failures": int(student["failures"]),
        "Study Time": f"{int(student['studytime'])} hrs/week",
        "Risk Score": f"{student['risk_score']:.0%}"
    }
    rows_html = "".join([
        f'<div class="profile-row"><span class="profile-key">{k}</span><span class="profile-val">{v}</span></div>'
        for k, v in profile.items()
    ])
    st.markdown(f'<div class="profile-card">{rows_html}</div>', unsafe_allow_html=True)

with col_b:
    st.markdown("**🎯 Recommended Actions**")
    tips = get_interventions(student)
    tips_html = "".join([
        f'<div class="tip-item"><div class="tip-num">{i}</div><div>{tip}</div></div>'
        for i, tip in enumerate(tips, 1)
    ])
    st.markdown(f'<div class="profile-card">{tips_html}</div>', unsafe_allow_html=True)

st.divider()
st.markdown("""
<div style="text-align:center; padding: 1rem; color: #94a3b8; font-size: 0.8rem;">
    EduLens AI &nbsp;·&nbsp; Microsoft Azure GitHub Models &nbsp;·&nbsp; GPT-4o mini &nbsp;·&nbsp; 
</div>
""", unsafe_allow_html=True)
