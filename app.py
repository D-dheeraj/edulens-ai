import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from openai import OpenAI

st.set_page_config(page_title="EduLens AI", page_icon="🎓", layout="wide")

st.markdown("""
<style>
    .main-header { font-size: 2rem; font-weight: 700; color: #1a1a2e; margin-bottom: 0.25rem; }
    .sub-header { color: #6b7280; font-size: 1rem; margin-bottom: 2rem; }
    .insight-box { background: #eff6ff; border-left: 4px solid #3b82f6; padding: 1rem 1.25rem;
        border-radius: 0 8px 8px 0; margin: 1rem 0; font-size: 0.9rem; color: #1e40af; }
    .reasoning-step { background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 0.75rem 1rem; margin: 0.5rem 0; font-size: 0.875rem; }
    .step-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase;
        letter-spacing: 0.08em; margin-bottom: 4px; }
    .step-analyze { color: #7c3aed; }
    .step-find { color: #0284c7; }
    .step-recommend { color: #16a34a; }
    .foundry-badge { background: #dbeafe; color: #1e40af; font-size: 0.75rem;
        padding: 3px 10px; border-radius: 100px; font-weight: 500; }
</style>
""", unsafe_allow_html=True)


GITHUB_TOKEN = ""
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
You reason step by step. Always structure your response in exactly 3 parts:
STEP 1 - ANALYZE: (what the data shows)
STEP 2 - FINDING: (key insight or pattern)
STEP 3 - RECOMMENDATION: (specific action for the teacher)
Be concise, specific, and use the actual numbers provided."""

    user_msg = f"""Dataset summary:
{data_summary}

Teacher's question: {question}

Answer with the 3-step reasoning format."""

    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_msg}
        ],
        max_tokens=500,
        temperature=0.3
    )
    return response.choices[0].message.content


def parse_reasoning(response_text):
    steps = {"analyze": "", "finding": "", "recommendation": ""}
    lines = response_text.split("\n")
    current = None
    for line in lines:
        line = line.strip()
        if "STEP 1" in line.upper() or "ANALYZE" in line.upper():
            current = "analyze"
            line = line.split(":", 1)[-1].strip() if ":" in line else ""
        elif "STEP 2" in line.upper() or "FINDING" in line.upper():
            current = "finding"
            line = line.split(":", 1)[-1].strip() if ":" in line else ""
        elif "STEP 3" in line.upper() or "RECOMMENDATION" in line.upper():
            current = "recommendation"
            line = line.split(":", 1)[-1].strip() if ":" in line else ""
        if current and line:
            steps[current] += " " + line
    if not any(steps.values()):
        steps["finding"] = response_text
    return steps


def get_interventions(row):
    tips = []
    if row["absences"] > 10:
        tips.append("Attendance counselling — absences significantly above average")
    if row["studytime"] < 2:
        tips.append("Study group enrollment — studying less than 2 hrs/week")
    if row["failures"] > 0:
        tips.append(f"Remedial support — {int(row['failures'])} prior course failure(s)")
    if row["G1"] < 8 or row["G2"] < 8:
        tips.append("Early grade intervention — mid-term scores show downward trend")
    if not tips:
        tips.append("Continue monitoring — borderline risk, no single dominant factor")
    return tips


# ── SIDEBAR ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎓 EduLens AI")
    st.markdown("*Student Performance Analytics*")
    st.divider()

    uploaded = st.file_uploader("Upload student CSV", type=["csv"])

    st.divider()
    st.markdown("**GitHub Models API Key**")
    token_input = st.text_input(
        "Paste your token here",
        type="password",
        placeholder="github_pat_xxx...",
        help="Get from github.com/marketplace/models"
    )
    if token_input:
        st.success("Token saved!")

    st.divider()
    st.markdown("**Built with**")
    st.markdown("🔷 Microsoft Azure · GitHub Models\n\n🤖 GPT-4o mini (OpenAI)\n\n🐍 Python · Pandas · scikit-learn\n\n📊 Plotly · Streamlit")
    st.markdown('<span class="foundry-badge">Microsoft AI Infrastructure</span>',
                unsafe_allow_html=True)


# ── MAIN ──────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">EduLens AI</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">AI-powered student analytics & early intervention system — powered by Microsoft Azure GitHub Models</div>',
            unsafe_allow_html=True)

if not uploaded:
    st.info("👈 Upload student-mat.csv from the sidebar to begin.")
    st.stop()

df_raw = load_and_process(uploaded)
model, df, accuracy, features = train_model(df_raw)

total = len(df)
at_risk = int(df["at_risk"].sum())
avg_grade = df["G3"].mean()
high_risk = int((df["risk_score"] > 0.7).sum())

# ── KPI ROW ──────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Total students", total)
c2.metric("At-risk students", at_risk,
          delta=f"{round(at_risk/total*100,1)}% of class", delta_color="inverse")
c3.metric("High-risk (>70%)", high_risk, delta_color="inverse")
c4.metric("Avg final grade", f"{avg_grade:.1f}/20")
c5.metric("Model accuracy", f"{accuracy:.0%}")

st.divider()

# ── AI QUERY BOX ─────────────────────────────────────────────────────────────
st.markdown("### Ask EduLens AI")
st.markdown("Type any question about your students — AI will reason step by step.")

col_q, col_btn = st.columns([4, 1])
with col_q:
    question = st.text_input(
        "Your question",
        placeholder="Which students are most at risk of failing? What is causing low performance?",
        label_visibility="collapsed"
    )
with col_btn:
    ask_btn = st.button("Ask AI ↗", use_container_width=True)

if ask_btn and question:
    if not token_input:
        st.warning("Please paste your GitHub Models token in the sidebar first.")
    else:
        data_summary = f"""
Total students: {total}
At-risk students: {at_risk} ({round(at_risk/total*100,1)}%)
High-risk (>70% score): {high_risk}
Average final grade: {avg_grade:.1f}/20
Average absences: {df['absences'].mean():.1f} days
Model accuracy: {accuracy:.0%}
Top risk factors: prior failures, absences, low G1/G2 grades
        """
        with st.spinner("EduLens AI is reasoning..."):
            try:
                response = ask_github_ai(question, data_summary, token_input)
                steps = parse_reasoning(response)

                st.markdown("**Reasoning trace** — powered by Microsoft Azure GitHub Models")

                if steps["analyze"]:
                    st.markdown(f"""<div class="reasoning-step">
                        <div class="step-label step-analyze">Step 1 — Analyze</div>
                        {steps['analyze'].strip()}
                    </div>""", unsafe_allow_html=True)

                if steps["finding"]:
                    st.markdown(f"""<div class="reasoning-step">
                        <div class="step-label step-find">Step 2 — Finding</div>
                        {steps['finding'].strip()}
                    </div>""", unsafe_allow_html=True)

                if steps["recommendation"]:
                    st.markdown(f"""<div class="reasoning-step">
                        <div class="step-label step-recommend">Step 3 — Recommendation</div>
                        {steps['recommendation'].strip()}
                    </div>""", unsafe_allow_html=True)

            except Exception as e:
                st.error(f"Error: {str(e)}. Please check your token.")

st.divider()

# ── CHARTS ───────────────────────────────────────────────────────────────────
st.markdown("### Analytics")
tab1, tab2, tab3 = st.tabs(["Risk distribution", "Performance trends", "Feature importance"])

with tab1:
    fig = px.histogram(df, x="risk_score", color="at_risk",
                       color_discrete_map={0: "#16a34a", 1: "#dc2626"},
                       labels={"risk_score": "Risk score", "at_risk": "At risk"},
                       title="Student risk score distribution", nbins=20)
    fig.update_layout(bargap=0.1)
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    fig2 = px.scatter(df, x="G1", y="G3", color="risk_score",
                      color_continuous_scale="RdYlGn_r",
                      labels={"G1": "First period grade", "G3": "Final grade"},
                      title="First period grade vs final grade",
                      hover_data=["absences", "failures"])
    st.plotly_chart(fig2, use_container_width=True)

with tab3:
    feat_df = pd.DataFrame({"feature": features,
                             "importance": model.feature_importances_})
    feat_df = feat_df.sort_values("importance", ascending=True)
    fig3 = px.bar(feat_df, x="importance", y="feature", orientation="h",
                  title="What predicts student risk most",
                  color="importance", color_continuous_scale="Blues")
    st.plotly_chart(fig3, use_container_width=True)

st.divider()

# ── AT-RISK TABLE ─────────────────────────────────────────────────────────────
st.markdown("### At-risk student list")
threshold = st.slider("Risk threshold", 0.3, 0.9, 0.5, 0.05)
risky = df[df["risk_score"] >= threshold].copy()
risky = risky[["G1", "G2", "G3", "absences", "failures",
               "studytime", "risk_score"]].sort_values("risk_score", ascending=False)
risky["risk_score"] = risky["risk_score"].apply(lambda x: f"{x:.0%}")
risky.index = range(1, len(risky) + 1)
st.dataframe(risky, use_container_width=True)
st.caption(f"{len(risky)} students above {threshold:.0%} risk threshold")

st.divider()

# ── INTERVENTIONS ─────────────────────────────────────────────────────────────
st.markdown("### Intervention recommendations")
top5 = df.nlargest(5, "risk_score").reset_index()
labels = [f"Student #{i+1} — Risk: {row['risk_score']:.0%}" for i, row in top5.iterrows()]
selected = st.selectbox("Choose student", labels)
idx = labels.index(selected)
student = top5.iloc[idx]

col_a, col_b = st.columns(2)
with col_a:
    st.markdown("**Student profile**")
    for k, v in {
        "First period grade": f"{int(student['G1'])}/20",
        "Second period grade": f"{int(student['G2'])}/20",
        "Final grade": f"{int(student['G3'])}/20",
        "Absences": int(student["absences"]),
        "Prior failures": int(student["failures"]),
        "Study time (hrs/wk)": int(student["studytime"]),
        "Risk score": f"{student['risk_score']:.0%}"
    }.items():
        st.markdown(f"- **{k}:** {v}")

with col_b:
    st.markdown("**Recommended interventions**")
    for i, tip in enumerate(get_interventions(student), 1):
        st.markdown(f"{i}. {tip}")

st.divider()
st.caption("EduLens AI · Microsoft Azure GitHub Models · GPT-4o mini · Hackathon 2026")