import streamlit as st
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Employee Churn Predictor",
    page_icon="👥",
    layout="centered"
)

# ── Generate synthetic training data based on real HR dataset patterns ────────
@st.cache_resource
def train_model():
    np.random.seed(42)
    n = 3000

    # Simulate realistic HR data patterns
    satisfaction   = np.random.beta(2, 2, n)
    last_eval      = np.random.beta(3, 2, n)
    num_projects   = np.random.randint(2, 8, n)
    avg_hours      = np.random.normal(200, 40, n).clip(80, 320).astype(int)
    tenure         = np.random.choice([2,3,4,5,6,7,8,10], n, p=[0.2,0.2,0.15,0.15,0.1,0.1,0.05,0.05])
    work_accident  = np.random.choice([0, 1], n, p=[0.86, 0.14])
    promoted       = np.random.choice([0, 1], n, p=[0.97, 0.03])
    dept_enc       = np.random.randint(0, 10, n)
    salary_enc     = np.random.choice([0, 1, 2], n, p=[0.48, 0.43, 0.09])  # low, medium, high

    # Churn probability based on real patterns
    churn_prob = (
        0.4 * (satisfaction < 0.4).astype(float) +
        0.3 * (avg_hours > 240).astype(float) +
        0.2 * (num_projects >= 6).astype(float) +
        0.2 * (promoted == 0).astype(float) * (tenure >= 5).astype(float) +
        0.1 * (salary_enc == 0).astype(float) +
        np.random.normal(0, 0.1, n)
    ).clip(0, 1)

    left = (churn_prob > 0.45).astype(int)

    X = np.column_stack([
        satisfaction, last_eval, num_projects, avg_hours,
        tenure, work_accident, promoted, dept_enc, salary_enc
    ])
    y = left

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)

    # Encoders to map user inputs
    departments = ['IT', 'RandD', 'accounting', 'hr', 'management',
                   'marketing', 'product_mng', 'sales', 'support', 'technical']
    salaries = ['high', 'low', 'medium']  # alphabetical = LabelEncoder order

    return model, departments, salaries

model, departments, salaries = train_model()

# ── UI ────────────────────────────────────────────────────────────────────────
st.title("👥 Employee Churn Predictor")
st.markdown(
    "Enter an employee's details below to predict whether they are likely "
    "to **leave** or **stay** at the company."
)

st.divider()

col1, col2 = st.columns(2)

with col1:
    satisfaction   = st.slider("Satisfaction Level", 0.0, 1.0, 0.5, 0.01)
    last_eval      = st.slider("Last Evaluation Score", 0.0, 1.0, 0.7, 0.01)
    num_projects   = st.slider("Number of Projects", 1, 10, 4)
    avg_hours      = st.slider("Avg Monthly Hours", 80, 320, 160)

with col2:
    tenure         = st.slider("Tenure (Years)", 1, 15, 3)
    work_accident  = st.selectbox("Work Accident?", ["No", "Yes"])
    promoted       = st.selectbox("Promoted in Last 5 Years?", ["No", "Yes"])
    department     = st.selectbox("Department", departments)
    salary         = st.selectbox("Salary Level", ["low", "medium", "high"])

st.divider()

if st.button("🔍 Predict Churn Risk", use_container_width=True):

    dept_enc = departments.index(department)
    sal_enc  = salaries.index(salary)  # alphabetical: high=0, low=1, medium=2

    X_input = np.array([[
        satisfaction,
        last_eval,
        num_projects,
        avg_hours,
        tenure,
        1 if work_accident == "Yes" else 0,
        1 if promoted == "Yes" else 0,
        dept_enc,
        sal_enc
    ]])

    prediction   = model.predict(X_input)[0]
    probability  = model.predict_proba(X_input)[0]
    churn_prob   = round(probability[1] * 100, 1)
    stay_prob    = round(probability[0] * 100, 1)

    st.subheader("📊 Prediction Result")

    if prediction == 1:
        st.error("⚠️ This employee is **likely to LEAVE** the company.")
    else:
        st.success("✅ This employee is **likely to STAY** at the company.")

    col_a, col_b = st.columns(2)
    col_a.metric("Churn Risk",            f"{churn_prob}%")
    col_b.metric("Retention Likelihood",  f"{stay_prob}%")

    st.progress(int(churn_prob))

    st.divider()
    st.subheader("💡 Key Risk Factors")
    flags = 0
    if satisfaction < 0.4:
        st.warning("🔴 Low satisfaction level — a major churn driver.")
        flags += 1
    if avg_hours > 240:
        st.warning("🔴 Very high monthly hours — burnout risk.")
        flags += 1
    if num_projects >= 6:
        st.warning("🔴 High number of projects — overload risk.")
        flags += 1
    if tenure == 4 and prediction == 1:
        st.warning("🔴 4-year tenure with churn risk — common pattern.")
        flags += 1
    if promoted == "No" and tenure >= 5:
        st.info("🟡 No promotion in 5+ years — potential dissatisfaction.")
        flags += 1
    if flags == 0:
        st.success("🟢 No major risk factors detected.")

st.divider()
st.caption(
    "Built by Mustajab Hussain · "
    "[LinkedIn](https://www.linkedin.com/in/mustajab-hussain-312475283/) · "
    "[GitHub](https://github.com/jabiyan) · "
    "[Upwork](https://www.upwork.com/freelancers/~011cf146030b8908fd)"
)
