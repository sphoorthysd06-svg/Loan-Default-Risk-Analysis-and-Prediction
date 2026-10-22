import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# =========================================================
# PAGE SETTINGS
# =========================================================

st.set_page_config(
    page_title="Loan Default Risk Analysis",
    page_icon="🏦",
    layout="wide"
)

# ============================================================
# FIN-CARE OFFICE BACKGROUND
# ============================================================

import base64

def set_background(image_file):

    with open(image_file, "rb") as file:
        image_data = base64.b64encode(
            file.read()
        ).decode()

    st.markdown(
        f"""
        <style>

        .stApp {{
            background-image:
                linear-gradient(
                    rgba(255, 255, 255, 0.88),
                    rgba(255, 255, 255, 0.88)
                ),
                url("data:image/png;base64,{image_data}");

            background-size: cover;
            background-position: center top;
            background-repeat: no-repeat;
            background-attachment: fixed;
        }}

        </style>
        """,
        unsafe_allow_html=True
    )

set_background("Fin-Care Office.png")


# =========================================================
# LOAD CLEANED CSV
# =========================================================

file_path = Path(__file__).parent / "loan_default_cleaned.csv"

df = pd.read_csv(file_path)


# =========================================================
# CLEAN COLUMN NAMES
# =========================================================

df.columns = (
    df.columns
    .str.strip()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace("%", "Percentage", regex=False)
)


# Rename CIBIL column
if "Credit_Score_Cibil" in df.columns:
    df.rename(
        columns={
            "Credit_Score_Cibil": "Credit_Score"
        },
        inplace=True
    )


# =========================================================
# DATA TYPES
# =========================================================

df["Monthly_Income"] = (
    df["Monthly_Income"]
    .astype(str)
    .str.replace(",", "", regex=False)
)

df["Monthly_Income"] = pd.to_numeric(
    df["Monthly_Income"],
    errors="coerce"
)

df["Loan_Amount"] = pd.to_numeric(
    df["Loan_Amount"],
    errors="coerce"
)

df["Loan_Amount_Lakhs"] = (
    df["Loan_Amount_Lakhs"]
    .astype(str)
    .str.replace("L", "", regex=False)
)

df["Loan_Amount_Lakhs"] = pd.to_numeric(
    df["Loan_Amount_Lakhs"],
    errors="coerce"
)

df["Interest_Rate"] = pd.to_numeric(
    df["Interest_Rate"],
    errors="coerce"
)

df["Interest_Rate_Percentage"] = (
    df["Interest_Rate_Percentage"]
    .astype(str)
    .str.replace("%", "", regex=False)
)

df["Interest_Rate_Percentage"] = pd.to_numeric(
    df["Interest_Rate_Percentage"],
    errors="coerce"
)

df["Credit_Score"] = pd.to_numeric(
    df["Credit_Score"],
    errors="coerce"
)

df["Default_Flag"] = pd.to_numeric(
    df["Default_Flag"],
    errors="coerce"
)

df["Debt_to_Income_Ratio"] = pd.to_numeric(
    df["Debt_to_Income_Ratio"],
    errors="coerce"
)

df["Loan_Approval_Date"] = pd.to_datetime(
    df["Loan_Approval_Date"],
    errors="coerce"
)

df["Year"] = df["Loan_Approval_Date"].dt.year


# =========================================================
# DERIVED COLUMNS
# =========================================================

df["Default_Status"] = np.where(
    df["Default_Flag"] == 1,
    "Default",
    "Non-Default"
)


df["Credit_Score_Category"] = np.select(
    [
        df["Credit_Score"] < 600,
        df["Credit_Score"] < 700
    ],
    [
        "Poor",
        "Average"
    ],
    default="Good"
)


df["DTI_Category"] = np.select(
    [
        df["Debt_to_Income_Ratio"] < 0.30,
        df["Debt_to_Income_Ratio"] < 0.50
    ],
    [
        "Low",
        "Medium"
    ],
    default="High"
)


df["Risk_Score"] = (
    (850 - df["Credit_Score"])
    + (df["Debt_to_Income_Ratio"] * 100)
    + (df["Interest_Rate"] * 2)
)


df["Risk_Level"] = np.select(
    [
        df["Risk_Score"] >= 250,
        df["Risk_Score"] >= 150
    ],
    [
        "High",
        "Medium"
    ],
    default="Low"
)


# =========================================================
# MACHINE LEARNING
# =========================================================

features = [
    "Loan_Type",
    "Age_Group",
    "Employment_Type",
    "Monthly_Income",
    "Loan_Amount",
    "Tenure_Months",
    "Interest_Rate",
    "EMI",
    "Credit_Score",
    "Previous_Default",
    "Existing_Loans",
    "Debt_to_Income_Ratio"
]


categorical_features = [
    "Loan_Type",
    "Age_Group",
    "Employment_Type",
    "Previous_Default"
]


numeric_features = [
    "Monthly_Income",
    "Loan_Amount",
    "Tenure_Months",
    "Interest_Rate",
    "EMI",
    "Credit_Score",
    "Existing_Loans",
    "Debt_to_Income_Ratio"
]


X = df[features]
y = df["Default_Flag"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "num",
            StandardScaler(),
            numeric_features
        )
    ]
)


# =========================================================
# LOGISTIC REGRESSION
# =========================================================

logistic_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            LogisticRegression(
                max_iter=3000,
                class_weight="balanced"
            )
        )
    ]
)

logistic_model.fit(X_train, y_train)

log_pred = logistic_model.predict(X_test)

log_prob = logistic_model.predict_proba(
    X_test
)[:, 1]


# =========================================================
# RANDOM FOREST
# =========================================================

rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=8,
                class_weight="balanced",
                random_state=42
            )
        )
    ]
)

rf_model.fit(X_train, y_train)

rf_pred = rf_model.predict(X_test)

rf_prob = rf_model.predict_proba(
    X_test
)[:, 1]


# =========================================================
# MODEL METRICS
# =========================================================

log_accuracy = accuracy_score(
    y_test,
    log_pred
)

log_precision = precision_score(
    y_test,
    log_pred,
    zero_division=0
)

log_recall = recall_score(
    y_test,
    log_pred,
    zero_division=0
)

log_f1 = f1_score(
    y_test,
    log_pred,
    zero_division=0
)

log_auc = roc_auc_score(
    y_test,
    log_prob
)


rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

rf_precision = precision_score(
    y_test,
    rf_pred,
    zero_division=0
)

rf_recall = recall_score(
    y_test,
    rf_pred,
    zero_division=0
)

rf_f1 = f1_score(
    y_test,
    rf_pred,
    zero_division=0
)

rf_auc = roc_auc_score(
    y_test,
    rf_prob
)


# =========================================================
# LOGIN
# =========================================================

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False


if not st.session_state.logged_in:

    st.title("🏦 Loan Default Risk Analysis")

    st.subheader(
        "Analyse | Predict | Minimise Risk"
    )

    st.write(
        "Fin-Care Starpowerz Digital Technology Pvt. Ltd."
    )

    st.divider()

    st.subheader("🔐 Login")

    user_id = st.text_input(
        "User ID",
        placeholder="Enter User ID"
    )

    password = st.text_input(
        "Password",
        type="password",
        placeholder="Enter Password"
    )

    if st.button(
        "🔐 Login",
        use_container_width=True
    ):

        if (
            user_id == "sony"
            and password == "Sony@4961"
        ):

            st.session_state.logged_in = True

            st.rerun()

        else:

            st.error(
                "Invalid User ID or Password"
            )

    st.stop()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("🏦 Loan Default Risk")

st.sidebar.write(
    "Fin-Care Starpowerz Digital Technology Pvt. Ltd."
)

st.sidebar.divider()


page = st.sidebar.radio(
    "Menu",
    [
        "🏠 Home",
        "📊 Dashboard",
        "📈 Loan Analytics",
        "⚠️ Risk Analysis",
        "📋 Transactions",
        "🔮 Loan Prediction",
        "🤖 Model Performance"
    ]
)


st.sidebar.divider()


if st.sidebar.button(
    "🚪 Logout",
    use_container_width=True
):

    st.session_state.logged_in = False

    st.rerun()


# =========================================================
# HOME
# =========================================================

if page == "🏠 Home":

    st.title(
        "🏦 Welcome to Loan Default Risk Analysis"
    )

    st.subheader(
        "Analyse | Predict | Minimise Risk"
    )

    st.write(
        "This Streamlit application analyzes loan data "
        "to identify default patterns, risk factors "
        "and loan default prediction."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:

        with st.container(border=True):

            st.subheader("📊 Dashboard")

            st.write(
                "View loan KPIs, default distribution, "
                "loan trends and loan performance."
            )

    with col2:

        with st.container(border=True):

            st.subheader("⚠️ Risk Analysis")

            st.write(
                "Analyze credit score, DTI, interest rate "
                "and other loan risk indicators."
            )

    with col3:

        with st.container(border=True):

            st.subheader("🔮 Loan Prediction")

            st.write(
                "Use machine learning to predict "
                "the probability of loan default."
            )

    st.divider()

    st.subheader("📌 Project Information")

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "**Company:** Fin-Care Starpowerz "
            "Digital Technology Pvt. Ltd."
        )

        st.write(
            "**Location:** Vijayapura, Karnataka"
        )

        st.write(
            "**Domain:** Financial Services / Loan Lending"
        )

    with col2:

        st.write(
            "**Project:** Loan Default Risk Analysis "
            "and Prediction"
        )

        st.write(
            "**Dataset:** 2,000 loan records"
        )

        st.write(
            "**Technology:** Python, Machine Learning, "
            "Streamlit, Power BI, MySQL"
        )


# =========================================================
# DASHBOARD
# =========================================================

elif page == "📊 Dashboard":

    st.title(
        "📊 Loan Default Risk Dashboard"
    )

    st.caption(
        "Loan Default Risk Analysis | Fin-Care"
    )

    # KPI ROW
    col1, col2, col3, col4, col5 = st.columns(5)

    total_loans = len(df)

    default_loans = int(
        df["Default_Flag"].sum()
    )

    non_default_loans = int(
        (df["Default_Flag"] == 0).sum()
    )

    default_rate = (
        df["Default_Flag"].mean() * 100
    )

    avg_credit = df["Credit_Score"].mean()

    with col1:

        st.metric(
            "Total Loans",
            f"{total_loans:,}"
        )

    with col2:

        st.metric(
            "Default Loans",
            f"{default_loans:,}"
        )

    with col3:

        st.metric(
            "Non-Default Loans",
            f"{non_default_loans:,}"
        )

    with col4:

        st.metric(
            "Default Rate",
            f"{default_rate:.2f}%"
        )

    with col5:

        st.metric(
            "Average Credit Score",
            f"{avg_credit:.2f}"
        )

    st.divider()

    # ROW 1
    col1, col2, col3 = st.columns(3)

    with col1:

        with st.container(border=True):

            st.subheader(
                "Loan Default Distribution"
            )

            data = (
                df["Default_Status"]
                .value_counts()
                .reset_index()
            )

            data.columns = [
                "Status",
                "Count"
            ]

            fig = px.pie(
                data,
                names="Status",
                values="Count",
                hole=0.55
            )

            fig.update_traces(
                textinfo="percent",
                hovertemplate=(
                    "%{label}<br>"
                    "Loans: %{value}<br>"
                    "%{percent}"
                    "<extra></extra>"
                )
            )

            fig.update_layout(
                height=330,
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col2:

        with st.container(border=True):

            st.subheader(
                "Defaults by Loan Type"
            )

            data = (
                df.groupby("Loan_Type")[
                    "Default_Flag"
                ]
                .sum()
                .reset_index()
                .sort_values(
                    "Default_Flag",
                    ascending=False
                )
            )

            fig = px.bar(
                data,
                x="Loan_Type",
                y="Default_Flag",
                labels={
                    "Default_Flag":
                    "Default Loans",
                    "Loan_Type":
                    "Loan Type"
                },
                text="Default_Flag"
            )

            fig.update_traces(
                textposition="outside"
            )

            fig.update_layout(
                height=330,
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col3:

        with st.container(border=True):

            st.subheader(
                "Year-wise Default Trend"
            )

            data = (
                df.groupby("Year")[
                    "Default_Flag"
                ]
                .mean()
                .mul(100)
                .reset_index()
            )

            fig = px.line(
                data,
                x="Year",
                y="Default_Flag",
                markers=True,
                labels={
                    "Default_Flag":
                    "Default Rate (%)"
                }
            )

            fig.update_layout(
                height=330,
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # ROW 2
    col1, col2, col3 = st.columns(3)

    with col1:

        with st.container(border=True):

            st.subheader(
                "Defaults by Credit Score"
            )

            data = (
                df.groupby(
                    "Credit_Score_Category"
                )["Default_Flag"]
                .mean()
                .mul(100)
                .reset_index()
            )

            fig = px.bar(
                data,
                x="Credit_Score_Category",
                y="Default_Flag",
                text="Default_Flag",
                labels={
                    "Credit_Score_Category":
                    "Credit Category",
                    "Default_Flag":
                    "Default Rate (%)"
                }
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            fig.update_layout(
                height=330,
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col2:

        with st.container(border=True):

            st.subheader(
                "Defaults by Employment Type"
            )

            data = (
                df.groupby(
                    "Employment_Type"
                )["Default_Flag"]
                .sum()
                .reset_index()
                .sort_values(
                    "Default_Flag",
                    ascending=False
                )
            )

            fig = px.bar(
                data,
                x="Employment_Type",
                y="Default_Flag",
                text="Default_Flag",
                labels={
                    "Employment_Type":
                    "Employment Type",
                    "Default_Flag":
                    "Default Loans"
                }
            )

            fig.update_layout(
                height=330,
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col3:

        with st.container(border=True):

            st.subheader(
                "Defaults by Age Group"
            )

            data = (
                df.groupby(
                    "Age_Group"
                )["Default_Flag"]
                .sum()
                .reset_index()
            )

            fig = px.bar(
                data,
                x="Age_Group",
                y="Default_Flag",
                text="Default_Flag",
                labels={
                    "Age_Group":
                    "Age Group",
                    "Default_Flag":
                    "Default Loans"
                }
            )

            fig.update_layout(
                height=330,
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10
                )
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================================
# LOAN ANALYTICS
# =========================================================

elif page == "📈 Loan Analytics":

    st.title(
        "📈 Loan Analytics"
    )

    # ROW 1
    col1, col2, col3 = st.columns(3)

    with col1:

        with st.container(border=True):

            st.subheader(
                "Loan Type Distribution"
            )

            data = (
                df["Loan_Type"]
                .value_counts()
                .reset_index()
            )

            data.columns = [
                "Loan Type",
                "Count"
            ]

            fig = px.pie(
                data,
                names="Loan Type",
                values="Count",
                hole=0.45
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col2:

        with st.container(border=True):

            st.subheader(
                "Employment Type Distribution"
            )

            data = (
                df["Employment_Type"]
                .value_counts()
                .reset_index()
            )

            data.columns = [
                "Employment Type",
                "Count"
            ]

            fig = px.bar(
                data,
                x="Employment Type",
                y="Count",
                text="Count"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col3:

        with st.container(border=True):

            st.subheader(
                "Age Group Distribution"
            )

            data = (
                df["Age_Group"]
                .value_counts()
                .reset_index()
            )

            data.columns = [
                "Age Group",
                "Count"
            ]

            fig = px.bar(
                data,
                x="Age Group",
                y="Count",
                text="Count"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # ROW 2
    col1, col2, col3 = st.columns(3)

    with col1:

        with st.container(border=True):

            st.subheader(
                "Average Loan Amount"
            )

            data = (
                df.groupby("Loan_Type")[
                    "Loan_Amount"
                ]
                .mean()
                .reset_index()
            )

            fig = px.bar(
                data,
                x="Loan_Type",
                y="Loan_Amount",
                text="Loan_Amount"
            )

            fig.update_traces(
                texttemplate="₹%{text:,.0f}",
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col2:

        with st.container(border=True):

            st.subheader(
                "Average Credit Score"
            )

            data = (
                df.groupby("Loan_Type")[
                    "Credit_Score"
                ]
                .mean()
                .reset_index()
            )

            fig = px.bar(
                data,
                x="Loan_Type",
                y="Credit_Score",
                text="Credit_Score"
            )

            fig.update_traces(
                texttemplate="%{text:.0f}",
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col3:

        with st.container(border=True):

            st.subheader(
                "Average EMI by Loan Type"
            )

            data = (
                df.groupby("Loan_Type")[
                    "EMI"
                ]
                .mean()
                .reset_index()
            )

            fig = px.bar(
                data,
                x="Loan_Type",
                y="EMI",
                text="EMI"
            )

            fig.update_traces(
                texttemplate="₹%{text:,.0f}",
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # ROW 3
    col1, col2, col3 = st.columns(3)

    with col1:

        with st.container(border=True):

            st.subheader(
                "Loan Amount by Year"
            )

            data = (
                df.groupby("Year")[
                    "Loan_Amount"
                ]
                .sum()
                .reset_index()
            )

            fig = px.line(
                data,
                x="Year",
                y="Loan_Amount",
                markers=True
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col2:

        with st.container(border=True):

            st.subheader(
                "Loan Approvals by Year"
            )

            data = (
                df["Year"]
                .value_counts()
                .sort_index()
                .reset_index()
            )

            data.columns = [
                "Year",
                "Loans"
            ]

            fig = px.bar(
                data,
                x="Year",
                y="Loans",
                text="Loans"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col3:

        with st.container(border=True):

            st.subheader(
                "Default Rate by Loan Type"
            )

            data = (
                df.groupby("Loan_Type")[
                    "Default_Flag"
                ]
                .mean()
                .mul(100)
                .reset_index()
            )

            fig = px.bar(
                data,
                x="Loan_Type",
                y="Default_Flag",
                text="Default_Flag"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================================
# RISK ANALYSIS
# =========================================================

elif page == "⚠️ Risk Analysis":

    st.title(
        "⚠️ Loan Risk Analysis"
    )

    # ROW 1
    col1, col2, col3 = st.columns(3)

    with col1:

        with st.container(border=True):

            st.subheader(
                "Risk Level Distribution"
            )

            data = (
                df["Risk_Level"]
                .value_counts()
                .reset_index()
            )

            data.columns = [
                "Risk Level",
                "Count"
            ]

            fig = px.pie(
                data,
                names="Risk Level",
                values="Count",
                hole=0.45
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col2:

        with st.container(border=True):

            st.subheader(
                "Default Rate by Risk Level"
            )

            data = (
                df.groupby("Risk_Level")[
                    "Default_Flag"
                ]
                .mean()
                .mul(100)
                .reset_index()
            )

            fig = px.bar(
                data,
                x="Risk_Level",
                y="Default_Flag",
                text="Default_Flag"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col3:

        with st.container(border=True):

            st.subheader(
                "Credit Score Category"
            )

            data = (
                df.groupby(
                    "Credit_Score_Category"
                )["Default_Flag"]
                .mean()
                .mul(100)
                .reset_index()
            )

            fig = px.bar(
                data,
                x="Credit_Score_Category",
                y="Default_Flag",
                text="Default_Flag"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # ROW 2
    col1, col2, col3 = st.columns(3)

    with col1:

        with st.container(border=True):

            st.subheader(
                "DTI Risk Analysis"
            )

            data = (
                df.groupby("DTI_Category")[
                    "Default_Flag"
                ]
                .mean()
                .mul(100)
                .reset_index()
            )

            fig = px.bar(
                data,
                x="DTI_Category",
                y="Default_Flag",
                text="Default_Flag"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col2:

        with st.container(border=True):

            st.subheader(
                "Previous Default Analysis"
            )

            data = (
                df.groupby("Previous_Default")[
                    "Default_Flag"
                ]
                .mean()
                .mul(100)
                .reset_index()
            )

            fig = px.bar(
                data,
                x="Previous_Default",
                y="Default_Flag",
                text="Default_Flag"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col3:

        with st.container(border=True):

            st.subheader(
                "Interest Rate Analysis"
            )

            data = (
                df.groupby("Year")[
                    "Interest_Rate"
                ]
                .mean()
                .reset_index()
            )

            fig = px.line(
                data,
                x="Year",
                y="Interest_Rate",
                markers=True
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    # ROW 3
    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.subheader(
                "Risk Score Distribution"
            )

            fig = px.histogram(
                df,
                x="Risk_Score",
                nbins=20
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col2:

        with st.container(border=True):

            st.subheader(
                "Credit Score vs Default"
            )

            fig = px.scatter(
                df,
                x="Credit_Score",
                y="Debt_to_Income_Ratio",
                color="Default_Status",
                hover_data=[
                    "Loan_ID",
                    "Loan_Type",
                    "Loan_Amount"
                ]
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )


# =========================================================
# TRANSACTIONS
# =========================================================

elif page == "📋 Transactions":

    st.title(
        "📋 Loan Transactions"
    )

    st.write(
        f"Total Records: {len(df):,}"
    )

    search = st.text_input(
        "🔍 Search Loan ID"
    )

    display_df = df.copy()

    if search:

        display_df = display_df[
            display_df["Loan_ID"]
            .astype(str)
            .str.contains(
                search,
                case=False,
                na=False
            )
        ]

    columns = [
        "Loan_ID",
        "Loan_Type",
        "Age_Group",
        "Employment_Type",
        "Monthly_Income",
        "Loan_Amount",
        "Tenure_Months",
        "Interest_Rate",
        "EMI",
        "Credit_Score",
        "Previous_Default",
        "Existing_Loans",
        "Debt_to_Income_Ratio",
        "Loan_Approval_Date",
        "Outstanding_Amount",
        "Repayment_Status",
        "Default_Flag"
    ]

    st.dataframe(
        display_df[columns],
        use_container_width=True,
        height=550
    )


# =========================================================
# LOAN PREDICTION
# =========================================================

elif page == "🔮 Loan Prediction":

    st.title(
        "🔮 Loan Default Prediction"
    )

    st.write(
        "Enter the applicant details to predict "
        "loan default risk."
    )

    col1, col2, col3 = st.columns(3)

    with col1:

        loan_type = st.selectbox(
            "Loan Type",
            sorted(
                df["Loan_Type"]
                .dropna()
                .unique()
            )
        )

        age_group = st.selectbox(
            "Age Group",
            sorted(
                df["Age_Group"]
                .dropna()
                .unique()
            )
        )

        employment = st.selectbox(
            "Employment Type",
            sorted(
                df["Employment_Type"]
                .dropna()
                .unique()
            )
        )

        previous_default = st.selectbox(
            "Previous Default",
            sorted(
                df["Previous_Default"]
                .dropna()
                .unique()
            )
        )

    with col2:

        monthly_income = st.number_input(
            "Monthly Income",
            min_value=0.0,
            value=30000.0
        )

        loan_amount = st.number_input(
            "Loan Amount",
            min_value=0.0,
            value=500000.0
        )

        tenure = st.number_input(
            "Tenure Months",
            min_value=1,
            value=36
        )

        interest = st.number_input(
            "Interest Rate",
            min_value=0.0,
            value=12.0
        )

    with col3:

        emi = st.number_input(
            "EMI",
            min_value=0.0,
            value=15000.0
        )

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=700
        )

        existing_loans = st.number_input(
            "Existing Loans",
            min_value=0,
            value=1
        )

        dti = st.number_input(
            "Debt to Income Ratio",
            min_value=0.0,
            max_value=2.0,
            value=0.30
        )

    st.divider()

    if st.button(
        "🔮 Predict Loan Risk",
        use_container_width=True
    ):

        input_data = pd.DataFrame({

            "Loan_Type": [loan_type],

            "Age_Group": [age_group],

            "Employment_Type": [employment],

            "Monthly_Income": [
                monthly_income
            ],

            "Loan_Amount": [
                loan_amount
            ],

            "Tenure_Months": [
                tenure
            ],

            "Interest_Rate": [
                interest
            ],

            "EMI": [
                emi
            ],

            "Credit_Score": [
                credit_score
            ],

            "Previous_Default": [
                previous_default
            ],

            "Existing_Loans": [
                existing_loans
            ],

            "Debt_to_Income_Ratio": [
                dti
            ]
        })

        prediction = logistic_model.predict(
            input_data
        )[0]

        probability = logistic_model.predict_proba(
            input_data
        )[0][1]

        st.divider()

        if prediction == 1:

            st.error(
                "⚠️ Loan Default Risk Detected"
            )

            st.metric(
                "Default Probability",
                f"{probability * 100:.2f}%"
            )

        else:

            st.success(
                "✅ Lower Loan Default Risk"
            )

            st.metric(
                "Default Probability",
                f"{probability * 100:.2f}%"
            )


# =========================================================
# MODEL PERFORMANCE
# =========================================================

elif page == "🤖 Model Performance":

    st.title(
        "🤖 Machine Learning Model Performance"
    )

    # KPI
    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Logistic Accuracy",
            f"{log_accuracy * 100:.2f}%"
        )

    with col2:

        st.metric(
            "Logistic ROC-AUC",
            f"{log_auc * 100:.2f}%"
        )

    with col3:

        st.metric(
            "Random Forest Accuracy",
            f"{rf_accuracy * 100:.2f}%"
        )

    with col4:

        st.metric(
            "Random Forest ROC-AUC",
            f"{rf_auc * 100:.2f}%"
        )

    st.divider()

    # MODEL COMPARISON
    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.subheader(
                "Model Performance Comparison"
            )

            model_data = pd.DataFrame({
                "Model": [
                    "Logistic Regression",
                    "Random Forest"
                ],
                "Accuracy": [
                    log_accuracy * 100,
                    rf_accuracy * 100
                ],
                "ROC-AUC": [
                    log_auc * 100,
                    rf_auc * 100
                ]
            })

            long_data = model_data.melt(
                id_vars="Model",
                value_vars=[
                    "Accuracy",
                    "ROC-AUC"
                ],
                var_name="Metric",
                value_name="Score"
            )

            fig = px.bar(
                long_data,
                x="Model",
                y="Score",
                color="Metric",
                barmode="group",
                text="Score"
            )

            fig.update_traces(
                texttemplate="%{text:.1f}%",
                textposition="outside"
            )

            fig.update_layout(
                yaxis_title="Score (%)"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col2:

        with st.container(border=True):

            st.subheader(
                "Model Evaluation"
            )

            result_data = pd.DataFrame({
                "Metric": [
                    "Accuracy",
                    "Precision",
                    "Recall",
                    "F1 Score",
                    "ROC-AUC"
                ],
                "Logistic Regression": [
                    log_accuracy,
                    log_precision,
                    log_recall,
                    log_f1,
                    log_auc
                ],
                "Random Forest": [
                    rf_accuracy,
                    rf_precision,
                    rf_recall,
                    rf_f1,
                    rf_auc
                ]
            })

            st.dataframe(
                result_data.style.format({
                    "Logistic Regression":
                    "{:.2%}",

                    "Random Forest":
                    "{:.2%}"
                }),
                use_container_width=True
            )

    st.divider()

    # CONFUSION MATRIX
    col1, col2 = st.columns(2)

    with col1:

        with st.container(border=True):

            st.subheader(
                "Random Forest Confusion Matrix"
            )

            cm = confusion_matrix(
                y_test,
                rf_pred
            )

            fig = px.imshow(
                cm,
                text_auto=True,
                x=[
                    "Non-Default",
                    "Default"
                ],
                y=[
                    "Non-Default",
                    "Default"
                ],
                labels={
                    "x": "Predicted",
                    "y": "Actual"
                }
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

    with col2:

        with st.container(border=True):

            st.subheader(
                "Random Forest Feature Importance"
            )

            rf_preprocessor = (
                rf_model
                .named_steps[
                    "preprocessor"
                ]
            )

            rf_classifier = (
                rf_model
                .named_steps[
                    "model"
                ]
            )

            feature_names = (
                rf_preprocessor
                .get_feature_names_out()
            )

            importance = pd.DataFrame({
                "Feature": feature_names,
                "Importance":
                    rf_classifier.feature_importances_
            })

            importance["Feature"] = (
                importance["Feature"]
                .str.replace(
                    "cat__",
                    "",
                    regex=False
                )
                .str.replace(
                    "num__",
                    "",
                    regex=False
                )
            )

            importance = (
                importance
                .sort_values(
                    "Importance",
                    ascending=False
                )
                .head(12)
            )

            fig = px.bar(
                importance.sort_values(
                    "Importance"
                ),
                x="Importance",
                y="Feature",
                orientation="h",
                text="Importance"
            )

            fig.update_traces(
                texttemplate="%{text:.3f}",
                textposition="outside"
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )