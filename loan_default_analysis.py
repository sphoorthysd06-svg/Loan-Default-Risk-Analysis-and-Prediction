# MAJOR PROJECT - 1
# LOAN DEFAULT RISK ANALYSIS AND PREDICTION
# Company: Fin-Care Starpowerz Digital Technology Pvt. Ltd.

# 1. IMPORT LIBRARIES

import pymysql
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score

# 2.LOAD CLEANED CSV
df = pd.read_csv("loan_data.csv")

print("CSV Loaded:", df.shape)
print(df.head())

# LOAD DATA FROM MYSQL
conn = pymysql.connect(
    host="localhost",
    user="root",
    password="system",
    database="loan_default_db"
)

df = pd.read_sql(
    "SELECT * FROM loan_data",
    conn
)

conn.close()

print("MySQL Data Loaded:", df.shape)

# 3.DATA CLEANING
df.columns = (
    df.columns.str.strip()
    .str.replace(" ", "_")
    .str.replace("(", "", regex=False)
    .str.replace(")", "", regex=False)
    .str.replace("%", "Percentage", regex=False)
)

df.rename(
    columns={"Credit_Score_Cibil": "Credit_Score"},
    inplace=True
)

df["Monthly_Income"] = (
    df["Monthly_Income"]
    .astype(str)
    .str.replace(",", "", regex=False)
    .astype(float)
)

df["Loan_Approval_Date"] = pd.to_datetime(
    df["Loan_Approval_Date"],
    errors="coerce"
)

df.drop_duplicates(inplace=True)

print("\nMissing Values:", df.isnull().sum().sum())
print("Duplicate Rows:", df.duplicated().sum())

# NUMERICAL STATISTICS
num_cols = [
    "Monthly_Income", "Loan_Amount", "Tenure_Months",
    "Interest_Rate", "EMI", "Credit_Score",
    "Existing_Loans", "Debt_to_Income_Ratio", "Default_Flag"
]

for col in num_cols:
    data = df[col].dropna().to_numpy()

    print("\n", col)
    print("Min:", np.min(data))
    print("Max:", np.max(data))
    print("Mean:", round(np.mean(data), 2))
    print("Median:", np.median(data))
    print("Std:", round(np.std(data), 2))
    print("Variance:", round(np.var(data), 2))

# FEATURE CREATION
df["Loan_to_Income"] = df["Loan_Amount"] / (df["Monthly_Income"] * 12)
df["EMI_to_Income"] = df["EMI"] / df["Monthly_Income"]

df["Risk_Score"] = (
    (850 - df["Credit_Score"])
    + (df["Debt_to_Income_Ratio"] * 100)
    + (df["Interest_Rate"] * 2)
)

# RISK LEVEL
df["Risk_Level"] = np.where(
    df["Risk_Score"] >= 250, "High",
    np.where(df["Risk_Score"] >= 150, "Medium", "Low")
)

# RISK SCORE STATISTICS
risk = df["Risk_Score"].to_numpy()

print("\nRisk Score")
print("Min:", round(np.min(risk), 2))
print("Max:", round(np.max(risk), 2))
print("Mean:", round(np.mean(risk), 2))
print("Median:", round(np.median(risk), 2))
print("Std:", round(np.std(risk), 2))
print("Variance:", round(np.var(risk), 2))

# RISK LEVEL COUNT
print("\nRisk Level Count:")
print(df["Risk_Level"].value_counts())

# DEFAULT RATE BY RISK LEVEL
print("\nDefault Rate by Risk Level:")
print(df.groupby("Risk_Level")["Default_Flag"].mean() * 100)

# RISK FLAGS
df["Credit_Risk"] = np.where(
    df["Credit_Score"] < 600, "High", "Low"
)

df["DTI_Risk"] = np.where(
    df["Debt_to_Income_Ratio"] > 0.50, "High", "Low"
)

df["Interest_Risk"] = np.where(
    df["Interest_Rate"] > 15, "High", "Low"
)

print("\nFeature Creation Completed!")
print(df[[
    "Loan_ID",
    "Loan_to_Income",
    "EMI_to_Income",
    "Risk_Score",
    "Risk_Level",
    "Credit_Risk",
    "DTI_Risk",
    "Interest_Risk"
]].head())

# 4.BASIC ANALYSIS
print("\n===== BASIC ANALYSIS =====")
print("Total Loans:", len(df))
print("Default Loans:", (df["Default_Flag"] == 1).sum())
print("Non-Default Loans:", (df["Default_Flag"] == 0).sum())
print("Default Rate:", round(df["Default_Flag"].mean() * 100, 2), "%")

print("\nDescriptive Statistics:")
print(df.describe())

# 5.DEFAULT DISTRIBUTION
sns.countplot(x="Default_Flag", data=df)
plt.title("Default vs Non-Default Loans")
plt.xlabel("Default Flag")
plt.ylabel("Number of Loans")
plt.show()

# 6.DEFAULT BY LOAN TYPE
loan_default = df.groupby("Loan_Type")["Default_Flag"].mean() * 100

print("\nDefault Rate by Loan Type:")
print(loan_default)

loan_default.plot(kind="bar", figsize=(8, 5))
plt.title("Default Rate by Loan Type")
plt.ylabel("Default Rate (%)")
plt.xticks(rotation=45)
plt.show()

# 7.CREDIT SCORE ANALYSIS
df["Credit_Category"] = pd.cut(
    df["Credit_Score"],
    bins=[0, 599, 699, 900],
    labels=["Poor", "Average", "Good"]
)

credit_default = df.groupby(
    "Credit_Category",
    observed=False
)["Default_Flag"].mean() * 100

print("\nDefault Rate by Credit Category:")
print(credit_default)

credit_default.plot(kind="bar", figsize=(7, 5))
plt.title("Default Rate by Credit Score Category")
plt.ylabel("Default Rate (%)")
plt.show()

# 8.DTI ANALYSIS
df["DTI_Category"] = pd.cut(
    df["Debt_to_Income_Ratio"],
    bins=[0, 0.2999, 0.4999, 1],
    labels=["Low", "Medium", "High"]
)

dti_default = df.groupby(
    "DTI_Category",
    observed=False
)["Default_Flag"].mean() * 100

print("\nDefault Rate by DTI Category:")
print(dti_default)

dti_default.plot(kind="bar", figsize=(7, 5))
plt.title("Default Rate by Debt-to-Income Category")
plt.ylabel("Default Rate (%)")
plt.show()

# 9.YEAR-WISE DEFAULT RATE
df["Year"] = df["Loan_Approval_Date"].dt.year

year_default = df.groupby("Year")["Default_Flag"].mean() * 100

print("\nYear-wise Default Rate:")
print(year_default)

year_default.plot(kind="line", marker="o", figsize=(8, 5))
plt.title("Year-wise Default Rate")
plt.ylabel("Default Rate (%)")
plt.grid(True)
plt.show()


# 10. CORRELATION HEATMAP
num_cols = [
    "Monthly_Income",
    "Loan_Amount",
    "Tenure_Months",
    "Interest_Rate",
    "EMI",
    "Credit_Score",
    "Existing_Loans",
    "Debt_to_Income_Ratio",
    "Default_Flag"
]

sns.heatmap(
    df[num_cols].corr(),
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Loan Default Correlation Heatmap")
plt.show()

# 11.PIE CHART - DEFAULT DISTRIBUTION
df["Default_Flag"].value_counts().plot(
    kind="pie",
    labels=["Non-Default", "Default"],
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Loan Default Distribution")
plt.ylabel("")
plt.show()

# 12. MACHINE LEARNING
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

# Encode categorical columns
for col in [
    "Loan_Type",
    "Age_Group",
    "Employment_Type",
    "Previous_Default"
]:
    df[col] = LabelEncoder().fit_transform(df[col].astype(str))

X = df[features]
y = df["Default_Flag"]

# 13. TRAIN TEST SPLIT
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# 14. LOGISTIC REGRESSION
logistic = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        max_iter=3000,
        class_weight="balanced"
    ))
])

logistic.fit(X_train, y_train)

log_pred = logistic.predict(X_test)
log_prob = logistic.predict_proba(X_test)[:, 1]

# 15. DECISION TREE
tree = DecisionTreeClassifier(
    max_depth=6,
    class_weight="balanced",
    random_state=42
)

tree.fit(X_train, y_train)

tree_pred = tree.predict(X_test)
tree_prob = tree.predict_proba(X_test)[:, 1]

# 16. RANDOM FOREST
rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=8,
    class_weight="balanced",
    random_state=42
)

rf.fit(X_train, y_train)

rf_pred = rf.predict(X_test)
rf_prob = rf.predict_proba(X_test)[:, 1]

# RANDOM FOREST FEATURE IMPORTANCE

importance = pd.DataFrame({
    "Feature": X.columns,
    "Importance": rf.feature_importances_
})

importance = importance.sort_values(
    "Importance",
    ascending=False
)

importance["Importance"] = importance["Importance"].round(2)

print(importance)

importance.to_csv("feature_importance.csv", index=False)

# 17. MODEL RESULTS
models = {
    "Logistic Regression": (log_pred, log_prob),
    "Decision Tree": (tree_pred, tree_prob),
    "Random Forest": (rf_pred, rf_prob)
}

results = []

for name, (pred, prob) in models.items():

    results.append([
        name,
        accuracy_score(y_test, pred),
        roc_auc_score(y_test, prob)
    ])

results = pd.DataFrame(
    results,
    columns=["Model", "Accuracy", "ROC_AUC"]
)

print("\n===== MODEL COMPARISON =====")
print(results)

# 18. MODEL COMPARISON GRAPH
results.set_index("Model").plot(
    kind="bar",
    figsize=(8, 5)
)

plt.title("Machine Learning Model Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.show()

# 19. RANDOM FOREST EVALUATION
print("\n===== RANDOM FOREST RESULTS =====")

print("Accuracy:",
      accuracy_score(y_test, rf_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_pred,
        target_names=["Non-Default", "Default"]
    )
)

# 20. CONFUSION MATRIX
cm = confusion_matrix(y_test, rf_pred)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=["Non-Default", "Default"],
    yticklabels=["Non-Default", "Default"]
)

plt.title("Loan Default Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

# 21. FEATURE IMPORTANCE
importance = pd.Series(
    rf.feature_importances_,
    index=features
).sort_values(ascending=False)

print("\n===== FEATURE IMPORTANCE =====")
print(importance)

importance.plot(
    kind="barh",
    figsize=(8, 6)
)

plt.title("Loan Default Risk Factor Importance")
plt.xlabel("Importance")
plt.show()
