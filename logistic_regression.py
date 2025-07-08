# ✅ STEP 1: Install required packages (run in terminal/command line if not installed)
# pip install pandas numpy statsmodels matplotlib seaborn scipy

# ✅ STEP 2: Import libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2

# ✅ STEP 3: Load dataset
url = "https://raw.githubusercontent.com/StatQuest/logistic_regression_demo/master/processed.cleveland.data"
df = pd.read_csv(url, header=None)

# ✅ STEP 4: Rename columns
df.columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg",
    "thalach", "exang", "oldpeak", "slope", "ca", "thal", "hd"
]

# ✅ STEP 5: Handle missing values
df.replace("?", np.nan, inplace=True)
df["ca"] = pd.to_numeric(df["ca"], errors='coerce')
df["thal"] = pd.to_numeric(df["thal"], errors='coerce')
df.dropna(inplace=True)

# ✅ STEP 6: Convert categorical columns
df["sex"] = df["sex"].map({0: "F", 1: "M"}).astype("category")

for col in ["cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]:
    df[col] = df[col].astype("category")

# ✅ STEP 7: Encode target variable
df["hd"] = df["hd"].apply(lambda x: "Healthy" if x == 0 else "Unhealthy")
df["hd"] = df["hd"].astype("category")
df["hd_bin"] = df["hd"].map({"Healthy": 0, "Unhealthy": 1}).astype(float)

# ===== SIMPLE LOGISTIC REGRESSION (sex only) =====
X_sex = pd.get_dummies(df["sex"], drop_first=True)
X_sex = sm.add_constant(X_sex).astype(float)
y = df["hd_bin"]

logit_sex = sm.Logit(y, X_sex).fit()
print(logit_sex.summary())

ll_null = logit_sex.llnull
ll_proposed = logit_sex.llf
pseudo_r2 = 1 - (ll_proposed / ll_null)
print("\nMcFadden's R²:", pseudo_r2)

chi2_val = 2 * (ll_proposed - ll_null)
p_val = chi2.sf(chi2_val, df=1)
print("Model p-value:", p_val)

# ✅ Plot: Probability of heart disease by sex
df["prob_hd_sex"] = logit_sex.predict(X_sex)

sns.boxplot(data=df, x="sex", y="prob_hd_sex", palette="Set2")
plt.title("Predicted Probability of Heart Disease by Sex")
plt.ylabel("Probability")
plt.xlabel("Sex")
plt.tight_layout()
plt.show()

# ===== FULL LOGISTIC REGRESSION (all features) =====
X_all = df.drop(columns=["hd", "hd_bin", "prob_hd_sex"])
X_all = pd.get_dummies(X_all, drop_first=True)
X_all = sm.add_constant(X_all).astype(float)

logit_full = sm.Logit(y, X_all).fit()
print(logit_full.summary())

ll_null_full = logit_full.llnull
ll_proposed_full = logit_full.llf
pseudo_r2_full = 1 - (ll_proposed_full / ll_null_full)
print("\nMcFadden's R² (full model):", pseudo_r2_full)

chi2_val_full = 2 * (ll_proposed_full - ll_null_full)
df_chi = len(logit_full.params) - 1
p_val_full = chi2.sf(chi2_val_full, df=df_chi)
print("Model p-value (full model):", p_val_full)

# ✅ Plot: Predicted probabilities (full model)
df["prob_hd_full"] = logit_full.predict(X_all)
df_sorted = df.sort_values("prob_hd_full").reset_index()

plt.figure(figsize=(12, 6))
sns.scatterplot(data=df_sorted, x=df_sorted.index, y="prob_hd_full",
                hue="hd", palette="Set1", s=80, marker="X")
plt.title("Predicted Probability of Heart Disease (Full Model)")
plt.xlabel("Index")
plt.ylabel("Probability of Heart Disease")
plt.tight_layout()
plt.show()
