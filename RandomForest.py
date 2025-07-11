import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.manifold import MDS
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.preprocessing import LabelEncoder

# Load the data
url = "http://archive.ics.uci.edu/ml/machine-learning-databases/heart-disease/processed.cleveland.data"
columns = [
    "age", "sex", "cp", "trestbps", "chol", "fbs", "restecg", "thalach",
    "exang", "oldpeak", "slope", "ca", "thal", "hd"
]
df = pd.read_csv(url, header=None, names=columns)

# Replace '?' with np.nan
df.replace("?", np.nan, inplace=True)

# Convert numerical columns from string to float
for col in ["ca", "thal"]:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Convert types
df["sex"] = df["sex"].map({0: "F", 1: "M"})
df["hd"] = df["hd"].apply(lambda x: "Healthy" if int(x) == 0 else "Unhealthy")

# Convert columns to category
categorical_cols = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal", "hd"]
for col in categorical_cols:
    df[col] = df[col].astype("category")

# Impute missing values (Simple strategy)
df_imputed = df.copy()
imputer = SimpleImputer(strategy="most_frequent")
df_imputed[df.columns] = imputer.fit_transform(df)

# Encode categorical features
df_encoded = pd.get_dummies(df_imputed.drop("hd", axis=1))
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df_imputed["hd"])  # 0 = Healthy, 1 = Unhealthy

# Build Random Forest with OOB
rf = RandomForestClassifier(n_estimators=1000, oob_score=True, random_state=42)
rf.fit(df_encoded, y)

# Print OOB score and confusion matrix
print("OOB Score:", rf.oob_score_)

# Confusion matrix on training data (since we didn't split here)
y_pred = rf.predict(df_encoded)
cm = confusion_matrix(y, y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
disp.plot()
plt.title("Confusion Matrix on Full Data")
plt.show()

# Plot OOB Error over Trees
oob_error = [1 - oob for oob in rf.oob_score_ * np.ones(len(rf.estimators_))]
plt.plot(range(1, len(oob_error)+1), oob_error)
plt.xlabel("Number of Trees")
plt.ylabel("OOB Error Rate")
plt.title("OOB Error Rate vs Number of Trees")
plt.show()

# MDS using the similarity (proximity approximation)
# scikit-learn does not directly support proximities, so we use MDS on RF leaf indices
leaf_indices = rf.apply(df_encoded)
similarity_matrix = np.equal(leaf_indices[:, None], leaf_indices).mean(axis=2)
distance_matrix = 1 - similarity_matrix

mds = MDS(n_components=2, dissimilarity="precomputed", random_state=42)
mds_coords = mds.fit_transform(distance_matrix)

mds_df = pd.DataFrame({
    "MDS1": mds_coords[:, 0],
    "MDS2": mds_coords[:, 1],
    "Status": label_encoder.inverse_transform(y)
})

plt.figure(figsize=(10, 6))
sns.scatterplot(data=mds_df, x="MDS1", y="MDS2", hue="Status")
plt.title("MDS Plot Using Random Forest Proximities (approx.)")
plt.show()
