import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt

# ----------------------------
# Example 1: 15 useful, 4085 useless features
# ----------------------------

np.random.seed(42)
n = 1000       # observations
p = 5000       # total features
real_p = 15    # true predictors

# Generate X and y
X = np.random.randn(n, p)
y = X[:, :real_p].sum(axis=1) + np.random.randn(n)

# Split data (2/3 train, 1/3 test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

# Ridge Regression (alpha = 0)
ridge = RidgeCV(alphas=np.logspace(-6, 6, 13), scoring='neg_mean_squared_error', cv=10)
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge_mse = mean_squared_error(y_test, ridge_pred)

# Lasso Regression (alpha = 1)
lasso = LassoCV(alphas=None, cv=10, random_state=42, max_iter=10000)
lasso.fit(X_train, y_train)
lasso_pred = lasso.predict(X_test)
lasso_mse = mean_squared_error(y_test, lasso_pred)

# ElasticNet Regression (alpha = 0.5)
elastic = ElasticNetCV(l1_ratio=0.5, cv=10, random_state=42, max_iter=10000)
elastic.fit(X_train, y_train)
elastic_pred = elastic.predict(X_test)
elastic_mse = mean_squared_error(y_test, elastic_pred)

print("MSE Ridge     (alpha=0):", ridge_mse)
print("MSE Lasso     (alpha=1):", lasso_mse)
print("MSE ElasticNet(alpha=0.5):", elastic_mse)

# Try multiple alpha values for ElasticNet (alpha = l1_ratio)
alphas = np.linspace(0, 1, 11)
mse_results = []

for alpha in alphas:
    enet = ElasticNetCV(l1_ratio=alpha, cv=10, random_state=42, max_iter=10000)
    enet.fit(X_train, y_train)
    pred = enet.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    mse_results.append({"alpha": alpha, "mse": mse})

results_df1 = pd.DataFrame(mse_results)
print("\nElasticNet results (Example 1):")
print(results_df1)

# ----------------------------
# Example 2: 1500 useful, 3500 useless features
# ----------------------------

np.random.seed(42)
real_p = 1500

# Generate X and y again
X = np.random.randn(n, p)
y = X[:, :real_p].sum(axis=1) + np.random.randn(n)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=1/3, random_state=42)

mse_results = []
for alpha in alphas:
    enet = ElasticNetCV(l1_ratio=alpha, cv=10, random_state=42, max_iter=10000)
    enet.fit(X_train, y_train)
    pred = enet.predict(X_test)
    mse = mean_squared_error(y_test, pred)
    mse_results.append({"alpha": alpha, "mse": mse})

results_df2 = pd.DataFrame(mse_results)
print("\nElasticNet results (Example 2):")
print(results_df2)
