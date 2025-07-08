from sklearn.linear_model import RidgeCV, Lasso, ElasticNet
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import numpy as np

# Generate synthetic regression data
X, y = make_regression(n_samples=100, n_features=20, noise=0.1, random_state=42)

# Split into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Alpha values to test
alphas = np.logspace(-4, 4, 100)

# Ridge Regression
ridge = RidgeCV(alphas=alphas)  # store_cv_values removed
ridge.fit(X_train, y_train)
print("MSE Ridge     (best alpha={}): {}".format(ridge.alpha_, mean_squared_error(y_test, ridge.predict(X_test))))

# Lasso Regression
lasso = Lasso(alpha=1.0)
lasso.fit(X_train, y_train)
print("MSE Lasso     (alpha=1):", mean_squared_error(y_test, lasso.predict(X_test)))

# ElasticNet Regression
enet = ElasticNet(alpha=1.0, l1_ratio=0.5)
enet.fit(X_train, y_train)
print("MSE ElasticNet(alpha=1, l1_ratio=0.5):", mean_squared_error(y_test, enet.predict(X_test)))
