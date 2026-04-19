# ==============================
# House Price Prediction Project
# ==============================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score

import joblib

# ------------------------------
# 1. Load Dataset
# ------------------------------
df = pd.read_csv("data/train.csv")

# Drop ID column
df.drop("Id", axis=1, inplace=True)

# ------------------------------
# 2. Target Variable
# ------------------------------
y = np.log1p(df["SalePrice"])  # log transform (important for Kaggle)
X = df.drop("SalePrice", axis=1)

# ------------------------------
# 3. Separate Numerical & Categorical
# ------------------------------
num_features = X.select_dtypes(include=["int64", "float64"]).columns
cat_features = X.select_dtypes(include=["object"]).columns

# ------------------------------
# 4. Preprocessing Pipelines
# ------------------------------

# Numerical pipeline
num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Categorical pipeline
cat_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine pipelines
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_features),
    ("cat", cat_pipeline, cat_features)
])

# ------------------------------
# 5. Full Pipeline with Model
# ------------------------------
model = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", Ridge(alpha=10))
])

# ------------------------------
# 6. Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ------------------------------
# 7. Train Model
# ------------------------------
model.fit(X_train, y_train)

# ------------------------------
# 8. Predictions
# ------------------------------
y_pred = model.predict(X_test)

# Convert back from log scale
y_pred_exp = np.expm1(y_pred)
y_test_exp = np.expm1(y_test)

# ------------------------------
# 9. Evaluation
# ------------------------------
rmse = np.sqrt(mean_squared_error(y_test_exp, y_pred_exp))
r2 = r2_score(y_test_exp, y_pred_exp)

print("RMSE:", rmse)
print("R2 Score:", r2)

# ------------------------------
# 10. Save Model
# ------------------------------
joblib.dump(model, "model.pkl")

print("Model saved successfully!")

import matplotlib.pyplot as plt
plt.scatter(y_test_exp, y_pred_exp)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()