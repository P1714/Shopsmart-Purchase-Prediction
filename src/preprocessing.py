"""Reusable preprocessing components for the ShopSmart purchase-prediction model.

Kept in a separate module so the saved pipeline (models/logistic_regression_purchase_model.joblib)
can be loaded from any script or notebook that can import this file.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

TARGET = "Purchase"
ID_COL = "Customer_ID"

CATEGORICAL_FEATURES = [
    "Gender", "Device_Type", "Traffic_Source", "Membership_Type", "Payment_Method",
]
NUMERIC_FEATURES = [
    "Age", "Session_Duration_Min", "Pages_Viewed", "Products_Viewed", "Cart_Items",
    "Cart_Value_INR", "Discount_Percent", "Previous_Orders", "Days_Since_Last_Visit",
    "Email_Open_Rate", "Customer_Rating", "Returns_Last_Year",
]


class IQRCapper(BaseEstimator, TransformerMixin):
    """Caps (winsorises) each numeric column at Q1 - k*IQR and Q3 + k*IQR.

    Bounds are learned on the training data only, then applied to any new data,
    so no information from the test set or from future customers leaks in.
    """

    def __init__(self, k=1.5):
        self.k = k

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        q1, q3 = np.nanpercentile(X, [25, 75], axis=0)
        iqr = q3 - q1
        self.lower_ = q1 - self.k * iqr
        self.upper_ = q3 + self.k * iqr
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        return np.clip(np.asarray(X, dtype=float), self.lower_, self.upper_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def build_preprocessor():
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("cap_outliers", IQRCapper(k=1.5)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("impute", SimpleImputer(strategy="most_frequent")),
        ("encode", OneHotEncoder(drop="first", handle_unknown="ignore")),
    ])
    return ColumnTransformer([
        ("num", numeric, NUMERIC_FEATURES),
        ("cat", categorical, CATEGORICAL_FEATURES),
    ])


def build_pipeline(**lr_params):
    params = dict(max_iter=5000, random_state=42)
    params.update(lr_params)
    return Pipeline([
        ("preprocess", build_preprocessor()),
        ("model", LogisticRegression(**params)),
    ])
