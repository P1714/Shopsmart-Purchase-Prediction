"""Score new ShopSmart customers with the trained Logistic Regression model.

Run from the repository root:
    python -m src.predict data/sample_new_customers.csv outputs/predictions.csv
    python -m src.predict data/sample_new_customers.csv outputs/predictions.csv --threshold 0.6

The input CSV needs the same feature columns as the training data
(Customer_ID optional, Purchase not needed). Missing values are allowed:
the saved pipeline imputes, caps outliers, encodes and scales automatically.
"""
import argparse
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

# Importing from src.preprocessing also registers IQRCapper, which joblib needs to load the model.
from src.preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "logistic_regression_purchase_model.joblib"


def segment(p):
    """Map a purchase probability to a marketing segment."""
    if p >= 0.75:
        return "High intent"
    if p >= 0.55:
        return "Medium intent"
    return "Low intent / at risk"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input_csv", help="CSV of customers to score")
    ap.add_argument("output_csv", nargs="?", default="outputs/predictions.csv", help="where to write the scores")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="probability cut-off for predicting a purchase (default 0.5; ~0.6 balances both classes)")
    args = ap.parse_args()

    df = pd.read_csv(args.input_csv)
    missing = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES if c not in df.columns]
    if missing:
        sys.exit(f"Input is missing required columns: {missing}")

    model = joblib.load(MODEL_PATH)
    proba = model.predict_proba(df[NUMERIC_FEATURES + CATEGORICAL_FEATURES])[:, 1]

    out = df.copy()
    out["Purchase_Probability"] = np.round(proba, 4)
    out["Predicted_Purchase"] = (proba >= args.threshold).astype(int)
    out["Segment"] = [segment(p) for p in proba]
    Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    print(f"Scored {len(out):,} customers -> {args.output_csv}")
    print(out["Segment"].value_counts().to_string())


if __name__ == "__main__":
    main()
