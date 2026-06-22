"""Train and use a customer churn prediction model.

Examples:
    python churn_prediction.py train --data data/Telco_customer_churn.xlsx
    python churn_prediction.py predict --input data/Telco_customer_churn.xlsx
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TARGET_COLUMN = "Churn Value"

# These fields either identify the customer, duplicate the target, or are only
# known after churn happens. Keeping them would inflate validation scores.
DEFAULT_DROP_COLUMNS = {
    "CustomerID",
    "Count",
    "Country",
    "State",
    "Lat Long",
    "Churn Label",
    "Churn Value",
    "Churn Score",
    "CLTV",
    "Churn Reason",
}


def read_table(path: str | Path) -> pd.DataFrame:
    """Read CSV or Excel data into a dataframe."""
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    if suffix == ".csv":
        return pd.read_csv(path)

    raise ValueError(f"Unsupported file type: {path.suffix}. Use .csv, .xls, or .xlsx.")


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize blank strings and numeric-looking object columns."""
    df = df.copy()
    df = df.replace(r"^\s*$", pd.NA, regex=True)

    for column in df.columns:
        if df[column].dtype == "object":
            try:
                df[column] = pd.to_numeric(df[column])
            except (TypeError, ValueError):
                pass

    return df


def build_feature_frame(df: pd.DataFrame, drop_columns: Iterable[str]) -> pd.DataFrame:
    """Return model features after removing unavailable or leaking columns."""
    columns_to_drop = [column for column in drop_columns if column in df.columns]
    return df.drop(columns=columns_to_drop)


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    """Build a preprocessing and classification pipeline."""
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number", "bool"]).columns.tolist()

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_transformer, numeric_features),
            ("categorical", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    classifier = LogisticRegression(
        class_weight="balanced",
        max_iter=2000,
        random_state=42,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def train(args: argparse.Namespace) -> None:
    df = clean_data(read_table(args.data))
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Training data must contain target column {TARGET_COLUMN!r}.")

    X = build_feature_frame(df, DEFAULT_DROP_COLUMNS)
    y = df[TARGET_COLUMN].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=args.test_size,
        random_state=args.random_state,
        stratify=y,
    )

    model = build_pipeline(X_train)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_probability = model.predict_proba(X_test)[:, 1]

    print("Validation metrics")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1 score:  {f1_score(y_test, y_pred):.4f}")
    print(f"ROC AUC:   {roc_auc_score(y_test, y_probability):.4f}")
    print("\nConfusion matrix [[TN, FP], [FN, TP]]")
    print(confusion_matrix(y_test, y_pred))
    print("\nClassification report")
    print(classification_report(y_test, y_pred, target_names=["Not churned", "Churned"]))

    model_package = {
        "model": model,
        "features": X.columns.tolist(),
        "drop_columns": sorted(DEFAULT_DROP_COLUMNS),
        "threshold": args.threshold,
    }
    Path(args.model_out).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model_package, args.model_out)
    print(f"\nSaved model to {args.model_out}")

    if args.predictions_out:
        write_predictions(
            model_package=model_package,
            source_df=df,
            output_path=args.predictions_out,
            threshold=args.threshold,
        )


def write_predictions(
    model_package: dict,
    source_df: pd.DataFrame,
    output_path: str | Path,
    threshold: float,
) -> None:
    features = model_package["features"]
    model = model_package["model"]

    X = build_feature_frame(source_df, DEFAULT_DROP_COLUMNS)
    X = X.reindex(columns=features)

    probabilities = model.predict_proba(X)[:, 1]
    predictions = (probabilities >= threshold).astype(int)

    output = source_df.copy()
    output["Predicted Churn Probability"] = probabilities
    output["Predicted Churn Value"] = predictions
    output["Predicted Churn Label"] = output["Predicted Churn Value"].map(
        {0: "No", 1: "Yes"}
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)

    predicted_churn_rate = probabilities.mean()
    predicted_customer_rate = predictions.mean()
    print(f"Saved predictions to {output_path}")
    print(f"Average predicted churn probability: {predicted_churn_rate:.2%}")
    print(f"Customers predicted to churn at threshold {threshold:.2f}: {predicted_customer_rate:.2%}")


def predict(args: argparse.Namespace) -> None:
    model_package = joblib.load(args.model)
    df = clean_data(read_table(args.input))
    write_predictions(
        model_package=model_package,
        source_df=df,
        output_path=args.output,
        threshold=args.threshold if args.threshold is not None else model_package["threshold"],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Predict customer churn rate.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="Train and evaluate the churn model.")
    train_parser.add_argument("--data", default="data/Telco_customer_churn.xlsx")
    train_parser.add_argument("--model-out", default="models/churn_model.joblib")
    train_parser.add_argument("--predictions-out", default="outputs/churn_predictions.csv")
    train_parser.add_argument("--test-size", type=float, default=0.2)
    train_parser.add_argument("--random-state", type=int, default=42)
    train_parser.add_argument("--threshold", type=float, default=0.5)
    train_parser.set_defaults(func=train)

    predict_parser = subparsers.add_parser(
        "predict",
        help="Predict churn for new customers using a saved model.",
    )
    predict_parser.add_argument("--input", required=True)
    predict_parser.add_argument("--model", default="models/churn_model.joblib")
    predict_parser.add_argument("--output", default="outputs/churn_predictions.csv")
    predict_parser.add_argument("--threshold", type=float)
    predict_parser.set_defaults(func=predict)

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
