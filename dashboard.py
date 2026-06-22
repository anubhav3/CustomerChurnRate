"""Interactive dashboard for churn prediction outputs.

Run:
    streamlit run dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


DEFAULT_PREDICTIONS_PATH = Path("outputs/churn_predictions.csv")
PROBABILITY_COLUMN = "Predicted Churn Probability"
PREDICTION_COLUMN = "Predicted Churn Label"
PREDICTION_VALUE_COLUMN = "Predicted Churn Value"
ACTUAL_VALUE_COLUMN = "Churn Value"
ACTUAL_LABEL_COLUMN = "Churn Label"
CUSTOMER_ID_COLUMN = "CustomerID"


st.set_page_config(
    page_title="Customer Churn Predictions",
    page_icon="",
    layout="wide",
)

sns.set_theme(style="whitegrid", palette="Set2")


@st.cache_data
def load_predictions(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if PROBABILITY_COLUMN not in df.columns or PREDICTION_COLUMN not in df.columns:
        raise ValueError(
            "Prediction file must include "
            f"{PROBABILITY_COLUMN!r} and {PREDICTION_COLUMN!r}."
        )
    return df


def filter_by_options(
    df: pd.DataFrame,
    column: str,
    label: str,
    default_all: bool = True,
) -> pd.DataFrame:
    if column not in df.columns:
        return df

    options = sorted(df[column].dropna().astype(str).unique())
    default = options if default_all else []
    selected = st.sidebar.multiselect(label, options=options, default=default)

    if not selected:
        return df.iloc[0:0]

    return df[df[column].astype(str).isin(selected)]


def format_percent(value: float) -> str:
    return f"{value:.1%}"


def get_actual_labels(df: pd.DataFrame) -> pd.Series | None:
    if ACTUAL_VALUE_COLUMN in df.columns:
        return df[ACTUAL_VALUE_COLUMN].astype(int)
    if ACTUAL_LABEL_COLUMN in df.columns:
        return df[ACTUAL_LABEL_COLUMN].map({"No": 0, "Yes": 1}).astype("Int64")
    return None


def get_predicted_labels(df: pd.DataFrame) -> pd.Series:
    if PREDICTION_VALUE_COLUMN in df.columns:
        return df[PREDICTION_VALUE_COLUMN].astype(int)
    return df[PREDICTION_COLUMN].map({"No": 0, "Yes": 1}).astype(int)


def calculate_performance(df: pd.DataFrame) -> dict[str, float] | None:
    y_true = get_actual_labels(df)
    if y_true is None:
        return None

    y_pred = get_predicted_labels(df)
    y_probability = df[PROBABILITY_COLUMN]

    valid_rows = y_true.notna() & y_pred.notna() & y_probability.notna()
    y_true = y_true[valid_rows].astype(int)
    y_pred = y_pred[valid_rows].astype(int)
    y_probability = y_probability[valid_rows]

    if y_true.empty:
        return None

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    if y_true.nunique() > 1:
        metrics["roc_auc"] = roc_auc_score(y_true, y_probability)

    return metrics


def plot_confusion_matrix(df: pd.DataFrame) -> plt.Figure | None:
    y_true = get_actual_labels(df)
    if y_true is None:
        return None

    y_pred = get_predicted_labels(df)
    valid_rows = y_true.notna() & y_pred.notna()
    y_true = y_true[valid_rows].astype(int)
    y_pred = y_pred[valid_rows].astype(int)

    if y_true.empty:
        return None

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    matrix_df = pd.DataFrame(
        matrix,
        index=["Actual No", "Actual Yes"],
        columns=["Predicted No", "Predicted Yes"],
    )

    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(matrix_df, annot=True, fmt=",d", cmap="Blues", cbar=False, ax=ax)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Prediction")
    ax.set_ylabel("Actual")
    fig.tight_layout()
    return fig


def plot_roc_curve(df: pd.DataFrame) -> plt.Figure | None:
    y_true = get_actual_labels(df)
    if y_true is None:
        return None

    y_probability = df[PROBABILITY_COLUMN]
    valid_rows = y_true.notna() & y_probability.notna()
    y_true = y_true[valid_rows].astype(int)
    y_probability = y_probability[valid_rows]

    if y_true.nunique() < 2:
        return None

    false_positive_rate, true_positive_rate, _ = roc_curve(y_true, y_probability)
    auc = roc_auc_score(y_true, y_probability)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(false_positive_rate, true_positive_rate, label=f"ROC AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_title("ROC Curve")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def plot_probability_distribution(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(
        data=df,
        x=PROBABILITY_COLUMN,
        hue=PREDICTION_COLUMN,
        bins=30,
        kde=True,
        ax=ax,
    )
    ax.set_title("Predicted Churn Probability Distribution")
    ax.set_xlabel("Predicted churn probability")
    ax.set_ylabel("Customers")
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
    fig.tight_layout()
    return fig


def plot_segment_churn_rate(df: pd.DataFrame, segment_column: str) -> plt.Figure:
    segment_summary = (
        df.groupby(segment_column, dropna=False)[PROBABILITY_COLUMN]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    segment_summary[segment_column] = segment_summary[segment_column].astype(str)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(
        data=segment_summary,
        x=segment_column,
        y=PROBABILITY_COLUMN,
        ax=ax,
    )
    ax.set_title(f"Average Predicted Churn Probability by {segment_column}")
    ax.set_xlabel(segment_column)
    ax.set_ylabel("Average predicted churn probability")
    ax.tick_params(axis="x", rotation=35)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    fig.tight_layout()
    return fig


def plot_tenure_vs_probability(df: pd.DataFrame) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.scatterplot(
        data=df,
        x="Tenure Months",
        y=PROBABILITY_COLUMN,
        hue=PREDICTION_COLUMN,
        alpha=0.55,
        ax=ax,
    )
    ax.set_title("Tenure vs Predicted Churn Probability")
    ax.set_xlabel("Tenure months")
    ax.set_ylabel("Predicted churn probability")
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f"{y:.0%}"))
    fig.tight_layout()
    return fig


st.title("Customer Churn Predictions")

prediction_path = st.sidebar.text_input(
    "Prediction CSV path",
    value=str(DEFAULT_PREDICTIONS_PATH),
)

try:
    predictions = load_predictions(prediction_path)
except Exception as exc:
    st.error(f"Could not load predictions: {exc}")
    st.stop()

filtered = predictions.copy()
filtered = filter_by_options(filtered, "Predicted Churn Label", "Predicted label")
filtered = filter_by_options(filtered, "Contract", "Contract")
filtered = filter_by_options(filtered, "Internet Service", "Internet service")
filtered = filter_by_options(filtered, "Payment Method", "Payment method")

probability_range = st.sidebar.slider(
    "Predicted churn probability",
    min_value=0.0,
    max_value=1.0,
    value=(0.0, 1.0),
    step=0.01,
)
filtered = filtered[
    filtered[PROBABILITY_COLUMN].between(probability_range[0], probability_range[1])
]

total_customers = len(filtered)
predicted_churners = int((filtered[PREDICTION_COLUMN] == "Yes").sum())
predicted_churn_rate = predicted_churners / total_customers if total_customers else 0
average_probability = (
    filtered[PROBABILITY_COLUMN].mean() if total_customers else 0
)

metric_cols = st.columns(4)
metric_cols[0].metric("Customers", f"{total_customers:,}")
metric_cols[1].metric("Predicted churners", f"{predicted_churners:,}")
metric_cols[2].metric("Predicted churn rate", format_percent(predicted_churn_rate))
metric_cols[3].metric(
    "Avg churn probability",
    format_percent(average_probability),
)

if filtered.empty:
    st.warning("No customers match the selected filters.")
    st.stop()

st.subheader("Model Performance")
performance = calculate_performance(filtered)
if performance is None:
    st.info("Actual churn labels are not available in this prediction file.")
else:
    st.caption(
        "Metrics are calculated against actual churn labels in the currently "
        "filtered rows. Use all rows for the broadest view."
    )
    performance_cols = st.columns(5)
    performance_cols[0].metric("Accuracy", format_percent(performance["accuracy"]))
    performance_cols[1].metric("Precision", format_percent(performance["precision"]))
    performance_cols[2].metric("Recall", format_percent(performance["recall"]))
    performance_cols[3].metric("F1 score", format_percent(performance["f1"]))
    if "roc_auc" in performance:
        performance_cols[4].metric("ROC AUC", f"{performance['roc_auc']:.3f}")
    else:
        performance_cols[4].metric("ROC AUC", "N/A")

    performance_chart_cols = st.columns(2)
    confusion_fig = plot_confusion_matrix(filtered)
    roc_fig = plot_roc_curve(filtered)

    if confusion_fig is not None:
        with performance_chart_cols[0]:
            st.pyplot(confusion_fig, clear_figure=True)
    if roc_fig is not None:
        with performance_chart_cols[1]:
            st.pyplot(roc_fig, clear_figure=True)

chart_cols = st.columns(2)
with chart_cols[0]:
    st.pyplot(plot_probability_distribution(filtered), clear_figure=True)

with chart_cols[1]:
    segment_options = [
        column
        for column in [
            "Contract",
            "Internet Service",
            "Payment Method",
            "Tenure Months",
            "Monthly Charges",
            "Senior Citizen",
        ]
        if column in filtered.columns
    ]
    segment_column = st.selectbox("Segment chart", segment_options)
    st.pyplot(plot_segment_churn_rate(filtered, segment_column), clear_figure=True)

if "Tenure Months" in filtered.columns:
    st.pyplot(plot_tenure_vs_probability(filtered), clear_figure=True)

top_risk_columns = [
    column
    for column in [
        CUSTOMER_ID_COLUMN,
        "City",
        "Contract",
        "Internet Service",
        "Payment Method",
        "Tenure Months",
        "Monthly Charges",
        "Total Charges",
        PROBABILITY_COLUMN,
        PREDICTION_COLUMN,
    ]
    if column in filtered.columns
]

st.subheader("Highest-Risk Customers")
top_risk = (
    filtered.sort_values(PROBABILITY_COLUMN, ascending=False)
    .loc[:, top_risk_columns]
    .head(50)
)
st.dataframe(
    top_risk,
    width="stretch",
    hide_index=True,
)

st.download_button(
    "Download filtered predictions",
    data=filtered.to_csv(index=False),
    file_name="filtered_churn_predictions.csv",
    mime="text/csv",
)
