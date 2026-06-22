"""Single-customer churn prediction dashboard.

Run:
    streamlit run customer_dashboard.py
"""

from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

from churn_prediction import clean_data


MODEL_PATH = Path("models/churn_model.joblib")

YES_NO = ["No", "Yes"]
INTERNET_OPTIONS = ["DSL", "Fiber optic", "No"]
NO_YES_NO_INTERNET = ["No", "Yes", "No internet service"]


@st.cache_resource
def load_model_package(path: str) -> dict:
    return joblib.load(path)


def build_customer_input() -> dict:
    st.subheader("Customer Profile")
    profile_cols = st.columns(3)
    city = profile_cols[0].text_input("City", value="Los Angeles")
    zip_code = profile_cols[1].number_input(
        "Zip Code",
        min_value=90001,
        max_value=96161,
        value=93552,
        step=1,
    )
    gender = profile_cols[2].selectbox("Gender", ["Female", "Male"])

    location_cols = st.columns(2)
    latitude = location_cols[0].number_input(
        "Latitude",
        min_value=32.0,
        max_value=42.5,
        value=36.391777,
        step=0.000001,
        format="%.6f",
    )
    longitude = location_cols[1].number_input(
        "Longitude",
        min_value=-125.0,
        max_value=-114.0,
        value=-119.730885,
        step=0.000001,
        format="%.6f",
    )

    relationship_cols = st.columns(3)
    senior_citizen = relationship_cols[0].selectbox("Senior Citizen", YES_NO)
    partner = relationship_cols[1].selectbox("Partner", YES_NO)
    dependents = relationship_cols[2].selectbox("Dependents", YES_NO)

    st.subheader("Services")
    service_cols = st.columns(3)
    tenure_months = service_cols[0].number_input(
        "Tenure Months",
        min_value=0,
        max_value=72,
        value=29,
        step=1,
    )
    phone_service = service_cols[1].selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = service_cols[2].selectbox(
        "Multiple Lines",
        ["No", "Yes", "No phone service"],
    )

    internet_cols = st.columns(3)
    internet_service = internet_cols[0].selectbox("Internet Service", INTERNET_OPTIONS)
    online_security = internet_cols[1].selectbox("Online Security", NO_YES_NO_INTERNET)
    online_backup = internet_cols[2].selectbox("Online Backup", NO_YES_NO_INTERNET)

    support_cols = st.columns(4)
    device_protection = support_cols[0].selectbox(
        "Device Protection",
        NO_YES_NO_INTERNET,
    )
    tech_support = support_cols[1].selectbox("Tech Support", NO_YES_NO_INTERNET)
    streaming_tv = support_cols[2].selectbox("Streaming TV", NO_YES_NO_INTERNET)
    streaming_movies = support_cols[3].selectbox(
        "Streaming Movies",
        NO_YES_NO_INTERNET,
    )

    st.subheader("Account")
    account_cols = st.columns(3)
    contract = account_cols[0].selectbox(
        "Contract",
        ["Month-to-month", "One year", "Two year"],
    )
    paperless_billing = account_cols[1].selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = account_cols[2].selectbox(
        "Payment Method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )

    charges_cols = st.columns(2)
    monthly_charges = charges_cols[0].number_input(
        "Monthly Charges",
        min_value=0.0,
        max_value=200.0,
        value=70.35,
        step=0.05,
    )
    total_charges = charges_cols[1].number_input(
        "Total Charges",
        min_value=0.0,
        max_value=10000.0,
        value=1397.48,
        step=0.05,
    )

    return {
        "City": city,
        "Zip Code": zip_code,
        "Latitude": latitude,
        "Longitude": longitude,
        "Gender": gender,
        "Senior Citizen": senior_citizen,
        "Partner": partner,
        "Dependents": dependents,
        "Tenure Months": tenure_months,
        "Phone Service": phone_service,
        "Multiple Lines": multiple_lines,
        "Internet Service": internet_service,
        "Online Security": online_security,
        "Online Backup": online_backup,
        "Device Protection": device_protection,
        "Tech Support": tech_support,
        "Streaming TV": streaming_tv,
        "Streaming Movies": streaming_movies,
        "Contract": contract,
        "Paperless Billing": paperless_billing,
        "Payment Method": payment_method,
        "Monthly Charges": monthly_charges,
        "Total Charges": total_charges,
    }


def predict_customer(model_package: dict, customer_input: dict, threshold: float) -> dict:
    features = model_package["features"]
    model = model_package["model"]

    customer_df = pd.DataFrame([customer_input])
    customer_df = clean_data(customer_df).reindex(columns=features)

    probability = float(model.predict_proba(customer_df)[0, 1])
    prediction_value = int(probability >= threshold)

    return {
        "probability": probability,
        "prediction_value": prediction_value,
        "prediction_label": "Yes" if prediction_value else "No",
    }


def main() -> None:
    st.set_page_config(
        page_title="Single Customer Churn Prediction",
        page_icon="",
        layout="wide",
    )

    st.title("Single Customer Churn Prediction")

    try:
        model_package = load_model_package(str(MODEL_PATH))
    except Exception as exc:
        st.error(f"Could not load model from {MODEL_PATH}: {exc}")
        st.stop()

    threshold = st.sidebar.slider(
        "Churn threshold",
        min_value=0.0,
        max_value=1.0,
        value=float(model_package.get("threshold", 0.5)),
        step=0.01,
    )

    with st.form("customer_prediction_form"):
        customer_input = build_customer_input()
        submitted = st.form_submit_button("Predict churn")

    if submitted:
        result = predict_customer(model_package, customer_input, threshold)

        result_cols = st.columns(3)
        result_cols[0].metric(
            "Churn prediction",
            "Will churn" if result["prediction_value"] else "Will not churn",
        )
        result_cols[1].metric("Churn probability", f"{result['probability']:.1%}")
        result_cols[2].metric("Decision threshold", f"{threshold:.0%}")

        if result["prediction_value"]:
            st.warning("This customer is above the selected churn threshold.")
        else:
            st.success("This customer is below the selected churn threshold.")

        result_row = pd.DataFrame(
            [
                {
                    **customer_input,
                    "Predicted Churn Probability": result["probability"],
                    "Predicted Churn Label": result["prediction_label"],
                }
            ]
        )
        st.subheader("Prediction Record")
        st.dataframe(result_row, width="stretch", hide_index=True)


if __name__ == "__main__":
    main()
