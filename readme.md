# Customer Churn Rate Prediction

This project trains a machine learning model to predict whether a customer is likely to churn. It uses the Telco customer churn dataset in `data/Telco_customer_churn.xlsx` and writes churn probabilities plus churn labels to a CSV output file.

## Project Files

- `churn_prediction.py`: command-line script for training and prediction.
- `api.py`: FastAPI service for single-customer predictions over HTTP.
- `customer_dashboard.py`: single-customer prediction dashboard.
- `dashboard_performance.py`: interactive dashboard for generated prediction outputs.
- `eda_customer_churn.ipynb`: exploratory data analysis notebook.
- `data/Telco_customer_churn.xlsx`: source dataset.
- `requirements.txt`: Python package dependencies.
- `outputs/churn_predictions.csv`: generated prediction output.
- `models/churn_model.joblib`: generated trained model file.

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Exploratory Data Analysis

Open and run:

```text
eda_customer_churn.ipynb
```

The notebook explores dataset shape, missing values, churn distribution, numeric feature distributions, churn by categorical features, churn reasons, and numeric correlations.

## Train the Model

Run:

```bash
python churn_prediction.py train --data data/Telco_customer_churn.xlsx
```

This command:

- reads the Excel dataset
- cleans missing and numeric-looking values
- removes leakage columns such as `Churn Label`, `Churn Score`, and `Churn Reason`
- trains a logistic regression churn model
- prints validation metrics
- saves the model to `models/churn_model.joblib`
- saves predictions to `outputs/churn_predictions.csv`

## Predict Churn

After training, run:

```bash
python churn_prediction.py predict --input data/Telco_customer_churn.xlsx
```

This loads the saved model and writes predictions to:

```text
outputs/churn_predictions.csv
```

The output includes:

- `Predicted Churn Probability`
- `Predicted Churn Value`
- `Predicted Churn Label`

## Useful Options

Train with a custom model path:

```bash
python churn_prediction.py train --model-out models/my_model.joblib
```

Write predictions to a custom file:

```bash
python churn_prediction.py predict \
  --input data/Telco_customer_churn.xlsx \
  --output outputs/my_predictions.csv
```

Change the churn classification threshold:

```bash
python churn_prediction.py predict \
  --input data/Telco_customer_churn.xlsx \
  --threshold 0.6
```

The default threshold is `0.5`. A customer with churn probability greater than or equal to the threshold is predicted as churned.

## Prediction Dashboard

Run the local dashboard:

```bash
streamlit run dashboard_performance.py
```

Then open:

```text
http://localhost:8501
```

The dashboard reads `outputs/churn_predictions.csv` by default and shows prediction metrics, churn probability distributions, segment-level churn risk, highest-risk customers, and a filtered CSV download.

## Single-Customer Prediction Dashboard

Run the form-based dashboard for scoring one customer at a time:

```bash
streamlit run customer_dashboard.py
```

The app loads `models/churn_model.joblib`, lets you enter customer details, and returns a churn probability plus a churn/not-churn prediction using the selected threshold.

## Prediction API

Run the API locally:

```bash
uvicorn api:app --host 0.0.0.0 --port 8000
```

Check service health:

```bash
curl http://localhost:8000/health
```

Send a single-customer prediction request:

```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "City": "Los Angeles",
    "Zip Code": 93552,
    "Latitude": 36.391777,
    "Longitude": -119.730885,
    "Gender": "Female",
    "Senior Citizen": "No",
    "Partner": "No",
    "Dependents": "No",
    "Tenure Months": 29,
    "Phone Service": "Yes",
    "Multiple Lines": "No",
    "Internet Service": "Fiber optic",
    "Online Security": "No",
    "Online Backup": "No",
    "Device Protection": "No",
    "Tech Support": "No",
    "Streaming TV": "No",
    "Streaming Movies": "No",
    "Contract": "Month-to-month",
    "Paperless Billing": "Yes",
    "Payment Method": "Electronic check",
    "Monthly Charges": 70.35,
    "Total Charges": 1397.48
  }'
```

Open the automatic API docs at:

```text
http://localhost:8000/docs
```

## Docker Deployment

Build the container image:

```bash
docker build -t customer-churn-dashboard .
```

Run the Streamlit dashboard:

```bash
docker run --rm -p 8501:8501 customer-churn-dashboard
```

Then open:

```text
http://localhost:8501
```

Run batch predictions inside the same image by overriding the default command:

```bash
docker run --rm \
  -v "$(pwd)/data:/app/data" \
  -v "$(pwd)/outputs:/app/outputs" \
  customer-churn-dashboard \
  python churn_prediction.py predict \
    --input data/Telco_customer_churn.xlsx \
    --output outputs/churn_predictions.csv
```

For production, mount or download fresh customer input data at runtime instead of baking raw data into the image. The image includes the trained model artifact from `models/churn_model.joblib` and the latest prediction output from `outputs/churn_predictions.csv` for dashboard startup.

Run the single-customer dashboard from the same image:

```bash
docker run --rm -p 8502:8501 customer-churn-dashboard \
  streamlit run customer_dashboard.py \
    --server.address=0.0.0.0 \
    --server.port=8501
```

Then open:

```text
http://localhost:8502
```

Run the prediction API from the same image:

```bash
docker run --rm -p 8000:8000 customer-churn-dashboard \
  uvicorn api:app --host 0.0.0.0 --port 8000
```

Then open:

```text
http://localhost:8000/docs
```

## Cloud API Deployment

This repo includes `Dockerfile.api` for deploying only the FastAPI service.

Build and test the API image locally:

```bash
docker build -f Dockerfile.api -t customer-churn-api .
docker run --rm -p 8000:8000 customer-churn-api
```

Then open:

```text
http://localhost:8000/docs
```

Deploy on Render:

1. Push this project to GitHub.
2. In Render, create a new Blueprint or Web Service from the GitHub repo.
3. Use the included `render.yaml`, or choose Docker and set the Dockerfile path to:

```text
./Dockerfile.api
```

4. Set the health check path to:

```text
/health
```

After deployment, Render will give you a public URL similar to:

```text
https://customer-churn-api.onrender.com
```

Use:

```text
https://customer-churn-api.onrender.com/docs
https://customer-churn-api.onrender.com/predict
```

## Model Pipeline

The script uses a scikit-learn pipeline:

```text
customer data
-> clean missing and numeric values
-> drop leakage columns
-> preprocess numeric and categorical features
-> logistic regression classifier
-> churn probability and churn label
```

Numeric columns are imputed with the median and scaled. Categorical columns are imputed with the most common value and one-hot encoded.

## Notes

Use a clean virtual environment for this project. The script and notebook depend on compatible versions of `pandas`, `scikit-learn`, `openpyxl`, `joblib`, `matplotlib`, and `seaborn`.
