# Customer Churn Rate Prediction

This project trains a machine learning model to predict whether a customer is likely to churn. It uses the Telco customer churn dataset in `data/Telco_customer_churn.xlsx` and writes churn probabilities plus churn labels to a CSV output file.

## Project Files

- `churn_prediction.py`: command-line script for training and prediction.
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
streamlit run dashboard.py
```

Then open:

```text
http://localhost:8501
```

The dashboard reads `outputs/churn_predictions.csv` by default and shows prediction metrics, churn probability distributions, segment-level churn risk, highest-risk customers, and a filtered CSV download.

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
