FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY churn_prediction.py dashboard_performance.py customer_dashboard.py api.py ./
COPY models ./models
COPY outputs ./outputs

EXPOSE 8501

CMD ["streamlit", "run", "dashboard_performance.py", "--server.address=0.0.0.0", "--server.port=8501"]
