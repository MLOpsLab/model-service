FROM python:3.9-slim

WORKDIR /app

# System dependencies
RUN apt-get update && \
    apt-get install -y build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY main.py /app/model_service.py

RUN pip install --no-cache-dir \
    fastapi uvicorn pandas scikit-learn mlflow

EXPOSE 8000

CMD ["uvicorn", "model_service:app", "--host", "0.0.0.0", "--port", "8000"]