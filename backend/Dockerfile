FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# We must use curl for the healthcheck as specified
RUN apt-get update && apt-get install -y curl && rm -rf /var/lib/apt/lists/*
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
