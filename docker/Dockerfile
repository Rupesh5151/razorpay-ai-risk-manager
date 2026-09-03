FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY api/         ./api/
COPY rzp/         ./rzp/
COPY audit/       ./audit/
COPY webhooks/    ./webhooks/
COPY coldstart/   ./coldstart/
COPY batch/       ./batch/
COPY mlops/       ./mlops/
COPY artifacts/   ./artifacts/
COPY .env.example .env.example

EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]