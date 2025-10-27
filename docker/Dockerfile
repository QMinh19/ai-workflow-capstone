FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY . /app

ENV LOG_DIR=/app/logs
ENV MODEL_DIR=/app/models
ENV REPORT_DIR=/app/reports

EXPOSE 5000

CMD ["gunicorn", "-b", "0.0.0.0:5000", "src.api.app:app", "--workers", "2"]
