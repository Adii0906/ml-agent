FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PYTHONUNBUFFERED=1 MLAGENT_DATA_DIR=/app/var
EXPOSE 8000
CMD ["python", "main.py"]
