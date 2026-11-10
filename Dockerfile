# Single-container image (e.g. Hugging Face Spaces): FastAPI serves the API and the built SPA.
FROM node:20-slim AS build-frontend
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm ci
COPY frontend/ ./
RUN VITE_API_URL="" npm run build

FROM python:3.12-slim
WORKDIR /app/backend
COPY backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt
COPY backend/ ./
COPY data/ /app/data/
COPY --from=build-frontend /app/frontend/dist ./static
ENV PORT=7860 PYTHONUNBUFFERED=1 MLAGENT_DATA_DIR=/app/backend/var
EXPOSE 7860
CMD ["python", "main.py"]
