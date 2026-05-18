# ML Engineer: Autonomous AI Agent 🚀

An end-to-end ML workflow automation tool powered by Groq's Llama models — from research to deployment, all in one place.

---

## What Is This?

**ML Engineer** is an AI-powered assistant that handles the full machine learning pipeline for you. Instead of stitching together multiple tools and scripts, you get a single interface that:

- Searches and ranks research papers from arXiv
- Analyzes your dataset automatically
- Recommends the best ML algorithm for your data (with reasoning)
- Trains models in one click with real-time progress updates
- Generates professional reports with charts and insights

Think of it as your personal ML co-pilot — you bring the data and the ideas, it handles the heavy lifting.

> **Note:** All AI features require your own Groq API key. No fallbacks or mock responses are used.

---

## How It's Built

```
┌─────────────────────────────────────────────────────────┐
│                    Frontend (React)                     │
│  - Futuristic UI with animations                        │
│  - Paper Search, Data Lab, Model Engine, Insights      │
└──────────────────────┬──────────────────────────────────┘
                       │ HTTP & WebSockets
┌──────────────────────▼──────────────────────────────────┐
│                  Backend (FastAPI)                      │
│  - File Uploads, WebSocket Streaming, Experiment Storage│
└──────────────────────┬──────────────────────────────────┘
                       │
        ┌──────────────┴──────────────┐
        │                              │
┌───────▼───────┐             ┌──────▼───────┐
│   LLM Engine  │             │  ML Pipeline  │
│ - Groq Llama  │             │ - Scikit-learn│
│ - Local Ollama│             │ - Preprocessing│
└───────────────┘             └──────────────┘
```

The frontend (React) communicates with the backend (FastAPI) over HTTP and WebSockets. The backend connects to two subsystems: the LLM engine (Groq or a local Ollama instance) and the ML pipeline (scikit-learn for training and preprocessing).

---

## Features at a Glance

| Feature | What It Does |
|---|---|
| **Paper Search** | Search arXiv and get AI-ranked results relevant to your topic |
| **Data Lab** | Upload a CSV and get instant statistics, missing value detection, and data profiling |
| **AI Recommendation** | Receive a suggested algorithm for your dataset, with an explanation of why |
| **Auto-Train** | Train a model with one click; watch real-time progress via WebSocket |
| **Insights** | Download a full ML report with charts and next-step recommendations |

---

## Getting Started

Follow these steps in order. It should take less than five minutes.

### Step 1 — Get a Groq API Key

If you don't have one yet, sign up at [console.groq.com](https://console.groq.com) and create a free API key.

### Step 2 — Add Your API Key

In the **root directory** of this project, create a file named `.env` and add the following line:

```env
GROQ_API_KEY=gsk_your_actual_key_here
```

Replace `gsk_your_actual_key_here` with your real key. This file is never committed to version control.

### Step 3 — Start the Backend

Open a terminal, navigate to the `backend` folder, and run:

```bash
cd backend
python -u main.py
```

The backend will start on `http://localhost:8000`.

### Step 4 — Start the Frontend

Open a **second terminal**, navigate to the `frontend` folder, and run:

```bash
cd frontend
npm install   # only needed the first time
npm run dev
```

The frontend will start on `http://localhost:5173`.

### Step 5 — Open the App

Visit **[http://localhost:5173](http://localhost:5173)** in your browser. You're ready to go.

---

## Try It Out

A sample dataset is included at `data/sample.csv`. Use it to walk through the full workflow end-to-end without needing your own data:

1. Upload `sample.csv` in the **Data Lab** tab
2. Review the auto-generated data profile
3. Click **Get AI Recommendation** to see which algorithm fits best
4. Hit **Train** and watch the model build in real time
5. Head to **Insights** to download your report

---

## Tech Stack

| Layer | Technology |
|---|---|
| Frontend | React, Vite, Framer Motion |
| Backend | FastAPI, SQLite |
| AI / LLM | Groq Llama 3.3-70B, Local Ollama (optional) |
| ML | Scikit-learn, Pandas, NumPy |

---

## Troubleshooting

**The backend won't start.**
Make sure Python 3.9+ is installed and your `.env` file is in the root directory (not inside `backend/`).

**I'm getting API errors.**
Double-check that your Groq API key is correct and has not exceeded its rate limit.

**The frontend shows a blank screen.**
Ensure the backend is running before opening the frontend. Check the browser console for errors.

**npm install fails.**
Make sure Node.js 18+ is installed. Run `node --version` to check.

Built for ML engineers who want to focus on ideas, not boilerplate.