# Hate Comment Detection — Presentation

## Title Slide
- Project: Hate Comment Detection
- Author: Atanu (or your name)
- Date: December 11, 2025

---

## Slide 1 — Agenda
- Problem statement
- Dataset
- Preprocessing
- Model & Training
- Results & Metrics
- API / Deployment
- Demo
- Next steps

---

## Slide 2 — Problem Statement
- Detect hateful content in short text (tweets/comments)
- Three classes: Hate Speech, Offensive Language, Neither/Clean
- Goal: High precision & recall for production use

---

## Slide 3 — Dataset
- Source: `data/archive/labeled_data.csv`
- Raw samples: 24,783 tweets
- After preprocessing & balancing: 57,546 samples (15.3K per class in final splits)
- Train/Val/Test: 46,036 / 5,754 / 5,756

---

## Slide 4 — Preprocessing
- Steps:
  - Expand contractions
  - Remove URLs, mentions, emojis
  - Normalize repeated characters and punctuation
  - Optional stopword handling
  - Oversampling to balance classes
- Files: `src/preprocess_dataset.py`

---

## Slide 5 — Model & Training
- Architecture: DistilBERT (`distilbert-base-uncased`)
- Framework: PyTorch + Hugging Face Transformers
- Hyperparams: 2 epochs, batch_size=16, lr=2e-5, max_length=128
- Training script: `src/train_pytorch.py`

---

## Slide 6 — Results
- Test accuracy: **97.03%**
- Precision: **0.9709**
- F1-score: **0.9701**
- Notes: Balanced dataset improved performance vs. unbalanced baseline (~92.46%)

---

## Slide 7 — API & Inference
- FastAPI server: `src/api.py`
- Endpoints: `/health`, `/predict`, `/predict-batch`
- Startup script: `scripts/run_api.py`
- Example: `curl -X POST http://localhost:8000/predict -d '{"text":"I hate this"}'`

---

## Slide 8 — Demo Plan
- Start server locally
- Show sample predictions (clean / offensive / hate)
- Show API docs at `/docs`

---

## Slide 9 — Next Steps
- Add authentication and rate limiting
- Deploy via Docker / Kubernetes
- Add monitoring & logging (Prometheus/Grafana)
- Expand to multi-language support

---

## Appendix
- Repo structure, important files, model path: `models/hate-detection-balanced`
- Contact / acknowledgements
