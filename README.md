# LawGPT

A legal AI platform I built to answer Indian law queries using retrieval-augmented generation, a legal knowledge graph, and LLM-powered structured reasoning.

The system indexes court judgments, retrieves relevant cases for a given query, maps facts to constitutional articles and legal acts, and generates a structured legal analysis — Rule, Application, Conclusion, and cited Sections. It also predicts likely case outcomes using a trained classifier, with a confidence gate that suppresses unreliable predictions rather than surfacing them.

---

## What it does

**Legal Research** — Ask a legal question. The system retrieves relevant judgments, enriches context with the knowledge graph, and generates a streamed legal analysis via an LLM.

**Precedent Search** — Search the indexed case database for similar judgments. Returns ranked results with relevance scoring.

**Case Assessment** — Enter case details to get an outcome prediction (Allowed / Dismissed / Partial). If confidence is below threshold, the system refuses to predict and redirects to the reasoning pipeline instead.

---

## How it's built

The retrieval pipeline uses **BAAI/bge-m3** — a hybrid dense + sparse model — because legal text has exact statutory phrases that dense-only models miss. Top-20 candidates come back from ChromaDB, then a **cross-encoder reranker** (ms-marco-MiniLM-L-6-v2) handles precision before anything goes to the LLM.

On top of that sits a **Graph-RAG layer** built with NetworkX. It extracts facts from the query, maps them to constitutional articles and IPC sections, and injects that structured context into the prompt alongside the retrieved documents.

The **outcome classifier** is a Logistic Regression model trained on 500 labelled cases using bge-m3 embeddings. Same embedder used end-to-end — training and inference — so there's no dimension mismatch.

The **confidence gate** is the part I'm most deliberate about. In a legal context, a wrong confident answer is worse than no answer. Below a reliability threshold, the system doesn't predict — it tells the user to use the research module instead.

---

## Stack

```
Embeddings      BAAI/bge-m3 (1024-dim, hybrid dense + sparse)
Reranker        cross-encoder/ms-marco-MiniLM-L-6-v2
Vector DB       ChromaDB
LLM             Ollama — phi3:latest
Knowledge Graph NetworkX
Classifier      Scikit-learn (Logistic Regression)
Frontend        Streamlit
Language        Python
```

---

## Project structure

```
app.py                  Streamlit app — all three modules
build_vector_db.py      Ingest JSONL → ChromaDB
train_classifier.py     Train outcome classifier
chunk_documents.py      Semantic chunking
graph_rag.py            Graph-RAG pipeline
legal_graph.py          Knowledge graph — facts to articles/acts
fact_mapper.py          Query → fact extraction
test_graph_rag.py       Graph-RAG tests
models/                 Saved classifier
input_docs/             User uploaded documents
```

---

## Setup

You'll need Python 3.9+, and [Ollama](https://ollama.ai) running locally with phi3 pulled:

```bash
ollama pull phi3:latest
```

Then:

```bash
git clone https://github.com/varshreddyy/LAWGPT.git
cd LAWGPT
pip install -r requirements.txt
```

Build the vector database (put your `dev.jsonl` in `data/`):

```bash
python build_vector_db.py
```

Train the classifier (put `legal_ai_dataset_500_entries.csv` in root):

```bash
python train_classifier.py
```

Run:

```bash
streamlit run app.py
```

---

## Dataset

- 10,000+ Indian Supreme Court and High Court judgment records in JSONL format
- 500 labelled cases for classifier training (Allowed / Dismissed / Partial)
- Knowledge graph covering constitutional articles and key IPC sections

---

## What's next

- Docker setup
- FastAPI backend for proper production serving
- Expanded knowledge graph
- Evaluation harness with logged retrieval metrics

---

## Disclaimer

This is a research project. Not a substitute for actual legal advice.
# lawgpt
