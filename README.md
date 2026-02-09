
# RAG Evaluation - e5-base + Ollama (Job Assessment)

This repository implements a **Retrieval-Augmented Generation (RAG)** pipeline
following the technical assessment instructions.
It is 100% open source and runs locally (no paid API keys required).

## Models (Open Source, local)
- **Embeddings:** intfloat/e5-base
- **Question Answering:** Ollama model `qwen2.5:1.5b`
- **Vector Store:** FAISS (local)

## Corpus
Use a 10-15 page technical PDF for evaluation. Place the PDF in `data/` and update the notebook path if needed.
For this assessment, the RAG dataset used `data/DSP_API_Guide-200-en.pdf`.

## Pipeline
1. PDF ingestion & chunking
2. e5-base embeddings
3. FAISS vector index
4. Retrieval (top-k)
5. Local LLM generation via Ollama (prompt with context only)
6. Evaluation using RAGAS (answer relevancy + faithfulness) + cosine similarity

## Ollama Setup
1. Install Ollama.
2. Start the Ollama server:
   ```bash
   ollama serve
   ```
3. Pull the model:
   ```bash
   ollama pull qwen2.5:1.5b
   ```
4. Optional quick check:
   ```bash
   ollama run qwen2.5:1.5b "Say hello in French."
   ```

## Project Setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Place the PDF documentation inside `data/` and run the notebook:
```bash
jupyter notebook notebook/demo_pipeline.ipynb
```
Update `data/ground_truth.json` with at least 5 questions/answers from your PDF before running evaluation.

## Evaluation Notes
RAGAS faithfulness and Answer Relevancy uses an LLM. This project uses a **local Ollama model** for both generation and RAGAS evaluation via a LangChain wrapper. Cosine similarity is computed between generated answers and ground-truth answers using e5-base.

## Project Structure

### Source Files (`src/`)
- **`ingest.py`** — PDF ingestion and text chunking
  - `extract_text(pdf_path)` — Extracts text from PDF using pdfplumber
  - `chunk_text(text, chunk_size=1024, overlap=200)` — Splits text into overlapping chunks

- **`embeddings.py`** — Embedding generation and query encoding
  - `load_e5()` — Loads the e5-base model from HuggingFace
  - `embed_passages(model, chunks)` — Embeds document chunks with proper "passage:" prefix
  - `embed_query(model, query)` — Embeds user query with proper "query:" prefix

- **`vector_store.py`** — FAISS vector index and retrieval
  - `VectorStore` class — Manages FAISS index and stores chunk texts
  - `add(embeddings, texts)` — Adds chunks and their embeddings to the index
  - `search(query_emb, k=5)` — Retrieves top-k most similar chunks

- **`qa.py`** — Local LLM question answering
  - `LocalLLMQA` class — Wraps Qwen 2.5 (1.5B) model via LangChain
  - `answer(question, contexts)` — Generates answers using only provided context

- **`evaluate.py`** — Evaluation metrics
  - `evaluate_cosine_similarity(answers, ground_truths)` — Computes semantic similarity using e5-base
  - `evaluate_ragas(questions, answers, contexts, llm, ground_truths)` — RAGAS evaluation (faithfulness + answer relevancy)

### Key Files
- **`notebook/demo_pipeline.ipynb`** — Main pipeline notebook with full workflow
- **`data/ground_truth.json`** — Question/answer pairs for evaluation
- **`data/DSP_API_Guide-200-en.pdf`** — RAG dataset (technical documentation)
- **`evaluation_results_*.csv`** — Output evaluation metrics (can be converted to Excel)


## Deliverables
- `notebook/demo_pipeline.ipynb` runs the end-to-end pipeline.
- `evaluation_results_*.csv` contains exported evaluation results for different chunk sizes.
