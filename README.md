# 🌾 Farmer Query Assistant

A multilingual RAG application that lets farmers ask agricultural questions in any regional language and receive expert answers in the same language — powered by a pruned Qwen3-0.6B model and Databricks Vector Search over 207,000+ real farmer Q&A pairs.

---

## Architecture

```
                        ┌──────────────────────────────────┐
                        │         Databricks App            │
                        │                                  │
  User query            │   ┌────────────┐                 │
  (any language) ──────►│   │  Gradio UI │                 │
                        │   └─────┬──────┘                 │
                        │         │                        │
                        │         ▼                        │
                        │   ┌─────────────────────────┐   │
                        │   │  Databricks Vector       │   │
                        │   │  Search                  │   │
                        │   │  (query_embeddings_index)│   │
                        │   │  Top-3 similar Q&A pairs │   │
                        │   └──────────┬──────────────┘   │
                        │              │ context           │
                        │         ┌────▼──────────────┐   │
                        │         │  Qwen3-0.6B        │   │
                        │         │  (pruned 50%)      │   │
                        │         │  Model Serving     │   │
                        │         └────────────────────┘   │
                        │              │ answer             │
                        │              ▼                   │
                        │   Response (same language)       │
                        └──────────────────────────────────┘
```

---

## Technologies Used

### Databricks
| Technology | Purpose |
|---|---|
| **Databricks Apps** | Hosts the Gradio web interface on port 8080 |
| **Databricks Vector Search** | Semantic similarity search over 207K+ farmer Q&A pairs |
| **Databricks Model Serving** | Serves the pruned Qwen model as a REST endpoint |
| **Unity Catalog** | Stores the data table, vector index, and registered model under `workspace.farmer_queries` |
| **MLflow** | Tracks pruning experiments and registers the model to Unity Catalog |
| **Delta Lake** | Stores the raw Q&A dataset with Change Data Feed enabled for index sync |
| **Databricks SDK v0.67.0** | Workspace client and OAuth auth for the app |
| **BGE Large EN** (`databricks-bge-large-en`) | Managed embedding model for the vector index |

### Open-Source Models & Libraries
| Model / Library | Purpose |
|---|---|
| **Qwen/Qwen3-0.6B** | Base LLM — small, multilingual causal language model |
| **50% structured pruning** (custom) | Covariance-based MLP pruning on layers 10–20 using WANDA-style scoring |
| **lm-eval (EleutherAI)** | Evaluates model accuracy on SciQ before and after pruning |
| **Gradio** | Web UI framework |
| **databricks-vectorsearch** | Python client for Vector Search |
| **transformers 4.51.3** | Model loading, tokenisation, and inference |
| **sentence-transformers** | Embedding generation |

### Dataset
`questionsv4.csv` — 207,101 real farmer queries and expert answers covering crops, pests, fertilisers, government schemes, irrigation, and more.

---

## Project Structure

```
├── Model_compression.ipynb     # Prune Qwen3-0.6B and register to Unity Catalog
├── Farmer_Query_AI_App.ipynb   # Data ingestion, Vector Search setup, permissions
├── app.py                      # Gradio app — Databricks Apps entry point
├── questionsv4.csv             # Farmer Q&A dataset (207K rows)
└── README.md
```

---

## How to Run

Run the two notebooks **in order**, then deploy the app.

### Step 1 — Model_compression.ipynb

Run each cell top to bottom:

| Cell | What it does |
|---|---|
| 1 | Installs dependencies |
| 2 | Imports libraries |
| 3 | Please give huggingface KEY to load models. Loads `Qwen/Qwen3-0.6B` from Hugging Face |
| 4 | Defines pruning classes (`Prune`, `algo1_functor_mod`) |
| 5 | Loads SciQ calibration dataset |
| 6 | Builds DataLoader for calibration |
| 7 | **Runs pruning** (50% sparsity, layers 10–20) and evaluates on SciQ |
| 8 | Registers pruned model to `workspace.farmer_queries.qwen_dense_model` via MLflow |
| 9 | Verifies model signature in Unity Catalog |

After Cell 8 completes, go to **Databricks UI → Serving → Create Endpoint** and point it to `workspace.farmer_queries.qwen_dense_model`. Note the endpoint ID for Step 2.

---

### Step 2 — Farmer_Query_AI_App.ipynb

Run each cell top to bottom:

| Cell | What it does |
|---|---|
| 1 | Installs dependencies |
| 2 | Sets config constants (catalog, schema, endpoint names) |
| 3 | Downloads and loads the 207K Q&A dataset |
| 4 | Saves dataset to Unity Catalog with a primary key column |
| 5 | Creates Vector Search endpoint and delta sync index |
| 6 | Waits for the index to finish syncing |
| 7 | Finds the app service principal client ID automatically |
| 8 | Grants Unity Catalog permissions (`USE CATALOG`, `USE SCHEMA`, `SELECT`) |
| 9 | Grants `CAN_QUERY` on the serving endpoint |
| 10 | Verifies model signature |
| 11 | **Smoke tests** the full RAG pipeline with 3 sample queries |

---

### Step 3 — Deploy app.py

In **Databricks UI → Apps → Create App**, point it to `app.py`. The app starts on `0.0.0.0:8080`.

---

## Demo

Once the app is running, open it via the Databricks Apps URL.

**Try these prompts:**

| Prompt | What it tests |
|---|---|
| `What is the best fertilizer for wheat?` | English query with RAG retrieval |
| `How do I control aphids in mustard?` | Specific pest query matched from dataset |
| `What is the fertilizer dose for wheat per bigha?` | Precise answer (urea 12kg, SSP 17kg, MOP 4–8kg/bigha) |
| `मेरी फसल में कीट लग गए हैं। मुझे क्या करना चाहिए?` | Hindi multilingual response |

**What happens when you submit:**
1. The app retrieves the 3 most similar Q&A pairs from the vector index as context
2. Builds a prompt with those pairs and your question
3. Calls the pruned Qwen endpoint via `requests.post()` directly
4. Strips any echoed prompt and repeated sentences from the response
5. Returns a clean 2–4 sentence answer in the same language as your question

---

## Configuration

Edit the constants at the top of `app.py` if your names differ:

```python
CATALOG_NAME  = "workspace"
SCHEMA_NAME   = "farmer_queries"
INDEX_NAME    = "workspace.farmer_queries.query_embeddings_index"
MODEL_ENDPOINT = "qwen_dense_model"
```

---

## Known Issues & Fixes

| Issue | Cause | Fix |
|---|---|---|
| `"more than one authorization method"` on `VectorSearchClient` | `w.config.token` is `None` in OAuth (Databricks Apps) | Extract bearer via `w.config.authenticate()` |
| `'dict' object has no attribute 'as_dict'` on `.query()` | SDK v0.67.0 calls `.as_dict()` on plain dicts internally | Bypass SDK — call endpoint directly via `requests.post()` to `/serving-endpoints/.../invocations` |
| Model echoes the full prompt in response | Pruned model loses instruction-following ability | Split on `Answer:` and take only the last segment |
| Model repeats the same sentence many times | 50% pruning degrades the model's ability to stop generating | Take only the first answer before the first repeated `Answer:` label |
| `PRINCIPAL_DOES_NOT_EXIST` on GRANT | Unity Catalog needs the client ID (UUID), not the display name | Use `sp.application_id` from `w.service_principals.list()` |
| `PARSE_SYNTAX_ERROR` on multiple GRANTs | `spark.sql()` only accepts one statement at a time | Run each GRANT as a separate `spark.sql()` call |
