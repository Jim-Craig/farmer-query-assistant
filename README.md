# 🌾 Farmer Query Assistant

A multilingual RAG application that lets farmers ask agricultural questions in any regional language and receive expert answers in the same language — powered by a custom-pruned Qwen3-0.6B model and Databricks Vector Search over 207,000+ real farmer Q&A pairs.

---

## Architecture

```
  ┌─────────────────────────────────────────────────────────────────────┐
  │                        Databricks App                               │
  │                                                                     │
  │   User query (any language)                                         │
  │        │                                                            │
  │        ▼                                                            │
  │   ┌─────────────┐                                                   │
  │   │  Gradio UI  │                                                   │
  │   └──────┬──────┘                                                   │
  │          │ query text                                               │
  │          ▼                                                          │
  │   ┌──────────────────────────────┐                                  │
  │   │   Databricks Vector Search   │  similarity_search (top 3)       │
  │   │   query_embeddings_index     │◄────────────────────────────     │
  │   │   (BGE Large EN embeddings)  │                                  │
  │   └──────────────┬───────────────┘                                  │
  │                  │ top-3 Q&A pairs as context                       │
  │                  ▼                                                  │
  │   ┌──────────────────────────────┐                                  │
  │   │   Prompt Builder             │                                  │
  │   │   context + question         │                                  │
  │   └──────────────┬───────────────┘                                  │
  │                  │                                                  │
  │                  ▼                                                  │
  │   ┌──────────────────────────────────────────────┐                 │
  │   │   Qwen3-0.6B  ◄── INNOVATION: Custom         │                 │
  │   │   (pruned 50%, layers 10-20)   Covariance +  │                 │
  │   │   Databricks Model Serving     WANDA Pruning  │                 │
  │   └──────────────┬───────────────────────────────┘                 │
  │                  │ raw prediction                                   │
  │                  ▼                                                  │
  │   ┌──────────────────────────────┐                                  │
  │   │   Response Cleaner           │                                  │
  │   │   strip echo + dedup         │                                  │
  │   └──────────────┬───────────────┘                                  │
  │                  │                                                  │
  │                  ▼                                                  │
  │   Answer (same language as query)                                   │
  └─────────────────────────────────────────────────────────────────────┘
```

---

## 💡 Innovation: Custom Covariance + WANDA Pruning

Instead of using an off-the-shelf quantised model, this project implements a **custom structured pruning algorithm** that reduces the Qwen3-0.6B MLP intermediate dimension by 50% on layers 10–20, while preserving as much accuracy as possible.

### Why this matters
Standard quantisation (4-bit, 8-bit) reduces memory but keeps all neurons. Structured pruning physically removes neurons — making the model smaller, faster at inference, and deployable on lower-cost serving hardware — while the WANDA-based scoring ensures the most important neurons are kept.

### How the algorithm works

```
For each target layer (10 to 20):
│
├── 1. HOOK — register a pre-hook on down_proj
│         Captures input activations X during a forward pass
│         Builds covariance matrix C = Xᵀ X  (shape: d_int × d_int)
│         Raises InterruptExecution to stop forward pass early (saves GPU memory)
│
├── 2. SCORE — rank neurons using WANDA-style importance
│         diagonal(C)  →  activation magnitude per neuron
│           ×
│         ‖W_down_proj‖  →  weight magnitude per neuron
│         ─────────────────────────────────────────────
│         score_i = C_ii × ‖w_i‖   (higher = more important)
│
├── 3. SELECT — keep top-k neurons
│         k = (1 - sparsity) × d_int
│         topk_indices = argsort(scores, descending)[:k]
│         Build selection matrix S_k  (d_int × k)
│
└── 4. PROJECT — rewrite weight matrices into pruned subspace
          up_proj   (hidden → d_int)  →  new_up_proj   (hidden → k)
          gate_proj (hidden → d_int)  →  new_gate_proj (hidden → k)
          down_proj (d_int → hidden)  →  new_down_proj (k → hidden)

          W_U_k = [up_proj; gate_proj]ᵀ @ S_k
          W_D_k = down_proj @ S_k
```

### Key design choices

| Choice | Why |
|---|---|
| **Covariance matrix** instead of random or magnitude-only scoring | Captures how neurons co-activate — more informative than weight magnitude alone |
| **WANDA multiplier** (activation × weight norm) | Combines data-driven activation importance with weight magnitude, outperforming either alone |
| **InterruptExecution hook** | Stops the forward pass immediately after the target layer, avoiding unnecessary computation across the full network |
| **Layers 10–20 only** | Early and late layers are more critical for language understanding; middle layers have more redundancy |
| **S_k projection matrix** | Cleanly rewrites all three weight matrices (up, gate, down) into the pruned subspace with no zeroed-out dead weights |

### Before vs After

| Metric | Before Pruning | After Pruning |
|---|---|---|
| MLP intermediate dim (layers 10-20) | d_int | 0.5 × d_int |
| Parameters reduced | — | ~15–20% fewer total |
| Evaluated on | SciQ benchmark | SciQ benchmark |

---

## Technologies Used

### Databricks
| Technology | Purpose |
|---|---|
| **Databricks Apps** | Hosts the Gradio web interface on port 8080 |
| **Databricks Vector Search** | Semantic similarity search over 207K+ farmer Q&A pairs |
| **Databricks Model Serving** | Serves the pruned Qwen model as a REST endpoint |
| **Unity Catalog** | Stores data table, vector index, and registered model under `workspace.farmer_queries` |
| **MLflow** | Tracks pruning experiments and registers the model to Unity Catalog |
| **Delta Lake** | Stores the Q&A dataset with Change Data Feed enabled for index sync |
| **Databricks SDK v0.67.0** | Workspace client and OAuth auth for the app |
| **BGE Large EN** (`databricks-bge-large-en`) | Managed embedding model for the vector index |

### Open-Source Models & Libraries
| Model / Library | Purpose |
|---|---|
| **Qwen/Qwen3-0.6B** | Base LLM — small, multilingual causal language model |
| **Custom covariance + WANDA pruning** | Structured MLP pruning on layers 10–20 at 50% sparsity |
| **lm-eval (EleutherAI)** | Evaluates accuracy on SciQ before and after pruning |
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

| Cell | What it does |
|---|---|
| 1 | Installs dependencies |
| 2 | Imports libraries |
| 3 | PLEASE ADD HUGGINGFACE KEY TO DOWNLOAD MODELS AND Loads `Qwen/Qwen3-0.6B` from Hugging Face |
| 4 | Defines pruning classes (`Prune`, `algo1_functor_mod`) |
| 5 | Loads SciQ calibration dataset |
| 6 | Builds DataLoader for calibration |
| 7 | Runs pruning (50% sparsity, layers 10–20) and evaluates on SciQ |
| 8 | Registers pruned model to `workspace.farmer_queries.qwen_dense_model` via MLflow |
| 9 | Verifies model signature in Unity Catalog |

After Cell 8, go to **Databricks UI → Serving → Create Endpoint** and point it to `workspace.farmer_queries.qwen_dense_model`.

### Step 2 — Farmer_Query_AI_App.ipynb

| Cell | What it does |
|---|---|
| 1 | Installs dependencies |
| 2 | Sets config constants |
| 3 | Downloads and loads the 207K Q&A dataset |
| 4 | Saves to Unity Catalog with primary key column |
| 5 | Creates Vector Search endpoint and delta sync index |
| 6 | Waits for index to finish syncing |
| 7 | Finds the app service principal client ID |
| 8 | Grants Unity Catalog permissions |
| 9 | Grants `CAN_QUERY` on the serving endpoint |
| 10 | Verifies model signature |
| 11 | Smoke tests the full RAG pipeline with 3 sample queries |

### Step 3 — Deploy app.py

In **Databricks UI → Apps → Create App**, point it to `app.py`. The app starts on `0.0.0.0:8080`.

---

## Demo

**Try these prompts:**

| Prompt | What it tests |
|---|---|
| `What is the best fertilizer for wheat?` | English query with RAG retrieval |
| `How do I control aphids in mustard?` | Specific pest query matched from dataset |
| `What is the fertilizer dose for wheat per bigha?` | Precise answer (urea 12kg, SSP 17kg, MOP 4–8kg/bigha) |
| `मेरी फसल में कीट लग गए हैं। मुझे क्या करना चाहिए?` | Hindi multilingual response |

---

## Known Issues & Fixes

| Issue | Cause | Fix |
|---|---|---|
| `"more than one authorization method"` | `w.config.token` is `None` in OAuth environments | Extract bearer via `w.config.authenticate()` |
| `'dict' object has no attribute 'as_dict'` | SDK v0.67.0 calls `.as_dict()` on plain dicts | Bypass SDK — use `requests.post()` to `/serving-endpoints/.../invocations` |
| Model echoes the full prompt | Pruned model loses instruction-following | Split on `Answer:` and take the last segment |
| Model repeats the same sentence | 50% pruning degrades the model's stopping ability | Take only the first answer before the first repeated `Answer:` label |
| `PRINCIPAL_DOES_NOT_EXIST` on GRANT | Unity Catalog needs the client ID (UUID), not the display name | Use `sp.application_id` from `w.service_principals.list()` |
| `PARSE_SYNTAX_ERROR` on multiple GRANTs | `spark.sql()` only accepts one statement at a time | Run each GRANT as a separate `spark.sql()` call |
