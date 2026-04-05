import gradio as gr
from databricks.vector_search.client import VectorSearchClient
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
import os

# Configuration
CATALOG_NAME = "workspace"
SCHEMA_NAME = "farmer_queries"
INDEX_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.query_embeddings_index"
ENDPOINT_NAME = "farmer_query_endpoint"

# Initialize clients
# In Databricks Apps, WorkspaceClient auto-detects OAuth credentials
# Don't pass explicit token to avoid "more than one authorization method" error

# For Databricks App environment (OAuth is auto-configured)
w = WorkspaceClient()
token = w.config.authenticate()           # → {"Authorization": "Bearer dapi..."}
bearer = token.get("Authorization", "").replace("Bearer ", "")

vsc = VectorSearchClient(
    workspace_url=w.config.host,
    personal_access_token=bearer,         # real token, never None
    disable_notice=True
)

def answer_farmer_query(query: str) -> str:
    if not query or query.strip() == "":
        return "Please enter a question."
    
    try:
        # 🔍 Vector Search
        results = vsc.get_index(index_name=INDEX_NAME).similarity_search(
            query_text=query,
            columns=["questions", "answers"],
            num_results=3
        )

        result = results.get("result", {})
        rows = result.get("data_array") or result.get("data") or []

        context = "\n\n".join([
            f"Q: {r[0]}\nA: {r[1]}"
            for r in rows if len(r) >= 2
        ]) if rows else "No relevant context found."

        # 🧠 Prompt
        prompt = f"""You are an agricultural expert assistant helping farmers.
Use the context below to answer the question. Follow these rules strictly:
- Answer in the SAME LANGUAGE as the question
- Be concise and practical (2-4 sentences max)
- Do NOT repeat yourself
- Do NOT include any instructions, notes, or meta-commentary in your answer
- Stop after giving the answer. Do not add anything after it.
 
Context:
{context}
 
Question:
{query}
 
Answer (in the same language as the question, no repetition):"""

        # 🤖 LLM Call (CHANGE endpoint if needed)
        response = w.serving_endpoints.query(
            name="qwen_dense_model",
            dataframe_records=[{"inputs": prompt}]
        )
 
        raw = response.predictions[0]
        # Step 1: Strip the echoed prompt — take everything after the last Answer label
        
        for marker in ['Answer (in the same language as the question, no repetition):', 'Answer:']:
            if marker in raw:
                raw = raw.split(marker)[-1].strip()
                break
 
        # Step 2: Extract just the first clean sentence/paragraph before repetition starts
        # The model repeats the same answer — just take the first occurrence
        raw = raw.strip('"').strip()
        first_answer = raw.split('Answer:')[0].strip().strip('"').strip()
 
        return first_answer if first_answer else raw

    except Exception as e:
        print(f"Error: {e}")
        return "⚠️ System temporarily unavailable. Please try again."


# Create Gradio Interface
demo = gr.Interface(
    fn=answer_farmer_query,
    inputs=gr.Textbox(
        label="Your Question (Any Language)",
        placeholder="Ask your farming question in any language...",
        lines=3
    ),
    outputs=gr.Textbox(
        label="Answer",
        lines=8
    ),
    title="🌾 Farmer Query Assistant",
    description="Ask agricultural questions in your regional language and get expert answers. Powered by Databricks RAG and Foundation Models.",
    examples=[
        ["How do I control pests in my crop?"],
        ["मेरी फसल में कीट लग गए हैं। मुझे क्या करना चाहिए?"],
        ["What is the best fertilizer for wheat?"],
    ],
    theme=gr.themes.Soft(),
    allow_flagging="never"
)

if __name__ == "__main__":
    # Databricks Apps requires explicit server configuration
    demo.launch(server_name="0.0.0.0", server_port=8080, share=True)