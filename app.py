import gradio as gr
from databricks.vector_search.client import VectorSearchClient
import os

# Configuration
CATALOG_NAME = "workspace"
SCHEMA_NAME = "farmer_queries"
INDEX_NAME = f"{CATALOG_NAME}.{SCHEMA_NAME}.query_embeddings_index"
ENDPOINT_NAME = "farmer_query_endpoint"

# Initialize Vector Search client
vsc = VectorSearchClient(disable_notice=True)


def answer_farmer_query(query: str) -> str:
    """
    Answer farmer queries in their regional language using RAG.
    Uses Databricks Foundation Model APIs for multilingual support.
    """
    if not query or query.strip() == "":
        return "Please enter a question."
    
    try:
        # Step 1: Vector Search - Find relevant Q&A pairs
        results = vsc.get_index(index_name=INDEX_NAME).similarity_search(
            query_text=query,
            columns=["questions", "answers"],
            num_results=3
        )
        
        rows = results.get("result", {}).get("data_array", [])
        
        if rows and len(rows) > 0:
            # Build context from retrieved Q&A pairs
            context = "\n\n".join([
                f"Q: {r[0]}\nA: {r[1]}"
                for r in rows if len(r) >= 2
            ])
        else:
            context = "No relevant context found in the knowledge base."
        
        # Step 2: Build prompt for the LLM
        prompt = f"""You are an agricultural expert assistant helping farmers.
Answer clearly and practically using the context below.
Respond in the SAME LANGUAGE as the question.
If the context doesn't contain relevant information, provide general agricultural guidance.

Context:
{context}

Question:
{query}

Answer:"""
        
        # Step 3: Generate answer using Databricks Foundation Model API
        # Note: Replace with your preferred model endpoint
        # For production, use: databricks-meta-llama-3-1-70b-instruct or similar
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
        
        w = WorkspaceClient()
        response = w.serving_endpoints.query(
            name="databricks-meta-llama-3-1-70b-instruct",
            messages=[
                ChatMessage(
                    role=ChatMessageRole.SYSTEM,
                    content="You are a helpful agricultural expert assistant."
                ),
                ChatMessage(
                    role=ChatMessageRole.USER,
                    content=prompt
                )
            ],
            max_tokens=500
        )
        
        answer = response.choices[0].message.content
        return answer
        
    except Exception as e:
        return f"Error processing your query: {str(e)}\n\nPlease make sure the vector search index is properly set up and the model endpoint is available."


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
    demo.launch()
