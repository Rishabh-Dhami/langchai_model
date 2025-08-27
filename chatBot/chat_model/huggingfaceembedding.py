from langchain_huggingface import HuggingFaceInferenceAPIEmbeddings


# --- Initialization ---
# Create an instance of the embeddings class.
# It automatically uses the environment variable for the token.
embeddings = HuggingFaceInferenceAPIEmbeddings(
    repo_id="sentence-transformers/all-MiniLM-L6-v2"
)


# --- Usage ---
# Now you can use the 'embeddings' object just like any other in LangChain.

# 1. To embed a single user query
query_text = "What is the best way to travel in Mumbai?"
query_embedding = embeddings.embed_query(query_text)

print(f"Embedding for the query has {len(query_embedding)} dimensions.")


# 2. To embed multiple documents for a vector database
documents_to_embed = [
    "The local train network is the lifeline of Mumbai.",
    "Auto-rickshaws are a common mode of transport for short distances.",
    "During monsoon, it's wise to check for waterlogging before traveling."
]
doc_embeddings = embeddings.embed_documents(documents_to_embed)

print(f"\nSuccessfully embedded {len(doc_embeddings)} documents.")
print(f"Each document embedding has {len(doc_embeddings[0])} dimensions.")