from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

text = """
LangChain is a framework for developing applications powered by language models.
It provides abstractions for prompt management, chaining, agents, and memory.
Semantic chunking helps in splitting text into meaningful parts rather than raw token counts.
"""

# Create a semantic chunker with your embedding model
chunker = SemanticChunker(embedding)

# Split the text into chunks
chunks = chunker.split_text(text)

print("Chunks:")
for i, c in enumerate(chunks, 1):
    print(f"{i}: {c}\n")