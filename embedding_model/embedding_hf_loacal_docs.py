from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(model_name = "sentence-transformers/all-MiniLM-L6-v2")

doc = [
    "Delhi is the capital of india",
    "My name is Rishabh Singh Dhami",
    "My love is Shanti Devi"
]

vector = embedding.embed_documents(doc)

print(str(vector))