from langchain_huggingface import HuggingFaceEmbeddings
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

document = [
    "Football is the most popular sport in the world",
    "A standard football match is played between two teams of eleven players",
    "Famous players like Lionel Messi and Cristiano Ronaldo have inspired millions",
    "The FIFA World Cup is the biggest international football tournament",
    "Fans celebrate football for its passion, teamwork, and unforgettable goals"
]

embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vector = embedding.embed_documents(document)

query = "How many players are in each football team during a match?"
q_vector = embedding.embed_query(query)

# Cosine similarity
score = cosine_similarity([q_vector], vector)[0]

# Rank results
# ranked = sorted(list(enumerate(score)), key=lambda x: x[1], reverse=True)

# for idx, sim in ranked:
#     print(f"Doc: {document[idx]} | Score: {sim:.4f}")


ranked = sorted(list(enumerate(score)), key=lambda x:x[1])[-1]

print(document[ranked[0]])