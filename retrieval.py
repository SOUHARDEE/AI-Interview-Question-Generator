import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

with open("questions.json", "r") as f:
    data = json.load(f)

questions = [item["question"] for item in data]

print("Loading embedding model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Generating embeddings...")
embeddings = model.encode(questions)
embeddings = np.array(embeddings).astype("float32")

dimension = embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(embeddings)

print("FAISS index built successfully!\n")


def search(query, skill=None, difficulty=None, k=3):
    filtered_data = data

    if skill:
        filtered_data = [
            item for item in filtered_data
            if item["skill"].lower() == skill.lower()
        ]

    if difficulty:
        filtered_data = [
            item for item in filtered_data
            if item["difficulty"].lower() == difficulty.lower()
        ]

    if not filtered_data:
        return [], None

    filtered_questions = [item["question"] for item in filtered_data]

    filtered_embeddings = model.encode(filtered_questions)
    filtered_embeddings = np.array(filtered_embeddings).astype("float32")

    temp_index = faiss.IndexFlatL2(filtered_embeddings.shape[1])
    temp_index.add(filtered_embeddings)

    query_embedding = model.encode([query])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = temp_index.search(
        query_embedding,
        min(k, len(filtered_questions))
    )

    retrieved_questions = [filtered_questions[i] for i in indices[0]]

    return retrieved_questions, distances[0]