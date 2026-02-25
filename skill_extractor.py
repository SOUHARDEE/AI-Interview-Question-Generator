import numpy as np
from sentence_transformers import SentenceTransformer
from skills import skill_list

model = SentenceTransformer("all-MiniLM-L6-v2")

skill_embeddings = model.encode(skill_list)
skill_embeddings = np.array(skill_embeddings).astype("float32")


def extract_skills_from_jd(job_description, top_k=5):
    jd_embedding = model.encode([job_description])
    jd_embedding = np.array(jd_embedding).astype("float32")

    similarities = np.dot(skill_embeddings, jd_embedding.T).squeeze()

    top_indices = similarities.argsort()[-top_k:][::-1]

    extracted = [skill_list[i] for i in top_indices]

    return extracted