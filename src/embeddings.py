import numpy as np
from sentence_transformers import SentenceTransformer

# MODEL_PATH = "nomic-ai/nomic-embed-text-v1"
MODEL_PATH = "sentence-transformers/all-MiniLM-L6-v2"


class Embeddings:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)

    def compute_embedding(self, text_content: str):
        payload = f"{text_content.strip('/wiki/')}"
        return self.model.encode(payload)

    def compute_embeddings_for_nav_links(self, nav_links):
        links = list(nav_links.values())
        batch = [f"{link.strip('/wiki/')}" for link in links]
        embeddings = self.model.encode(batch)
        embeddings_dict = {
            link: embedding for link, embedding in zip(links, embeddings)
        }
        return embeddings_dict

    @staticmethod
    def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
        if a.shape != b.shape:
            print(f"Shape mismatch: {a.shape} vs {b.shape}")
            return 0.0
        if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
            return 0.0
        return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
