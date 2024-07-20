import numpy as np
from sentence_transformers import SentenceTransformer

MODEL_PATH = "nomic-ai/nomic-embed-text-v1"


class Embeddings:
    def __init__(self):
        self.model = SentenceTransformer(MODEL_PATH, trust_remote_code=True)

    def compute_embedding(self, text_content: str):
        return self.model.encode(f"classification: {text_content}")

    def compute_embeddings_for_nav_links(self, nav_links):
        links = list(nav_links.values())
        embeddings = self.model.encode([f"classification: {link}" for link in links])
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
