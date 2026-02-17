import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def compute_similarity(book_vectors: np.ndarray, user_vector: np.ndarray):
    user_vector = user_vector / (np.linalg.norm(user_vector) + 1e-9)
    return cosine_similarity(book_vectors, user_vector.reshape(1, -1)).flatten()
