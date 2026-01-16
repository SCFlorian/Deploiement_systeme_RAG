# =======================
# Librairies nécessaires
# =======================
import numpy as np
# ===================================
# Test de similarité avec le cosinus
# ===================================
import numpy as np

def cos_test(vectorstore, i=0, j=1):
    """
    Test de similarité cosinus entre deux documents stockés dans FAISS
    """
    vec1 = vectorstore.index.reconstruct(i)
    vec2 = vectorstore.index.reconstruct(j)

    vec1 = np.array(vec1)
    vec2 = np.array(vec2)

    similarite = np.dot(vec1, vec2) / (
        np.linalg.norm(vec1) * np.linalg.norm(vec2)
    )

    return similarite
