# Librairies nécessaires
from rag.evaluation import query_test
import os

# Éléments que nous voulons tester
index_path="faiss_index_long"
query="Un festival en été"
k=3
threshold=0.4
csv_file="evaluation.csv"

# Génération de la fonction afin de tester quelques questions sur notre base d'index
test = query_test(
    index_path,
    query,
    k,
    threshold,
    csv_file
)
