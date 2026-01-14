# Librairies nécessaires
import os
import csv
from langchain_community.vectorstores import FAISS
from langchain_mistralai.embeddings import MistralAIEmbeddings
from dotenv import load_dotenv

# Permet de contrer le problème libomp.dylib. Pas la solution la plus élégante mais la plus efficace pour le moment
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
load_dotenv()

# Fonction d'évaluation de la capacité de nos index en fonction d'une question
# Enregistrement dans un fichier csv
def query_test(
        index_path,
        query,
        k,
        threshold,
        csv_file
):
    # ===================
    # Embeddings (recréés ici)
    # ===================
    embeddings = MistralAIEmbeddings(
        api_key=os.getenv("MISTRAL_KEY")
    )

    # ===================
    # Chargement FAISS
    # ===================
    vectorstore = FAISS.load_local(
        index_path,
        embeddings,
        allow_dangerous_deserialization=True
    )

    docs_with_scores = vectorstore.similarity_search_with_score(query, k=k)

    filtered_docs = [
        (doc, score)
        for doc, score in docs_with_scores
        if score < threshold
    ]

    # ===================
    # CSV
    # ===================
    write_header = not os.path.exists(csv_file)

    with open(csv_file, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        if write_header:
            writer.writerow([
                "index",
                "question",
                "k",
                "threshold",
                "doc_id",
                "score",
                "contenu"
            ])

        for doc, score in filtered_docs:
            writer.writerow([
                index_path,
                query,
                k,
                threshold,
                doc.metadata.get("doc_id"),
                round(score, 4),
                doc.page_content[:300]
            ])

    # ===================
    # Debug console
    # ===================
    for doc, score in filtered_docs:
        print("-" * 80)
        print("Index :", index_path)
        print("Doc ID :", doc.metadata.get("doc_id"))
        print("Score :", round(score, 4))
        print(doc.page_content[:300])
        print("-" * 80)

    return filtered_docs

