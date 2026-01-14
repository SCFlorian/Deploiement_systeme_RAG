from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


# ======================
# VERSION LONGUE (chunk)
# ======================
def generation_embeddings(df_long):
    texts = df_long["text_for_embedding"].astype(str).tolist()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    documents = splitter.create_documents(
        texts=texts,
        metadatas=[
            {
                "doc_id": i,
                "Date de début": df_long.iloc[i]["firstdate_begin"],
                "Date de fin": df_long.iloc[i]["lastdate_end"],
                "Lieu": df_long.iloc[i]["location_name"],
                "Adresse postale": df_long.iloc[i]["location_address"]
            }
            for i in range(len(texts))
        ]
    )

    return documents


# ======================
# VERSION COURTE (1 doc)
# ======================
def generation_embeddings_short(df_short):
    documents = []

    for i, row in df_short.iterrows():
        documents.append(
            Document(
                page_content=row["text_for_embedding"],
                metadata={
                    "doc_id": i,
                    "Date de fin": row["lastdate_end"],
                    "Lieu": row["location_name"],
                    "Adresse postale": row["location_address"],
                    "version": "short"
                }
            )
        )

    return documents
