# Import Document directly from langchain.schema
from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain.retrievers import EnsembleRetriever
import os

# from embeddings import texts_meta

def rag_save_and_retriever():
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vector_store = FAISS.load_local("medgpt_faiss_index", embeddings, allow_dangerous_deserialization=True )

    docstore = vector_store.docstore
    id_map = vector_store.index_to_docstore_id
    # Reconstruct docs list
    docs = []
    for idx, doc_id in id_map.items():
        doc = docstore.search(doc_id)
        docs.append(doc)

    dense_retriever = vector_store.as_retriever(search_kwargs={"k": 5})

    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5

    hybrid_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, dense_retriever],
    weights=[0.4, 0.6]
    )

    return hybrid_retriever