from langchain.text_splitter import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
import os
import sys

from langchain.schema import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_core.documents import Document

# Initialize splitter and embedder
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# FAISS index placeholder
dim = embedder.get_sentence_embedding_dimension()
faiss_index = faiss.IndexFlatL2(dim)
texts_meta = []  # store (text, source)

def chunk_and_embeddings(df_train_symcat, df_medquad, df_pubmed):
    print("Building new FAISS index from datasets...")

    #Medquad embeddings
    for _, row in df_medquad.iterrows():
        doc = row['question'] + " " + row['answer']
        for chunk in splitter.split_text(doc):
            emb = embedder.encode(chunk)
            faiss_index.add(np.array([emb]))
            texts_meta.append({'text': chunk, 'source': 'MedQuAD'})
            # print("Medquad : ",len(texts_meta))
            print(f"\rMedquad :  {len(texts_meta)}", end="")
            sys.stdout.flush()

    #Pubmed embeddings
    for _, row in df_pubmed.iterrows():
        doc = row['Title'] + " " + row['Abstract']
        for chunk in splitter.split_text(doc):
            emb = embedder.encode(chunk)
            faiss_index.add(np.array([emb]))
            texts_meta.append({'text': chunk, 'source': 'PubMed'})
            # print("pubmed : ",len(texts_meta))
            print(f"\rpubmed :  {len(texts_meta)}", end="")
            sys.stdout.flush()
    
    # SymCat embedding: include both explicit and implicit symptoms
    # df_train_symcat = df_train_symcat[:100000]
    for _, row in df_train_symcat.iterrows():
        # Join the lists of symptoms into strings before concatenation
        explicit_symptoms_str = ", ".join(row['explicit_symptoms'])
        implicit_symptoms_str = ", ".join(row['implicit_symptoms'])
        # Combine explicit and implicit symptoms into one document
        doc = (
            "Explicit symptoms: " + explicit_symptoms_str + ". "
            "Implicit symptoms: " + implicit_symptoms_str + "."
        )
        # Split into chunks
        for chunk in splitter.split_text(doc):
            emb = embedder.encode(chunk)
            faiss_index.add(np.array([emb]))
            texts_meta.append({
                'text': chunk,
                'source': 'SymCat'
            })
            # print("Symcat : ",len(texts_meta))
            print(f"\rSymcat :  {len(texts_meta)}", end="")
            sys.stdout.flush()
    
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    docs = [Document(page_content=m['text'], metadata={'source': m['source']}) for m in texts_meta]
    # Build FAISS store
    vector_store = FAISS.from_documents(docs, embeddings)
    # Persist locally for faster reloads
    vector_store.save_local("medgpt_faiss_index")
    return texts_meta


def text_embedding(df_train_symcat, df_medquad, df_pubmed, INDEX_DIR):
    if os.path.exists(INDEX_DIR):
        return texts_meta
    else:
        return chunk_and_embeddings(df_train_symcat, df_medquad, df_pubmed)
    # return chunk_and_embeddings(df_train_symcat, df_medquad, df_pubmed)