from retriever import rag_save_and_retriever
from system_prompt import system_prompt
from clean_text import clean_text
from embeddings import text_embedding
# from datasets import df_train_symcat, df_medquad, df_pubmed

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from google import genai


print("Started")
# Creating the vector store and embeddings of the datasets
# INDEX_DIR = "C:/Users/arunp/Documents/project/M.Tech Mini Project/MedGPT/medgpt_faiss_index"
# embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
# texts_meta = text_embedding(df_train_symcat, df_medquad, df_pubmed, INDEX_DIR)
# # print("Text embedding completed")
# retriever = rag_save_and_retriever()
# print("Retriver initiated")
    
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("Gemini_API_KEY")
# Initialize Gemini LLM instead of ChatOpenAI
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0,
)
# Initialize the GenAI client
genai_client = genai.Client()
chat_session = genai_client.chats.create(model="gemini-2.5-flash", history=[])

# Create the RAG chain
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

# doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
# rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=doc_chain)
# print("RAG chain created")
retriever,doc_chain, rag_chain = None, None, None

is_initialized = False
def initialize_medgpt_model():
    global is_initialized
    global doc_chain, rag_chain, retriever
    if is_initialized:
        print("MedGPT model already initialized.")
        return
    
    print("Initializing MedGPT model...")
    retriever = rag_save_and_retriever()
    print("Retriver initiated")
    doc_chain = create_stuff_documents_chain(llm=llm, prompt=prompt)
    rag_chain = create_retrieval_chain(retriever=retriever, combine_docs_chain=doc_chain)
    print("RAG chain created")
    is_initialized = True
    os.system("cls")

def append_to_history(entry: str):
    global history
    history += entry + "\n"

def reset_history():
    global history
    global is_initialized
    is_initialized = False
    history = ""


# # Test the chain
# response = rag_chain.invoke({"input": "What are effective treatments for viral fever?"})
# print(response["answer"])

history: str = ""
def medgpt(query):

    if not is_initialized:
        raise RuntimeError("MedGPT not initialized yet")
    
    append_to_history(f"You: {query}")
    global history, rag_chain
    if not rag_chain:
        raise RuntimeError("RAG chain not initialized yet")
    rag_output = rag_chain.invoke({"input": history})
    bot_answer = rag_output["answer"]
    append_to_history(f"MedGPT: {bot_answer}")
    cleaned_response = clean_text(rag_output["answer"])

    return cleaned_response


# os.system("cls")
# initialize_medgpt_model()
# while True:
#     user_q = input("\n👤 You: ")
#     if user_q.lower() in ("exit", "quit"):
#         break

#     # history += f"User: {user_q}\n"

#     response = medgpt(user_q)
#     # response = rag_chain.invoke({"input": user_q})
#     # answer= clean_text(response["answer"])
#     # history += f"Assistant: {answer}\n"
    
#     print(f"\n\n Assistant: {response}")
