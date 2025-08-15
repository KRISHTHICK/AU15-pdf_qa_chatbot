import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OllamaEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.llms import Ollama
import os

# ----------------- SETTINGS -----------------
DATA_DIR = "data"
VECTOR_DB_DIR = "vectordb"

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(VECTOR_DB_DIR, exist_ok=True)

# ----------------- STREAMLIT UI -----------------
st.set_page_config(page_title="📄 PDF Q&A with Ollama", layout="wide")
st.title("📄 Chat with Your PDF (Local LLM + RAG)")

uploaded_file = st.file_uploader("Upload a PDF file", type=["pdf"])

if uploaded_file:
    pdf_path = os.path.join(DATA_DIR, uploaded_file.name)
    with open(pdf_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.success(f"✅ File saved: {uploaded_file.name}")

    # ----------------- LOAD & SPLIT -----------------
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    # ----------------- EMBEDDINGS -----------------
    embeddings = OllamaEmbeddings(model="llama2")

    # ----------------- VECTOR STORE -----------------
    vectorstore = FAISS.from_documents(split_docs, embeddings)
    retriever = vectorstore.as_retriever()

    # ----------------- LLM -----------------
    llm = Ollama(model="llama2")

    # ----------------- RAG CHAIN -----------------
    qa_chain = ConversationalRetrievalChain.from_llm(llm, retriever)

    # ----------------- CHAT SESSION -----------------
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    user_query = st.text_input("Ask a question about the PDF:")

    if st.button("Get Answer"):
        if user_query:
            result = qa_chain({"question": user_query, "chat_history": st.session_state.chat_history})
            answer = result["answer"]

            st.session_state.chat_history.append((user_query, answer))
            st.markdown(f"**You:** {user_query}")
            st.markdown(f"**Bot:** {answer}")
