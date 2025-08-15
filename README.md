# AU15-pdf_qa_chatbot
Gen Ai 

Step 3 – How It Works

Upload PDF → Saved in data/.

Load & Split → Uses RecursiveCharacterTextSplitter for overlapping chunks.

Embed with Ollama → Converts chunks into vectors using OllamaEmbeddings and stores them in FAISS.

RAG Retrieval → Searches vector DB for relevant chunks.

LLM Query → Sends retrieved context + question to Ollama LLM.

Display Answer → Shows conversation in Streamlit.
