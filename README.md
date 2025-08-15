# AU15-pdf_qa_chatbot
Gen Ai 

Step 3 – How It Works

Upload PDF → Saved in data/.

Load & Split → Uses RecursiveCharacterTextSplitter for overlapping chunks.

Embed with Ollama → Converts chunks into vectors using OllamaEmbeddings and stores them in FAISS.

RAG Retrieval → Searches vector DB for relevant chunks.

LLM Query → Sends retrieved context + question to Ollama LLM.

Display Answer → Shows conversation in Streamlit.

Step 4 – Running the App

Start Ollama and pull a model (example with llama2):

ollama pull llama2


Run the app:

streamlit run app.py


Open in your browser and start chatting with your PDF.
