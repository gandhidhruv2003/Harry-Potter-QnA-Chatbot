import os
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
load_dotenv()

def extract_chunks(pdf_path, chunk_dir="data"):
    os.makedirs(chunk_dir, exist_ok=True)
    chunk_files = sorted([f for f in os.listdir(chunk_dir) if f.startswith("chunk_") and f.endswith(".txt")])

    # If chunks already exist in files, load and return them
    if chunk_files:
        chunks = []
        for filename in chunk_files:
            with open(os.path.join(chunk_dir, filename), "r", encoding="utf-8") as f:
                chunks.append(f.read())
        return chunks

    # Otherwise extract and split
    reader = PdfReader(pdf_path)
    raw_text = ""
    for page in reader.pages:
        raw_text += page.extract_text() + "\n"

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "],
    )
    chunks = text_splitter.split_text(raw_text)

    # Save each chunk as chunk_1.txt, chunk_2.txt, etc.
    for i, chunk in enumerate(chunks, start=1):
        with open(os.path.join(chunk_dir, f"chunk_{i}.txt"), "w", encoding="utf-8") as f:
            f.write(chunk)

    return chunks

def load_llm():
    return ChatGroq(model="llama-3.3-70b-versatile", temperature=0.8, api_key=os.getenv("GROQ_API_KEY"))

def load_embeddings():
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

def get_vectorstore(chunks, embeddings, persist_dir="data"):
    os.makedirs(persist_dir, exist_ok=True)
    faiss_path = os.path.join(persist_dir, "faiss_index")

    if os.path.exists(faiss_path + ".pkl") and os.path.exists(faiss_path + ".index"):
        import pickle
        with open(faiss_path + ".pkl", "rb") as f:
            index = pickle.load(f)
        return FAISS.load_local(faiss_path, embeddings, index=index)

    vectorstore = FAISS.from_texts(chunks, embedding=embeddings)
    vectorstore.save_local(faiss_path)
    return vectorstore

# Pipeline
chunks = extract_chunks("harrypotter.pdf")
llm = load_llm()
embeddings = load_embeddings()
vectorstore = get_vectorstore(chunks, embeddings)
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    memory=memory
)
