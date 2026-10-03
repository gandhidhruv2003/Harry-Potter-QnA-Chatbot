import os
import re
import json
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.documents import Document
from dotenv import load_dotenv
load_dotenv()

BOOK_START_CHAPTERS = {
    "THE BOY WHO LIVED":
        "Harry Potter and the Philosopher's Stone",

    "THE WORST BIRTHDAY":
        "Harry Potter and the Chamber of Secrets",

    "OWL POST":
        "Harry Potter and the Prisoner of Azkaban",

    "THE RIDDLE HOUSE":
        "Harry Potter and the Goblet of Fire",

    "DUDLEY DEMENTED":
        "Harry Potter and the Order of the Phoenix",

    "THE OTHER MINISTER":
        "Harry Potter and the Half-Blood Prince",

    "THE DARK LORD ASCENDING":
        "Harry Potter and the Deathly Hallows"
}

def detect_chapter(text):
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    for i, line in enumerate(lines):
        if re.match(r"^CHAPTER\s+([A-Z0-9'\-]+)$", line, re.IGNORECASE):
            chapter_number = line
            chapter_title = None

            if i + 1 < len(lines):
                chapter_title = lines[i + 1].strip()
            return chapter_number, chapter_title
    return None, None

def extract_chunks(pdf_path):
    reader = PdfReader(pdf_path)

    documents = []
    
    current_book = None
    current_chapter_number = None
    current_chapter_title = None

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text()

        if not text:
            continue               

        chapter_number, chapter_title = detect_chapter(text)

        if chapter_number:
            current_chapter_number = chapter_number
            current_chapter_title = chapter_title

            # Detect which Harry Potter book we are currently in
            if chapter_title:
                normalized_title = " ".join(
                    chapter_title.upper().split()
                )

                if normalized_title in BOOK_START_CHAPTERS:
                    current_book = BOOK_START_CHAPTERS[
                        normalized_title
                    ]

        document = Document(
            page_content=text,
            metadata={
                "book_name": current_book,
                "chapter_number": current_chapter_number,
                "chapter_title": current_chapter_title,
                "page": page_number
            }
        )
        
        documents.append(document)

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=500,
        separators=["\n\n", "\n", ".", "!", "?", ",", " "],
    )
    chunks = text_splitter.split_documents(documents)
    
    for chunk_id, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = chunk_id

    return chunks

def save_chunks(chunks, path="data/chunks.jsonl"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for chunk in chunks:
            record = {
                "page_content": chunk.page_content,
                "metadata": chunk.metadata
            }

            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )


def load_chunks(path="data/chunks.jsonl"):
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            chunks.append(
                Document(
                    page_content=record["page_content"],
                    metadata=record["metadata"]
                )
            )
    return chunks

def load_llm(llm_model):
    return ChatGroq(model=llm_model, temperature=0.2, api_key=os.getenv("GROQ_API_KEY"), max_tokens=4096)

def load_embeddings(embedding_model):
    return HuggingFaceEmbeddings(model_name=embedding_model)

def get_vectorstore(pdf_path, embeddings, persist_dir="data/faiss_index", chunks_path="data/chunks.jsonl"):
    faiss_file = os.path.join(persist_dir, "index.faiss")
    pkl_file = os.path.join(persist_dir, "index.pkl")

    if (os.path.exists(faiss_file) and os.path.exists(pkl_file) and os.path.exists(chunks_path)):
        print("Loading existing FAISS index...")
        vectorstore = FAISS.load_local(persist_dir, embeddings, allow_dangerous_deserialization=True)
        chunks = load_chunks(chunks_path)
        return vectorstore, chunks
    
    print("Creating chunks...")
    chunks = extract_chunks(pdf_path)
    print(f"Total chunks: {len(chunks)}")
    print("Creating FAISS index...")
    vectorstore = FAISS.from_documents(chunks, embedding=embeddings)
    vectorstore.save_local(persist_dir)
    save_chunks(chunks, chunks_path)
    return vectorstore, chunks

# Pipeline
llm_model = os.getenv("llm_model")
embedding_model = os.getenv("embedding_model")
llm = load_llm(llm_model)
embeddings = load_embeddings(embedding_model)
vectorstore, chunks = get_vectorstore("harrypotter.pdf", embeddings)
dense_retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 4
hybrid_retriever = EnsembleRetriever(retrievers=[bm25_retriever, dense_retriever], id_key="chunk_id")
memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", output_key="answer", return_messages=True)

qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=hybrid_retriever,
    memory=memory,
    return_source_documents=True
)
