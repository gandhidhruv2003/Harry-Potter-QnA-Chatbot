from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
import re
from chatbot import qa_chain

load_dotenv()

# LangSmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "")

app = Flask(__name__)

def get_source_snippet(text, max_sentences=2):
    text = " ".join(text.split())
    sentences = re.split(r'(?<=[.!?])\s+', text)
    sentences = [
        sentence.strip()
        for sentence in sentences
        if sentence.strip()
    ]

    return " ".join(sentences[:max_sentences])

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json()
    question = data.get("query", "").strip()
    
    if not question:
        return jsonify({"response": "Please ask something."})
    answer = qa_chain.invoke({"question": question})

    sources = []

    for doc in answer["source_documents"]:
        sources.append({
        "book_name": doc.metadata.get("book_name"),
        "chapter_number": doc.metadata.get("chapter_number"),
        "chapter_title": doc.metadata.get("chapter_title"),
        "page": doc.metadata.get("page"),
        "text": get_source_snippet(doc.page_content)
    })

    return jsonify({
        "response": answer["answer"],
        "sources": sources
    })

if __name__ == "__main__":
    print("Flask server starting...")
    app.run(debug=True)
