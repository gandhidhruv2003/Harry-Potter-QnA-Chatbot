from flask import Flask, request, jsonify, render_template
from dotenv import load_dotenv
import os
from chatbot import qa_chain

load_dotenv()

## LangSmith tracking
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = os.getenv("LANGCHAIN_PROJECT", "")

app = Flask(__name__)

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
    return jsonify({"response": answer['answer']})

if __name__ == "__main__":
    print("🚀 Flask server starting...")
    app.run(debug=True)
