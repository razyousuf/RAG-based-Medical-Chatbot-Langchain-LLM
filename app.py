from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os

from medi_chat.helpers.index_store import IndexStore
from medi_chat.utils.logger import logger
from medi_chat.utils.exception import AppException

from medi_chat.config.constants import (
    DATA_PATH, INDEX_NAME, EMBED_DIMENSION, EMBED_MODEL
)

# LangChain imports (adapt if you use different names/versions)
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

load_dotenv()

app = Flask(__name__, template_folder="templates", static_folder="static")

# Configure index + LLM parameters via env or defaults
DATA_PATH = DATA_PATH
INDEX_NAME = INDEX_NAME
EMBED_DIM = EMBED_DIMENSION
LLM_MODEL = EMBED_MODEL

# Initialize IndexStore once (will connect to Pinecone client)
try:
    index_store = IndexStore(data_path=DATA_PATH, index_name=INDEX_NAME, embedding_dim=EMBED_DIM)
    # Ensure index exists (create if needed). You can comment out create if you only read.
    index_store.create_index_if_not_exists()
    logger.info("IndexStore ready.")
except Exception as e:
    logger.exception("Failed to initialize IndexStore at startup")
    index_store = None  # app will still run but endpoints should handle this

# Initialize LLM (for retrieval QA)
def get_llm():
    # wrap model. Replace ChatOpenAI with your Gemini wrapper if you prefer.
    return ChatOpenAI(model=LLM_MODEL, temperature=0)

@app.route("/")
def home():
    return render_template("chat.html")  # your front-end template

@app.route("/store", methods=["POST"])
def store():
    """Trigger document processing & upsert to Pinecone."""
    try:
        if not index_store:
            raise AppException("IndexStore not initialized", __import__("sys"))
        # optionally accept a body with {"data_path": "..."} to override
        payload = request.get_json(silent=True) or {}
        data_path = payload.get("data_path", None)
        docs = None
        if data_path:
            docs = index_store.prepare_documents()
        vector_store = index_store.upsert_documents(docs=docs)
        return jsonify({"message": "Documents stored successfully", "index": INDEX_NAME}), 200
    except AppException as ae:
        logger.error(str(ae))
        return jsonify({"error": str(ae)}), 500
    except Exception as e:
        logger.exception("Unexpected error during store")
        return jsonify({"error": str(e)}), 500

@app.route("/query", methods=["POST"])
def query():
    """
    Expects JSON: {"query": "your question", "k": 3}
    Returns JSON with answer and sources (if available).
    """
    try:
        if not index_store:
            raise AppException("IndexStore not initialized", __import__("sys"))
        payload = request.get_json()
        if not payload or "query" not in payload:
            return jsonify({"error": "Missing 'query' in request body"}), 400

        query_text = payload["query"]
        k = int(payload.get("k", 3))

        retriever = index_store.get_retriever(k=k)
        llm = get_llm()

        # Build a simple RetrievalQA chain. You can swap for a custom chain/prompt.
        qa_chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)

        result = qa_chain.run(query_text)
        # If `run` returns a string (no sources), wrap accordingly. Some versions of LangChain return dicts.
        response = {"answer": result}
        return jsonify(response), 200

    except AppException as ae:
        logger.error(str(ae))
        return jsonify({"error": str(ae)}), 500
    except Exception as e:
        logger.exception("Unexpected error during query")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    # Use production WSGI (gunicorn) in production. Flask dev server for now.
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", 8080)), debug=os.getenv("FLASK_DEBUG", "True") == "True")
