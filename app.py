# app.py
import os
import sys
from flask import Flask, render_template, request
from dotenv import load_dotenv

from medi_chat.src.rag.prompt import PromptRepository
from medi_chat.src.rag.helper import EmbeddingLoader

from medi_chat.src.utils.logger import logger
from medi_chat.src.utils.exception import AppException

from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate


class MedicalChatbotWebApp:
    """
    Encapsulates the RAG wiring and Flask routes,
    in a clean OOP style.
    """

    def __init__(self, index_name: str = "medi-chat"):
        load_dotenv()
        self.index_name = index_name

        # Keep env handling identical
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")
        os.environ["PINECONE_API_KEY"] = self.pinecone_api_key or ""
        os.environ["OPENAI_API_KEY"] = self.openai_api_key or ""

        # Flask
        self.app = Flask(__name__)

        # RAG pipeline objects
        self.embeddings = None
        self.docsearch = None
        self.retriever = None
        self.llm = None
        self.prompt = None
        self.qa_chain = None
        self.rag_chain = None

        # Build pipeline & routes
        self._wire_rag_pipeline()
        self._register_routes()

    # ---------- Pipeline wiring ----------
    def _wire_rag_pipeline(self):
        try:
            logger.info("Loading HuggingFace embeddings")
            self.embeddings = EmbeddingLoader.load_embeddings()

            logger.info(f"Connecting to existing Pinecone index '{self.index_name}'")
            self.docsearch = PineconeVectorStore.from_existing_index(
                index_name=self.index_name,
                embedding=self.embeddings,
            )

            self.retriever = self.docsearch.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3},  # unchanged
            )

            logger.info("Initializing ChatOpenAI and chains")
            self.llm = ChatOpenAI(model="gpt-4o")  # unchanged

            self.prompt = ChatPromptTemplate.from_messages(
                [("system", PromptRepository.get_system_prompt()), ("human", "{input}")]
            )

            self.qa_chain = create_stuff_documents_chain(self.llm, self.prompt)
            self.rag_chain = create_retrieval_chain(self.retriever, self.qa_chain)

            logger.info("RAG pipeline ready")
        except Exception as e:
            logger.exception("Failed to wire RAG pipeline")
            raise AppException(e, sys)

    # ---------- Routes ----------
    def _register_routes(self):
        app = self.app

        @app.route("/")
        def index():
            return render_template("chat.html")

        @app.route("/get", methods=["GET", "POST"])
        def chat():
            try:
                msg = request.form["msg"]
                logger.info(f"User query: {msg}")
                response = self.rag_chain.invoke({"input": msg})
                answer = response["answer"]
                logger.info(f"Model answer: {answer}")
                return str(answer)
            except Exception as e:
                logger.exception("Error during /get handling")
                raise AppException(e, sys)


# Global `app` Flask object so Docker/gunicorn still works unchanged
web = MedicalChatbotWebApp(index_name="medi-chat")
app = web.app

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8080, debug=True)
