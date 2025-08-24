import os
import sys
from flask import Flask, render_template, request
from dotenv import load_dotenv

from medi_chat.src.rag.prompt import PromptRepository
from medi_chat.src.rag.embeddings_loader import EmbeddingLoader

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
            # 1) embeddings
            self.embeddings = EmbeddingLoader.load_embeddings()

            # 2) vector store / retriever
            self.docsearch = PineconeVectorStore.from_existing_index(
                index_name=self.index_name,
                embedding=self.embeddings,
            )
            self.retriever = self.docsearch.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 3},
            )

            # 3) LLM
            self.llm = ChatOpenAI(model="gpt-4o")

            # 4) PROMPT — add {context} so stuff chain can inject retrieved docs
            system_msg = PromptRepository.get_system_prompt() + "\n\nContext:\n{context}"
            self.prompt = ChatPromptTemplate.from_messages(
                [("system", system_msg), ("human", "{input}")]
            )

            # 5) chains
            self.qa_chain = create_stuff_documents_chain(self.llm, self.prompt)
            self.rag_chain = create_retrieval_chain(self.retriever, self.qa_chain)
        except Exception as e:
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
