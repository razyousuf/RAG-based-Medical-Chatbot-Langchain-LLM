from dotenv import load_dotenv
import os
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

from src.rag.helper import DocumentProcessor, EmbeddingLoader

from src.utils.logger import logger
from src.utils.exception import AppException


class MedicalIndexer:
    """
    Small, focused class that:
      1) Loads and minimally filters PDFs
      2) Splits documents into chunks
      3) Ensures a Pinecone index exists
      4) Uploads embeddings to Pinecone

    Usage:
        MedicalIndexer(data_dir="data/", index_name="medi-chat").run()
    """

    def __init__(self, data_dir: str, index_name: str = "medi-chat"):
        load_dotenv()

        self.data_dir = data_dir
        self.index_name = index_name

        # Read API keys exactly as before
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")

        # Keep environment propagation identical
        os.environ["PINECONE_API_KEY"] = self.pinecone_api_key or ""
        os.environ["OPENAI_API_KEY"] = self.openai_api_key or ""

        # Lazy-init objects
        self._pc = None
        self._embeddings = None

    # ---------- Internal helpers (single-responsibility, easy to read) ----------
    def _init_clients(self):
        try:
            logger.info("Initializing Pinecone and Embeddings clients")
            self._pc = Pinecone(api_key=self.pinecone_api_key)
            self._embeddings = EmbeddingLoader.load_embeddings()
            logger.info("Clients initialized successfully")
        except Exception as e:
            logger.exception("Failed to initialize clients")
            raise AppException(e, sys)

    def _ensure_index(self):
        try:
            logger.info(f"Ensuring Pinecone index '{self.index_name}' exists")
            # same dimension/metric/region as before
            if not self._pc.has_index(self.index_name):
                self._pc.create_index(
                    name=self.index_name,
                    dimension=384,
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1"),
                )
            logger.info(f"Index '{self.index_name}' is ready")
        except Exception as e:
            logger.exception("Failed to ensure/create index")
            raise AppException(e, sys)

    def _prepare_documents(self):
        try:
            logger.info(f"Loading PDFs from: {self.data_dir}")
            extracted = DocumentProcessor.load_pdfs(data=self.data_dir)
            logger.info(f"Loaded {len(extracted)} documents")

            minimal = DocumentProcessor.filter_docs(extracted)
            chunks = DocumentProcessor.split_docs(minimal)
            logger.info(f"Split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.exception("Failed during document preparation")
            raise AppException(e, sys)

    def _upsert_documents(self, chunks):
        try:
            logger.info("Upserting document embeddings to Pinecone")
            # identical behavior to your working version
            PineconeVectorStore.from_documents(
                documents=chunks,
                index_name=self.index_name,
                embedding=self._embeddings,
            )
            logger.info("Upsert complete")
        except Exception as e:
            logger.exception("Failed to upsert documents into Pinecone")
            raise AppException(e, sys)

    # ---------- Public API ----------
    def run(self):
        self._init_clients()
        self._ensure_index()
        chunks = self._prepare_documents()
        self._upsert_documents(chunks)


if __name__ == "__main__":
    import sys
    try:
        MedicalIndexer(data_dir="data/", index_name="medical-chatbot").run()
    except Exception as e:
        # Ensure any unexpected error is formatted nicely for logs/console
        print(AppException(e, sys))
        raise
