"""
IndexStore: manage Pinecone index lifecycle and vector-store operations.

Responsibilities:
- create index if missing
- prepare documents (load -> filter -> split)
- upsert chunks into Pinecone (via langchain_pinecone)
- return a LangChain retriever for querying
"""

import os
import time
from typing import List, Optional

from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document

from src.pdf_loader import PDFLoader
from src.text_splitter import TextSplitter
from src.huggingface_embedder import HuggingFaceEmbedder
from utils.logger import logger
from utils.exception import AppException

from config.constants import (
    DATA_PATH, INDEX_NAME, PINECONE_DIMENSION,
    PINECONE_METRIC, PINECONE_CLOUD, PINECONE_REGION
)

load_dotenv()


class IndexStore:
    def __init__(self):
        load_dotenv()

        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self.openai_api_key = os.getenv("OPENAI_API_KEY")

        os.environ["PINECONE_API_KEY"] = self.pinecone_api_key
        os.environ["OPENAI_API_KEY"] = self.openai_api_key

        self.pinecone_client = Pinecone(api_key=self.pinecone_api_key)
        self.index_name = INDEX_NAME

    # ---------- Index lifecycle ----------
    def index_exists(self) -> bool:
        try:
            existing = self.pinecone_client.list_indexes()
            return self.index_name in existing
        except Exception as e:
            logger.exception("Error checking existing indexes")
            raise AppException(e, __import__("sys")) from e

    def create_index_if_not_exists(self, wait_for_ready: bool = True, timeout: int = 30) -> None:
        try:
            if not self.pinecone_client.has_index(INDEX_NAME):
                self.pinecone_client.create_index(
                    name=INDEX_NAME,
                    dimension=PINECONE_DIMENSION,
                    metric=PINECONE_METRIC,
                    spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
                )
                # optional wait until ready
                if wait_for_ready:
                    logger.info("Waiting for index to be ready...")
                    start = time.time()
                    while time.time() - start < timeout:
                        desc = self.pinecone_client.describe_index(self.index_name)
                        status = desc.status.get("ready", False) if desc and hasattr(desc, "status") else True
                        if status:
                            logger.info("Index ready.")
                            break
                        time.sleep(1)
                    else:
                        logger.warning("Index creation timed out; continue and try later.")
            else:
                logger.info(f"Pinecone index '{self.index_name}' already exists.")
        except Exception as e:
            logger.exception("Failed to create or verify Pinecone index")
            raise AppException(e, __import__("sys")) from e

    # ---------- Data preparation ----------
    def prepare_documents(self) -> List[Document]:
        """
        Load PDFs, filter boilerplate, split into chunks.
        Uses external modular components (PDFLoader, TextSplitter).
        """
        try:
            pdf_loader = PDFLoader(data_path=DATA_PATH)
            raw_docs = pdf_loader.load_documents()
            logger.info(f"Loaded {len(raw_docs)} raw documents from {DATA_PATH}")

            minimal_docs = pdf_loader.filter_minimal_docs(raw_docs)
            logger.info(f"{len(minimal_docs)} documents after minimal filtering")

            splitter = TextSplitter(chunk_size=500, chunk_overlap=20)
            chunks = splitter.split_documents(minimal_docs)
            logger.info(f"Split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.exception("Error preparing documents")
            raise AppException(e, __import__("sys")) from e

    # ---------- Embeddings ----------
    def get_embeddings(self):
        try:
            embedder = HuggingFaceEmbedder()
            embeddings = embedder.get_embeddings()
            logger.info("Loaded HuggingFace embeddings")
            return embeddings
        except Exception as e:
            logger.exception("Error loading embeddings")
            raise AppException(e, __import__("sys")) from e

    # ---------- Upsert (store) ----------
    def upsert_documents(self, docs: Optional[List[Document]] = None) -> PineconeVectorStore:
        """
        Prepare docs if not provided, then create index (if needed) and upsert to Pinecone
        Returns: PineconeVectorStore object
        """
        try:
            if docs is None:
                docs = self.prepare_documents()

            embeddings = self.get_embeddings()
            self.create_index_if_not_exists()

            logger.info(f"Upserting {len(docs)} documents into Pinecone index '{self.index_name}'")
            vector_store = PineconeVectorStore.from_documents(
                documents=docs,
                index_name=self.index_name,
                embedding=embeddings,
            )
            logger.info("Upsert complete.")
            return vector_store
        except Exception as e:
            logger.exception("Error upserting documents")
            raise AppException(e, __import__("sys")) from e

    # ---------- Access existing store ----------
    def get_vector_store(self) -> PineconeVectorStore:
        try:
            embeddings = self.get_embeddings()
            vector_store = PineconeVectorStore.from_existing_index(index_name=self.index_name, embedding=embeddings)
            logger.info("Connected to existing Pinecone index via PineconeVectorStore.")
            return vector_store
        except Exception as e:
            logger.exception("Error connecting to existing vector store")
            raise AppException(e, __import__("sys")) from e

    # ---------- Convenience retrieval helper ----------
    def get_retriever(self, k: int = 3, search_type: str = "similarity"):
        """
        Return a LangChain retriever built from the vector store.
        """
        try:
            vs = self.get_vector_store()
            retriever = vs.as_retriever(search_type=search_type, search_kwargs={"k": k})
            logger.info(f"Built retriever (k={k}, type={search_type})")
            return retriever
        except Exception as e:
            logger.exception("Error building retriever")
            raise AppException(e, __import__("sys")) from e
