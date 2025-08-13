
import os
from typing import List
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from medi_chat.src.oop.exception import AppException
from medi_chat.src.oop.logger import logger
import sys

class PineconeIndexer:
    """Handles Pinecone index lifecycle and vector store creation."""

    def __init__(self, api_key: str | None = None, cloud: str = "aws", region: str = "us-east-1", metric: str = "cosine"):
        try:
            load_dotenv()
            api_key = api_key or os.environ.get("PINECONE_API_KEY")
            if not api_key:
                raise ValueError("PINECONE_API_KEY is not set. Provide it explicitly or via environment.")
            self.pc = Pinecone(api_key=api_key)
            self.cloud = cloud
            self.region = region
            self.metric = metric
            logger.debug(f"Initialized PineconeIndexer (cloud={cloud}, region={region}, metric={metric})")
        except Exception as e:
            logger.error("Failed to initialize PineconeIndexer", exc_info=True)
            raise AppException(e, sys)

    def ensure_index(self, name: str, dimension: int) -> None:
        try:
            if not self.pc.has_index(name):
                logger.info(f"Creating Pinecone index '{name}' (dim={dimension}, metric={self.metric})")
                self.pc.create_index(
                    name=name,
                    dimension=dimension,
                    metric=self.metric,
                    spec=ServerlessSpec(cloud=self.cloud, region=self.region),
                )
            else:
                logger.info(f"Index '{name}' already exists")
        except Exception as e:
            logger.error("Failed to ensure index", exc_info=True)
            raise AppException(e, sys)

    def index_from_documents(self, name: str, documents: List[Document], embedding) -> PineconeVectorStore:
        try:
            logger.info(f"Upserting {len(documents)} documents to index '{name}'")
            return PineconeVectorStore.from_documents(documents=documents, index_name=name, embedding=embedding)
        except Exception as e:
            logger.error("Failed to create vector store from documents", exc_info=True)
            raise AppException(e, sys)

    def index_from_existing(self, name: str, embedding) -> PineconeVectorStore:
        try:
            logger.info(f"Connecting to existing index '{name}'")
            return PineconeVectorStore.from_existing_index(index_name=name, embedding=embedding)
        except Exception as e:
            logger.error("Failed to connect to existing index", exc_info=True)
            raise AppException(e, sys)
