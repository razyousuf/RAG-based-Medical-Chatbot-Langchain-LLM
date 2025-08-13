
from langchain.embeddings import HuggingFaceEmbeddings
from src.oop.exception import AppException
from src.oop.logger import logger
import sys

class EmbeddingsFactory:
    """Creating embeddings implementations."""

    @staticmethod
    def huggingface(model_name: str) -> HuggingFaceEmbeddings:
        try:
            logger.info(f"Loading HuggingFace embeddings: {model_name}")
            emb = HuggingFaceEmbeddings(model_name=model_name)
            return emb
        except Exception as e:
            logger.error("Failed to load HuggingFace embeddings", exc_info=True)
            raise AppException(e, sys)
