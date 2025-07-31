from langchain.embeddings import HuggingFaceEmbeddings
from utils.logger import logger
from utils.exception import AppException
import sys

class HuggingFaceEmbedder:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        self.model_name = model_name
        try:
            self.embeddings = HuggingFaceEmbeddings(model_name=self.model_name)
            logger.info(f"Loaded HuggingFace embeddings model: {self.model_name}")
        except Exception as e:
            logger.error("Error occurred while loading HuggingFace embeddings")
            raise AppException(e, sys)

    def get_embeddings(self):
        return self.embeddings