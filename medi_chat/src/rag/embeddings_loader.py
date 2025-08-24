from langchain_community.embeddings import HuggingFaceEmbeddings  # old
# from langchain_huggingface import HuggingFaceEmbeddings


from medi_chat.src.utils.logger import logger
from medi_chat.src.utils.exception import AppException
import sys
class EmbeddingLoader:
    """
    Wraps HuggingFaceEmbeddings loader into a simple class.
    """

    @staticmethod
    def load_embeddings():
        try:
            logger.info("Downloading HuggingFace embeddings (all-MiniLM-L6-v2)")
            return HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        except Exception as e:
            logger.exception("Failed to download HuggingFace embeddings")
            raise AppException(e, sys)
