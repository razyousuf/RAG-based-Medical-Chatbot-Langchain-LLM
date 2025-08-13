
from typing import List
from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.schema import Document
from src.oop.exception import AppException
from src.oop.logger import logger
import sys

class PDFDataLoader:
    """Loads PDF documents from a directory using LangChain loaders."""

    def __init__(self, glob_pattern: str = "*.pdf"):
        self.glob_pattern = glob_pattern

    def load(self, data_dir: str) -> List[Document]:
        try:
            logger.info(f"Loading PDFs from: {data_dir} (pattern: {self.glob_pattern})")
            loader = DirectoryLoader(data_dir, glob=self.glob_pattern, loader_cls=PyPDFLoader)
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} documents")
            return docs
        except Exception as e:
            logger.error("Failed to load PDFs", exc_info=True)
            raise AppException(e, sys)
