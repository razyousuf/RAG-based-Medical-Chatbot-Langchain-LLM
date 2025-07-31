from langchain.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.schema import Document
from typing import List
from utils.logger import logger
from utils.exception import AppException
import sys

class PDFLoader:
    def __init__(self, data_path: str):
        self.data_path = data_path

    def load_documents(self) -> List[Document]:
        try:
            loader = DirectoryLoader(self.data_path, glob="*.pdf", loader_cls=PyPDFLoader)
            documents = loader.load()
            logger.info(f"Loaded {len(documents)} documents from {self.data_path}")
            return documents
        except Exception as e:
            logger.error("Error occurred while loading documents")
            raise AppException(e, sys)

    @staticmethod
    def filter_minimal_docs(docs: List[Document]) -> List[Document]:
        try:
            minimal_docs = [
                Document(page_content=doc.page_content, metadata={"source": doc.metadata.get("source")})
                for doc in docs if doc.page_content.strip()
            ]
            logger.info(f"Filtered {len(minimal_docs)} minimal documents")
            return minimal_docs
        except Exception as e:
            logger.error("Error occurred while filtering documents")
            raise AppException(e, sys)