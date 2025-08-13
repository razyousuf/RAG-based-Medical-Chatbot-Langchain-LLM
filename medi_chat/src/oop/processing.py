
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from medi_chat.src.oop.exception import AppException
from medi_chat.src.oop.logger import logger
import sys

class DocumentProcessor:
    """Filters and splits documents into chunks."""

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 20):
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    @staticmethod
    def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
        """Keep only page_content and 'source' in metadata."""
        try:
            minimal_docs: List[Document] = []
            for doc in docs:
                src = doc.metadata.get("source")
                minimal_docs.append(Document(page_content=doc.page_content, metadata={"source": src}))
            logger.debug(f"Filtered to minimal docs: {len(minimal_docs)}")
            return minimal_docs
        except Exception as e:
            logger.error("Failed during filter_to_minimal_docs", exc_info=True)
            raise AppException(e, sys)

    def split(self, docs: List[Document]) -> List[Document]:
        try:
            chunks = self.splitter.split_documents(docs)
            logger.info(f"Split into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error("Failed during text split", exc_info=True)
            raise AppException(e, sys)
