from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from typing import List
from utils.logger import logger
from utils.exception import AppException
import sys

class TextSplitter:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 20):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap
        )

    def split_documents(self, docs: List[Document]) -> List[Document]:
        try:
            chunks = self.splitter.split_documents(docs)
            logger.info(f"Split documents into {len(chunks)} chunks")
            return chunks
        except Exception as e:
            logger.error("Error occurred while splitting documents")
            raise AppException(e, sys)