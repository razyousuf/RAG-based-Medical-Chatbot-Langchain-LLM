import os
from typing import List

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_community.embeddings import HuggingFaceEmbeddings

from logger import logger
from exception import AppException
import sys


class DocumentProcessor:
    """
    Handles loading, filtering, and splitting PDF documents.
    Keeps original functionality intact but in OOP style.
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir

    def load_pdfs(self) -> List[Document]:
        try:
            logger.info(f"Loading PDFs from directory: {self.data_dir}")
            loader = PyPDFDirectoryLoader(self.data_dir)
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} PDF documents")
            return docs
        except Exception as e:
            logger.exception("Failed to load PDFs")
            raise AppException(e, sys)

    def filter_docs(self, docs: List[Document]) -> List[Document]:
        try:
            logger.info("Filtering documents to minimal set")
            # original code returned docs as-is, but kept as function
            return docs
        except Exception as e:
            logger.exception("Failed during document filtering")
            raise AppException(e, sys)

    def split_docs(self, docs: List[Document]) -> List[Document]:
        try:
            logger.info("Splitting documents into smaller chunks")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len,
            )
            chunks = text_splitter.split_documents(docs)
            logger.info(f"Split into {len(chunks)} text chunks")
            return chunks
        except Exception as e:
            logger.exception("Failed during text splitting")
            raise AppException(e, sys)


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
