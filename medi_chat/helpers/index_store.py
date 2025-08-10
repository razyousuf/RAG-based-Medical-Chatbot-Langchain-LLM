# src/index_store.py
import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI

from src.pdf_loader import PDFLoader
from src.huggingface_embedder import HuggingFaceEmbedder
from src.text_splitter import TextSplitter


class IndexStore:
    def __init__(self, data_path: str, index_name: str = "medi-chat"):
        load_dotenv()

        self.data_path = data_path
        self.index_name = index_name
        self.pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        self.openai_api_key = os.environ.get("OPENAI_API_KEY")

        if not self.pinecone_api_key or not self.openai_api_key:
            raise ValueError("Missing API keys. Ensure PINECONE_API_KEY and OPENAI_API_KEY are set.")

        os.environ["PINECONE_API_KEY"] = self.pinecone_api_key
        os.environ["OPENAI_API_KEY"] = self.openai_api_key

        self.pinecone_client = Pinecone(api_key=self.pinecone_api_key)

    def prepare_documents(self):
        """Loads, filters, and splits PDF documents into text chunks."""
        pdf_loader = PDFLoader(self.data_path)
        extracted_docs = pdf_loader.load()
        filtered_docs = pdf_loader.filter_minimal_docs(extracted_docs)

        splitter = TextSplitter(chunk_size=500, chunk_overlap=20)
        return splitter.split(filtered_docs)

    def get_embeddings(self):
        """Initialises HuggingFace embeddings."""
        return HuggingFaceEmbedder(model_name='sentence-transformers/all-MiniLM-L6-v2').load_embeddings()

    def create_index_if_not_exists(self, dimension: int = 384, metric: str = "cosine"):
        """Creates Pinecone index if it doesn't already exist."""
        if not self.pinecone_client.has_index(self.index_name):
            self.pinecone_client.create_index(
                name=self.index_name,
                dimension=dimension,
                metric=metric,
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )

    def store_documents(self):
        """Uploads document chunks to Pinecone."""
        text_chunks = self.prepare_documents()
        embeddings = self.get_embeddings()

        self.create_index_if_not_exists()
        return PineconeVectorStore.from_documents(
            documents=text_chunks,
            index_name=self.index_name,
            embedding=embeddings,
        )

    def get_vector_store(self):
        """Returns an existing Pinecone vector store."""
        embeddings = self.get_embeddings()
        return PineconeVectorStore.from_existing_index(
            index_name=self.index_name,
            embedding=embeddings
        )

    def retrieve_answer(self, query: str):
        """Retrieves an answer for the given query using RetrievalQA."""
        vector_store = self.get_vector_store()

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
        qa = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vector_store.as_retriever(),
            return_source_documents=True
        )

        result = qa.invoke({"query": query})
        return {
            "answer": result["result"],
            "sources": [doc.metadata for doc in result["source_documents"]]
        }


if __name__ == "__main__":
    index_store = IndexStore(data_path="data/")
    index_store.store_documents()
    print("Documents successfully stored in Pinecone.")
