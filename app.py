
import os
import sys
import tempfile
import streamlit as st
from dotenv import load_dotenv
from medi_chat.src.oop.config import AppConfig
from medi_chat.src.oop.data_loader import PDFDataLoader
from medi_chat.src.oop.processing import DocumentProcessor
from medi_chat.src.oop.embeddings import EmbeddingsFactory
from medi_chat.src.oop.indexer import PineconeIndexer
from medi_chat.src.oop.rag import RAGPipeline
from medi_chat.src.oop.prompts import SYSTEM_PROMPT
from medi_chat.src.oop.logger import logger
from medi_chat.src.oop.exception import AppException

def main():
    load_dotenv()
    cfg = AppConfig()

    st.set_page_config(page_title="Medical Chatbot", layout="centered")
    st.title("🩺 Medical Chatbot")

    # Sidebar Pinecone index viewer
    with st.sidebar:
        st.header("Pinecone Indexes")
        try:
            pinecone_api = os.environ.get("PINECONE_API_KEY")
            if pinecone_api:
                indexer_for_list = PineconeIndexer(api_key=pinecone_api)
                st.write(indexer_for_list.pc.list_indexes())
            else:
                st.warning("Set PINECONE_API_KEY to view indexes")
        except Exception as e:
            st.error(str(e))

    # File uploader
    uploaded_files = st.file_uploader("Drag & drop PDF(s) here", type="pdf", accept_multiple_files=True)

    if uploaded_files and st.button("📚 Build/Update Index from Uploaded PDFs"):
        with tempfile.TemporaryDirectory() as tmpdir:
            for file in uploaded_files:
                filepath = os.path.join(tmpdir, file.name)
                with open(filepath, "wb") as f:
                    f.write(file.getbuffer())

            try:
                loader = PDFDataLoader()
                docs = loader.load(tmpdir)

                processor = DocumentProcessor(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
                minimal_docs = processor.filter_to_minimal_docs(docs)
                chunks = processor.split(minimal_docs)

                embeddings = EmbeddingsFactory.huggingface(cfg.embed_model)
                pinecone_api = os.environ.get("PINECONE_API_KEY")
                indexer = PineconeIndexer(api_key=pinecone_api, cloud=cfg.pinecone_cloud, region=cfg.pinecone_region, metric=cfg.pinecone_metric)
                indexer.ensure_index(cfg.index_name, dimension=cfg.embed_dimension)
                indexer.index_from_documents(cfg.index_name, chunks, embeddings)

                st.success(f"Indexed {len(chunks)} chunks into '{cfg.index_name}'")
            except Exception as e:
                st.error(str(AppException(e, sys)))
                logger.error("Index build failed", exc_info=True)

    # Chat interface
    st.header("💬 Chat with your Medical Assistant")
    user_input = st.text_input("Ask a question")
    if st.button("Send") and user_input:
        try:
            embeddings = EmbeddingsFactory.huggingface(cfg.embed_model)
            indexer = PineconeIndexer(cloud=cfg.pinecone_cloud, region=cfg.pinecone_region, metric=cfg.pinecone_metric)
            vectorstore = indexer.index_from_existing(cfg.index_name, embeddings)

            rag = RAGPipeline(vectorstore, openai_model=cfg.openai_model, system_prompt=SYSTEM_PROMPT, k=3)
            answer = rag.answer(user_input)

            st.markdown(f"**Answer:** {answer}")
        except Exception as e:
            st.error(str(AppException(e, sys)))
            logger.error("Chat query failed", exc_info=True)

if __name__ == "__main__":
    main()
