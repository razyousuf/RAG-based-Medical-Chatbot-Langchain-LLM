
import os
from dotenv import load_dotenv
from src.oop.config import AppConfig
from src.oop.data_loader import PDFDataLoader
from src.oop.processing import DocumentProcessor
from src.oop.embeddings import EmbeddingsFactory
from src.oop.indexer import PineconeIndexer

def main():
    load_dotenv()
    cfg = AppConfig()

    # Load documents
    loader = PDFDataLoader(glob_pattern="*.pdf")
    docs = loader.load(cfg.data_dir)

    # Filter and split
    processor = DocumentProcessor(chunk_size=cfg.chunk_size, chunk_overlap=cfg.chunk_overlap)
    minimal = processor.filter_to_minimal_docs(docs)
    chunks = processor.split(minimal)

    # Embeddings
    embeddings = EmbeddingsFactory.huggingface(cfg.embed_model)

    # Pinecone index & upsert
    pinecone_api = os.environ.get("PINECONE_API_KEY")
    indexer = PineconeIndexer(api_key=pinecone_api, cloud=cfg.pinecone_cloud, region=cfg.pinecone_region, metric=cfg.pinecone_metric)
    indexer.ensure_index(cfg.index_name, dimension=cfg.embed_dimension)
    _ = indexer.index_from_documents(cfg.index_name, chunks, embeddings)
    print(f"Indexed {len(chunks)} chunks into Pinecone index '{cfg.index_name}'.")

if __name__ == "__main__":
    main()
