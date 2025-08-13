
import os
import pytest

from src.oop.config import AppConfig
from src.oop.embeddings import EmbeddingsFactory
from src.oop.indexer import PineconeIndexer

@pytest.mark.skipif(
    not (os.environ.get("PINECONE_API_KEY") and os.environ.get("OPENAI_API_KEY")),
    reason="Requires PINECONE_API_KEY and OPENAI_API_KEY for live integration."
)
def test_existing_index_load():
    cfg = AppConfig()
    embeddings = EmbeddingsFactory.huggingface(cfg.embed_model)
    indexer = PineconeIndexer()
    vs = indexer.index_from_existing(cfg.index_name, embeddings)
    retriever = vs.as_retriever(search_kwargs={"k":1})
    # Just ensure retriever can be created
    assert retriever is not None
