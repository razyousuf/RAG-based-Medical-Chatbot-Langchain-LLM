
import pytest
from langchain.schema import Document
from src.oop.processing import DocumentProcessor

def test_filter_to_minimal_docs():
    docs = [
        Document(page_content="A", metadata={"source":"s1", "foo":"bar"}),
        Document(page_content="B", metadata={"foo":"no_source"}),
    ]
    minimal = DocumentProcessor.filter_to_minimal_docs(docs)
    assert len(minimal)==2
    assert minimal[0].metadata=={"source":"s1"}
    assert minimal[1].metadata=={"source":None}
    assert minimal[0].page_content=="A"

def test_split_roundtrip():
    docs = [Document(page_content="a"*1200, metadata={"source":"s"})]
    proc = DocumentProcessor(chunk_size=500, chunk_overlap=20)
    chunks = proc.split(docs)
    # Expect >= 3 chunks for 1200 chars at chunk_size 500 with overlap
    assert len(chunks) >= 3
    assert all("source" in c.metadata for c in chunks)
