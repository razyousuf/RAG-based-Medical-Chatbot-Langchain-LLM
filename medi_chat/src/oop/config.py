
from dataclasses import dataclass

@dataclass(frozen=True)
class AppConfig:
    data_dir: str = "data/"
    index_name: str = "medi-chat"
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    chunk_size: int = 500
    chunk_overlap: int = 20
    pinecone_cloud: str = "aws"
    pinecone_region: str = "us-east-1"
    pinecone_metric: str = "cosine"
    openai_model: str = "gpt-4o"
    # Dimensions for the chosen embedding model.
    # all-MiniLM-L6-v2 returns 384-d vectors.
    embed_dimension: int = 384
