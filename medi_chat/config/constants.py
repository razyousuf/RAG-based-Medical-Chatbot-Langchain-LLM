# medi_chat/config/constants.py

import os
from dotenv import load_dotenv

load_dotenv()

# Paths
DATA_PATH = os.path.join("medi_chat", "data")  # Folder containing PDFs

# Embedding and Pinecone
INDEX_NAME = "medi-chat"
EMBED_DIMENSION = 384
EMBED_METRIC = "cosine"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PINECONE_CLOUD = "aws"
PINECONE_REGION = "us-east-1"

# API Keys (loaded from env in main scripts)
PINECONE_API_KEY_ENV = "PINECONE_API_KEY"
OPENAI_API_KEY_ENV = "OPENAI_API_KEY"

# Retrieval
RETRIEVAL_K = 3

# Flask settings
FLASK_HOST = "0.0.0.0"
FLASK_PORT = 8080
FLASK_DEBUG = True