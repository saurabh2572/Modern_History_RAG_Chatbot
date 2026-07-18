import json
import logging
import os
from pathlib import Path

import yaml
from dotenv import load_dotenv
from huggingface_hub import login
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------------------------------------------
# Logging
# ---------------------------------------------------
Path("logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename="logs/all_errors.log",
    level=logging.ERROR,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

logger = logging.getLogger("all_errors")

# ---------------------------------------------------
# Load Environment Variables
# ---------------------------------------------------
load_dotenv()

# ---------------------------------------------------
# Load Configuration
# ---------------------------------------------------
try:
    with open("./config.yaml", "r", encoding="utf-8") as stream:
        CONFIG = yaml.safe_load(stream)

except yaml.YAMLError as e:
    logger.error(f"Error loading config.yaml: {e}", exc_info=True)
    raise

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
INPUT_FILE = Path(CONFIG["PATH"]["CHUNKED_DATA"])
OUTPUT_FILE = Path(CONFIG["PATH"]["VECTOR_EMBEDDING"])

OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------
# Hugging Face Login
# ---------------------------------------------------
try:
    hf_token = os.getenv("HF_TOKEN")

    if not hf_token:
        raise ValueError("HF_TOKEN not found in .env file.")

    login(token=hf_token)

    print("Successfully logged into Hugging Face.")

except Exception as e:
    logger.error(f"Error logging into Hugging Face: {e}", exc_info=True)
    raise

# ---------------------------------------------------
# Load Embedding Model
# ---------------------------------------------------
try:
    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={
            "device": "cpu",
            "trust_remote_code": True,
        },
        encode_kwargs={
            "normalize_embeddings": True,
        },
    )

    print("Embedding model loaded successfully.")

except Exception as e:
    logger.error(f"Error loading embedding model: {e}", exc_info=True)
    raise

# ---------------------------------------------------
# Read JSON Chunk File
# ---------------------------------------------------
try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    if not isinstance(chunks, list):
        raise ValueError("Chunked data must be a JSON list.")

    valid_chunks = []

    for chunk in chunks:
        text = chunk.get("text", "").strip()

        if not text:
            continue

        valid_chunks.append(chunk)

    chunks = valid_chunks

    print(f"Loaded {len(chunks)} chunks from JSON.")

except Exception as e:
    logger.error(f"Error reading JSON chunk file: {e}", exc_info=True)
    raise

# ---------------------------------------------------
# Generate Embeddings
# ---------------------------------------------------
try:
    texts = [chunk["text"] for chunk in chunks]

    embeddings = embedding_model.embed_documents(texts)

    print(f"Generated embeddings for {len(texts)} chunks.")

except Exception as e:
    logger.error(f"Error generating embeddings: {e}", exc_info=True)
    raise

# ---------------------------------------------------
# Save Embeddings with Chunk Metadata
# ---------------------------------------------------
try:
    results = []

    for chunk, embedding in zip(chunks, embeddings):
        metadata = chunk.get("metadata", {})

        result = {
            "id": chunk.get("id"),
            "source": chunk.get("source"),
            "book_title": chunk.get("book_title"),
            "text": chunk.get("text"),
            "embedding": embedding,
            "metadata": {
                "file_name": metadata.get("file_name"),
                "file_path": metadata.get("file_path"),
                "book_title": metadata.get("book_title"),
                "chapter": metadata.get("chapter"),
                "section": metadata.get("section"),
                "header": metadata.get("header"),
                "section_index": metadata.get("section_index"),
                "chunk_index": metadata.get("chunk_index"),
                "local_chunk_index": metadata.get("local_chunk_index"),
                "page_start": metadata.get("page_start"),
                "page_end": metadata.get("page_end"),
                "char_count": metadata.get("char_count"),
                "estimated_token_count": metadata.get("estimated_token_count"),
                "chunk_size": metadata.get("chunk_size"),
                "chunk_overlap": metadata.get("chunk_overlap"),
                "embedding_model": "BAAI/bge-small-en-v1.5",
                "embedding_dimension": len(embedding),
                "normalized_embedding": True,
            },
        }

        results.append(result)

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Successfully saved {len(results)} embeddings to:")
    print(OUTPUT_FILE)

except Exception as e:
    logger.error(f"Error saving embeddings: {e}", exc_info=True)
    raise