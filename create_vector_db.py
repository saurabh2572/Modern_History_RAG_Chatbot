import json
import logging
from pathlib import Path

import yaml
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
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
# Load Configuration
# ---------------------------------------------------
try:
    with open("./config.yaml", "r", encoding="utf-8") as stream:
        CONFIG = yaml.safe_load(stream)

except yaml.YAMLError as exc:
    logger.error(f"Error loading config.yaml: {exc}", exc_info=True)
    raise

# ---------------------------------------------------
# Paths
# ---------------------------------------------------
INPUT_FILE = Path(CONFIG["PATH"]["VECTOR_EMBEDDING"])
FAISS_PATH = Path(CONFIG["PATH"]["FAISS_VECTOR_STORE"])

FAISS_PATH.mkdir(parents=True, exist_ok=True)

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
# Load Embedding JSON
# ---------------------------------------------------
try:
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("Vector embedding file must contain a JSON list.")

    print(f"Loaded {len(data)} embedded chunks.")

except Exception as e:
    logger.error(f"Error reading embedding file: {e}", exc_info=True)
    raise

# ---------------------------------------------------
# Create Documents with Full Metadata
# ---------------------------------------------------
documents = []

for item in data:
    text = item.get("text", "").strip()

    if not text:
        continue

    chunk_metadata = item.get("metadata", {})

    metadata = {
        "id": item.get("id"),
        "source": item.get("source"),
        "book_title": item.get("book_title"),
        "file_name": chunk_metadata.get("file_name"),
        "file_path": chunk_metadata.get("file_path"),
        "chapter": chunk_metadata.get("chapter"),
        "section": chunk_metadata.get("section"),
        "header": chunk_metadata.get("header"),
        "section_index": chunk_metadata.get("section_index"),
        "chunk_index": chunk_metadata.get("chunk_index"),
        "local_chunk_index": chunk_metadata.get("local_chunk_index"),
        "page_start": chunk_metadata.get("page_start"),
        "page_end": chunk_metadata.get("page_end"),
        "char_count": chunk_metadata.get("char_count"),
        "estimated_token_count": chunk_metadata.get("estimated_token_count"),
        "chunk_size": chunk_metadata.get("chunk_size"),
        "chunk_overlap": chunk_metadata.get("chunk_overlap"),
        "embedding_model": chunk_metadata.get("embedding_model"),
        "embedding_dimension": chunk_metadata.get("embedding_dimension"),
        "normalized_embedding": chunk_metadata.get("normalized_embedding"),
    }

    documents.append(
        Document(
            page_content=text,
            metadata=metadata,
        )
    )

print(f"Created {len(documents)} LangChain documents.")

# ---------------------------------------------------
# Create FAISS Vector Store
# ---------------------------------------------------
try:
    vector_store = FAISS.from_documents(
        documents=documents,
        embedding=embedding_model,
    )

    vector_store.save_local(str(FAISS_PATH))

    print(f"FAISS Vector Store saved to: {FAISS_PATH}")

except Exception as e:
    logger.error(f"Error creating FAISS vector store: {e}", exc_info=True)
    raise