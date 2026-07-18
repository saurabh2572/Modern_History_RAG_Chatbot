import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import asyncio
import yaml
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_huggingface import HuggingFaceEmbeddings
from langsmith import traceable
from sentence_transformers import CrossEncoder

load_dotenv()

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

except yaml.YAMLError as e:
    logger.error(f"Error loading config.yaml: {e}", exc_info=True)
    raise


class Retriever:
    def __init__(self):
        try:
            # ----------------------------------------
            # Load Embedding Model
            # ----------------------------------------
            self.embedding_model = HuggingFaceEmbeddings(
                model_name="BAAI/bge-small-en-v1.5",
                model_kwargs={
                    "device": "cpu",
                    "trust_remote_code": True,
                },
                encode_kwargs={
                    "normalize_embeddings": True,
                },
            )

            # ----------------------------------------
            # Load Re-ranker Model
            # ----------------------------------------
            self.reranker = CrossEncoder(
                "BAAI/bge-reranker-base",
                max_length=512,
                device="cpu",
            )

            # ----------------------------------------
            # Load FAISS Vector Store
            # ----------------------------------------
            persist_dir = CONFIG["PATH"]["FAISS_VECTOR_STORE"]

            self.vector_store = FAISS.load_local(
                persist_dir,
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True,
            )

            retriever_config = CONFIG.get("RETRIEVAR_CONFIG", {})

            self.retrieval_top_k = retriever_config.get("RETRIEVAL_TOP_K", 20)
            self.rerank_top_k = retriever_config.get("RERANK_TOP_K", 5)

            print("Retriever with re-ranker initialized successfully.")

        except Exception as e:
            logger.error(f"Error initializing Retriever: {e}", exc_info=True)
            raise

    @traceable(name="FAISS Initial Retrieval")
    def retrieve_documents(
        self,
        rephrased_query: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
    ) -> List[Document]:
        try:
            similar_docs = self.vector_store.similarity_search(
                query=rephrased_query,
                k=self.retrieval_top_k,
                filter=metadata_filter,
            )

            return similar_docs

        except Exception as e:
            logger.error(
                f"Error retrieving documents from FAISS: {e}",
                exc_info=True,
            )
            raise Exception("[Error] Unable to retrieve documents from FAISS.")

    @traceable(name="BGE Re-ranking")
    def rerank_documents(
        self,
        query: str,
        documents: List[Document],
    ) -> List[Tuple[Document, float]]:
        try:
            if not documents:
                return []

            pairs = [
                [query, doc.page_content]
                for doc in documents
            ]

            scores = self.reranker.predict(pairs)

            reranked_docs = sorted(
                zip(documents, scores),
                key=lambda item: item[1],
                reverse=True,
            )

            return [
                (doc, float(score))
                for doc, score in reranked_docs[: self.rerank_top_k]
            ]

        except Exception as e:
            logger.error(
                f"Error re-ranking documents: {e}",
                exc_info=True,
            )
            raise Exception("[Error] Unable to re-rank retrieved documents.")

    @traceable(name="FAISS Retrieval With BGE Re-ranker")
    def get_context(
        self,
        rephrased_query: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> str:
        try:
            initial_docs = self.retrieve_documents(
                rephrased_query=rephrased_query,
                metadata_filter=metadata_filter,
            )

            reranked_docs = self.rerank_documents(
                query=rephrased_query,
                documents=initial_docs,
            )

            context_blocks = []

            for rank, (doc, rerank_score) in enumerate(reranked_docs, start=1):
                metadata = doc.metadata or {}

                if include_metadata:
                    context_block = f"""
[Retrieved Chunk {rank}]
Re-rank Score: {rerank_score}
Source: {metadata.get("source")}
Book: {metadata.get("book_title")}
Chapter: {metadata.get("chapter")}
Section: {metadata.get("section")}
Header: {metadata.get("header")}
Pages: {metadata.get("page_start")} - {metadata.get("page_end")}
Chunk Index: {metadata.get("chunk_index")}

Content:
{doc.page_content}
""".strip()
                else:
                    context_block = doc.page_content

                context_blocks.append(context_block)

            return "\n\n---\n\n".join(context_blocks)

        except Exception as e:
            logger.error(
                f"Error building re-ranked context: {e}",
                exc_info=True,
            )
            raise Exception("[Error] Unable to build re-ranked context.")

    @traceable(name="FAISS Retrieval With Re-rank Scores")
    def get_context_with_scores(
        self,
        rephrased_query: str,
        metadata_filter: Optional[Dict[str, Any]] = None,
        include_metadata: bool = True,
    ) -> List[Dict[str, Any]]:
        try:
            initial_docs = self.retrieve_documents(
                rephrased_query=rephrased_query,
                metadata_filter=metadata_filter,
            )

            reranked_docs = self.rerank_documents(
                query=rephrased_query,
                documents=initial_docs,
            )

            results = []

            for rank, (doc, rerank_score) in enumerate(reranked_docs, start=1):
                item = {
                    "rank": rank,
                    "rerank_score": rerank_score,
                    "text": doc.page_content,
                    "metadata": doc.metadata,
                }

                if not include_metadata:
                    item.pop("metadata")

                results.append(item)

            return results

        except Exception as e:
            logger.error(
                f"Error retrieving re-ranked documents with scores: {e}",
                exc_info=True,
            )
            raise Exception("[Error] Unable to retrieve re-ranked documents.")