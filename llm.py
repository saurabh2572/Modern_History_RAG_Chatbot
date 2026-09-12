import os
import sqlite3

import faiss
import numpy as np

from huggingface_hub import InferenceClient
from sentence_transformers import SentenceTransformer
from langchain_core.messages import AIMessage


class HuggingFaceLLM:

    def __init__(
        self,
        model: str,
        token: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        embedding_model: str = "BAAI/bge-small-en-v1.5",
        similarity_threshold: float = 0.90,
    ):

        # ==================================================
        # Validate HF Token
        # ==================================================

        if not token:
            raise ValueError(
                "HF_TOKEN is missing from .env"
            )

        # ==================================================
        # HuggingFace Client
        # ==================================================

        self.client = InferenceClient(
            provider="auto",
            api_key=token,
        )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        # ==================================================
        # Semantic Cache Configuration
        # ==================================================

        self.embedding_model_name = embedding_model
        self.similarity_threshold = similarity_threshold

        # ==================================================
        # Cache Paths
        # ==================================================

        self.cache_directory = "./cache"

        os.makedirs(
            self.cache_directory,
            exist_ok=True
        )

        self.database_path = (
            "./cache/cache_database.sqlite"
        )

        self.faiss_index_path = (
            "./cache/cache.index"
        )

        # ==================================================
        # Load Embedding Model
        # ==================================================

        print(
            f"Loading embedding model: "
            f"{self.embedding_model_name}"
        )

        self.embedding_model = (
            SentenceTransformer(
                self.embedding_model_name
            )
        )

        # BGE-small-en-v1.5 produces 384-dimensional
        # embeddings.

        self.embedding_dimension = 384

        # ==================================================
        # Initialize SQLite Cache
        # ==================================================

        self._initialize_cache()

        # ==================================================
        # Initialize / Load FAISS Index
        # ==================================================

        self._initialize_faiss()

    # ======================================================
    # Initialize SQLite Cache
    # ======================================================

    def _initialize_cache(self):

        with sqlite3.connect(
            self.database_path
        ) as conn:

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS
                cache_database (

                    id INTEGER PRIMARY KEY AUTOINCREMENT,

                    question TEXT NOT NULL,

                    answer TEXT NOT NULL

                )
                """
            )

            columns = {
                row[1]
                for row in conn.execute(
                    "PRAGMA table_info(cache_database)"
                )
            }

            # Earlier cache files stored only question and answer.  SQLite
            # cannot add an AUTOINCREMENT primary key with ALTER TABLE, so
            # migrate those files before any lookup relies on a cache ID.
            if "id" not in columns:
                conn.execute(
                    """
                    CREATE TABLE cache_database_migrated (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        question TEXT NOT NULL,
                        answer TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    INSERT INTO cache_database_migrated (question, answer)
                    SELECT question, answer
                    FROM cache_database
                    ORDER BY rowid
                    """
                )
                conn.execute("DROP TABLE cache_database")
                conn.execute(
                    "ALTER TABLE cache_database_migrated "
                    "RENAME TO cache_database"
                )
                print("Migrated semantic cache database to the current schema.")

            conn.commit()

    # ======================================================
    # Initialize FAISS
    # ======================================================

    def _initialize_faiss(self):

        with sqlite3.connect(self.database_path) as conn:
            rows = conn.execute(
                "SELECT id, question FROM cache_database ORDER BY id"
            ).fetchall()

        self.cache_ids = [row[0] for row in rows]

        if os.path.exists(
            self.faiss_index_path
        ):

            print(
                "Loading existing FAISS cache..."
            )

            self.index = faiss.read_index(
                self.faiss_index_path
            )

        else:

            print(
                "Creating new FAISS cache..."
            )

            self.index = faiss.IndexFlatIP(
                self.embedding_dimension
            )

        # The database is the source of truth.  Rebuild a missing or stale
        # FAISS file so vector positions always match cache_ids.
        if self.index.ntotal != len(rows):
            print("Rebuilding FAISS cache to match the SQLite cache.")
            self.index = faiss.IndexFlatIP(
                self.embedding_dimension
            )

            if rows:
                embeddings = np.array(
                    [self._create_embedding(row[1]) for row in rows],
                    dtype="float32"
                )
                self.index.add(embeddings)

            faiss.write_index(
                self.index,
                self.faiss_index_path
            )

    # ======================================================
    # Create Embedding
    # ======================================================

    def _create_embedding(
        self,
        question: str
    ):

        embedding = (
            self.embedding_model.encode(
                question,
                normalize_embeddings=True
            )
        )

        return embedding

    # ======================================================
    # LLM Invoke
    # ======================================================

    def invoke(
        self,
        prompt: str
    ) -> AIMessage:

        response = (
            self.client
            .chat
            .completions
            .create(

                model=self.model,

                messages=[

                    {
                        "role": "user",
                        "content": prompt,
                    }

                ],

                temperature=self.temperature,

                max_tokens=self.max_tokens,
            )
        )

        content = (
            response
            .choices[0]
            .message
            .content
        )

        return AIMessage(
            content=content
        )

    # ======================================================
    # Generate Final Answer
    # ======================================================

    def generate(
        self,
        prompt: str
    ) -> str:

        response = self.invoke(
            prompt
        )

        return response.content

    # ======================================================
    # Semantic Cache Lookup
    # ======================================================

    def lookup_cache(
        self,
        question: str,
        threshold: float = None
    ):

        # --------------------------------------------------
        # Use default threshold if not provided
        # --------------------------------------------------

        if threshold is None:
            threshold = (
                self.similarity_threshold
            )

        # --------------------------------------------------
        # Check if FAISS contains anything
        # --------------------------------------------------

        if self.index.ntotal == 0:

            print(
                "SEMANTIC CACHE MISS "
                "(FAISS index is empty)"
            )

            return None

        # --------------------------------------------------
        # Create query embedding
        # --------------------------------------------------

        embedding = self._create_embedding(
            question
        )

        # FAISS expects float32
        embedding = np.array(
            [embedding],
            dtype="float32"
        )

        # --------------------------------------------------
        # Semantic Similarity Search
        # --------------------------------------------------

        scores, indices = (
            self.index.search(
                embedding,
                k=1
            )
        )

        similarity = float(
            scores[0][0]
        )

        faiss_index = int(
            indices[0][0]
        )

        print(
            f"Semantic similarity: "
            f"{similarity:.4f}"
        )

        # --------------------------------------------------
        # No valid result
        # --------------------------------------------------

        if faiss_index == -1:

            print(
                "SEMANTIC CACHE MISS"
            )

            return None

        # --------------------------------------------------
        # Similarity Threshold
        # --------------------------------------------------

        if similarity < threshold:

            print(
                f"SEMANTIC CACHE MISS "
                f"(similarity {similarity:.4f} "
                f"< threshold {threshold:.4f})"
            )

            return None

        # --------------------------------------------------
        # FAISS position → SQLite cache ID
        # --------------------------------------------------

        cache_id = self.cache_ids[faiss_index]

        # --------------------------------------------------
        # Retrieve cached answer
        # --------------------------------------------------

        with sqlite3.connect(
            self.database_path
        ) as conn:

            row = conn.execute(

                """
                SELECT
                    question,
                    answer

                FROM cache_database

                WHERE id = ?
                """,

                (cache_id,)

            ).fetchone()

        # --------------------------------------------------
        # Cache Hit
        # --------------------------------------------------

        if row:

            cached_question = row[0]
            cached_answer = row[1]

            print(
                "SEMANTIC CACHE HIT"
            )

            print(
                f"Matched Question: "
                f"{cached_question}"
            )

            print(
                f"Similarity: "
                f"{similarity:.4f}"
            )

            return cached_answer

        # --------------------------------------------------
        # Cache Miss
        # --------------------------------------------------

        print(
            "SEMANTIC CACHE MISS"
        )

        return None

    # ======================================================
    # Update Semantic Cache
    # ======================================================

    def update_cache(
        self,
        question: str,
        answer: str
    ):

        # --------------------------------------------------
        # Create embedding
        # --------------------------------------------------

        embedding = self._create_embedding(
            question
        )

        embedding = np.array(
            [embedding],
            dtype="float32"
        )

        # --------------------------------------------------
        # Insert into SQLite
        # --------------------------------------------------

        with sqlite3.connect(
            self.database_path
        ) as conn:

            cursor = conn.execute(

                """
                INSERT INTO cache_database
                (
                    question,
                    answer
                )

                VALUES (?, ?)
                """,

                (
                    question,
                    answer
                )

            )

            cache_id = cursor.lastrowid

            conn.commit()

        # --------------------------------------------------
        # Add embedding to FAISS
        # --------------------------------------------------

        self.index.add(
            embedding
        )

        self.cache_ids.append(cache_id)

        # --------------------------------------------------
        # Persist FAISS index
        # --------------------------------------------------

        faiss.write_index(
            self.index,
            self.faiss_index_path
        )

        print(
            f"SEMANTIC CACHE UPDATED "
            f"(ID: {cache_id})"
        )

    # ======================================================
    # Generate With Semantic Cache
    # ======================================================

    def generate_with_cache(
        self,
        prompt: str
    ):

        # --------------------------------------------------
        # Step 1: Semantic Cache Lookup
        # --------------------------------------------------

        cached_answer = (
            self.lookup_cache(
                prompt
            )
        )

        # --------------------------------------------------
        # Step 2: Cache HIT
        # --------------------------------------------------

        if cached_answer is not None:

            return cached_answer

        # --------------------------------------------------
        # Step 3: Cache MISS → Call LLM
        # --------------------------------------------------

        print(
            "Calling HuggingFace LLM..."
        )

        answer = self.generate(
            prompt
        )

        # --------------------------------------------------
        # Step 4: Store New Response
        # --------------------------------------------------

        self.update_cache(
            prompt,
            answer
        )

        # --------------------------------------------------
        # Step 5: Return Answer
        # --------------------------------------------------

        return answer
