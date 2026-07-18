import hashlib
import os
import sqlite3

from huggingface_hub import InferenceClient
from langchain_core.messages import AIMessage


class HuggingFaceLLM:

    def __init__(
        self,
        model: str,
        token: str,
        temperature: float = 0.0,
        max_tokens: int = 2048,
    ):

        if not token:
            raise ValueError(
                "HF_TOKEN is missing from .env"
            )

        self.client = InferenceClient(
            provider="auto",
            api_key=token,
        )

        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

        self.database_path = (
            "./cache/cache_database.sqlite"
        )

        self._initialize_cache()

    # ======================================================
    # Initialize Cache
    # ======================================================

    def _initialize_cache(self):

        with sqlite3.connect(
            self.database_path
        ) as conn:

            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS
                cache_database (

                    cache_key TEXT PRIMARY KEY,

                    question TEXT NOT NULL,

                    answer TEXT NOT NULL

                )
                """
            )

            conn.commit()

    # ======================================================
    # Create Cache Key
    # ======================================================

    def _create_cache_key(
        self,
        question: str
    ) -> str:

        cache_input = (

            f"{self.model}|"

            f"{self.temperature}|"

            f"{question.strip().lower()}"

        )

        return hashlib.sha256(

            cache_input.encode(
                "utf-8"
            )

        ).hexdigest()

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
    # Cache Lookup
    # ======================================================

    def lookup_cache(
        self,
        question: str
    ):

        cache_key = self._create_cache_key(
            question
        )

        with sqlite3.connect(
            self.database_path
        ) as conn:

            row = conn.execute(

                """
                SELECT answer

                FROM cache_database

                WHERE cache_key = ?

                """,

                (cache_key,)

            ).fetchone()


        if row:

            print(
                "FINAL ANSWER CACHE HIT"
            )

            return row[0]


        print(
            "FINAL ANSWER CACHE MISS"
        )

        return None

    # ======================================================
    # Cache Update
    # ======================================================

    def update_cache(
        self,
        question: str,
        answer: str
    ):

        cache_key = self._create_cache_key(
            question
        )

        with sqlite3.connect(
            self.database_path
        ) as conn:

            conn.execute(

                """
                INSERT OR REPLACE INTO
                cache_database

                (

                    cache_key,

                    question,

                    answer

                )

                VALUES (?, ?, ?)

                """,

                (

                    cache_key,

                    question,

                    answer,

                )

            )

            conn.commit()


        print(
            "FINAL ANSWER CACHE UPDATED"
        )