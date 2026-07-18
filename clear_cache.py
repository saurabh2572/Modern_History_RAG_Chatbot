import sqlite3

DB_PATH = "./cache/langchain_llm_cache.sqlite"

with sqlite3.connect(DB_PATH) as conn:
    conn.execute("DELETE FROM full_llm_cache")
    conn.commit()

print("All LLM cache entries cleared successfully.")

