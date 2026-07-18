# RAG Chatbot with LangGraph, HuggingFace LLM, and Streamlit



This project implements a Retrieval-Augmented Generation (RAG) chatbot using **LangGraph** for workflow orchestration, a **HuggingFace LLM** as the core model, and **Streamlit** as a simple web UI. It includes:

- A modular RAG workflow built as a graph (`StateGraph`)
- Query rephrasing, input guardrails, and context retrieval
- Structured output via `PydanticOutputParser`
- A custom LLM cache to short‑circuit repeated questions
- A history‑teacher persona grounded in a Modern Indian History knowledge base

---

## Architecture
 ![Project Architecture](images/architecture.png)

## Features

- **LangGraph‑based workflow**: Nodes for cache lookup, query rephrasing, guardrails, retrieval, and answer generation wired with conditional edges. [web:49][web:52]
- **Custom LLM wrapper**: Uses a HuggingFace model with temperature and HF token configuration.
- **RAG pipeline**: Retrieves context from your knowledge base (via `Retriever`) and generates grounded answers.
- **Input guardrails**: Routes user queries to:
  - RAG path (`rag`)
  - Capability message (`capability`)
  - Jailbreak block (`jailbreak`)
- **JSON‑structured answers**: Enforced via `PydanticOutputParser` and format instructions.
- **Caching**: Caches final answers per question so repeated queries are served instantly from cache.
- **Streamlit UI**: Simple chat‑style frontend to interact with the RAG bot.

---

## Project Structure

```bash
RAG_Chatbot/
├── graph.py              # LangGraph workflow & nodes
├── llm.py                # HuggingFaceLLM wrapper (invoke, generate, cache)
├── retriever.py          # Retriever with re-ranker for knowledge base context
├── schema.py             # GraphState & Pydantic models for structured outputs
├── prompts.py            # QUERY_REPHRASE_PROMPT, INPUT_GUARDRAIL_PROMPT, ANSWER_PROMPT
├── frontend.py           # Streamlit app entry point
├── config.yaml           # LLM and app configuration
├── logs/                 # Error logs
├── cache/                # LangChain & custom cache files
├── .env                  # Environment variables (HF_TOKEN, etc.)
└── README.md             # Project documentation
```

---

## Requirements

Install dependencies (simplify or expand as needed):

```bash
pip install \
  langchain \
  langgraph \
  langsmith \
  streamlit \
  pydantic \
  python-dotenv \
  langchain-community \
  huggingface_hub \
  yaml
```

You may need additional packages depending on your retriever (e.g., FAISS, sentence-transformers, etc.). [web:51][web:56]

---

## Configuration

### 1. Environment Variables

Create a `.env` file in the project root:

```env
HF_TOKEN=your_huggingface_token_here
```

And load it in Python:

```python
from dotenv import load_dotenv
load_dotenv()
```

### 2. `config.yaml`

Example structure:

```yaml
LLM_CONFIG:
  MODEL: "mistralai/Mixtral-8x7B-Instruct-v0.1"   # or any HF model you use
  TEMPERATURE: 0.2
```

Adjust according to your chosen HuggingFace model and parameters.

---

## Core Architecture

### Graph State

`GraphState` holds all data passed between nodes, including:

- `messages`: conversation history (LangChain `BaseMessage` list)
- `rephrased_query`: latest normalized question
- `guardrail_decision`: one of `rag`, `capability`, `jailbreak`
- `context`: retrieved knowledge base snippets
- `answer`: final answer returned to the user
- `cache_hit`: boolean flag indicating cache usage
- `cached_answer`: cached response when available

Defining these fields ensures LangGraph persists them across nodes. [web:49][web:52]

### Nodes

The graph is built from the following nodes:

- `cache_lookup`: Checks if the latest question is in the custom cache and sets `cache_hit` / `cached_answer`.
- `cached_answer`: Returns the cached answer and terminates the workflow on cache hits.
- `query_rephraser`: Rephrases the user question using `QUERY_REPHRASE_PROMPT` and `RephrasedQueryOutput`.
- `input_guardrails`: Uses `INPUT_GUARDRAIL_PROMPT` to classify the query (rag / capability / jailbreak).
- `capability_reply`: Explains the bot’s capabilities when a non‑RAG query is detected.
- `jailbreak_reply`: Blocks jailbreak / unsafe requests.
- `retrieve_context`: Fetches relevant documents from the knowledge base via `Retriever`.
- `generate_answer`: Runs the main RAG answer generation, parses JSON with `AnswerOutput`, and updates the cache.

### Routing Logic

Conditional edges orchestrate control flow:

- From `cache_lookup`:
  - `cache_hit` → `cached_answer` → `END`
  - `cache_miss` → `query_rephraser`
- From `input_guardrails`:
  - `rag` → `retrieve_context` → `generate_answer` → `END`
  - `capability` → `capability_reply` → `END`
  - `jailbreak` → `jailbreak_reply` → `END`

This ensures repeated questions bypass the full RAG pipeline and are served from cache. [web:49][web:52]

---

## HuggingFace LLM Integration

`HuggingFaceLLM` is responsible for:

- Initializing the HF model with token and configuration.
- Providing `.invoke(prompt)` that returns a message object with `.content` (string).
- Implementing `lookup_cache(question)` and `update_cache(question, answer)` to store and retrieve answers.
- Optionally exposing `.generate(prompt)` for raw text generation.

When used with `PydanticOutputParser`, the prompts enforce strict JSON output, which is then parsed into Pydantic models. [web:43][web:47]

---

## Streamlit Frontend

`frontend.py` wires the graph into a simple chat UI:

```python
import streamlit as st
from graph import graph  # compiled LangGraph

st.title("📚 RAG Chatbot (Modern Indian History)")

if "conversation" not in st.session_state:
    st.session_state.conversation = []

user_input = st.chat_input("Ask a question about Modern Indian History...")

if user_input:
    st.session_state.conversation.append({"role": "user", "content": user_input})

    with st.spinner("Thinking..."):
        # Convert stored conversation to LangChain messages here
        messages = ...  # build list of BaseMessage objects
        result = graph.invoke({"messages": messages})

    st.session_state.conversation.append({"role": "assistant", "content": result["answer"]})

for msg in st.session_state.conversation:
    if msg["role"] == "user":
        st.chat_message("user").markdown(msg["content"])
    else:
        st.chat_message("assistant").markdown(msg["content"])
```

Run the app:

```bash
streamlit run frontend.py
```



Paste `mermaid_code` into https://mermaid.live to see a graphical view of nodes and edges. [web:26][web:27][web:32]

---

## Troubleshooting

### 1. Cache not short‑circuiting

- Ensure `cache_hit` and `cached_answer` are declared in `GraphState`.
- Verify `cache_router` returns `"cache_hit"` / `"cache_miss"` and matches `add_conditional_edges` keys.
- Use `draw_ascii()` / `draw_mermaid()` to confirm `cache_lookup` branch wiring.



---

---

## License

Add your preferred license here (MIT, Apache‑2.0, etc.).