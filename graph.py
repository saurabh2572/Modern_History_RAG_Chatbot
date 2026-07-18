from pathlib import Path
import logging
import os
import yaml

from dotenv import load_dotenv

from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

from langgraph.graph import StateGraph, START, END
from langsmith import traceable

from pydantic import BaseModel, Field

from schema import (
    GraphState,
    InputGuardrailOutput,
    RephrasedQueryOutput,
    AnswerOutput,
    RetrievedContextOutput,
)
from prompts import QUERY_REPHRASE_PROMPT, ANSWER_PROMPT, INPUT_GUARDRAIL_PROMPT
from retriever import Retriever
from llm import HuggingFaceLLM


# ==========================================================
# Environment & Paths
# ==========================================================
load_dotenv()

Path("logs").mkdir(parents=True, exist_ok=True)
Path("cache").mkdir(parents=True, exist_ok=True)


# ==========================================================
# Logging
# ==========================================================
logging.basicConfig(
    filename="logs/all_errors.log",
    level=logging.ERROR,
    format="%(asctime)s %(levelname)s %(message)s",
)

logger = logging.getLogger(__name__)



# ==========================================================
# Load Configuration
# ==========================================================
try:
    with open("./config.yaml", "r", encoding="utf-8") as stream:
        CONFIG = yaml.safe_load(stream)
except Exception:
    logger.exception("Unable to load config.yaml")
    raise


# ==========================================================
# RAG Workflow
# ==========================================================
class RAGWorkflow:
    def __init__(self):
        try:
            # ------------------------------
            # LLM Wrapper
            # ------------------------------
            self.llm = HuggingFaceLLM(
                model=CONFIG["LLM_CONFIG"]["MODEL"],
                temperature=CONFIG["LLM_CONFIG"]["TEMPERATURE"],
                token=os.getenv("HF_TOKEN"),
            )

            # ------------------------------
            # Parsers
            # ------------------------------
            self.guardrail_parser = PydanticOutputParser(
                pydantic_object=InputGuardrailOutput
            )
            self.query_parser = PydanticOutputParser(
                pydantic_object=RephrasedQueryOutput
            )
            self.answer_parser = PydanticOutputParser(
                pydantic_object=AnswerOutput
            )

            # ------------------------------
            # Retriever
            # ------------------------------
            self.retriever = Retriever()

            print("RAG workflow initialized successfully.")

        except Exception:
            logger.exception("Error initializing RAG workflow")
            raise

    # ------------------------------------------------------
    # Cache Lookup
    # ------------------------------------------------------
    @traceable(name="Cache Lookup")
    def cache_lookup(self, state: GraphState):
        """Check if the latest user question is already cached."""

        question = state["messages"][-1].content

        print("\n========== CACHE LOOKUP ==========")
        print(f"Question: {repr(question)}")

        cached_answer = self.llm.lookup_cache(question)
        print(f"Cached Answer: {repr(cached_answer)}")

        if cached_answer is not None:
            print("CACHE HIT → will route to cached_answer node and END")
            return {
                "cached_answer": cached_answer,
                "cache_hit": True,
            }

        print("CACHE MISS → will route to query_rephraser (full RAG)")
        return {
            "cache_hit": False,
        }

    def cache_router(self, state: GraphState) -> str:
        cache_hit_flag = state.get("cache_hit", False)
        print(f">>> cache_router: cache_hit = {cache_hit_flag!r}")
        return "cache_hit" if cache_hit_flag else "cache_miss"

    @traceable(name="Cached Answer")
    def cached_answer(self, state: GraphState):
        """Return the cached answer without running RAG nodes."""
        print(">>> cached_answer node executed (serving from cache)")
        return {
            "answer": state["cached_answer"]
        }

    # ------------------------------------------------------
    # Query Rephraser
    # ------------------------------------------------------
    @traceable(name="Query Rephraser")
    def query_rephraser(self, state: GraphState) -> dict:
        try:
            print(">>> query_rephraser node executed (cache miss path)")
            messages = state["messages"]

            if not messages:
                raise ValueError("No messages found in graph state.")

            latest_question = messages[-1].content
            history = messages[-5:-1]

            history_text = (
                "\n".join(f"{msg.type}: {msg.content}" for msg in history)
                if history
                else "No previous conversation."
            )

            prompt = QUERY_REPHRASE_PROMPT.format(
                history=history_text,
                question=latest_question,
                format_instructions=self.query_parser.get_format_instructions(),
            )

            response = self.llm.invoke(prompt)
            parsed_response = self.query_parser.parse(response.content)

            return {
                "rephrased_query": parsed_response.rephrased_query
            }

        except Exception:
            logger.exception("Error occurred while rephrasing the query.")
            # Fallback: use original question
            return {
                "rephrased_query": state["messages"][-1].content
            }

    # ------------------------------------------------------
    # Input Guardrails
    # ------------------------------------------------------
    @traceable(name="Input Guardrails")
    def input_guardrails(self, state: GraphState):
        """Decide whether to run RAG, reply with capability, or block jailbreak."""
        print(">>> input_guardrails node executed")

        prompt = INPUT_GUARDRAIL_PROMPT.format(
            question=state["rephrased_query"],
            format_instructions=self.guardrail_parser.get_format_instructions(),
        )

        response = self.llm.invoke(prompt)
        parsed = self.guardrail_parser.parse(response.content)

        # parsed.decision is expected to be one of: "rag", "capability", "jailbreak"
        return {
            "guardrail_decision": parsed.decision
        }

    def guardrail_router(self, state: GraphState) -> str:
        """Route based on guardrail decision: rag / capability / jailbreak."""
        decision = state["guardrail_decision"]
        print(f">>> guardrail_router decision: {decision}")
        return decision

    @traceable(name="Capability Reply")
    def capability_reply(self, state: GraphState):
        print(">>> capability_reply node executed")
        return {
            "answer": (
                "Hi, I am your History Teacher. I can answer questions related to "
                "Modern Indian History from the provided knowledge base."
            )
        }

    @traceable(name="Jailbreak")
    def jailbreak_reply(self, state: GraphState):
        print(">>> jailbreak_reply node executed")
        return {
            "answer": "Jailbreak detected: Response is blocked."
        }

    # ------------------------------------------------------
    # Retrieve Context
    # ------------------------------------------------------
    @traceable(name="Context Retrieval")
    def retrieve_context(self, state: GraphState) -> dict:
        try:
            print(">>> retrieve_context node executed (RAG path)")
            context = self.retriever.get_context(
                rephrased_query=state["rephrased_query"],
                include_metadata=True,
            )

            parsed_context = RetrievedContextOutput(context=context)
            return {
                "context": parsed_context.context
            }

        except Exception:
            logger.exception("Error during retrieval")
            raise

    # ------------------------------------------------------
    # Generate Answer
    # ------------------------------------------------------
    @traceable(name="Answer Generation")
    def generate_answer(self, state: GraphState) -> dict:
        print(">>> generate_answer node executed (RAG path)")

        question = state["messages"][-1].content

        prompt = ANSWER_PROMPT.format(
            context=state["context"],
            question=question,
            format_instructions=self.answer_parser.get_format_instructions(),
        )

        # Generate answer from LLM
        answer_text = self.llm.generate(prompt)
        parsed_answer = self.answer_parser.parse(answer_text)
        final_answer = parsed_answer.answer

        # Update custom cache with the final answer
        self.llm.update_cache(
            question=question,
            answer=final_answer,
        )

        print(">>> cache updated with new answer (RAG path)")
        return {
            "answer": final_answer
        }


# ==========================================================
# Build Graph
# ==========================================================
workflow = RAGWorkflow()
builder = StateGraph(GraphState)

# Nodes
builder.add_node("cache_lookup", workflow.cache_lookup)
builder.add_node("cached_answer", workflow.cached_answer)
builder.add_node("query_rephraser", workflow.query_rephraser)
builder.add_node("input_guardrails", workflow.input_guardrails)
builder.add_node("retrieve_context", workflow.retrieve_context)
builder.add_node("generate_answer", workflow.generate_answer)
builder.add_node("capability_reply", workflow.capability_reply)
builder.add_node("jailbreak_reply", workflow.jailbreak_reply)

# Edges
builder.add_edge(START, "cache_lookup")

builder.add_conditional_edges(
    "cache_lookup",
    workflow.cache_router,
    {
        "cache_hit": "cached_answer",
        "cache_miss": "query_rephraser",
    },
)

builder.add_edge("cached_answer", END)

builder.add_edge("query_rephraser", "input_guardrails")

builder.add_conditional_edges(
    "input_guardrails",
    workflow.guardrail_router,
    {
        "rag": "retrieve_context",
        "capability": "capability_reply",
        "jailbreak": "jailbreak_reply",
    },
)

builder.add_edge("retrieve_context", "generate_answer")
builder.add_edge("generate_answer", END)

builder.add_edge("capability_reply", END)
builder.add_edge("jailbreak_reply", END)

graph = builder.compile()


# ==========================================================
# Helper: run a single query
# ==========================================================
def run_query(messages):
    """
    messages: list of LangChain-style messages (e.g. ChatMessage)
    where the last element is the latest user question.
    """

    state = graph.invoke({"messages": messages})

    answer = state["answer"]
    cache_hit = state.get("cache_hit", False)

    if cache_hit:
        print("\n=== FINAL ANSWER (FROM CACHE) ===")
    else:
        print("\n=== FINAL ANSWER (FROM RAG) ===")

    print(answer)

    return answer