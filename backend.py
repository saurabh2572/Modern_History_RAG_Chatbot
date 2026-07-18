from typing import Dict, List, Literal, Optional
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field

from graph import graph

# ---------------------------------------------------
# FastAPI App
# ---------------------------------------------------
app = FastAPI(
    title="Modern RAG Chatbot API",
    description="FastAPI backend for LangGraph-based Modern RAG chatbot.",
    version="1.0.0",
)

# ---------------------------------------------------
# CORS
# ---------------------------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------
# In-Memory Session Store
# ---------------------------------------------------
chat_sessions: Dict[str, List[Dict[str, str]]] = {}


# ---------------------------------------------------
# Request / Response Schemas
# ---------------------------------------------------
class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str


class ChatRequest(BaseModel):
    message: str = Field(..., description="User question")
    session_id: Optional[str] = Field(
        default=None,
        description="Existing chat session ID. If not provided, a new one is created.",
    )


class ChatResponse(BaseModel):
    session_id: str
    answer: str
    messages: List[ChatMessage]


class ClearChatRequest(BaseModel):
    session_id: str


class ClearChatResponse(BaseModel):
    session_id: str
    cleared: bool


# ---------------------------------------------------
# Helpers
# ---------------------------------------------------
def convert_to_langchain_messages(messages: List[Dict[str, str]]):
    conversation = []

    for msg in messages:
        if msg["role"] == "user":
            conversation.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            conversation.append(AIMessage(content=msg["content"]))

    return conversation


# ---------------------------------------------------
# Routes
# ---------------------------------------------------
@app.get("/")
def root():
    return {
        "message": "Modern RAG Chatbot API is running."
    }


@app.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        session_id = request.session_id or str(uuid4())

        if session_id not in chat_sessions:
            chat_sessions[session_id] = []

        chat_sessions[session_id].append(
            {
                "role": "user",
                "content": request.message,
            }
        )

        conversation = convert_to_langchain_messages(
            chat_sessions[session_id]
        )

        result =  await graph.ainvoke(
            {
                "messages": conversation,
            }
        )

        answer = result["answer"]

        chat_sessions[session_id].append(
            {
                "role": "assistant",
                "content": answer,
            }
        )

        return ChatResponse(
            session_id=session_id,
            answer=answer,
            messages=chat_sessions[session_id],
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error while processing chat request: {str(e)}",
        )


@app.get("/sessions/{session_id}", response_model=List[ChatMessage])
def get_session_messages(session_id: str):
    if session_id not in chat_sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found.",
        )

    return chat_sessions[session_id]


@app.post("/clear-chat", response_model=ClearChatResponse)
def clear_chat(request: ClearChatRequest):
    if request.session_id in chat_sessions:
        chat_sessions[request.session_id] = []

    return ClearChatResponse(
        session_id=request.session_id,
        cleared=True,
    )


@app.delete("/sessions/{session_id}")
def delete_session(session_id: str):
    if session_id not in chat_sessions:
        raise HTTPException(
            status_code=404,
            detail="Session not found.",
        )

    del chat_sessions[session_id]

    return {
        "session_id": session_id,
        "deleted": True,
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy"
    }