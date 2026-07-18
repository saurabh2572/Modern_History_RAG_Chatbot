import streamlit as st

from langchain_core.messages import HumanMessage, AIMessage
from graph import graph

# ---------------------------------------------------
# Page Configuration
# ---------------------------------------------------
st.set_page_config(
    page_title="Modern History Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Modern History RAG Chatbot")
st.markdown("Ask questions about Modern History using your local LangGraph + Hugging Face RAG pipeline.")

# ---------------------------------------------------
# Session State
# ---------------------------------------------------
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------
# Display Chat History
# ---------------------------------------------------
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# ---------------------------------------------------
# User Input
# ---------------------------------------------------
if prompt := st.chat_input("Ask a question..."):

    # Display User Message
    st.chat_message("user").markdown(prompt)

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    # ---------------------------------------------------
    # Convert to LangChain Messages
    # ---------------------------------------------------
    conversation = []

    for msg in st.session_state.messages:

        if msg["role"] == "user":
            conversation.append(
                HumanMessage(content=msg["content"])
            )

        else:
            conversation.append(
                AIMessage(content=msg["content"])
            )

    # ---------------------------------------------------
    # Invoke LangGraph
    # ---------------------------------------------------
    with st.spinner("Thinking..."):

        result = graph.invoke(
            {
                "messages": conversation
            }
        )

    answer = result["answer"]

    # ---------------------------------------------------
    # Display Assistant Response
    # ---------------------------------------------------
    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer
        }
    )

# ---------------------------------------------------
# Sidebar
# ---------------------------------------------------
with st.sidebar:

    st.header("Conversation")

    if st.button("🗑️ Clear Chat"):

        st.session_state.messages = []
        st.rerun()