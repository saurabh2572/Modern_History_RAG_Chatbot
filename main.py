from langchain_core.messages import HumanMessage, AIMessage
from graph import graph
from langsmith import traceable
from dotenv import load_dotenv
import os

load_dotenv()
conversation = [
    HumanMessage(content="Who won Fifa 2022?"),
]


@traceable(name="Modern-History-RAG")
def invoke_graph(messages):
    return graph.invoke({"messages": messages})

result = result = invoke_graph(conversation)

# print(result["rephrased_query"])
# print(result["context"])
print(result["answer"])