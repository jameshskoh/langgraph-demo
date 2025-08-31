from typing import Annotated

from langchain_core.messages import (
    AIMessage,
    SystemMessage,
    HumanMessage,
    BaseMessage,
)
from langgraph.graph import StateGraph, add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# just ignore this type error :)
builder = StateGraph(State)

builder.add_node(
    "chatbot",
    # note that this function takes a state, then returns a message or a list of message
    # this function fits perfectly into a LangChain model.invoke() as it takes a list of messages and returns a message
    lambda state: {
        "messages": [
            SystemMessage("This is system prompt."),
            HumanMessage("This is human's prompt."),
            AIMessage("This is chatbot's message."),
        ]
    },
)


builder.set_entry_point("chatbot")
builder.set_finish_point("chatbot")
graph = builder.compile()

result = graph.invoke(SystemMessage("This is the start of the graph."))
print(result)
