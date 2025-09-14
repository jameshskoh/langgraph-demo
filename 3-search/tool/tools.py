from typing import Annotated

from langchain_core.messages import BaseMessage
from langchain_tavily import TavilySearch
from langgraph.constants import END
from langgraph.graph import add_messages, MessagesState

tool = TavilySearch(max_results=30)
tools = [tool]


class State(MessagesState):
    messages: Annotated[list[BaseMessage], add_messages]


def route_tools(
    state: State,
):
    """
    Use in the conditional_edge to route to the ToolNode if the last message
    has tool calls. Otherwise, route to the end.
    """
    if isinstance(state, list):
        ai_message = state[-1]
    elif messages := state.get("messages", []):
        ai_message = messages[-1]
    else:
        raise ValueError(f"No messages found in input state to tool_edge: {state}")
    # https://python.langchain.com/api_reference/core/messages/langchain_core.messages.ai.AIMessage.html
    # AIMessage may have tool_calls attribute if tools are used
    if hasattr(ai_message, "tool_calls") and len(ai_message.tool_calls) > 0:
        return "tools"
    return END
