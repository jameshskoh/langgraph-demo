from enum import Enum

from langchain_core.runnables import Runnable
from langgraph.constants import END, START
from langgraph.graph import MessagesState, StateGraph
from langgraph.prebuilt import ToolNode

from .tools import tools


class GraphNode(Enum):
    CHATBOT = 1
    TOOLS = 2


def build_graph(llm: Runnable):
    graph_builder = StateGraph(MessagesState)

    def chatbot(state: MessagesState, model_with_tools: Runnable):
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    graph_builder.add_node(GraphNode.CHATBOT.name, (lambda state: chatbot(state, llm)))
    graph_builder.add_node(GraphNode.TOOLS.name, ToolNode(tools=tools))

    def should_use_tool(state: MessagesState):
        messages = state.get("messages", [])

        if not messages:
            return END

        last_message = messages[-1]

        if last_message.tool_calls:
            return GraphNode.TOOLS.name
        return END

    graph_builder.add_edge(START, GraphNode.CHATBOT.name)
    graph_builder.add_conditional_edges(
        GraphNode.CHATBOT.name,
        should_use_tool,
        {GraphNode.TOOLS.name: GraphNode.TOOLS.name, END: END},
    )
    graph_builder.add_edge(GraphNode.TOOLS.name, GraphNode.CHATBOT.name)

    return graph_builder.compile()
