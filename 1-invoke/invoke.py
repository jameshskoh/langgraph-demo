from typing import Annotated

from IPython.core.display import Image
from IPython.core.display_functions import display
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]


if __name__ == "__main__":
    graph_builder = StateGraph(State)

    load_dotenv()

    llm = init_chat_model(
        "qwen/qwen3-30b-a3b-2507",
        model_provider="openai",
        base_url="http://localhost:1234/v1",
        api_key="dummy",
    )

    def chatbot(state: State):
        return {"messages": [llm.invoke(state["messages"])]}

    graph_builder.add_node("chatbot", chatbot)
    graph_builder.add_edge(START, "chatbot")
    graph_builder.add_edge("chatbot", END)
    graph = graph_builder.compile()
    display(Image(graph.get_graph().draw_mermaid_png()))

    # ignore type check issue and trust that it will work :)
    result = graph.invoke(
        input=State(messages=SystemMessage("This is a demo LangGraph application."))
    )

    print(result)
