from typing import Annotated

from IPython.core.display import Image
from IPython.core.display_functions import display
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, BaseMessage
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


class State(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


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

    def stream_graph_updates(input_str: str):
        for event in graph.stream(State(messages=[HumanMessage(input_str)])):
            for value in event.values():
                print("Assistant:", value["messages"][-1].content)

    while True:
        try:
            user_input = input("User: ")
            if user_input.lower() in ["quit", "exit", "q"]:
                print("Goodbye!")
                break
            stream_graph_updates(user_input)
        except:
            # fallback if input() is not available
            user_input = "What do you know about LangGraph?"
            print("User: " + user_input)
            stream_graph_updates(user_input)
            break
