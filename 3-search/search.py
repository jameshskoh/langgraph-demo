from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langgraph.constants import END, START
from langgraph.graph import StateGraph

from tool.basic_tool_node import BasicToolNode
from tool.tools import tools, State, route_tools


def chatbot(state: State, model):
    return {"messages": [model.invoke(state["messages"])]}


if __name__ == "__main__":
    load_dotenv()
    llm = init_chat_model(
        "qwen/qwen3-30b-a3b-2507",
        model_provider="openai",
        base_url="http://localhost:1234/v1",
        api_key="dummy",
    ).bind_tools(tools)

    '''
    Define nodes
    '''
    graph_builder = StateGraph(State)
    # https://langchain-ai.github.io/langgraph/reference/graphs/#langgraph.graph.state.StateGraph.add_node
    graph_builder.add_node("chatbot", (lambda state: chatbot(state, llm)))

    tool_node = BasicToolNode(tools=tools)
    graph_builder.add_node("tools", tool_node)

    '''
    Define edges
    '''
    graph_builder.add_conditional_edges(
        "chatbot",
        route_tools,
        # The following dictionary lets you tell the graph to interpret the condition's outputs as a specific node
        # It defaults to the identity function, but if you
        # want to use a node named something else apart from "tools",
        # You can update the value of the dictionary to something else
        # e.g., "tools": "my_tools"
        {"tools": "tools", END: END},
    )
    graph_builder.add_edge(START, "chatbot")
    # Any time a tool is called, we return to the chatbot to decide the next step
    graph_builder.add_edge("tools", "chatbot")

    graph = graph_builder.compile()

    def stream_graph_updates(user_input: str):
        for event in graph.stream(
            {"messages": [{"role": "user", "content": user_input}]}
        ):
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
