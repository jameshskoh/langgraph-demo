from pprint import pprint

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.messages import SystemMessage, HumanMessage

from graph.graph import build_graph
from graph.tools import tools

if __name__ == "__main__":
    load_dotenv()

    llm = init_chat_model(
        "qwen/qwen3-30b-a3b-2507",
        model_provider="openai",
        base_url="http://localhost:1234/v1",
        api_key="dummy",
    ).bind_tools(tools)

    graph = build_graph(llm)

    result = graph.invoke(
        {
            "messages": [
                SystemMessage(
                    "You have a few tools at your disposal. Please try to fulfill the user's request and use the tools whenever necessary."
                ),
                HumanMessage("How's the weather in Johor Bahru?"),
            ]
        }
    )

    pprint(result)
