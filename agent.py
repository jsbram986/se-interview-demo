import json
import operator
from typing import Annotated, Literal

from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_core.messages import AnyMessage, SystemMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, START, StateGraph
from typing_extensions import TypedDict

from tools import get_weather_forecast

load_dotenv()

search_tool = DuckDuckGoSearchRun()
tools = [search_tool, get_weather_forecast]
tools_by_name = {tool.name: tool for tool in tools}

model = ChatOpenAI(model="gpt-4o", temperature=0)
model_with_tools = model.bind_tools(tools)


class MessagesState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]


def llm_call(state: MessagesState) -> dict:
    """Call the LLM with the current messages and available tools."""
    return {
        "messages": [
            model_with_tools.invoke(
                [
                    SystemMessage(
                        content="You are a helpful travel assistant. You help users plan trips and find "
                        "attractions, flights, hotels and other travel information. "
                        "Use get_weather_forecast for any question about the weather at a destination, "
                        "what to pack, or which days suit outdoor plans. "
                        "Use the web search tool for other current information such as attractions, "
                        "flights, hotels and events. "
                        "If a tool returns an error, tell the user plainly and suggest what they could try instead."
                    )
                ]
                + state["messages"]
            )
        ]
    }


def tool_node(state: MessagesState) -> dict:
    """Execute tool calls from the last message."""
    result = []
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        try:
            observation = tool.invoke(tool_call["args"])
            status = "success"
        except Exception as exc:  # keep the agent running if a tool fails
            observation = {"error": f"{type(exc).__name__}: {exc}"}
            status = "error"
        # Structured tool output (dicts) is serialized to JSON so the LLM receives it intact.
        content = observation if isinstance(observation, str) else json.dumps(observation)
        result.append(ToolMessage(content=content, tool_call_id=tool_call["id"], status=status))
    return {"messages": result}


def should_continue(state: MessagesState) -> Literal["tool_node", "__end__"]:
    """Determine whether to continue to tool execution or end."""
    last_message = state["messages"][-1]
    if last_message.tool_calls:
        return "tool_node"
    return END


def build_agent():
    graph_builder = StateGraph(MessagesState)

    graph_builder.add_node("llm_call", llm_call)
    graph_builder.add_node("tool_node", tool_node)

    graph_builder.add_edge(START, "llm_call")
    graph_builder.add_conditional_edges("llm_call", should_continue, ["tool_node", END])
    graph_builder.add_edge("tool_node", "llm_call")

    agent = graph_builder.compile()
    return agent