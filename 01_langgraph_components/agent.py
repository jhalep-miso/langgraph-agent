from langgraph.graph import StateGraph, END
from typing import List, TypedDict, Annotated
import operator
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, ToolMessage
from langchain.chat_models import init_chat_model

# from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_tavily import TavilySearch
import os
import getpass
from dotenv import load_dotenv


load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], operator.add]


class Agent:
    def __init__(self, model, tools, system=""):
        self.system = system
        graph = StateGraph(AgentState)
        graph.add_node("llm", self.call_model)
        graph.add_node("action", self.take_action)
        graph.add_conditional_edges(
            "llm", self.exists_action, {True: "action", False: END}
        )
        graph.add_edge("action", "llm")
        graph.set_entry_point("llm")
        self.graph = graph.compile()
        self.tools = {t.name: t for t in tools}
        self.model = model.bind_tools(tools)

    def exists_action(self, state: AgentState) -> bool:
        result = state['messages'][-1]
        return len(result.tool_calls) > 0

    def call_model(self, state: AgentState):
        messages = state['messages']
        if self.system:
            messages = [SystemMessage(content=self.system)] + messages

        message = self.model.invoke(messages)
        return {'messages': [message]}

    def take_action(self, state: AgentState):
        tool_calls = state['messages'][-1].tool_calls
        results = []
        for t in tool_calls:
            print(f"Calling tool {t['name']} with args {t['args']}")
            if not t['name'] in self.tools:
                print("\n ....bad tool name....")
                result = "bad tool name, retry"
            else:
                result = self.tools[t['name']].invoke(t['args'])

            results.append(
                ToolMessage(tool_call_id=t['id'], name=t['name'], content=str(result))
            )
        print("Back to the model!")
        return {'messages': results}


prompt = """You are a smart research assistant with access to a search engine tool. 

IMPORTANT: You can and should use the search tool to find current information about:
- Weather conditions in any location
- Recent news and events
- Current prices, stock information, or market data
- Any other real-time or frequently changing information

When a user asks about weather, current events, or any information that might be time-sensitive, 
USE THE SEARCH TOOL to get the most up-to-date information.

You are allowed to make multiple search calls (either together or in sequence). 
Only search when you are sure of what you want to find.
If you need to search for some information before asking a follow up question, you are allowed to do that!
"""

model = init_chat_model(
    "gemini-2.0-flash", model_provider="google_genai", temperature=0.0
)
tavily_tool = TavilySearch(max_results=4)
research_agent = Agent(model, [tavily_tool], system=prompt)


if not os.path.exists("graph.png"):
    graph_viz = research_agent.graph.get_graph().draw_mermaid_png()
    with open("graph.png", "wb") as f:
        f.write(graph_viz)


# messages = [HumanMessage(content="What is the weather in sf?")]
# messages = [HumanMessage(content="What is the weather in SF and LA?")]

query = "Who won the super bowl in 2023? In what state is the winning team headquarters located? \
What is the GDP of that state? Answer each question." 
messages = [HumanMessage(content=query)]

result = research_agent.graph.invoke({"messages": messages})

# pretty print the messages
for message in result['messages']:
    message.pretty_print()
