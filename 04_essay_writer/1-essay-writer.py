from langgraph.graph import StateGraph, END
from typing import List, TypedDict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import MemorySaver
from tavily import TavilyClient
import os
import getpass
from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

if not os.environ.get("GOOGLE_API_KEY"):
    os.environ["GOOGLE_API_KEY"] = getpass.getpass("Enter API key for Google Gemini: ")

memory = MemorySaver()
tavily = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])
model = init_chat_model(
    "gemini-2.0-flash-lite", model_provider="google_genai", temperature=0.0
)


class Queries(BaseModel):
    queries: List[str]


class AgentState(TypedDict):
    task: str
    plan: str
    draft: str
    critique: str
    content: List[str]
    revision_number: int
    max_revisions: int


PLAN_PROMPT = """You are an expert writer tasked with writing a high level outline of an essay. 
Write such an outline for the user provided topic. Give an outline of the essay along with any relevant notes 
or instructions for the sections."""

WRITER_PROMPT = """You are an essay assistant tasked with writing excellent 5-paragraph essays.
Generate the best essay possible for the user's request and the initial outline. 
If the user provides critique, respond with a revised version of your previous attempts. 
Utilize all the information below as needed: 

------

{content}"""

REFLECTION_PROMPT = """You are a teacher grading an essay submission. 
Generate critique and recommendations for the user's submission. 
Provide detailed recommendations, including requests for length, depth, style, etc."""

RESEARCH_PLAN_PROMPT = """You are a researcher charged with providing information that can 
be used when writing the following essay. Generate a list of search queries that will gather 
any relevant information. Only generate 3 queries max."""

RESEARCH_CRITIQUE_PROMPT = """You are a researcher charged with providing information that can 
be used when making any requested revisions (as outlined below). 
Generate a list of search queries that will gather any relevant information. Only generate 3 queries max."""


def plan_node(state: AgentState):
    messages = [SystemMessage(content=PLAN_PROMPT), HumanMessage(content=state['task'])]
    response = model.invoke(messages)
    return {"plan": response.content}


def research_plan_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=RESEARCH_PLAN_PROMPT),
            HumanMessage(content=state['task']),
        ]
    )
    content = state.get('content', [])
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}


def generation_node(state: AgentState):
    content = "\n\n".join(state['content'] or [])
    user_message = HumanMessage(
        content=f"{state['task']}\n\nHere is my plan:\n\n{state['plan']}"
    )
    messages = [
        SystemMessage(content=WRITER_PROMPT.format(content=content)),
        user_message,
    ]
    response = model.invoke(messages)
    return {
        "draft": response.content,
        "revision_number": state.get("revision_number", 1) + 1,
    }


def reflection_node(state: AgentState):
    messages = [
        SystemMessage(content=REFLECTION_PROMPT),
        HumanMessage(content=state['draft']),
    ]
    response = model.invoke(messages)
    return {"critique": response.content}


def research_critique_node(state: AgentState):
    queries = model.with_structured_output(Queries).invoke(
        [
            SystemMessage(content=RESEARCH_CRITIQUE_PROMPT),
            HumanMessage(content=state['critique']),
        ]
    )
    content = state['content'] or []
    for q in queries.queries:
        response = tavily.search(query=q, max_results=2)
        for r in response['results']:
            content.append(r['content'])
    return {"content": content}


def should_continue(state: AgentState):
    if state["revision_number"] > state["max_revisions"]:
        return END
    return "reflect"


builder = StateGraph(AgentState)

builder.add_node("planner", plan_node)
builder.add_node("generate", generation_node)
builder.add_node("reflect", reflection_node)
builder.add_node("research_plan", research_plan_node)
builder.add_node("research_critique", research_critique_node)

builder.set_entry_point("planner")

builder.add_conditional_edges(
    "generate", should_continue, {END: END, "reflect": "reflect"}
)

builder.add_edge("planner", "research_plan")
builder.add_edge("research_plan", "generate")
builder.add_edge("reflect", "research_critique")
builder.add_edge("research_critique", "generate")

graph = builder.compile(checkpointer=memory)

if not os.path.exists("graph-essay.png"):
    graph_viz = graph.get_graph().draw_mermaid_png()
    with open("graph-essay.png", "wb") as f:
        f.write(graph_viz)


thread = {"configurable": {"thread_id": "1"}}

for s in graph.stream(
    {
        'task': "what is the difference between langchain and langsmith",
        "max_revisions": 2,
        "revision_number": 1,
    },
    thread,
    stream_mode="values",
):
    print(s)
