from typing import List, Optional, Literal, TypedDict
from typing_extensions import TypedDict
from langgraph.graph import StateGraph
from langgraph.graph.message import add_messages
from langchain_openai import ChatOpenAI
from langgraph.types import Command
from langgraph.graph import StateGraph, MessagesState, START, END
from langchain_core.messages import HumanMessage, trim_messages
import openai
from dotenv import load_dotenv
import os

from helper_utils import make_supervisor_node
from teams import search_graph


load_dotenv()

llm_openai = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("OPENAI_API_KEY")
)

teams_supervisor_node = make_supervisor_node(llm_openai, ["search_team"])

def call_search_team(state: MessagesState) -> Command[Literal["supervisor"]]:
    response = search_graph.invoke({"messages": state["messages"][-1]})
    return Command(
        update={
            "messages": [
                HumanMessage(
                    content=response["messages"][-1].content, name="search_team"
                )
            ]
        },
        goto="supervisor",
    )

super_builder = StateGraph(MessagesState)
super_builder.add_node("supervisor", teams_supervisor_node)
super_builder.add_node("search_team", call_search_team)

super_builder.add_edge(START, "supervisor")
super_graph = super_builder.compile()

from IPython.display import Image, display

img = Image(super_graph.get_graph().draw_mermaid_png())

with open('saved_image.png', 'wb') as f:
    f.write(img.data)

for s in super_graph.stream(
    {
        "messages": [
            ("user", "Search for the five cheapest hotels in Madrid.")
        ],
    },
    {"recursion_limit": 150},
):
    print(s)
    print("---")