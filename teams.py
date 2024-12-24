from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from langgraph.graph import StateGraph, MessagesState, START, END
from typing import List, Optional, Literal

from helper_utils import make_supervisor_node
from tools import tavily_tool, scrape_webpages

import os
from dotenv import load_dotenv
load_dotenv()

llm_openai = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("OPENAI_API_KEY")
)
search_agent = create_react_agent(llm_openai, tools=[tavily_tool])

'''
Search Team
'''
def search_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = search_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="search")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )


web_scraper_agent = create_react_agent(llm_openai, tools=[scrape_webpages])


def web_scraper_node(state: MessagesState) -> Command[Literal["supervisor"]]:
    result = web_scraper_agent.invoke(state)
    return Command(
        update={
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="web_scraper")
            ]
        },
        # We want our workers to ALWAYS "report back" to the supervisor when done
        goto="supervisor",
    )


search_supervisor_node = make_supervisor_node(llm_openai, ["search", "web_scraper"])

search_builder = StateGraph(MessagesState)
search_builder.add_node("supervisor", search_supervisor_node)
search_builder.add_node("search", search_node)
search_builder.add_node("web_scraper", web_scraper_node)

search_builder.add_edge(START, "supervisor")
search_graph = search_builder.compile()