from langchain_core.messages import BaseMessage
import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()


class Orchestrator:

    def __init__(self):

        pass

    



llm_openai = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    max_tokens=None,
    timeout=None,
    max_retries=2,
    api_key=os.getenv("OPENAI_API_KEY")
)


