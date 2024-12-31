from state import MainState
from agent import LLMAgent
from tools import Tool
from models import LLMFactory
from dotenv import load_dotenv
import os


@Tool
def get_result_sum():
    return 5/342

def test(current_state, agent):
    result = agent.action(prompt = "Use the get_result_sum function at your disposal")
    current_state.add_message(result)

    return result




if __name__ == "__main__":
    # Create a factory instance
    # load_dotenv()
    factory = LLMFactory()
    # Create a GPT model
    gpt_model = factory.create_model("gpt",openai_api_key = os.getenv('OPENAI_API_KEY'))

    currentState = MainState()
    agent = LLMAgent(llm_model= gpt_model, tools = [get_result_sum])
    print(test(current_state=currentState, agent=agent))

