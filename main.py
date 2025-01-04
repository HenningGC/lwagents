from state import MainState
from agent import LLMAgent
from tools import Tool
from graph import *
from models import LLMFactory
from dotenv import load_dotenv
import os


@Tool
def get_result_sum():
    return 5/342

def get_test_sum():
    return 5+10

@node_router
def test_router(agent):
    prompt =[{"role": "system", "content": "You are an agent router and you decide which node to travel to next based on the task and results thus far. Your next answer must only return the node name."},
     {"role": "user", "content": "I want you to travel to testSum node"}]
    result = agent.action(prompt = prompt)

    return result


def print_val(val):

    print(val)
    return

def test_condition(state):
    state_history = state.history
    if 'tool_call_id' in state_history[0]['command_result']:
        return True

    return False

if __name__ == "__main__":
    # Create a factory instance
    load_dotenv()
    factory = LLMFactory()
    # Create a GPT model
    gpt_model = factory.create_model("gpt",openai_api_key = os.getenv('OPENAI_API_KEY'))

    MainState = MainState([])
    tool_agent = LLMAgent(llm_model= gpt_model, tools = [get_result_sum])
    router_agent = LLMAgent(llm_model= gpt_model)


    supervisor_node = Node(node_name='supervisor',
                           kind='START',
                           command=test_router,
                           parameters={"agent": router_agent})
    
    test_node = Node(node_name='testSum',
                     kind='STATE',
                     command=get_test_sum)
    
    terminal_node = Node(node_name='test_terminal',
                         kind='TERMINAL')
    

    edge1 = Edge(edge_name="edge1", condition=None)
    edge2 = Edge(edge_name="edge2", condition=None)
    #edge_c = Edge(edge_name="edge_w_condition", condition=test_condition, parameters={"state":MainState})

    graph = Graph(state=MainState)

    supervisor_node.connect(to_node=test_node, edge=edge1, graph=graph)
    test_node.connect(to_node=terminal_node, edge=edge2, graph=graph)

    graph.run(start_node=supervisor_node, streaming=True)

