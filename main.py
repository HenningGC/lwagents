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

# Acts as a node
def test(current_state, agent):
    prompt =[{"role": "system", "content": "You are a helpful assistant. Use the supplied tools to assist the user."},
     {"role": "user", "content": "Use the get_result_sum function at your disposal"}]
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
    agent = LLMAgent(llm_model= gpt_model, tools = [get_result_sum])


    supervisor_node = Node(node_name='supervisor',
                           kind='START',
                           command=test,
                           parameters={"current_state":MainState,
                                       "agent": agent})
    
    test_node = Node(node_name='test',
                     kind='STATE',
                     command=print_val,
                     parameters={'val':'hello'})
    
    terminal_node = Node(node_name='test_terminal',
                         kind='TERMINAL')
    

    edge1 = Edge(edge_name="edge1", condition=None)
    edge2 = Edge(edge_name="edge2", condition=None)
    edge_c = Edge(edge_name="edge_w_condition", condition=test_condition, parameters={"state":MainState})

    graph = Graph(state=MainState)

    supervisor_node.connect(to_node=test_node, edge=edge_c, graph=graph)
    test_node.connect(to_node=terminal_node, edge=edge2, graph=graph)

    graph.run(start_node=supervisor_node, streaming=True)

