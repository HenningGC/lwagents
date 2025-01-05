from ..lwagents.state import MainState
from ..lwagents.agent import LLMAgent
from ..lwagents.tools import Tool
from ..lwagents.graph import *
from ..lwagents.models import LLMFactory
from dotenv import load_dotenv
import os


@Tool
def get_result_sum():
    return 5/342

def get_sum(agent):
    result = agent.action([{"role": "system", "content":"You are an helpful assistant that uses his tools at their disposal"},
                  {"role": "user", "content":"Use the get_result_sum tool"}])
    
    return result

def get_division():
    return 8/2

def search_internet():
    return "RESULTS HAVE BEEN VERIFIED"

@node_router
def test_router(agent, state):
    prompt =[{"role": "system", "content": "You are an agent router and you decide which node to travel to next based on the task and results thus far. Your next answer must only return the node name."},
     {"role": "user", "content": f"You have the following nodes at your disposal: get_division, search_internet, get_sum, end. You have to decide the sequence of nodes to travel to based on based on this objective: get sum, then divide and search on the internet. These are the results thus far: {state.history}"}]
    result = agent.action(prompt = prompt)

    return result

if __name__ == "__main__":
    load_dotenv()
    factory = LLMFactory()

    gpt_model = factory.create_model("gpt",openai_api_key = os.getenv('OPENAI_API_KEY'))

    MainState = MainState([])
    tool_agent = LLMAgent(llm_model= gpt_model, tools = [get_result_sum])
    router_agent = LLMAgent(llm_model= gpt_model)


    supervisor_node = Node(node_name='supervisor',
                           kind='START',
                           command=test_router,
                           parameters={"agent": router_agent,
                                       "state":MainState})
    
    get_sum_node = Node(node_name='get_sum',
                     kind='STATE',
                     command=get_sum,
                     parameters={"agent":tool_agent})
    
    get_division_node = Node(node_name='get_division',
                             kind='STATE',
                             command=get_division)
    
    search_internet_node = Node(node_name='search_internet',
                                kind='STATE',
                                command=search_internet)
    
    end_node = Node(node_name='end',
                         kind='TERMINAL')
    

    edge = Edge(edge_name="edge1", condition=None)

    with Graph(state=MainState) as graph:
        supervisor_node.connect(to_node=get_sum_node, edge=edge)
        supervisor_node.connect(to_node=get_division_node, edge=edge)
        supervisor_node.connect(to_node=search_internet_node, edge=edge)
        supervisor_node.connect(to_node=end_node, edge=edge)
        get_sum_node.connect(to_node=supervisor_node, edge=edge)
        get_division_node.connect(to_node=supervisor_node, edge=edge)
        search_internet_node.connect(to_node=supervisor_node, edge=edge)

        graph.run(start_node=supervisor_node, streaming=True)

    MainState.print_history()
