from lwagents import *
from lwagents.state import get_global_agent_state, reset_global_agent_state
from dotenv import load_dotenv
import os

@Tool
def get_result_sum(val1: int,val2: int):
    return val1+val2

def get_sum(agent):
    result = agent.action([{"role": "system", "content":"You are an helpful assistant that uses his tools at their disposal"},
                            {"role": "user", "content":"Use the get_result_sum tool to sum 300+140"}], model_name="gpt-5-mini" ,temperature=0.8)
    
    # Access the global agent state to see what agents have done
    global_state = get_global_agent_state()
    print(f"Global agent actions so far: {len(global_state.history)} actions")
    print(graphState.get_last_entry())
    return result

def get_division():
    return 8/2

def search_internet():
    return "RESULTS HAVE BEEN VERIFIED"


def test_router(agent):
    global_state = get_global_agent_state()
    prompt =[{"role": "system", "content": "You are an agent router and you decide which node to travel to next based on the task and results thus far. Your next answer must only return the node name."},
     {"role": "user", "content": f"You have the following nodes at your disposal: get_division, search_internet, get_sum, end. You have to decide the sequence of nodes to travel to based on based on this objective: get sum, then divide and search on the internet. These are the results thus far: {global_state.history}"}]
    result = agent.action(prompt = prompt)
    
    # You can also access global state here to see all agent activities
    #print(f"Router agent executed. Total agent actions: {len(global_state.history)}")
    
    return GraphRequest(result=result.content, traversal=result.content)

if __name__ == "__main__":
    load_dotenv()
    
    # Reset global state at the beginning (optional, good for testing)
    reset_global_agent_state()

    gpt_model = create_model(model_type="gpt",api_key = os.getenv('OPENAI_API_KEY'))


    tool_agent = LLMAgent(name="tool_agent", llm_model= gpt_model, tools = [get_result_sum])
    router_agent = LLMAgent(name="router_agent", llm_model= gpt_model)

    supervisor_node = Node(node_name='supervisor',
                           kind='START',
                           command=test_router,
                           parameters={"agent": router_agent})
    
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
    global graphState
    graphState = GraphState()  # This is your local graph state
    
    with Graph(state=graphState) as graph:
        supervisor_node.connect(to_node=get_sum_node, edge=edge)
        supervisor_node.connect(to_node=get_division_node, edge=edge)
        supervisor_node.connect(to_node=search_internet_node, edge=edge)
        supervisor_node.connect(to_node=end_node, edge=edge)
        get_sum_node.connect(to_node=supervisor_node, edge=edge)
        get_division_node.connect(to_node=supervisor_node, edge=edge)
        search_internet_node.connect(to_node=supervisor_node, edge=edge)

        graph.run(start_node=supervisor_node, streaming=True, additional_log_entries={"test":"test"})

    # Print both local graph state and global agent state
    print("\n=== GRAPH STATE HISTORY ===")
    graphState.print_history()
    
    print("\n=== GLOBAL AGENT STATE HISTORY ===")
    global_agent_state = get_global_agent_state()
    global_agent_state.print_history()
    
    print(f"\nTotal agent actions performed: {len(global_agent_state.history)}")