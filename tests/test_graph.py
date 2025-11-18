from lwagents.graph import Graph, Node, Edge, NodeKind

# Define some example command functions
def start_command(**kwargs):
    print("Starting the workflow")
    return "Started"

def process_command(**kwargs):
    print(f"Processing with params: {kwargs}")
    return "Processed"

def decision_command(**kwargs):
    print("Making a decision")
    return "Decision made"

def end_command(**kwargs):
    print("Ending the workflow")
    return "Ended"

# Define edge conditions
def always_true():
    return True

def check_condition(**kwargs):
    return kwargs.get("proceed", True)

# Create nodes
start_node = Node(
    node_name="start",
    kind=NodeKind.STATE,
    command=start_command,
    parameters={}
)

process_node = Node(
    node_name="process",
    kind=NodeKind.STATE,
    command=process_command,
    parameters={"data": "test_data", "mode": "fast"}
)

decision_node = Node(
    node_name="decision",
    kind=NodeKind.STATE,
    command=decision_command,
    parameters={}
)

end_node = Node(
    node_name="end",
    kind=NodeKind.TERMINAL,
    command=end_command,
    parameters={}
)

# Create edges
start_edge = Edge(edge_name="to_process", condition=always_true)
process_edge = Edge(edge_name="to_decision", condition=None)
decision_edge_success = Edge(edge_name="to_end", condition=check_condition, parameters={"proceed": True})

# Construct the graph dictionary
graph_dict = {
    start_node: [
        {process_node: start_edge}
    ],
    process_node: [
        {decision_node: process_edge}
    ],
    decision_node: [
        {end_node: decision_edge_success}
    ],
    end_node: []  # Terminal node has no outgoing edges
}

# Test it
graph = Graph()
graph.construct_graph(graph_dict, start_name="start")

print(graph._graphDict)
# Run the graph
#graph.run(start_node=start_node, streaming=True)