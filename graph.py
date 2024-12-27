from state import MainState
from agent import LLMAgent
from models import Message
from dataclasses import dataclass
from pydantic import BaseModel, Field, SecretStr
from typing import Literal, Optional, Dict, List, Optional, Tuple

class Node(BaseModel):
    node_name: str = Field(..., description="The Node Name")
    kind: Literal["START", "STATE", "TERMINAL"] = Field(..., description="The kind of the Node")
    command: Optional[function] = Field(..., description="The function to run")

class Edge(BaseModel):
    edge_name: str = Field(..., description="The Edge Name")
    condition: Optional[function] = Field(..., description="Established Condition")

class GraphException(Exception):
    pass

@dataclass
class Graph:

    nodes_number: int
    edges_number: int

    def __init__(self):
        self._graphDict = {}
        self._MainState = MainState(None)

    def connect_edge(self, FROM: Node, TO: Node, WITH: Edge):
        if self._graphDict[FROM]:
            self._graphDict[FROM].append({TO,WITH})
        else:
            self._graphDict[FROM] = [{TO,WITH}]

    from typing import Dict, List, Tuple

    def construct_graph(self, graph_dict: Dict['Node', List[Dict['Node', 'Edge']]], start_name: str, *args, **kwargs):
        """
        Constructs a graph from the provided dictionary.

        Args:
            graph_dict (Dict[Node, List[Dict[Node, Edge]]]): A dictionary representing the graph.
                Keys are Node objects, and values are lists of dictionaries where each dictionary
                maps a connected Node object to an Edge object.
            start_name (str): The name of the start node.
            *args: Additional arguments (unused).
            **kwargs: Additional keyword arguments (unused).

        Raises:
            TypeError: If any key in the graph_dict is not a Node or if any value is not a list of dictionaries.
        """
        for node, edges in graph_dict.items():
            # Check that the key is a Node
            if not isinstance(node, Node):
                raise TypeError(f"Graph key {node} must be of type Node. Got {type(node)} instead.")

            # Check that the value is a list
            if not isinstance(edges, list):
                raise TypeError(f"Graph value for {node} must be of type List. Got {type(edges)} instead.")

            for edge in edges:
                # Check that each element in the list is a dictionary with one key-value pair
                if not (isinstance(edge, dict) and len(edge) == 1):
                    raise TypeError(f"Each edge in the list for node {node} must be a dictionary with one (Node, Edge) pair. Got {edge} instead.")

                # Extract the Node and Edge from the dictionary
                connected_node, edge_obj = next(iter(edge.items()))

                # Validate that the connected_node is a Node and edge_obj is an Edge
                if not isinstance(connected_node, Node):
                    raise TypeError(f"Key in edge dictionary must be of type Node. Got {type(connected_node)} instead.")
                if not isinstance(edge_obj, Edge):
                    raise TypeError(f"Value in edge dictionary must be of type Edge. Got {type(edge_obj)} instead.")

            # Check if the node is the start node and set its kind
            if node.node_name == start_name:
                node.kind = "START"

            # Store the processed edges in the graph's internal dictionary
            self._graphDict[node] = [(connected_node, edge_obj) for edge in edges for connected_node, edge_obj in edge.items()]


    def run(self, start_node, *args, **kwargs):
        
        if start_node.kind != "START":
            raise GraphException(f"Chosen starting Node: {start_node} is not of type START")
        
        current_node = start_node

        while current_node.kind != "TERMINAL" or current_node != None:

            if current_node:
                result = LLMAgent.run_command(current_node.command)

                self._MainState.add_message(Message(result))

                next_edge = current_node[result]

                if next_edge.condition:
                    result = next_edge.condition

                current_node = current_node[result]



        
                
            
                