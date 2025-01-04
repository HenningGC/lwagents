from state import MainState
from agent import LLMAgent
from dataclasses import dataclass
from pydantic import BaseModel, Field, SkipValidation
from typing import Literal, Optional, Dict, List, Optional, Tuple, Any
from typing_extensions import Self, override
from enum import Enum

class NodeKind(str, Enum):
    START = "START"
    STATE = "STATE"
    TERMINAL = "TERMINAL"

class Node(BaseModel):
    node_name: str = Field(..., description="The Node Name")
    kind: NodeKind = Field(..., description="The kind of the Node")
    command: Optional[callable] = None
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Parameters for the command")

    def connect(self, to_node: Self, edge: 'Edge', graph: 'Graph'):
        """
        Connects this node to another node using the given edge.

        Args:
            to_node (Node): The destination node.
            edge (Edge): The edge connecting the nodes.
            graph (Graph): The graph to which the connection belongs.
        """
        graph.connect_edge(FROM=self, TO=to_node, WITH=edge)

    class Config:
        arbitrary_types_allowed = True

    def __hash__(self):
        # Use a tuple of the field values to generate a hash
        return hash((self.node_name, self.kind))


class Edge(BaseModel):
    edge_name: str = Field(..., description="The Edge Name")
    condition: Optional[callable] = Field(None, description="A function that determines if the transition is valid")
    parameters: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Parameters for the function")

    class Config:
        arbitrary_types_allowed = True

class GraphException(Exception):
    pass

@dataclass
class Graph:

    def __init__(self):
        self._graphDict = {}
        self._MainState = MainState(None)

    def connect_edge(self, FROM: Node, TO: Node, WITH: Edge):
        """
        Connects two nodes with an edge in the graph.

        Args:
            FROM (Node): The starting node.
            TO (Node): The destination node.
            WITH (Edge): The edge connecting the nodes.
        """
        if FROM not in self._graphDict:
            self._graphDict[FROM] = []
        self._graphDict[FROM].append((TO, WITH))

    def get_edges(self, node: Node) -> List[Tuple[Node, Edge]]:
        """
        Retrieves edges connected to the given node.

        Args:
            node (Node): The node to query.

        Returns:
            List[Tuple[Node, Edge]]: A list of connected nodes and their edges.
        """
        return self._graphDict.get(node, [])

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

    @override
    def run(self, start_node, *args, **kwargs):
        print("Executing Graph...")
        if start_node.kind != "START":
            raise GraphException(f"Chosen starting Node: {start_node} is not of type START")
        
        current_node = start_node

        while current_node.kind != "TERMINAL" and current_node != None:
            print("Current_Node:", current_node, "Kind:",current_node.kind)
            result = None
            if current_node.command:
                result = current_node.command(**current_node.parameters)

            self._MainState.update_state(str(result))

            next_node = None
            for connected_node, edge in self._graphDict.get(current_node, []):
                if edge.condition:
                    if edge.parameters:  # Execute the edge's function
                        if edge.conditions(**edge.parameters):
                            next_node = connected_node
                            break
                    else:
                        if edge.conditions(**edge.parameters):
                            next_node = connected_node
                            break
                else:
                    next_node = connected_node
                    break

            if not next_node:
                raise GraphException(f"No valid transition from node: {current_node.node_name}")

            # Transition to the next node
            current_node = next_node
        print("Finished Graph Run")
        print(self._MainState.history)



        
                
            
                