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

class DirectTraversal:
    """
    Wrapper class for direct traversal instructions.
    Holds the name of the target node for traversal.
    """
    def __init__(self, target_node_name: str):
        self.target_node_name = target_node_name

def node_router(func):
    """
    Decorator to wrap the result of a function with DirectTraversal.

    Args:
        func (callable): The function to decorate.

    Returns:
        callable: The wrapped function.
    """
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        if isinstance(result, str):
            return DirectTraversal(target_node_name=result)
        return result
    return wrapper


class GraphException(Exception):
    pass

@dataclass
class Graph:

    def __init__(self, state):
        self._graphDict = {}
        self._MainState = state or MainState(None)

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
            if not isinstance(node, Node):
                raise TypeError(f"Graph key {node} must be of type Node. Got {type(node)} instead.")

            if not isinstance(edges, list):
                raise TypeError(f"Graph value for {node} must be of type List. Got {type(edges)} instead.")

            for edge in edges:
                if not (isinstance(edge, dict) and len(edge) == 1):
                    raise TypeError(f"Each edge in the list for node {node} must be a dictionary with one (Node, Edge) pair. Got {edge} instead.")

                connected_node, edge_obj = next(iter(edge.items()))

                if not isinstance(connected_node, Node):
                    raise TypeError(f"Key in edge dictionary must be of type Node. Got {type(connected_node)} instead.")
                if not isinstance(edge_obj, Edge):
                    raise TypeError(f"Value in edge dictionary must be of type Edge. Got {type(edge_obj)} instead.")

            
            if node.node_name == start_name:
                node.kind = "START"

            self._graphDict[node] = [(connected_node, edge_obj) for edge in edges for connected_node, edge_obj in edge.items()]
        


    @override
    def run(self, start_node, streaming=False, *args, **kwargs):
        """
        Executes the graph starting from the given start_node, with structured logging for each step.

        Args:
            start_node (Node): The starting node of the graph.
            streaming (bool): If True, print execution details in real-time.

        Raises:
            GraphException: If no valid transition is found or if conditions return non-boolean values.
        """
        if streaming:
            print("Executing Graph...")

        if start_node.kind != "START":
            raise GraphException(f"Chosen starting Node: {start_node.node_name} is not of type START")

        current_node = start_node
        step_number = 1 

        while current_node.kind != "TERMINAL" and current_node is not None:
            if streaming:
                print(f"Current_Node: {current_node.node_name}, Kind: {current_node.kind}")

            result = None
            if current_node.command:
                result = current_node.command(**current_node.parameters)
                if streaming:
                    print(f"{current_node.node_name} executed its command. Result: {result}")

            log_entry = {
                "step_number": step_number,
                "node_name": current_node.node_name,
                "node_kind": current_node.kind,
                "command_result": result,
                "transition": None 
            }

            self._MainState.update_state(log_entry)

            if streaming:
                print("State Global History", self._MainState.history)

            next_node = None
            if isinstance(result, DirectTraversal):
                target_node_name = result.target_node_name
                if streaming:
                    print(f"Direct traversal to node: {target_node_name}")
                next_node = None
                for connected_node, edge in self._graphDict.get(current_node, []):
                    if connected_node.node_name == target_node_name:
                        next_node = connected_node
                        log_entry["transition"] = (edge.edge_name, connected_node.node_name)
                        break
                if not next_node:
                    raise GraphException(f"Direct traversal failed: Node {target_node_name} not found")
            else:
                for connected_node, edge in self._graphDict.get(current_node, []):
                    if edge.condition:
                        if streaming:
                            print(f"Executing edge condition on {edge.edge_name}")
                        edge_result = edge.condition(**edge.parameters) if edge.parameters else edge.condition()

                        if streaming:
                            print(f"Edge {edge.edge_name} condition returned {edge_result}")

                        if not isinstance(edge_result, bool):
                            raise GraphException("Edge condition must return a Bool")

                        if edge_result:
                            next_node = connected_node
                            log_entry["transition"] = (edge.edge_name, connected_node.node_name)
                            break
                    else:
                        next_node = connected_node
                        log_entry["transition"] = (edge.edge_name, connected_node.node_name)
                        break

            if streaming:
                if next_node:
                    print(f"Traversing to Node: {next_node.node_name} through Edge: {edge.edge_name}")

            if not next_node:
                raise GraphException(f"No valid transition from node: {current_node.node_name}")

            current_node = next_node
            step_number += 1

        if streaming:
            print(self._MainState.history)
            print("Finished Graph Run")





        
                
            
                