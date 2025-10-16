from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, Field, SkipValidation
from typing_extensions import Self, override

from .agent import LLMAgent
from .state import GraphState


class NodeKind(str, Enum):
    START = "START"
    STATE = "STATE"
    TERMINAL = "TERMINAL"


class Node(BaseModel):
    node_name: str = Field(..., description="The Node Name")
    kind: NodeKind = Field(..., description="The kind of the Node")
    command: Optional[callable] = None
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Parameters for the command"
    )

    def connect(self, to_node: Self, edge: "Edge"):
        """
        Connects this node to another node using the given edge.

        Args:
            to_node (Node): The destination node.
            edge (Edge): The edge connecting the nodes.
            graph (Graph): The graph to which the connection belongs.
        """
        current_graph = BaseGraph.get_current_graph()
        current_graph.connect_edge(FROM=self, TO=to_node, WITH=edge)

    class Config:
        arbitrary_types_allowed = True

    def __hash__(self):
        return hash((self.node_name, self.kind))


class Edge(BaseModel):
    edge_name: str = Field(..., description="The Edge Name")
    condition: Optional[callable] = Field(
        None, description="A function that determines if the transition is valid"
    )
    parameters: Optional[Dict[str, Any]] = Field(
        default_factory=dict, description="Parameters for the function"
    )

    class Config:
        arbitrary_types_allowed = True


class DirectTraversal:
    """
    Wrapper class for direct traversal instructions.
    Holds the name of the target node for traversal.
    """

    def __init__(self, target_node_name: str):
        self.target_node_name = target_node_name


class GraphRequest(BaseModel):
    commands: Optional[List[SkipValidation[callable]]] = Field(
        None, description="The callback functions to execute"
    )
    parameters: Optional[Dict[str, Dict[str, Any]]] = Field(
        default_factory=dict, description="Parameters for the command"
    )
    result: Optional[Any] = Field(None, description="The result of the command")
    traversal: Optional[str] = Field(None, description="The traversal to the next node")
    update_additional_log_entries: Optional[Dict[str, Any]] = Field(
        default_factory=dict,
        description="Additional log entries to update with new values. The key is the name of the log entry and the value is the new value",
    )

    class Config:
        arbitrary_types_allowed = True


class GraphException(Exception):
    pass


class DirectTraversalRequest(BaseModel):
    traversal: str = Field(..., description="The traversal to the next node")


class BaseGraph:
    _current_graph = None  # Tracks the active graph context

    def __init__(self):
        self._graphDict = {}

    def __enter__(self):
        """
        Enter the context: set the current graph context.
        """
        BaseGraph._current_graph = self
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """
        Exit the context: clear the current graph context.
        """
        BaseGraph._current_graph = None

    @classmethod
    def get_current_graph(cls):
        """
        Retrieve the current graph in context.
        """
        if cls._current_graph is None:
            raise ValueError("No active graph context. Use 'with Graph() as graph:'")
        return cls._current_graph


@dataclass
class Graph(BaseGraph):

    def __init__(self, state=None):
        super().__init__()
        self._GraphState = state or GraphState([])

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

    def construct_graph(
        self,
        graph_dict: Dict["Node", List[Dict["Node", "Edge"]]],
        start_name: str,
        *args,
        **kwargs,
    ):
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
                raise TypeError(
                    f"Graph key {node} must be of type Node. Got {type(node)} instead."
                )

            if not isinstance(edges, list):
                raise TypeError(
                    f"Graph value for {node} must be of type List. Got {type(edges)} instead."
                )

            for edge in edges:
                if not (isinstance(edge, dict) and len(edge) == 1):
                    raise TypeError(
                        f"Each edge in the list for node {node} must be a dictionary with one (Node, Edge) pair. Got {edge} instead."
                    )

                connected_node, edge_obj = next(iter(edge.items()))

                if not isinstance(connected_node, Node):
                    raise TypeError(
                        f"Key in edge dictionary must be of type Node. Got {type(connected_node)} instead."
                    )
                if not isinstance(edge_obj, Edge):
                    raise TypeError(
                        f"Value in edge dictionary must be of type Edge. Got {type(edge_obj)} instead."
                    )

            if node.node_name == start_name:
                node.kind = "START"

            self._graphDict[node] = [
                (connected_node, edge_obj)
                for edge in edges
                for connected_node, edge_obj in edge.items()
            ]

    def run(
        self,
        start_node: Node,
        streaming=False,
        additional_log_entries: Dict = {},
        *args,
        **kwargs,
    ):
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
            raise GraphException(
                f"Chosen starting Node: {start_node.node_name} is not of type START"
            )

        current_node = start_node
        step_number = 1

        while current_node.kind != "TERMINAL" and current_node is not None:
            if streaming:
                print(
                    f"Current_Node: {current_node.node_name}, Kind: {current_node.kind}"
                )

            execution_result = None
            direct_traversal_request = None
            if current_node.command:
                execution_result = current_node.command(**current_node.parameters)
                if isinstance(execution_result, GraphRequest):
                    if execution_result.commands:
                        for command in execution_result.commands:
                            command(**execution_result.parameters)
                    if execution_result.update_additional_log_entries:
                        additional_log_entries.update(
                            execution_result.update_additional_log_entries
                        )
                    if execution_result.traversal:
                        direct_traversal_request = DirectTraversalRequest(
                            traversal=execution_result.traversal
                        )
                if streaming:
                    print(
                        f"{current_node.node_name} executed its command. Result: {execution_result}"
                    )

            log_entry = {
                "step_number": step_number,
                "node_name": current_node.node_name,
                "node_kind": current_node.kind,
                "command_result": execution_result,
                "transition": None,
                **additional_log_entries,
            }

            if streaming:
                print("📚 State Global History", self._GraphState.history)

            next_node = None

            if direct_traversal_request:
                target_node_name = direct_traversal_request.traversal
                if streaming:
                    print(f"➡️ Direct traversal to node: {target_node_name}")
                next_node = None
                for connected_node, edge in self._graphDict.get(current_node, []):
                    if connected_node.node_name == target_node_name:
                        next_node = connected_node
                        log_entry["transition"] = (
                            edge.edge_name,
                            connected_node.node_name,
                        )
                        break
                if not next_node:
                    raise GraphException(
                        f"🔴 Direct traversal failed: Node {target_node_name} not found 🔴"
                    )
            else:
                for connected_node, edge in self._graphDict.get(current_node, []):
                    if edge.condition:
                        if streaming:
                            print(f"Executing edge condition on {edge.edge_name}")
                        edge_result = (
                            edge.condition(**edge.parameters)
                            if edge.parameters
                            else edge.condition()
                        )

                        if streaming:
                            print(
                                f"Edge {edge.edge_name} condition returned {edge_result}"
                            )

                        if not isinstance(edge_result, bool):
                            raise GraphException("Edge condition must return a Bool")

                        if edge_result:
                            next_node = connected_node
                            log_entry["transition"] = (
                                edge.edge_name,
                                connected_node.node_name,
                            )
                            break
                    else:
                        next_node = connected_node
                        log_entry["transition"] = (
                            edge.edge_name,
                            connected_node.node_name,
                        )
                        break

            self._GraphState.update_state(**log_entry)
            if streaming:
                if next_node:
                    print(
                        f"Traversing to Node: {next_node.node_name} through Edge: {edge.edge_name}"
                    )

            if not next_node:
                raise GraphException(
                    f"No valid transition from node: {current_node.node_name}"
                )

            current_node = next_node
            step_number += 1

        if streaming:
            print(self._GraphState.history)
            print("Finished Graph Run")
