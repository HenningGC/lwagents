"""
Test suite for multi_step_action in sequential execution mode (CHAIN).
Tests the sequential execution of multiple LLM agent steps.
"""

from lwagents.tools import Tool
from lwagents.agent import LLMAgent
from lwagents.models import create_model
from lwagents._types import ExecutionMode, StepResult
from lwagents.state import AgentState, GraphState, get_global_agent_state, reset_global_agent_state
from lwagents.graph import Graph, Node, Edge, GraphRequest
from dotenv import load_dotenv
import os


# Define test tools
@Tool
def add_numbers(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b


@Tool
def multiply_numbers(a: int, b: int) -> int:
    """Multiply two numbers together."""
    return a * b


@Tool
def get_greeting(name: str) -> str:
    """Generate a greeting for a person."""
    return f"Hello, {name}!"


def test_sequential_multi_step_basic():
    """Test basic sequential multi-step execution without tools."""
    print("\n=== Test 1: Basic Sequential Multi-Step ===")
    
    load_dotenv()
    reset_global_agent_state()
    
    # Create model and agent
    gpt_model = create_model(
        model_type="openai",
        instance_params={"api_key": os.getenv("OPENAI_API_KEY")}
    )
    
    agent = LLMAgent(
        name="test_agent",
        llm_model=gpt_model,
        state=AgentState()
    )
    
    # Define sequential steps
    steps = [
        {
            "step": "step_1",
            "model_params": {
                "model": "gpt-4o-mini",
                "instructions": "You are a helpful assistant.",
                "input": [
                    {"role": "user", "content": "What is 5 + 3? Just answer with the number."}
                ]
            }
        },
        {
            "step": "step_2",
            "model_params": {
                "model": "gpt-4o-mini",
                "instructions": "You are a helpful assistant.",
                "input": [
                    {"role": "user", "content": "What is 10 * 2? Just answer with the number."}
                ]
            }
        },
        {
            "step": "step_3",
            "model_params": {
                "model": "gpt-4o-mini",
                "instructions": "You are a helpful assistant.",
                "input": [
                    {"role": "user", "content": "What is the capital of France? Answer with just the city name."}
                ]
            }
        }
    ]
    
    # Execute multi-step action in sequential mode
    results = agent.multi_step_action(
        steps=steps,
        mode=ExecutionMode.CHAIN
    )
    
    # Verify results
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    assert all(isinstance(r, StepResult) for r in results), "All results should be StepResult instances"
    
    print(f"✓ Executed {len(results)} sequential steps")
    for i, result in enumerate(results, 1):
        print(f"  Step {i} ({result.step}): {result.response.content[:50]}...")
    
    # Check global state
    global_state = get_global_agent_state()
    print(f"✓ Global agent state has {len(global_state.history)} actions recorded")
    
    return results


def test_sequential_multi_step_with_tools():
    """Test sequential multi-step execution with tool calling."""
    print("\n=== Test 2: Sequential Multi-Step with Tools ===")
    
    load_dotenv()
    reset_global_agent_state()
    
    # Create model and agent with tools
    gpt_model = create_model(
        model_type="openai",
        instance_params={"api_key": os.getenv("OPENAI_API_KEY")}
    )
    
    agent = LLMAgent(
        name="tool_agent",
        llm_model=gpt_model,
        tools=[add_numbers, multiply_numbers, get_greeting],
        state=AgentState()
    )
    
    # Define sequential steps with tool usage
    steps = [
        {
            "step": "addition",
            "model_params": {
                "model": "gpt-4o-mini",
                "instructions": "You are a helpful assistant that uses tools to solve math problems.",
                "input": [
                    {"role": "user", "content": "Use the add_numbers tool to calculate 15 + 27"}
                ]
            }
        },
        {
            "step": "multiplication",
            "model_params": {
                "model": "gpt-4o-mini",
                "instructions": "You are a helpful assistant that uses tools to solve math problems.",
                "input": [
                    {"role": "user", "content": "Use the multiply_numbers tool to calculate 8 * 7"}
                ]
            }
        },
        {
            "step": "greeting",
            "model_params": {
                "model": "gpt-4o-mini",
                "instructions": "You are a helpful assistant that uses tools.",
                "input": [
                    {"role": "user", "content": "Use the get_greeting tool to greet Alice"}
                ]
            }
        }
    ]
    
    # Execute multi-step action in sequential mode
    results = agent.multi_step_action(
        steps=steps,
        mode=ExecutionMode.CHAIN
    )
    
    # Verify results
    assert len(results) == 3, f"Expected 3 results, got {len(results)}"
    
    print(f"✓ Executed {len(results)} sequential steps with tools")
    for i, result in enumerate(results, 1):
        tools_used = result.response.tools_used or []
        print(f"  Step {i} ({result.step}):")
        print(f"    Tools used: {tools_used}")
        print(f"    Result: {result.response.content[:80]}...")
    
    # Check global state
    global_state = get_global_agent_state()
    print(f"✓ Global agent state has {len(global_state.history)} actions recorded")
    
    return results


def test_sequential_multi_step_with_context_passing():
    """Test sequential multi-step where later steps reference earlier results."""
    print("\n=== Test 3: Sequential Multi-Step with Context Passing ===")
    
    load_dotenv()
    reset_global_agent_state()
    
    # Create model and agent
    gpt_model = create_model(
        model_type="openai",
        instance_params={"api_key": os.getenv("OPENAI_API_KEY")}
    )
    
    agent = LLMAgent(
        name="context_agent",
        llm_model=gpt_model,
        tools=[add_numbers],
        state=AgentState()
    )
    
    # Step 1: Get a number
    step_1_results = agent.multi_step_action(
        steps=[{
            "step": "get_number",
            "model_params": {
                "model": "gpt-4o-mini",
                "instructions": "You are a helpful assistant.",
                "input": [
                    {"role": "user", "content": "Use add_numbers tool to add 10 and 20"}
                ]
            }
        }],
        mode=ExecutionMode.CHAIN
    )
    
    first_result = step_1_results[0].response.content
    print(f"✓ Step 1 result: {first_result}")
    
    # Step 2: Use the previous result
    step_2_results = agent.multi_step_action(
        steps=[{
            "step": "use_previous_result",
            "model_params": {
                "model": "gpt-4o-mini",
                "instructions": "You are a helpful assistant.",
                "input": [
                    {"role": "user", "content": f"The previous calculation gave us: {first_result}. Now use add_numbers to add 5 to the result 30."}
                ]
            }
        }],
        mode=ExecutionMode.CHAIN
    )
    
    second_result = step_2_results[0].response.content
    print(f"✓ Step 2 result: {second_result}")
    
    # Check global state captured both steps
    global_state = get_global_agent_state()
    print(f"✓ Global agent state has {len(global_state.history)} actions recorded")
    
    return step_1_results + step_2_results


def test_sequential_multi_step_empty_steps():
    """Test that multi_step_action handles empty steps list."""
    print("\n=== Test 4: Sequential Multi-Step with Empty Steps ===")
    
    load_dotenv()
    reset_global_agent_state()
    
    gpt_model = create_model(
        model_type="openai",
        instance_params={"api_key": os.getenv("OPENAI_API_KEY")}
    )
    
    agent = LLMAgent(
        name="empty_agent",
        llm_model=gpt_model,
        state=AgentState()
    )
    
    # Execute with empty steps
    results = agent.multi_step_action(
        steps=[],
        mode=ExecutionMode.CHAIN
    )
    
    assert len(results) == 0, "Expected 0 results for empty steps"
    print("✓ Empty steps handled correctly")
    
    return results


def test_sequential_multi_step_with_state_entry():
    """Test sequential multi-step with state entry tracking."""
    print("\n=== Test 5: Sequential Multi-Step with State Entry ===")
    
    load_dotenv()
    reset_global_agent_state()
    
    gpt_model = create_model(
        model_type="openai",
        instance_params={"api_key": os.getenv("OPENAI_API_KEY")}
    )
    
    agent = LLMAgent(
        name="state_agent",
        llm_model=gpt_model,
        state=AgentState()
    )
    
    # Define steps with state entry metadata
    steps = [
        {
            "step": "analysis",
            "state_entry": {"task_type": "analysis", "priority": "high"},
            "model_params": {
                "model": "gpt-4o-mini",
                "instructions": "You are an analytical assistant.",
                "input": [
                    {"role": "user", "content": "Analyze the number 42. Is it even or odd?"}
                ]
            }
        },
        {
            "step": "summary",
            "state_entry": {"task_type": "summary", "priority": "medium"},
            "model_params": {
                "model": "gpt-4o-mini",
                "instructions": "You are a summarizing assistant.",
                "input": [
                    {"role": "user", "content": "Summarize: Python is a programming language."}
                ]
            }
        }
    ]
    
    results = agent.multi_step_action(
        steps=steps,
        mode=ExecutionMode.CHAIN
    )
    
    assert len(results) == 2, f"Expected 2 results, got {len(results)}"
    print(f"✓ Executed {len(results)} steps with state entry metadata")
    
    for i, result in enumerate(results, 1):
        print(f"  Step {i} ({result.step}): {result.response.content[:50]}...")
    
    return results


def test_graph_execution_basic():
    """Test graph execution mode with a simple linear graph."""
    print("\n=== Test 6: Graph Execution Mode - Basic ===")
    
    load_dotenv()
    reset_global_agent_state()
    
    # Create model and agent
    gpt_model = create_model(
        model_type="openai",
        instance_params={"api_key": os.getenv("OPENAI_API_KEY")}
    )
    
    agent = LLMAgent(
        name="graph_agent",
        llm_model=gpt_model,
        tools=[add_numbers],
        state=AgentState()
    )
    
    # Define node commands
    def step1_command(agent):
        model_params = {
            "model": "gpt-4o-mini",
            "instructions": "You are a helpful assistant.",
            "input": [
                {"role": "user", "content": "Use add_numbers tool to add 5 and 10"}
            ]
        }
        result = agent.action(model_params=model_params)
        print(f"  Step 1 executed: {result.content}")
        return result
    
    def step2_command(agent):
        model_params = {
            "model": "gpt-4o-mini",
            "instructions": "You are a helpful assistant.",
            "input": [
                {"role": "user", "content": "Use add_numbers tool to add 20 and 30"}
            ]
        }
        result = agent.action(model_params=model_params)
        print(f"  Step 2 executed: {result.content}")
        return result
    
    # Create graph
    graph_state = GraphState()
    
    with Graph(state=graph_state) as graph:
        start_node = Node(
            node_name="start",
            kind="START",
            command=step1_command,
            parameters={"agent": agent}
        )
        
        process_node = Node(
            node_name="process",
            kind="STATE",
            command=step2_command,
            parameters={"agent": agent}
        )
        
        end_node = Node(
            node_name="end",
            kind="TERMINAL"
        )
        
        edge = Edge(edge_name="default", condition=None)
        
        start_node.connect(to_node=process_node, edge=edge)
        process_node.connect(to_node=end_node, edge=edge)
        
        # Execute in graph mode
        results = agent.multi_step_action(
            mode=ExecutionMode.GRAPH,
            graph=graph,
            streaming=True
        )
        
        print(f"✓ Graph execution completed with {len(results)} results")
        
        # Check graph state
        print(f"✓ Graph state has {len(graph_state.history)} entries")
        
    return results


def test_graph_execution_with_router():
    """Test graph execution mode with a router pattern."""
    print("\n=== Test 7: Graph Execution Mode - Router Pattern ===")
    
    load_dotenv()
    reset_global_agent_state()
    
    # Create models and agents
    gpt_model = create_model(
        model_type="openai",
        instance_params={"api_key": os.getenv("OPENAI_API_KEY")}
    )
    
    tool_agent = LLMAgent(
        name="tool_agent",
        llm_model=gpt_model,
        tools=[add_numbers, multiply_numbers],
        state=AgentState()
    )
    
    router_agent = LLMAgent(
        name="router_agent",
        llm_model=gpt_model,
        state=AgentState()
    )
    
    # Define node commands
    execution_count = {"count": 0}
    
    def router_command(agent):
        execution_count["count"] += 1
        count = execution_count["count"]
        
        if count == 1:
            next_node = "add_task"
        elif count == 2:
            next_node = "multiply_task"
        else:
            next_node = "end"
        
        print(f"  Router deciding: going to {next_node}")
        return GraphRequest(result=f"Routing to {next_node}", traversal=next_node)
    
    def add_command(agent):
        model_params = {
            "model": "gpt-4o-mini",
            "instructions": "You are a helpful math assistant.",
            "input": [
                {"role": "user", "content": "Use add_numbers to add 100 and 200"}
            ]
        }
        result = agent.action(model_params=model_params)
        print(f"  Addition task: {result.content}")
        return result
    
    def multiply_command(agent):
        model_params = {
            "model": "gpt-4o-mini",
            "instructions": "You are a helpful math assistant.",
            "input": [
                {"role": "user", "content": "Use multiply_numbers to multiply 7 and 8"}
            ]
        }
        result = agent.action(model_params=model_params)
        print(f"  Multiplication task: {result.content}")
        return result
    
    # Create graph with router pattern
    graph_state = GraphState()
    
    with Graph(state=graph_state) as graph:
        router_node = Node(
            node_name="router",
            kind="START",
            command=router_command,
            parameters={"agent": router_agent}
        )
        
        add_node = Node(
            node_name="add_task",
            kind="STATE",
            command=add_command,
            parameters={"agent": tool_agent}
        )
        
        multiply_node = Node(
            node_name="multiply_task",
            kind="STATE",
            command=multiply_command,
            parameters={"agent": tool_agent}
        )
        
        end_node = Node(
            node_name="end",
            kind="TERMINAL"
        )
        
        edge = Edge(edge_name="route", condition=None)
        
        # Connect router to all possible destinations
        router_node.connect(to_node=add_node, edge=edge)
        router_node.connect(to_node=multiply_node, edge=edge)
        router_node.connect(to_node=end_node, edge=edge)
        
        # Connect tasks back to router
        add_node.connect(to_node=router_node, edge=edge)
        multiply_node.connect(to_node=router_node, edge=edge)
        
        # Define steps
        steps = [
            {"step": "route_and_execute_1"},
            {"step": "route_and_execute_2"}
        ]
        
        # Execute in graph mode
        results = tool_agent.multi_step_action(
            steps=steps,
            mode=ExecutionMode.GRAPH,
            graph=graph,
            streaming=True
        )
        
        print(f"✓ Router graph execution completed")
        print(f"✓ Graph state has {len(graph_state.history)} entries")
        
        # Print graph history
        print("\n  Graph execution history:")
        for i, entry in enumerate(graph_state.history, 1):
            print(f"    {i}. {entry.get('node_name')} -> {entry.get('transition')}")
    
    return results


def test_graph_execution_with_conditions():
    """Test graph execution mode with conditional edges."""
    print("\n=== Test 8: Graph Execution Mode - Conditional Edges ===")
    
    load_dotenv()
    reset_global_agent_state()
    
    gpt_model = create_model(
        model_type="openai",
        instance_params={"api_key": os.getenv("OPENAI_API_KEY")}
    )
    
    agent = LLMAgent(
        name="conditional_agent",
        llm_model=gpt_model,
        tools=[add_numbers],
        state=AgentState()
    )
    
    # Track execution path
    execution_path = []
    
    def start_command():
        execution_path.append("start")
        print("  Start node executed")
        return "started"
    
    def task_a_command(agent):
        execution_path.append("task_a")
        model_params = {
            "model": "gpt-4o-mini",
            "instructions": "You are a helpful assistant.",
            "input": [
                {"role": "user", "content": "Use add_numbers to add 1 and 1"}
            ]
        }
        result = agent.action(model_params=model_params)
        print(f"  Task A executed: {result.content}")
        return result
    
    def task_b_command():
        execution_path.append("task_b")
        print("  Task B executed")
        return "task_b_result"
    
    # Condition functions
    def go_to_task_a():
        print("  Condition: Going to Task A")
        return True
    
    def go_to_task_b():
        print("  Condition: Going to Task B")
        return False  # This won't be taken
    
    # Create graph
    graph_state = GraphState()
    
    with Graph(state=graph_state) as graph:
        start_node = Node(
            node_name="start",
            kind="START",
            command=start_command
        )
        
        task_a_node = Node(
            node_name="task_a",
            kind="STATE",
            command=task_a_command,
            parameters={"agent": agent}
        )
        
        task_b_node = Node(
            node_name="task_b",
            kind="STATE",
            command=task_b_command
        )
        
        end_node = Node(
            node_name="end",
            kind="TERMINAL"
        )
        
        edge_to_a = Edge(edge_name="to_task_a", condition=go_to_task_a)
        edge_to_b = Edge(edge_name="to_task_b", condition=go_to_task_b)
        edge_to_end = Edge(edge_name="to_end", condition=None)
        
        start_node.connect(to_node=task_a_node, edge=edge_to_a)
        start_node.connect(to_node=task_b_node, edge=edge_to_b)
        task_a_node.connect(to_node=end_node, edge=edge_to_end)
        task_b_node.connect(to_node=end_node, edge=edge_to_end)
        
        steps = [{"step": "conditional_execution"}]
        
        results = agent.multi_step_action(
            steps=steps,
            mode=ExecutionMode.GRAPH,
            graph=graph,
            streaming=True
        )
        
        print(f"✓ Conditional graph executed")
        print(f"  Execution path: {' -> '.join(execution_path)}")
        assert "task_a" in execution_path, "Task A should have been executed"
        print(f"✓ Correct path taken based on conditions")
    
    return results


def test_graph_execution_error_handling():
    """Test that graph mode raises error when graph is not provided."""
    print("\n=== Test 9: Graph Execution Mode - Error Handling ===")
    
    load_dotenv()
    reset_global_agent_state()
    
    gpt_model = create_model(
        model_type="openai",
        instance_params={"api_key": os.getenv("OPENAI_API_KEY")}
    )
    
    agent = LLMAgent(
        name="error_agent",
        llm_model=gpt_model,
        state=AgentState()
    )
    
    steps = [{"step": "test_step"}]
    
    try:
        # Try to execute in graph mode without providing a graph
        results = agent.multi_step_action(
            steps=steps,
            mode=ExecutionMode.GRAPH,
            graph=None  # This should raise an error
        )
        print("✗ Should have raised ValueError")
        assert False, "Expected ValueError to be raised"
    except ValueError as e:
        print(f"✓ Correctly raised ValueError: {e}")
        assert "Graph must be provided" in str(e)
    
    return None


def run_all_tests():
    """Run all test cases."""
    print("=" * 60)
    print("Running Multi-Step Action Tests")
    print("=" * 60)
    
    try:
        # Sequential execution tests
        print("\n" + "=" * 60)
        print("SEQUENTIAL EXECUTION MODE TESTS")
        print("=" * 60)
        test_sequential_multi_step_basic()
        test_sequential_multi_step_with_tools()
        test_sequential_multi_step_with_context_passing()
        test_sequential_multi_step_empty_steps()
        test_sequential_multi_step_with_state_entry()
        
        # Graph execution tests
        print("\n" + "=" * 60)
        print("GRAPH EXECUTION MODE TESTS")
        print("=" * 60)
        test_graph_execution_basic()
        test_graph_execution_with_router()
        test_graph_execution_with_conditions()
        test_graph_execution_error_handling()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    run_all_tests()
