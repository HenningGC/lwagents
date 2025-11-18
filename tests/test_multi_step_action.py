"""
Test suite for multi_step_action in sequential execution mode (CHAIN).
Tests the sequential execution of multiple LLM agent steps.
"""

from lwagents.tools import Tool
from lwagents.agent import LLMAgent
from lwagents.models import create_model
from lwagents._types import ExecutionMode, StepResult
from lwagents.state import AgentState, get_global_agent_state, reset_global_agent_state
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


def run_all_tests():
    """Run all test cases."""
    print("=" * 60)
    print("Running Multi-Step Action Sequential Execution Tests")
    print("=" * 60)
    
    try:
        #test_sequential_multi_step_basic()
        test_sequential_multi_step_with_tools()
        # test_sequential_multi_step_with_context_passing()
        # test_sequential_multi_step_empty_steps()
        # test_sequential_multi_step_with_state_entry()
        
        print("\n" + "=" * 60)
        print("✓ All tests passed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        raise


if __name__ == "__main__":
    run_all_tests()
