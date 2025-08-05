"""A demo fo a simple routing agent with a logging decorator that can be turned on or off."""

import functools
import json
from openai import OpenAI
import inspect
import os
import time
from datetime import datetime
from pathlib import Path
from rich.console import Console
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
console = Console()


console.print(
    "\n[cyan bold]A basic routing agent with a logging decorator with on/off.[/]"
)


def log_tool_usage(enable_logging=True, log_file="./results/tracing_detailed.csv"):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if enable_logging:
                # Capture metadata
                timestamp = datetime.now().isoformat()
                function_name = func.__name__
                module_name = func.__module__

                # Get caller information
                frame = inspect.currentframe().f_back
                caller_filename = frame.f_code.co_filename
                caller_line = frame.f_lineno
                caller_function = frame.f_code.co_name

                # Get function signature and source file
                try:
                    source_file = inspect.getfile(func)
                    source_line = inspect.findsource(func)[1]
                except (OSError, TypeError):
                    source_file = "unknown"
                    source_line = "unknown"

                # Process ID and thread ID
                pid = os.getpid()
                thread_id = None
                try:
                    import threading

                    thread_id = threading.current_thread().ident
                except ImportError:
                    thread_id = "unknown"

                # Start timing
                start_time = time.perf_counter()

                # Console logging
                console.print(
                    f"[green italic]LOG: Tool '{function_name}' called at {timestamp}"
                )
                console.print(
                    f"[green italic]LOG: Called from {Path(caller_filename).name}:{caller_line} in {caller_function}()"
                )
                console.print(f"[green italic]LOG: Args: {args}, Kwargs: {kwargs}[/]")

            # Execute the function
            try:
                result = func(*args, **kwargs)
                success = True
                error_msg = None
            except Exception as e:
                result = None
                success = False
                error_msg = str(e)
                if enable_logging:
                    console.print(
                        f"[red]LOG: Tool '{function_name}' failed with error: {error_msg}[/]"
                    )
                raise

            if enable_logging:
                # End timing
                end_time = time.perf_counter()
                execution_time = end_time - start_time

                # Console logging for result
                if success:
                    console.print(
                        f"[green italic]LOG: Tool '{function_name}' completed in {execution_time:.4f}s"
                    )
                    console.print(f"[green italic]LOG: Returned: {result}[/]")

                # Prepare CSV data
                csv_data = {
                    "timestamp": timestamp,
                    "function_name": function_name,
                    "module_name": module_name,
                    "source_file": Path(source_file).name,
                    "source_line": source_line,
                    "caller_file": Path(caller_filename).name,
                    "caller_line": caller_line,
                    "caller_function": caller_function,
                    "pid": pid,
                    "thread_id": thread_id,
                    "execution_time": f"{execution_time:.4f}",
                    "success": success,
                    "args": str(args),
                    "kwargs": str(kwargs),
                    "result": str(result) if success else "ERROR",
                    "error_msg": error_msg or "",
                }

                # Write to CSV
                try:
                    # Create directory if it doesn't exist
                    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

                    # Check if file exists to write header
                    write_header = not Path(log_file).exists()

                    with open(log_file, "a", encoding="utf-8") as f:
                        if write_header:
                            # Write CSV header
                            headers = list(csv_data.keys())
                            f.write("|".join(headers) + "\n")

                        # Write data
                        values = [
                            str(v).replace("|", "&#124;") for v in csv_data.values()
                        ]  # Escape pipe characters
                        f.write("|".join(values) + "\n")

                except Exception as e:
                    console.print(f"[red]LOG: Failed to write to log file: {e}[/]")

            return result

        return wrapper

    return decorator


# Define the three tools with logging enabled
@log_tool_usage()
def tool_a(data: str) -> str:
    """Tool A: Processes data with method A"""
    return f"Tool A processed: {data}"


@log_tool_usage()
def tool_b(data: str) -> str:
    """Tool B: Processes data with method B"""
    return f"Tool B processed: {data}"


@log_tool_usage()
def tool_c(data: str) -> str:
    """Tool C: Processes data with method C"""
    return f"Tool C processed: {data}"


# Simple Agent class
class SimpleAgent:

    def __init__(self, model="gpt-4o-mini"):
        self.client = OpenAI()
        self.model = model
        self.tools = {"tool_a": tool_a, "tool_b": tool_b, "tool_c": tool_c}
        self.tool_definitions = [
            {
                "type": "function",
                "function": {
                    "name": "tool_a",
                    "description": "Tool A: Processes data with method A",
                    "parameters": {
                        "type": "object",
                        "properties": {"data": {"type": "string"}},
                        "required": ["data"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_b",
                    "description": "Tool B: Processes data with method B",
                    "parameters": {
                        "type": "object",
                        "properties": {"data": {"type": "string"}},
                        "required": ["data"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "tool_c",
                    "description": "Tool C: Processes data with method C",
                    "parameters": {
                        "type": "object",
                        "properties": {"data": {"type": "string"}},
                        "required": ["data"],
                    },
                },
            },
        ]

    def run(self, user_input: str):
        """Run the agent with a single argument"""
        print(f"Agent received input: {user_input}")

        # Create system prompt for routing
        system_prompt = """
        You are a simple routing agent. Based on the user's input:
        - If input contains 'alpha' or 'A', use tool_a
        - If input contains 'beta' or 'B', use tool_b  
        - If input contains 'gamma' or 'C', use tool_c
        - Otherwise, default to tool_a
        
        Always use exactly one tool and pass the user's input as the data parameter.
        """

        # Call OpenAI API
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_input},
            ],
            tools=self.tool_definitions,
            tool_choice="auto",
        )

        # Handle tool calls
        if response.choices[0].message.tool_calls:
            tool_call = response.choices[0].message.tool_calls[0]
            tool_name = tool_call.function.name
            args = json.loads(tool_call.function.arguments)

            # Call the appropriate tool
            if tool_name in self.tools:
                result = self.tools[tool_name](args["data"])
                print(f"Agent response: {result}")
                return result
        else:
            print(f"Agent response: {response.choices[0].message.content}")
            return response.choices[0].message.content


# Create the agent
agent = SimpleAgent()


# Main function to run the agent
def run_agent(user_input: str):
    """Run the agent with a single argument"""
    return agent.run(user_input)


# Example usage
if __name__ == "__main__":
    # Test cases
    test_inputs = [
        "Process this alpha data",
        "Handle beta information",
        "Work with gamma values",
        "Random input",
    ]

    for test_input in test_inputs:
        print(f"\n{'='*50}")
        run_agent(test_input)
