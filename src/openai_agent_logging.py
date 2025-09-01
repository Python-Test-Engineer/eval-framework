"""A demo fo a simple routing agent with a logging decorator that can be turned on or off."""

import functools
import os
import json
from openai import OpenAI
from dotenv import load_dotenv, find_dotenv
from rich.console import Console

console = Console()
load_dotenv(find_dotenv(), override=True)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

console.print(
    "\n[cyan bold]A basic routing agent with a logging decorator with on/off.[/]"
)


def log_tool_usage(enable_logging=True):

    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            if enable_logging:
                console.print(
                    f"[green italic]LOG: Tool '{func.__name__}' called with args: {args}, kwargs: {kwargs}"
                )
            result = func(*args, **kwargs)
            if enable_logging:
                console.print(
                    f"[green italic]LOG: Tool '{func.__name__}' returned: {result}[/]"
                )
                console.print(
                    f"[green italic]LOG: Tool '{func.__name__}' called with args: {args}, kwargs: {kwargs} \n\treturned: {result}[/]"
                )
                with open("./results/tracing.csv", "a") as f:
                    f.write(f"{func.__name__}|{args}|{kwargs}|{result}|\n")
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
