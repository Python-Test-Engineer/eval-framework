# LLM Eval for tool calling that can use a chain of two tools. We can check correct tools are called with correct inputs and also the correct chain flow occurs.

import os
from datetime import datetime

from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from rich.console import Console
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
console = Console()

MODEL = "gpt-4o-mini"
TEMPERATURE = 0
OUTPUT_DIR = "./src/case_study3/output/"
llm = ChatOpenAI(model=MODEL, temperature=TEMPERATURE)


def get_time_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_weather(location: str):
    """Call to get the current weather."""
    if location.lower() in ["munich"]:
        return "It's 10 degrees Celsius and cold."
    else:
        return "It's 40 degrees Celsius and sunny."


@tool
def check_seating_availability(location: str, seating_type: str):
    """Call to check seating availability."""
    if location.lower() == "munich" and seating_type.lower() == "outdoor":
        return "Yes, we still have seats available outdoors."
    elif location.lower() == "munich" and seating_type.lower() == "indoor":
        return "Yes, we have indoor seating available."
    else:
        return "Sorry, seating information for this location is unavailable."


@tool
def convert_c_to_f(centigrade: float, fahrenheit: float) -> float:
    """Given a temperature in Celsius, convert it to Fahrenheit.
    Uses the formula: °F = (°C × 1.8) + 32"""
    if centigrade is not None:
        return (centigrade * 1.8) + 32
    else:
        return "Sorry, I am unable to calculate the temperature."


@tool
def describe_fahrenheit_with_label(temperature: float) -> str:
    """Given a temperature in Fahrenheit, convert it to a lable of either COLD, MILD, WARM or HOT."""
    # this has leading | so that the output is easier to parse as we get 85|HOT in the output
    if temperature < 45:
        return "|COLD"
    elif temperature < 65:
        return "|MILD"
    elif temperature < 75:
        return "|WARM"
    elif temperature < 100:
        return "|HOT"
    else:
        return "|NONE"


@tool
def order_food(temp_desc: str) -> str:
    """Given a described temperature in text, order food based on the label."""
    if temp_desc == "COLD":
        food = "spicy_food"
    elif temp_desc == "MILD":
        food = "hot soup"
    elif temp_desc == "WARM":
        food = "cold_drink"
    elif temp_desc == "HOT":
        food = "iced_drink"
    else:
        food = "none"
    console.print(f"[cyan bold]<debug>TOOL ORDER_FOOD: {food}[/]")
    return console.print(f"[purple]RETURN ORDER_FOOD: {food}")


tools = [
    get_weather,
    check_seating_availability,
    convert_c_to_f,
    describe_fahrenheit_with_label,
    order_food,
]
llm_with_tools = llm.bind_tools(tools)
tools_called = ""

Q1 = """
How will the weather be in munich today? Do you still have indoor seats available?
"""
Q2 = """
What is 32 centigrade in fahrenheit?
"""
Q3 = """
What is 15 centigrade in fahrenheit? What is the label for this temperature?"""

Q4 = """
It is 15 centigrade. Please order food for this temperature."""


messages = [HumanMessage(Q4)]
# messages = [HumanMessage(Q2)]


console.print("[green]Starting...[/]")
llm_output = llm_with_tools.invoke(messages)
messages.append(llm_output)

tool_mapping = {
    "get_weather": get_weather,
    "check_seating_availability": check_seating_availability,
    "convert_c_to_f": convert_c_to_f,
    "describe_fahrenheit_with_label": describe_fahrenheit_with_label,
    "order_food": order_food,
}

tools_called = llm_output.tool_calls

for tool_call in llm_output.tool_calls:
    tool = tool_mapping[tool_call["name"].lower()]
    tool_output = tool.invoke(tool_call["args"])
    messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))


INPUT = ""
OUTPUT = ""

llm_with_tools.invoke(messages)
for message in messages:
    if isinstance(message, ToolMessage) or isinstance(message, HumanMessage):
        console.print(f"Message Type: {message.type}")
        console.print(f"Message: {message.content}")
        if message.type == "tool":
            OUTPUT += message.content
        if message.type == "human":
            INPUT += message.content
            INPUT = INPUT.replace("\n", "").replace("  ", " ").strip()


#################### EVALS01 ####################
log = f"{get_time_now()}|TOOL_CALLING|{MODEL}|{TEMPERATURE}|{INPUT}|{OUTPUT}|{tools_called}"
console.print(log)
with open(f"{OUTPUT_DIR}/02_tool_calling.csv", "a") as f:
    f.write(f"{log}\n")
#################################################
console.print(f"[green]Done! Logged to {OUTPUT_DIR}/01_tool_calling.csv[/]")
