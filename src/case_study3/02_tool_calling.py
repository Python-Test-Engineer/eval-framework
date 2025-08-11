# LLM Eval for tool calling that can use a chain of two tools. We can check correct tools are called with correct inputs and also the correct chain flow occurs.

import os
from datetime import datetime
from uuid import uuid4
from langchain_core.messages import HumanMessage, ToolMessage, AIMessage
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
        return "~It's 10 degrees Celsius and cold."
    else:
        return "~It's 40 degrees Celsius and sunny."


@tool
def check_seating_availability(location: str, seating_type: str):
    """Call to check seating availability."""
    if location.lower() == "munich" and seating_type.lower() == "outdoor":
        return "~Yes, we still have seats available outdoors."
    elif location.lower() == "munich" and seating_type.lower() == "indoor":
        return "~Yes, we have indoor seating available."
    else:
        return "~Sorry, seating information for this location is unavailable."


@tool
def convert_c_to_f(centigrade: float, fahrenheit: float) -> float:
    """Given a temperature in Celsius, convert it to Fahrenheit.
    Uses the formula: °F = (°C × 1.8) + 32"""
    if centigrade is not None:
        result = "~" + str((centigrade * 1.8) + 32)

        return result
    else:
        return "~Sorry, I am unable to calculate the temperature."


@tool
def describe_fahrenheit_with_label(temperature: float) -> str:
    """Given a temperature in Fahrenheit, describe it as COLD, MILD, WARM or HOT."""

    if temperature < 45:
        return "~COLD"
    elif temperature < 65:
        return "~MILD"
    elif temperature < 75:
        return "~WARM"
    elif temperature < 100:
        return "~HOT"
    elif temperature > 100:
        return "~ULTRA_HOT"
    else:
        return "~NONE"


@tool
def order_food(temp_desc: str) -> str:
    """The food to order for a given temperature description. Use this tool if the user wants to order some food or to pick a type of food needed."""
    console.print(f"[cyan bold]<debug>temp_desc: {temp_desc}[/]")
    if "COLD" in temp_desc:
        food = "~COLD_FOOD"
    elif "MILD" in temp_desc:
        food = "~MILD_FOOD"
    elif "WARM" in temp_desc:
        food = "~WARM_FOOD"
    elif "HOT" in temp_desc:
        food = "~HOT_FOOD"
    else:
        food = "~NONE"

    console.print(f"[cyan bold]<debug>TOOL ORDER_FOOD: {food}[/]")
    return food


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
What is 32 centigrade in fahrenheit and its label? What is the weather? 
"""
Q3 = """
What is 22 centigrade in fahrenheit? What is the label for this temperature?"""

Q4 = """
What is 35 centigrade in fahrenheit? What is the label for this temperature? Please order food for this temperature."""
Q5 = """ It is 45 centigrade. Calculate the temperature in fahrenheit. What is the label for this temperature? What food do I order?."""
Q6 = """
Describe 45 Centrigrade and what food should I get?
It is HOT. What food do I order? 
what is the weather in munich? 
What is 45 centigrade in fahrenheit?

"""

run_id = str(uuid4())
messages = [HumanMessage(Q6)]
# messages = [HumanMessage(Q5)]


console.print("[green]Starting...[/]")
console.print(f"[green]{messages}[/]")
llm_output = llm_with_tools.invoke(messages)
messages.append(llm_output)

tool_mapping = {
    "get_weather": get_weather,
    "check_seating_availability": check_seating_availability,
    "convert_c_to_f": convert_c_to_f,
    "describe_fahrenheit_with_label": describe_fahrenheit_with_label,
    "order_food": order_food,
}


tools_called = ":".join([tool_call["name"] for tool_call in llm_output.tool_calls])
for tool_call in llm_output.tool_calls:
    try:
        if tool_call["id"]:
            tool = tool_mapping[tool_call["name"].lower()]
            tool_id = tool_call["id"]
            tool_output = tool.invoke(tool_call["args"])
            print(
                f"Tool called: {tool_call['name'].lower()} with args: {tool_call['args']} and OUTPUT: {tool_output}"
            )

            messages.append(ToolMessage(tool_output, tool_call_id=tool_call["id"]))
    except Exception:
        console.print(f"[red]❌ Error during tool call: [/]")


INPUT = ""
OUTPUT = ""
try:
    llm_with_tools.invoke(messages)
except Exception:
    console.print("[red]❌ Error during LLM call: [/]")

for message in messages:

    # console.print(f"Message Type: {message.type}")
    # console.print(f"Message: {message.content}")
    if message.type == "tool":
        OUTPUT += message.content
    if message.type == "human":
        INPUT += message.content
        INPUT = INPUT.replace("\n", "").replace("  ", " ").strip()


#################### EVALS01 ####################
print("\nEVALS01\n")
log = f"{run_id}|{get_time_now()}|TOOL_CALLING|{MODEL}|{TEMPERATURE}|{INPUT}|{OUTPUT}|{tools_called}|"
console.print(log)
with open(f"{OUTPUT_DIR}/02_tool_calling.csv", "a") as f:
    f.write(f"{log}\n")
#################################################
console.print(f"\n[green]Done! Logged to {OUTPUT_DIR}/02_tool_calling.csv[/]")
