# LLM Eval for ReAct agent that can use a chain of tools. We can check correct tools are called with correct inputs and also the correct chain flow occurs.

import os
from datetime import datetime
from uuid import uuid4
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from rich.console import Console
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
console = Console()

MODEL = "gpt-4o-mini"
TEMPERATURE = 0
OUTPUT_DIR = "./src/case_study3/output/"
llm = ChatOpenAI(model=MODEL, temperature=TEMPERATURE)

DEBUG = ""


def get_time_now():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@tool
def get_weather(location: str):
    """Call to get the current weather."""
    console.print(f"[cyan bold]<debug>get_weather: {location}[/]")
    global DEBUG
    DEBUG += f"<debug>get_weather: {location}</debug>"
    if location.lower() in ["munich"]:
        return "~It's 10 degrees Celsius and cold."
    else:
        return "~It's 40 degrees Celsius and sunny."


@tool
def check_seating_availability(location: str, seating_type: str):
    """Call to check seating availability."""
    console.print(
        f"[cyan bold]<debug>check_seating_availability: {location}, {seating_type}[/]"
    )
    global DEBUG
    DEBUG += f"<debug>check_seating_availability: {location}, {seating_type}</debug>"
    if location.lower() == "munich" and seating_type.lower() == "outdoor":
        return "~Yes, we still have seats available outdoors."
    elif location.lower() == "munich" and seating_type.lower() == "indoor":
        return "~Yes, we have indoor seating available."
    else:
        return "~Sorry, seating information for this location is unavailable."


@tool
def convert_c_to_f(centigrade: float) -> float:
    """Given a temperature in Celsius, convert it to Fahrenheit.
    Uses the formula: °F = (°C × 1.8) + 32"""
    console.print(f"[cyan bold]<debug>convert_c_to_f: {centigrade}[/]")
    global DEBUG
    DEBUG += f"<debug>convert_c_to_f: {centigrade}</debug>"
    if centigrade is not None:
        result = "~" + str((centigrade * 1.8) + 32)
        return result
    else:
        return "~Sorry, I am unable to calculate the temperature."


@tool
def describe_fahrenheit_with_label(temperature: float) -> str:
    """Given a temperature in Fahrenheit, describe it as COLD, MILD, WARM or HOT."""
    console.print(f"[cyan bold]<debug>describe_fahrenheit_with_label: {temperature}[/]")
    global DEBUG
    DEBUG += f"<debug>describe_fahrenheit_with_label: {temperature}</debug>"
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
    global DEBUG
    DEBUG += f"<debug>temp_desc: {temp_desc}</debug>"
    if "COLD" in temp_desc:
        food = "~HOT_FOOD"
    elif "MILD" in temp_desc:
        food = "~WARM_FOOD"
    elif "WARM" in temp_desc:
        food = "~COOL_FOOD"
    elif "HOT" in temp_desc:
        food = "~COLD_FOOD"
    else:
        food = "~NONE"

    console.print(f"[cyan bold]<debug>TOOL ORDER_FOOD: {food}[/]")
    return food


# Define the prompt template for tool calling agent
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant that can help with weather, seating, temperature conversion, and food ordering.
    You have access to tools that can help you answer questions. 
    When you need to chain multiple tools together, make sure to use the output of one tool as input to the next.
    Always provide a clear final answer to the user's question.""",
        ),
        ("human", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create tools list
tools = [
    get_weather,
    check_seating_availability,
    convert_c_to_f,
    describe_fahrenheit_with_label,
    order_food,
]

# Create the tool calling agent (more stable than ReAct)
agent = create_tool_calling_agent(llm, tools, prompt)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=False,  # Set to False to avoid callback issues
    handle_parsing_errors=True,
    max_iterations=10,
    return_intermediate_steps=True,  # This helps with tool tracking
)

# Test questions
Q1 = """How will the weather be in munich today? Do you still have indoor seats available?"""
Q2 = """What is 32 centigrade in fahrenheit and its label? What is the weather?"""
Q3 = """It is 22 centigrade so what food do I order?"""
Q4 = """What is 35 centigrade in fahrenheit? What is the label for this temperature? Please order food for this temperature."""
Q5 = """It is 45 centigrade. Calculate the temperature in fahrenheit. What is the label for this temperature? What food do I order?"""
Q6 = """Describe 1 Centrigrade~ what is the weather in munich?~ What is 45 centigrade in fahrenheit?~ It is HOT. What food do I order?~"""

run_id = str(uuid4())

# Select question to run
question = Q3

console.print("[green]Starting Tool Calling Agent...[/]")
console.print(f"[green]Question: {question}[/]")

try:
    # Execute the agent
    result = agent_executor.invoke({"input": question})

    # Extract information for logging
    INPUT = question.replace("\n", "").replace("  ", " ").strip()
    OUTPUT = result.get("output", "")

    # Extract tools called from the intermediate steps
    tools_called = []
    if "intermediate_steps" in result and result["intermediate_steps"]:
        for step in result["intermediate_steps"]:
            # step[0] is the AgentAction, step[1] is the observation
            if hasattr(step[0], "tool"):
                tools_called.append(step[0].tool)

    tools_called_str = ":".join(tools_called) if tools_called else "none"

    console.print(f"[green]Agent Result: {result['output']}[/]")
    console.print(f"[yellow]Tools Called: {tools_called_str}[/]")

except Exception as e:
    console.print(f"[red]❌ Error during agent execution: {e}[/]")
    INPUT = question.replace("\n", "").replace("  ", " ").strip()
    OUTPUT = f"ERROR: {str(e)}"
    tools_called_str = "error"

#################### EVALS01 ####################
print("\nEVALS01\n")
log = f"{run_id}|{get_time_now()}|TOOL_CALLING_AGENT|{MODEL}|{TEMPERATURE}|{INPUT}|{OUTPUT}|{DEBUG}|{tools_called_str}|"
console.print(log)

# Ensure output directory exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

with open(f"{OUTPUT_DIR}/03_tool_calling_agent.csv", "a") as f:
    f.write(f"{log}\n")
#################################################
console.print(f"\n[green]Done! Logged to {OUTPUT_DIR}/03_tool_calling_agent.csv[/]")


# Optional: Run multiple questions for comprehensive evaluation
def run_all_questions():
    """Run all test questions for comprehensive evaluation"""
    questions = [Q1, Q2, Q3, Q4, Q5, Q6]

    for i, question in enumerate(questions, 1):
        console.print(f"\n[blue]--- Running Question {i} ---[/]")
        run_id = str(uuid4())

        try:
            result = agent_executor.invoke({"input": question})
            INPUT = question.replace("\n", "").replace("  ", " ").strip()
            OUTPUT = result.get("output", "")

            tools_called = []
            if "intermediate_steps" in result and result["intermediate_steps"]:
                for step in result["intermediate_steps"]:
                    if hasattr(step[0], "tool"):
                        tools_called.append(step[0].tool)
            tools_called_str = ":".join(tools_called) if tools_called else "none"

        except Exception as e:
            INPUT = question.replace("\n", "").replace("  ", " ").strip()
            OUTPUT = f"ERROR: {str(e)}"
            tools_called_str = "error"

        log = f"{run_id}|{get_time_now()}|TOOL_CALLING_AGENT|{MODEL}|{TEMPERATURE}|{INPUT}|{OUTPUT}|{DEBUG}|{tools_called_str}|"

        with open(f"{OUTPUT_DIR}/03_tool_calling_agent_all.csv", "a") as f:
            f.write(f"{log}\n")

        console.print(f"[green]Question {i} completed and logged[/]")


# Uncomment to run all questions
run_all_questions()
