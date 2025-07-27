import re

import csv

file = "./src/case_study2/labs/output/log_agent_hello_world.csv"
# Read the CSV and extract the agent string from the 4th column (index 3)
with open(file, "r") as file:
    reader = csv.reader(file, delimiter="|")
    for row in reader:
        agent_str = row[3]  # 4th column (index 3)
        dt = row[0]
        print("=" * 40)
        print(f"DT: {dt}:")
        print("=" * 40)
        break  # Just get the first row


def parse_agent_string(agent_str):
    """Extract key-value pairs from Agent string representation"""

    # Remove Agent() wrapper
    content = agent_str[6:-1] if agent_str.startswith("Agent(") else agent_str

    # Find all key=value patterns
    pattern = r"(\w+)=([^,=]+(?:\([^)]*\))?[^,]*)"
    matches = re.findall(pattern, content)

    result = {}
    for key, value in matches:
        # Clean up the value
        value = value.strip()

        # Convert basic types
        if value == "None":
            result[key] = None
        elif value == "True":
            result[key] = True
        elif value == "False":
            result[key] = False
        elif value == "[]":
            result[key] = []
        elif value == "{}":
            result[key] = {}
        elif value.startswith("'") and value.endswith("'"):
            result[key] = value[1:-1]  # Remove quotes
        else:
            result[key] = value

    return result


# Test with your data
# agent_str = "Agent(name='Hello World Assistant', instructions='You are a friendly assistant. Greet users warmly and be helpful.', prompt=None, handoff_description=None, handoffs=[], model='gpt-4o-mini', model_settings=ModelSettings(temperature=None, top_p=None, frequency_penalty=None, presence_penalty=None, tool_choice=None, parallel_tool_calls=None, truncation=None, max_tokens=None, reasoning=None, metadata=None, store=None, include_usage=None, response_include=None, extra_query=None, extra_body=None, extra_headers=None, extra_args=None), tools=[], mcp_servers=[], mcp_config={}, input_guardrails=[], output_guardrails=[], output_type=None, hooks=None, tool_use_behavior='run_llm_again', reset_tool_choice=True)"

result = parse_agent_string(agent_str)

print("Parsed result in dictionary:")
print(result)

print("Extracted key-value pairs:")
for key, value in result.items():
    print(f"{key}: {value}")
