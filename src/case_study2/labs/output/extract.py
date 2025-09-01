import re

import csv

file = "./src/case_study2/labs/output/log_agent_hello_world.csv"


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


# Read the CSV and extract the agent string from the 4th column (index 3)
with open(file, "r") as file:
    reader = csv.reader(file, delimiter="|")
    for row in reader:
        agent_str = row[3]  # 4th column (index 3)
        dt = row[0]
        print("=" * 40)
        print(f"DT: {dt}:")
        print("=" * 40)
        result = parse_agent_string(agent_str)

        print("Parsed result in dictionary:")
        print(result)  # Just get the first row
