# A way to test env keys - run `python 01_py_llm.py` to see if the OpenAI API key is set.

import os
import json
import requests
from dotenv import load_dotenv, find_dotenv
from rich.console import Console


console = Console()


load_dotenv(find_dotenv(), override=True)  # Load environment variables from .env file

openai_api_key = os.getenv("OPENAI_API_KEY")
if openai_api_key:
    console.print(
        f"[green]OpenAI API Key exists and begins {openai_api_key[:14]}...[/]"
    )
else:
    console.print("[red]OpenAI API Key not set[/]")


MODEL = "gpt-4o-mini"
console.print(f"[dark_orange]Model selected: {MODEL}[/]")

query = "What is an the difference between Pandas and AutoGen?"

# For demo, we will create our own OpenAI request class but in future we will use the OpenAI library.
#


#
# Temperature - The OpenAI API incorporates a hyperparameter known as temperature that affects the computation of token probabilities when generating output through the large language model. The temperature value ranges from 0 to 2, with lower values indicating greater determinism and higher values indicating more randomness. Default is 1.
#
# Low Temperature:
#
# The bag is full of mostly blue marbles, with a few red and green. Low temperature means you're very likely to pull a blue marble, but you might occasionally get a red or green.
#
# High Temperature:
#
# The bag is filled with a mix of colors, and all colors are equally likely. High temperature means you're equally likely to pull any color, including the less common ones.
#
# <img src="./images/temperature.png">


# We are essentially making a POST request to one endpoint and we create a convenience Class to pass in prompts, temperature and model.


class MyCustomOpenAI:

    def __init__(self, system_prompt=None, temperature=1.0, model=MODEL):
        self.model_endpoint = "https://api.openai.com/v1/chat/completions"
        self.temperature = temperature
        self.model = model
        self.system_prompt = system_prompt
        self.api_key = openai_api_key
        self.headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

    # Note that other than the API key to be authenticated, we pass nor reference to previous queries. Each request is STATELESS and is independent of any other request.

    # That is why we append previous output to new queries so that the LLM has the full context or 'picture'.

    # OpenAI was trained with 'messages' in the form of a list of messages - SYSTEM, USER, ASSISTANT - so using this is most effective.

    # SYSTEM will contain the 'character' and instruction set of the Agent, USER will be the input to the Agent and ASSISTANT will be the output.

    # TAKEAWAY: WWe are appending the last output to the list of 'messages' so that the Agent has knowledge of the context - the history of the conversation.

    def generate_text(self, prompt):
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ],
            "stream": False,  # no streaming or output
            "temperature": self.temperature,
        }
        # Use HTTP POST method from Requests library

        # In future examples we will use OpenAI library instead of requests directly but this is for demo purposes.
        response = requests.post(
            url=self.model_endpoint, headers=self.headers, data=json.dumps(payload)
        ).json()
        return response


# We create an instance of MyCustomOpenAI and pass in a request - just a convenience wrapper of making a REST API call.

# System Prompt sets up the character of the AI Agent and can be considered to be the API route we are creating and it it will be in Natural Language - explained later...

client = MyCustomOpenAI(
    model=MODEL,
    system_prompt="You give concise answers to questions with no more than 200 characters",
    temperature=0.0,  # as deterministic as possible
)

response = client.generate_text(query)


# We can drill down into 'response' to get our choice of data
# The 'answer' is in the message with 'role': 'assistant'
print("\n=====================================\n")
console.print(response)
# print(response["choices"][0])
# print(response["choices"][0]["message"])
print("=====================================\n\n")
print(f"Query: {query}\n")
console.print(response["choices"][0]["message"]["content"])
print("\n\n")
