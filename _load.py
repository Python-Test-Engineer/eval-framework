import os
from dotenv import load_dotenv, find_dotenv
from rich.console import Console

console = Console()

load_dotenv(find_dotenv(), override=True)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
MODEL = "gpt-4o-mini"


def get_llm_client(llm_choice):
    if llm_choice == "GROQ":
        client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=os.environ.get("GROQ_API_KEY"),
        )
        return client
    elif llm_choice == "OPENAI":
        load_dotenv()  # load environment variables from .env fil
        client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
        return client
    else:
        raise ValueError("Invalid LLM choice. Please choose 'GROQ' or 'OPENAI'.")


LLM_CHOICE = "OPENAI"
LLM_CHOICE = "GROQ"

if OPENAI_API_KEY:
    print(f"OPENAI_API_KEY exists and begins {OPENAI_API_KEY[:14]}...")
else:
    print("OPENAI_API_KEY not set")

if GROQ_API_KEY:
    print(f"GROQ_API_KEY exists and begins {GROQ_API_KEY[:14]}...")
else:
    print("GROQ_API_KEY not set")


client = get_llm_client(LLM_CHOICE)
if LLM_CHOICE == "GROQ":
    MODEL = "llama-3.3-70b-versatile"
else:
    MODEL = "gpt-4o-mini"

print(f"LLM_CHOICE: {LLM_CHOICE} - MODEL: {MODEL}")

