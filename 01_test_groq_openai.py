from langchain_openai import ChatOpenAI
import os
from dotenv import load_dotenv, find_dotenv
from rich.console import Console

console = Console()

load_dotenv(find_dotenv(), override=True)

# Load environment variables in a file called .env
# Print the key prefixes to help with any debugging
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

LLM_CHOICE = "OPENAI"
# LLM_CHOICE = "GROQ"

if OPENAI_API_KEY:
    print(f"OPENAI_API_KEY exists and begins {OPENAI_API_KEY[:14]}...")
else:
    print("OPENAI_API_KEY not set")

if GROQ_API_KEY:
    print(f"GROQ_API_KEY exists and begins {GROQ_API_KEY[:14]}...")
else:
    print("GROQ_API_KEY not set")


if LLM_CHOICE == "GROQ":
    MODEL = "llama-3.3-70b-versatile"
    BASE_URL = "https://api.groq.com/openai/v1"

    llm = ChatOpenAI(
        model=MODEL,  # or other Groq models
        openai_api_base=BASE_URL,  # omit for default OPENAI_API_BASE
        openai_api_key=GROQ_API_KEY,
    )
else:
    MODEL = "gpt-4o-mini"
    BASE_URL = "https://api.openai.com/v1"
    llm = ChatOpenAI(
        model=MODEL,  # or other Groq models
        openai_api_base=BASE_URL,  # omit for default OPENAI_API_BASE
        openai_api_key=OPENAI_API_KEY,
    )

print(f"LLM_CHOICE: {LLM_CHOICE} - MODEL: {MODEL}")


# Initialize ChatOpenAI with Groq's base URL


# Use it
response = llm.invoke("Hello, What is Groq")
print(response.content)
