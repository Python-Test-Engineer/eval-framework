# Pytest
import os

import pytest
import requests
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference
from langchain_groq import ChatGroq
from dotenv import load_dotenv, find_dotenv
from rich.console import Console

console = Console()

load_dotenv(find_dotenv(), override=True)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
MODEL = "gpt-4o-mini"

print("Starting...")


# Initialize Groq LLM
# llm = ChatGroq(
#     model="llama-3.3-70b-versatile",  # or other Groq models
#     groq_api_key=GROQ_API_KEY,  # or set GROQ_API_KEY env var
#     temperature=0.0,
# )
# # # Use it like any LangChain LLM
# response = llm.invoke("What is the capital of France?")
# console.print(f"[green]GROQ TEST: {response.content}[/]")
# power of LLM + method metric ->score
llm = ChatOpenAI(model=MODEL, temperature=0)
llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # or other Groq models
    groq_api_key=GROQ_API_KEY,  # or set GROQ_API_KEY env var
    temperature=0.0,
)


@pytest.mark.asyncio
async def test_context_precision():
    # create object of class for that specific metric

    langchain_llm = LangchainLLMWrapper(llm)
    context_precision = LLMContextPrecisionWithoutReference(llm=langchain_llm)
    question = "How many articles are there in the Selenium webdriver python course?"
    # Feed data -
    responseDict = requests.post(
        "https://rahulshettyacademy.com/rag-llm/ask",
        json={"question": question, "chat_history": []},
    ).json()
    print(responseDict)

    sample = SingleTurnSample(
        user_input=question,
        response=responseDict["answer"],
        retrieved_contexts=[
            responseDict["retrieved_docs"][0]["page_content"],
            responseDict["retrieved_docs"][1]["page_content"],
            responseDict["retrieved_docs"][2]["page_content"],
        ],
    )

    # score
    score = await context_precision.single_turn_ascore(sample)
    print("SCORE: ", score)

    assert score > 0.8
