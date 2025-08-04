# Pytest
# uv run pytest -vs .\src\case_study5\rag\pytest_examples\test_1_context_precision.py
import os

import pytest

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

llm = ChatGroq(
    model="llama-3.3-70b-versatile",  # or other Groq models
    groq_api_key=GROQ_API_KEY,  # or set GROQ_API_KEY env var
    temperature=0.0,
)
llm = ChatOpenAI(model=MODEL, temperature=0)


# we use asyncio just in case we use async functions like requests...
# we can also use sync functions and not use the decorator for asyncio.
@pytest.mark.asyncio
async def test_context_precision():
    # create object of class for that specific metric

    langchain_llm = LangchainLLMWrapper(llm)
    context_precision = LLMContextPrecisionWithoutReference(llm=langchain_llm)
    question = "What is Langgraph?"
    # Feed data -
    # responseDict = requests.post(
    #     "https://rahulshettyacademy.com/rag-llm/ask",
    #     json={"question": question, "chat_history": []},
    # ).json()
    # print(responseDict)

    sample = SingleTurnSample(
        user_input=question,
        response="""LangGraph is a library for building stateful, multi-actor applications with Large Language Models (LLMs), built on top of LangChain. It uses a finite state machine approach.""",
        retrieved_contexts=[
            """LangGraph is a library for building stateful, multi-actor applications with Large Language Models (LLMs), built on top of LangChain. It uses a finite state machine approach.
            """
        ],
    )

    # score
    score = await context_precision.single_turn_ascore(sample)
    print("SCORE: ", score)

    assert score > 0.8
