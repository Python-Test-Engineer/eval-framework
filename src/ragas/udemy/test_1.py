# Pytest 
import os

import pytest
import requests
from langchain_openai import ChatOpenAI
from ragas import SingleTurnSample
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import LLMContextPrecisionWithoutReference
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
MODEL = "gpt-4o-mini"

print("Starting...")


@pytest.mark.asyncio
async def test_context_precision():
    # create object of class for that specific metric

    # power of LLM + method metric ->score
    llm = ChatOpenAI(model=MODEL, temperature=0)
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
