import os

import pytest
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper

from dotenv import load_dotenv, find_dotenv

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
MODEL = "gpt-4o-mini"
load_dotenv(find_dotenv(), override=True)


@pytest.fixture
def llm_wrapper():
    llm = ChatOpenAI(model=MODEL, temperature=0)
    llm = ChatGroq(
        model="llama-3.3-70b-versatile",  # or other Groq models
        groq_api_key=GROQ_API_KEY,  # or set GROQ_API_KEY env var
        temperature=0.0,
    )
    llm = LangchainLLMWrapper(llm)
    return llm
