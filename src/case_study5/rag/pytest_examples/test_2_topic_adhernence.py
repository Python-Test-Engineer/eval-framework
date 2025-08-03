# Pytest
# uv run pytest -vs .\src\case_study5\rag\pytest_examples\test_2_topic_adhernence.py
import os
import pytest
from ragas import  MultiTurnSample
from ragas.messages import HumanMessage, AIMessage
from ragas.metrics import TopicAdherenceScore
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)

OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
MODEL = "gpt-4o-mini"
load_dotenv(find_dotenv(), override=True)
llm_OPENAI = ChatOpenAI(model=MODEL, temperature=0)
llm_GROQ = ChatGroq(
    model="llama-3.3-70b-versatile",  # or other Groq models
    groq_api_key=GROQ_API_KEY,  # or set GROQ_API_KEY env var
    temperature=0.0,
)


PROVIDER = "OPENAI"
PROVIDER = "GROQ"


@pytest.fixture
def llm_wrapper(llm=PROVIDER):
    if llm == "OPENAI":
        llm = llm_OPENAI
    elif llm == "GROQ":
        llm = llm_GROQ

    llm = LangchainLLMWrapper(llm)
    return llm


@pytest.mark.asyncio
async def test_topicAdherence(llm_wrapper, getData):
    topicScore = TopicAdherenceScore(llm=llm_wrapper)
    score = await topicScore.multi_turn_ascore(getData)
    print(score)
    assert score > 0.8


@pytest.fixture
def getData():
  
    conversation = [
        HumanMessage(content="how many articles are there in the selenium webdriver python course?"),
        AIMessage(content="There are 23 articles in the Selenium WebDriver Python course."),
        HumanMessage(content="How many downloadable resources are there in this course?"),
        AIMessage(content="There are 9 downloadable resources in the course.")

    ]
    reference = [""" 
    The AI should:
    1. Give results related to the selenium webdriver python course
    2. There are 23 articles and 9 downloadable resources in the course"""]
    sample = MultiTurnSample(user_input=conversation, reference_topics=reference)
    return sample
