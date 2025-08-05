# Pytest
# uv run pytest -vs .\src\case_study5\rag\pytest_examples\test_2_topic_adhernence.py
import os
import pytest
from ragas import MultiTurnSample
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


# we use asyncio just in case we use async functions like requests...
# we can also use sync functions and not use the decorator for asyncio.
@pytest.mark.asyncio
async def test_082_topicAdherence(llm_wrapper, getData):
    topicScore = TopicAdherenceScore(llm=llm_wrapper)
    score = await topicScore.multi_turn_ascore(getData)
    print(score)
    assert score > 0.8


@pytest.fixture
def getData():

    conversation = [
        HumanMessage(content="What is Langgraph?"),
        AIMessage(
            content="LangGraph is a library for building stateful, multi-actor applications with Large Language Models (LLMs), built on top of LangChain. It's designed to create complex, graph-based workflows where different components can interact and maintain state across multiple steps."
        ),
        HumanMessage(content="What are its key features?"),
        AIMessage(
            content="""            Key Features
            Graph-based Architecture: LangGraph represents your application as a directed graph where nodes are functions (often LLM calls or other operations) and edges define the flow between them.
            State Management: Unlike simple chains, LangGraph maintains state that persists across the entire workflow, allowing for complex multi-step reasoning and decision-making.."""
        ),
    ]
    reference = [
        """ 
    Facts about Langgraph:
    LangGraph is a library for building stateful, multi-actor applications with Large Language Models (LLMs), built on top of LangChain. It's designed to create complex, graph-based workflows where different components can interact and maintain state across multiple steps.
    Key Features
    Graph-based Architecture: LangGraph represents your application as a directed graph where nodes are functions (often LLM calls or other operations) and edges define the flow between them.
    State Management: Unlike simple chains, LangGraph maintains state that persists across the entire workflow, allowing for complex multi-step reasoning and decision-making.
    """
    ]
    sample = MultiTurnSample(user_input=conversation, reference_topics=reference)
    return sample
