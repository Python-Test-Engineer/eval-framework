import os
import pytest

from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from ragas import SingleTurnSample
from ragas.metrics import RubricsScore
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
async def test_083_rubric_score(llm_wrapper):
    sample = SingleTurnSample(
        user_input="Where is the Eiffel Tower located?",
        response="The Eiffel Tower is located in Paris.",
        reference="The Eiffel Tower is located in Paris.",
    )
    rubrics = {
        "score1_description": "The response is incorrect, irrelevant, or does not align with the ground truth.",
        "score2_description": "The response partially matches the ground truth but includes significant errors, omissions, or irrelevant information.",
        "score3_description": "The response generally aligns with the ground truth but may lack detail, clarity, or have minor inaccuracies.",
        "score4_description": "The response is mostly accurate and aligns well with the ground truth, with only minor issues or missing details.",
        "score5_description": "The response is fully accurate, aligns completely with the ground truth, and is clear and detailed.",
    }

    rubrics_score = RubricsScore(rubrics=rubrics, llm=llm_wrapper)
    score = await rubrics_score.single_turn_ascore(sample)
    print(score)
    assert score > 3
