from enum import Enum
import os
import asyncio

from ragas.dataset_schema import SingleTurnSample

from ragas.metrics import RubricsScore
from ragas.llms import LangchainLLMWrapper
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv, find_dotenv
from rich.console import Console
from openai import OpenAI

console = Console()

load_dotenv(find_dotenv(), override=True)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
MODEL = "gpt-4o-mini"

console.print("[cyan]Starting...[/]")


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
# LLM_CHOICE = "GROQ"

if OPENAI_API_KEY:
    console.print(
        f"[green]✅ OPENAI_API_KEY exists and begins {OPENAI_API_KEY[:14]}...[/]"
    )
else:
    console.print("[red bold]❌ OPENAI_API_KEY not set[/]")

if GROQ_API_KEY:
    console.print(f"[green]✅ GROQ_API_KEY exists and begins {GROQ_API_KEY[:14]}...[/]")

else:
    console.print("[red bold]❌ GROQ_API_KEY not set[/]")


client = get_llm_client(LLM_CHOICE)
if LLM_CHOICE == "GROQ":
    MODEL = "llama-3.3-70b-versatile"
else:
    MODEL = "gpt-4o-mini"

console.print(f"[green]✅ LLM_CHOICE: {LLM_CHOICE} - MODEL: {MODEL}[/]")


# Create the OpenAI LLM evaluator
evaluator_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model=MODEL,
        temperature=0,
        api_key=OPENAI_API_KEY,
    )
)

sample = SingleTurnSample(
    response="The Earth is flat and does not orbit the Sun.",
    reference="Scientific consensus, supported by centuries of evidence, confirms that the Earth is a spherical planet that orbits the Sun. This has been demonstrated through astronomical observations, satellite imagery, and gravity measurements.",
)

# RAGAS use score1_description giving a numberical result of 1 which then means "The response is entirely incorrect and fails to address any aspect of the reference."

# 1 = Entirely incorrect ← Your result
# 2 = Partially accurate but major errors
# 3 = Mostly accurate but lacks clarity/details
# 4 = Accurate and clear with minor issues
# 5 = Completely accurate and thorough
# If you change score1_description to score6_description the you will get 5 as the result but will mean "The response is entirely incorrect and fails to address any aspect of the reference."


class RubricScore(Enum):
    ENTIRELY_INCORRECT = "The response is entirely incorrect and fails to address any aspect of the reference."
    PARTIAL_ACCURACY_MAJOR_ERRORS = "The response contains partial accuracy but includes major errors or significant omissions that affect its relevance to the reference."
    MOSTLY_ACCURATE_LACKS_CLARITY = "The response is mostly accurate but lacks clarity, thoroughness, or minor details needed to fully address the reference."
    ACCURATE_MINOR_OMISSIONS = "The response is accurate and clear, with only minor omissions or slight inaccuracies in addressing the reference."
    COMPLETELY_ACCURATE = "The response is completely accurate, clear, and thoroughly addresses the reference without any errors or omissions."


# Convert enum to dictionary format expected by Ragas
rubrics = {
    "score1_description": RubricScore.ENTIRELY_INCORRECT.value,
    "score2_description": RubricScore.PARTIAL_ACCURACY_MAJOR_ERRORS.value,
    "score3_description": RubricScore.MOSTLY_ACCURATE_LACKS_CLARITY.value,
    "score4_description": RubricScore.ACCURATE_MINOR_OMISSIONS.value,
    "score5_description": RubricScore.COMPLETELY_ACCURATE.value,
}

# ORIGINAL WAY but I replaced with enum to make output result have both score and description
# rubrics = {
#     "score1_description": "The response is entirely incorrect and fails to address any aspect of the reference.",
#     "score2_description": "The response contains partial accuracy but includes major errors or significant omissions that affect its relevance to the reference.",
#     "score3_description": "The response is mostly accurate but lacks clarity, thoroughness, or minor details needed to fully address the reference.",
#     "score4_description": "The response is accurate and clear, with only minor omissions or slight inaccuracies in addressing the reference.",
#     "score5_description": "The response is completely accurate, clear, and thoroughly addresses the reference without any errors or omissions.",
# }
scorer = RubricsScore(rubrics=rubrics, llm=evaluator_llm)


# Function to get description from score number
def get_score_description(score_number):
    score_map = {
        1: RubricScore.ENTIRELY_INCORRECT,
        2: RubricScore.PARTIAL_ACCURACY_MAJOR_ERRORS,
        3: RubricScore.MOSTLY_ACCURATE_LACKS_CLARITY,
        4: RubricScore.ACCURATE_MINOR_OMISSIONS,
        5: RubricScore.COMPLETELY_ACCURATE,
    }
    return score_map.get(score_number, None)


def format_result_output(result):
    score_desc = get_score_description(result)
    if score_desc:
        return f"Score {result}: {score_desc.value}"
    else:
        return f"Unknown score: {result}"


# Async function to run the evaluation
async def run_evaluation():
    try:
        console.print("[yellow]Running evaluation...[/]")
        result = await scorer.single_turn_ascore(sample)
        console.print(f"[green]✅ Evaluation completed![/]")
        return result
    except Exception as e:
        console.print(f"[red]❌ Error during evaluation: {e}[/]")
        return None


# Execute the evaluation
if __name__ == "__main__":
    # Run the async function
    result = asyncio.run(run_evaluation())
    formatted_output = format_result_output(result)
    console.print(f"[bold cyan]Final Result: {formatted_output}[/]")
