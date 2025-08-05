# https://docs.ragas.io/en/v0.3.0/concepts/metrics/available_metrics/general_purpose/#rubrics-based-criteria-scoring


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
        model=MODEL,  # or "gpt-3.5-turbo" for faster/cheaper evaluation
        temperature=0,  # Set to 0 for consistent evaluation
        api_key=OPENAI_API_KEY,  # Replace with your actual API key
    )
)

# Your sample data
sample = SingleTurnSample(
    response="The Earth is flat and does not orbit the Sun.",
    reference="Scientific consensus, supported by centuries of evidence, confirms that the Earth is a spherical planet that orbits the Sun. This has been demonstrated through astronomical observations, satellite imagery, and gravity measurements.",
)

# Your rubrics
rubrics = {
    "score1_description": "The response is entirely incorrect and fails to address any aspect of the reference.",
    "score2_description": "The response contains partial accuracy but includes major errors or significant omissions that affect its relevance to the reference.",
    "score3_description": "The response is mostly accurate but lacks clarity, thoroughness, or minor details needed to fully address the reference.",
    "score4_description": "The response is accurate and clear, with only minor omissions or slight inaccuracies in addressing the reference.",
    "score5_description": "The response is completely accurate, clear, and thoroughly addresses the reference without any errors or omissions.",
}

# Create the scorer with the evaluator LLM
scorer = RubricsScore(rubrics=rubrics, llm=evaluator_llm)


# Run the evaluation
async def run_evaluation():
    result = await scorer.single_turn_ascore(sample)
    print(f"Evaluation Score: {result}")
    return result


# Execute the evaluation
if __name__ == "__main__":
    asyncio.run(run_evaluation())
