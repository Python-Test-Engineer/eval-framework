# ### Team of Agents with a supervisor


# https://www.youtube.com/watch?v=9HhcFiSgLok&list=PLNVqeXDm5tIqUIPQHLk5Xw5mpisruvsac&index=7


import os
from datetime import datetime

from openai import OpenAI
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

from pydantic import BaseModel, Field
from rich.console import Console
from dotenv import load_dotenv, find_dotenv
import warnings

warnings.filterwarnings("ignore")
console = Console()
load_dotenv(find_dotenv())
# Load environment variables from .env file
MODEL = "gpt-4o-mini"
TEMPERATURE = 0
LANGUAGE = "FRENCH"
SUBJECT = "AI, ML, Data Science, Programming, Web, Technology"
CONTENT_LENGTH = 100
MAX_LENGTH = 100
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    console.print(
        f"[green]OpenAI API Key exists and begins {OPENAI_API_KEY[:14]}...[/]"
    )
else:
    console.print("[red]OpenAI API Key not set[/]")


def get_report_date():
    """
    Returns the current date and time formatted as a string.
    """
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


class TransferNewsGrader(BaseModel):
    f"""Binary score for relevance check on {SUBJECT}news."""

    binary_score: str = Field(
        description=f"The article is about {SUBJECT}, 'yes' or 'no'"
    )


class ArticlePostabilityGrader(BaseModel):
    """Binary scores for postability check, word count, sensationalism, and language verification of a news article."""

    can_be_posted: str = Field(
        description="The article is ready to be posted, 'yes' or 'no'"
    )
    meets_word_count: str = Field(
        description=f"The article has at least {CONTENT_LENGTH} words, 'yes' or 'no'"
    )
    is_not_sensationalistic: str = Field(
        description="The article is NOT written in a sensationalistic style, 'yes' or 'no'"
    )
    is_language_french: str = Field(
        description=f"The language of the article is {LANGUAGE}, 'yes' or 'no'"
    )


class AgentState(TypedDict):
    article_state: str


def get_transfer_news_grade(state: AgentState) -> AgentState:
    # print(f"get_transfer_news_grade: Current state: {state}")
    # print("Evaluator: Reading article but doing nothing to change it...")
    return state


def evaluate_article(state: AgentState) -> AgentState:
    # print(f"evaluate_article: Current state: {state}")
    # print("News : Reading article but doing nothing to change it...")
    return state


def publisher(state: AgentState) -> AgentState:
    # print(f"publisher: Current state: {state}")
    # print("FINAL_STATE in publisher:", state)
    return state


def evaluator_router(state: AgentState) -> Literal["editor", "not_relevant"]:
    article = state["article_state"]
    INPUT = article
    print(f"evaluator_router: INPUT: {article}")
    MODEL = "gpt-4o-mini"
    TEMPERATURE = 0

    llm = ChatOpenAI(
        model=MODEL,
        temperature=TEMPERATURE,
    )
    structured_llm_grader = llm.with_structured_output(TransferNewsGrader)

    system = f"""You are a researcher that determines the content type of an article.
        Check if the article refers to {SUBJECT} area.
        Provide a binary score 'yes' or 'no' to indicate whether the article is technical in nature."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "News Article:\n\n {article}")]
    )
    # should_write = grade_prompt | structured_llm_grader
    evaluator = grade_prompt | structured_llm_grader
    result = evaluator.invoke({"article": article})

    # This is to get a different more detailed response from the LLM using OpenAI rather than LangChain
    client = OpenAI()
    completion = client.chat.completions.parse(
        model="gpt-4o",
        messages=[{"role": "user", "content": "News Article:\n\n {article}"}],
        response_format=TransferNewsGrader,
    )
    # The completion object has these main attributes:
    print(f"ID: {completion.id}")
    print(f"Object: {completion.object}")
    print(f"Created: {completion.created}")
    print(f"Model: {completion.model}")
    print(f"System fingerprint: {completion.system_fingerprint}")

    # Usage statistics
    usage = completion.usage
    print(f"Prompt tokens: {usage.prompt_tokens}")
    print(f"Completion tokens: {usage.completion_tokens}")
    print(f"Total tokens: {usage.total_tokens}")

    # Choice details
    choice = completion.choices[0]
    print(f"Finish reason: {choice.finish_reason}")
    print(f"Index: {choice.index}")

    # Message content
    message = choice.message
    print(f"Role: {message.role}")
    print(f"Content: {message.content}")
    print(f"Parsed: {message.parsed}")  # Your structured Pydantic object
    print(f"Refusal: {message.refusal}")  # If model refused to respond

    OUTPUT1 = completion.binary_score
    console.print(f"[green italic]binary_score: {result}[/]")

    OUTPUT = result.binary_score
    console.print(f"[green]Evaluator Result: {result}[/]")
    print(f"evaluator_router: OUTPUT: {OUTPUT}")

    #################### EVALS01 ####################
    #
    # This can be standardised during development
    # DATE|COMPONENT_CODE|MODEL|TEMPERATURE|INPUT|OUTPUT and any optional fields
    #
    with open(
        "./src/case_study1/langgraph/01_article_writer_should_write.csv",
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            f"{get_report_date()}|ARTICLE_WRITER|EVALUATOR|{MODEL}|{TEMPERATURE}|{INPUT}|{OUTPUT}|{result}|\n"
        )
    ##############################################

    if result.binary_score == "yes":
        print("NEXT: EDITOR")
        return "editor"
    else:
        print("NEXT: END")
        return "not_relevant"


def translate_article(state: AgentState) -> AgentState:
    # print(f"translate_article: Current state: {state}")
    article = state["article_state"]
    llm_translation = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    translation_system = f"""You are a translator converting articles into {LANGUAGE}. Translate the text accurately while maintaining the original tone and style."""
    translation_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", translation_system),
            ("human", "Article to translate:\n\n {article}"),
        ]
    )

    translator = translation_prompt | llm_translation

    result = translator.invoke({"article": article})

    INPUT = article
    result = translator.invoke({"article": article})
    OUTPUT = result
    #################### EVALS02 ####################
    #
    # This can be standardised during development
    # DATE|COMPONENT_CODE|MODEL|TEMPERATURE|INPUT|OUTPUT and any optional fields
    #
    with open(
        "./src/case_study1/langgraph/02_article_writer_translate.csv",
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            f"{get_report_date()}|ARTICLE_WRITER|TRANSLATE|{MODEL}|{TEMPERATURE}|{INPUT}|{OUTPUT}|{result}|\n"
        )
    ##############################################
    state["article_state"] = result.content
    return state


def expand_article(state: AgentState) -> AgentState:
    article = state["article_state"]
    llm_expansion = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    expansion_system = f"""You are a writer tasked with expanding the given article to at approximately {CONTENT_LENGTH} words, with some variation either side, while maintaining relevance, coherence, and the original tone."""
    expansion_prompt = ChatPromptTemplate.from_messages(
        [("system", expansion_system), ("human", "Original article:\n\n {article}")]
    )

    expander = expansion_prompt | llm_expansion

    print(f"expand_article: Current state: {state}")
    article = state["article_state"]
    INPUT = article
    result = expander.invoke({"article": article})
    # print(result)
    OUTPUT = result.content
    state["article_state"] = result.content
    #################### EVALS03 ####################
    #
    # This can be standardised during development
    # DATE|COMPONENT_CODE|MODEL|TEMPERATURE|INPUT|OUTPUT and any optional fields
    #
    with open(
        "./src/case_study1/langgraph/03_article_writer_expand.csv",
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            f"{get_report_date()}|ARTICLE_WRITER|EXPANDER|{MODEL}|{TEMPERATURE}|{INPUT}|{OUTPUT}|{result}|\n"
        )
    ##############################################
    return state


def editor_router(
    state: AgentState,
) -> Literal["translator", "publisher", "expander"]:
    TEMPERATURE = 0.5
    MODEL = "gpt-4o-mini"
    llm_postability = ChatOpenAI(model=MODEL, temperature=TEMPERATURE)
    structured_llm_postability_grader = llm_postability.with_structured_output(
        ArticlePostabilityGrader
    )

    postability_system = f"""You are a grader assessing whether a news article is ready to be posted, if it meets the minimum word count of {CONTENT_LENGTH} words, is not written in a sensationalistic style, and if it is in {LANGUAGE}. \n
        Evaluate the article for grammatical errors, completeness, appropriateness for publication, and EXAGERATED sensationalism. \n
        Also, confirm if the language used in the article is {LANGUAGE} and it meets the word count requirement. \n
        Provide four binary scores: one to indicate if the article can be posted ('yes' or 'no'), one for adequate word count ('yes' or 'no'), one for sensationalistic writing ('yes' or 'no'), and another if the language is {LANGUAGE} ('yes' or 'no')."""
    postability_grade_prompt = ChatPromptTemplate.from_messages(
        [("system", postability_system), ("human", "News Article:\n\n {article}")]
    )

    editor = postability_grade_prompt | structured_llm_postability_grader

    article = state["article_state"]
    result = editor.invoke({"article": article})
    print(f"news_chef_router: Current state: {state}")
    print("Editor result: ", result)
    INPUT = article
    OUTPUT = result

    #################### EVALS04 ####################
    #
    # This can be standardised during development
    # DATE|COMPONENT_CODE|MODEL|TEMPERATURE|INPUT|OUTPUT and any optional fields
    #
    with open(
        "./src/case_study1/langgraph/04_article_writer_publishable.csv",
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            f"{get_report_date()}|ARTICLE_WRITER|PUBLISHER|{MODEL}|{TEMPERATURE}|{INPUT[:75]}...|{OUTPUT}|{result}|\n"
        )
    ##############################################
    num_words = len(INPUT.split())

    console.print(f"[green]Number of Words: {num_words}[/]")
    if result.can_be_posted == "yes":
        return "publisher"
    elif result.is_language_french == "yes":
        if result.meets_word_count == "no" or result.is_not_sensationalistic == "no":
            return "expander"
    return "translator"


workflow = StateGraph(AgentState)

workflow.add_node("should_write_article", get_transfer_news_grade)
workflow.add_node("editor", evaluate_article)
workflow.add_node("translator", translate_article)
workflow.add_node("expander", expand_article)
workflow.add_node("publisher", publisher)

workflow.set_entry_point("should_write_article")

workflow.add_conditional_edges(
    "should_write_article", evaluator_router, {"editor": "editor", "not_relevant": END}
)
workflow.add_conditional_edges(
    "editor",
    editor_router,
    {"translator": "translator", "publisher": "publisher", "expander": "expander"},
)
workflow.add_edge("translator", "editor")
workflow.add_edge("expander", "editor")
workflow.add_edge("publisher", END)

app = workflow.compile()

# Run tests...

print("\n======================================")
print("FIRST EXAMPLE...\n")
initial_state = {"article_state": "Planning permission for 49 England Road sought"}
result = app.invoke(initial_state)

print("Final result:", result)

print("\n======================================")
print("SECOND EXAMPLE...\n")
initial_state = {"article_state": "AI and Machine learning in modern business"}
result = app.invoke(initial_state)

print("Final article:\n\n", result["article_state"])
