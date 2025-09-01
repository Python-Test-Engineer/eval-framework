# ### Team of Agents with a supervisor


# https://www.youtube.com/watch?v=9HhcFiSgLok&list=PLNVqeXDm5tIqUIPQHLk5Xw5mpisruvsac&index=7


import os
from datetime import datetime
import warnings
import time
from random import randint

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

from pydantic import BaseModel, Field
from rich.console import Console
from dotenv import load_dotenv, find_dotenv
import tiktoken
from config_langgraph import (
    PROVIDER,
    MODEL,
    TEMPERATURE,
    LANGUAGE,
    SUBJECT,
    CONTENT_LENGTH,
)

from datasets.blog_titles_list import blog_titles

print(len(blog_titles))
console = Console()
load_dotenv(find_dotenv(), override=True)

console.print(f"\n[cyan]Using {PROVIDER} provider with model {MODEL}[/]")


# MODEL = "gpt-4o-mini"
# TEMPERATURE = 0
# LANGUAGE = "FRENCH"
# SUBJECT = "AI, ML, Data Science, Programming, Web, Technology"
# CONTENT_LENGTH = 100

human_prompt = "News Article:\n\n {article}"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    console.print(f"[cyan]OpenAI API Key exists and begins {OPENAI_API_KEY[:14]}...[/]")
else:
    console.print("[red]OpenAI API Key not set[/]")


def count_tokens(text: str, model=MODEL) -> int:
    """Count tokens in text using tiktoken"""
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


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
    is_correct_language: str = Field(
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
    print(f"evaluator_router:\n\tINPUT: {article}")
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
        [("system", system), ("human", human_prompt)]
    )
    # should_write = grade_prompt | structured_llm_grader
    evaluator = grade_prompt | structured_llm_grader
    start = time.perf_counter()
    result = evaluator.invoke({"article": article})
    end = time.perf_counter()
    time_taken = end - start
    print(f"Execution time: {time_taken:.2f} seconds")
    print("RESULT:")
    print(result)
    print("END RESULT")
    OUTPUT = result.binary_score
    console.print(f"OUTPUT -> [green italic]binary_score: {OUTPUT}[/]")

    input_tokens = count_tokens(human_prompt, MODEL)
    print(f"01 Estimated input tokens: {input_tokens}")

    # Count output tokens
    output_tokens = count_tokens(str(result), MODEL)
    print(f"01 Estimated output tokens: {output_tokens}")
    print(f"01 Estimated total tokens: {input_tokens + output_tokens}")
    # NOTES:
    # In reality we add a trace_id to group the evaluations together and we add a span_id to identify the individual evaluation
    #################### EVALS01 ####################
    #
    # TRACE_ID|DATETIME|APP|SPAN_ID|MODEL|TEMPERATURE|INPUT|OUTPUT|INPUT_TOKENS|OUTPUT_TOKENS|TIME and any optional fields
    #
    with open(
        "./src/case_study1/langgraph/01_article_writer_should_write.csv",
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            f"{get_report_date()}|ARTICLE_WRITER|EVALUATOR|{MODEL}|{TEMPERATURE}|{INPUT}|{OUTPUT}|{time_taken:.2f}\n"
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

    start = time.perf_counter()

    result = translator.invoke({"article": article})
    result_dict = result.dict()
    print(f"Result as dict: {result_dict}")
    end = time.perf_counter()
    time_taken = end - start
    print(f"Execution time: {time_taken:.2f} seconds")
    OUTPUT = result
    
    
    #################### EVALS02 ####################
    with open(
        "./src/case_study1/langgraph/02_article_writer_translate.csv",
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            f"{get_report_date()}|ARTICLE_WRITER|TRANSLATE|{MODEL}|{TEMPERATURE}|{INPUT}|{time_taken:.2f}|{result_dict}\n"
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

    start = time.perf_counter()
    result = expander.invoke({"article": article})
    end = time.perf_counter()
    time_taken = end - start
    print(f"Execution time: {time_taken:.2f} seconds")
    OUTPUT = result.content
    print(type(result))
    result_dict = result.dict()
    print(f"Result as dict: {result_dict}")
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
            f"{get_report_date()}|ARTICLE_WRITER|EXPANDER|{MODEL}|{TEMPERATURE}|{INPUT}|{time_taken:.2f}|{result_dict}\n"
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
        Provide four binary scores: one to indicate if the article can be posted ('yes' or 'no'), one for adequate word count ('yes' or 'no'), one for not sensationalistic writing ('yes' or 'no'), and another if the language is {LANGUAGE} ('yes' or 'no')."""
    postability_grade_prompt = ChatPromptTemplate.from_messages(
        [("system", postability_system), ("human", human_prompt)]
    )

    editor = postability_grade_prompt | structured_llm_postability_grader

    article = state["article_state"]

    start = time.perf_counter()
    result = editor.invoke({"article": article})
    end = time.perf_counter()
    time_taken = end - start
    print(f"Execution time: {time_taken:.2f} seconds")
    print(f"news_chef_router: Current state: {state}")
    console.print(f"[dark_orange]Editor result: \n\t{result}[/]")
    INPUT = article
    OUTPUT = result
    input_tokens = count_tokens(human_prompt, MODEL)
    print(f"Estimated input tokens: {input_tokens}")

    # Count output tokens
    output_tokens = count_tokens(str(OUTPUT), MODEL)
    print(f"Estimated output tokens: {output_tokens}")
    print(f"Estimated total tokens: {input_tokens + output_tokens}")
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
            f"{get_report_date()}|ARTICLE_WRITER|PUBLISHER|{MODEL}|{TEMPERATURE}|{INPUT[:75]}...|{OUTPUT}|{input_tokens}|{output_tokens}|{time_taken:.2f}\n"
        )
    ##############################################
    num_words = len(INPUT.split())

    console.print(f"[green]Number of Words: {num_words}[/]")
    if result.can_be_posted == "yes":
        return "publisher"
    elif result.is_correct_language == "yes":
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

NUM_TITLES = len(blog_titles)
TITLE_LIMIT = randint(1, NUM_TITLES)  # Randomly choose a limit for testing

# run for all 30 title pairs
for i in range(len(blog_titles)):
    print("\n======================================")
    print("NON AI EXAMPLE...\n")
    non_ai_article = blog_titles[i][1]
    print(f"Non-AI article: {non_ai_article}")
    initial_state = {"article_state": blog_titles[i][1]}
    result = app.invoke(initial_state)

    # print("Final result:", result)

    print("\n======================================")
    print("AI EXAMPLE...\n")
    ai_article = blog_titles[i][0]
    print(f"AI article: {ai_article}")
    initial_state = {"article_state": blog_titles[i][0]}
    result = app.invoke(initial_state)

    # print("Final article:\n\n", result["article_state"])
