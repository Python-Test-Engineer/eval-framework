# ### Team of Agents with a supervisor


# https://www.youtube.com/watch?v=9HhcFiSgLok&list=PLNVqeXDm5tIqUIPQHLk5Xw5mpisruvsac&index=7


from datetime import datetime

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


from langgraph.graph import StateGraph, END
from typing import TypedDict, Literal

from pydantic import BaseModel, Field

import warnings

warnings.filterwarnings("ignore")


def get_report_date():
    """
    Returns the current date and time formatted as a string.
    """
    return datetime.now().strftime("%Y-%m-%d-%H-%M-%S")


class TransferNewsGrader(BaseModel):
    """Binary score for relevance check on football transfer news."""

    binary_score: str = Field(
        description="The article is about football transfers, 'yes' or 'no'"
    )


class ArticlePostabilityGrader(BaseModel):
    """Binary scores for postability check, word count, sensationalism, and language verification of a news article."""

    can_be_posted: str = Field(
        description="The article is ready to be posted, 'yes' or 'no'"
    )
    meets_word_count: str = Field(
        description="The article has at least 50 words, 'yes' or 'no'"
    )
    is_not_sensationalistic: str = Field(
        description="The article is NOT written in a sensationalistic style, 'yes' or 'no'"
    )
    is_language_german: str = Field(
        description="The language of the article is German, 'yes' or 'no'"
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


def translate_article(state: AgentState) -> AgentState:
    # print(f"translate_article: Current state: {state}")

    llm_translation = ChatOpenAI(model="gpt-4o-mini", temperature=0)

    translation_system = """You are a translator converting articles into German. Translate the text accurately while maintaining the original tone and style."""
    translation_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", translation_system),
            ("human", "Article to translate:\n\n {article}"),
        ]
    )

    translator = translation_prompt | llm_translation

    result = translator.invoke(
        {
            "article": "It has been reported that Messi will transfer from Real Madrid to FC Barcelona."
        }
    )
    article = state["article_state"]
    result = translator.invoke({"article": article})
    state["article_state"] = result.content
    return state


def expand_article(state: AgentState) -> AgentState:

    llm_expansion = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    expansion_system = """You are a writer tasked with expanding the given article to at least 200 words while maintaining relevance, coherence, and the original tone."""
    expansion_prompt = ChatPromptTemplate.from_messages(
        [("system", expansion_system), ("human", "Original article:\n\n {article}")]
    )

    expander = expansion_prompt | llm_expansion

    # print(f"expand_article: Current state: {state}")
    article = state["article_state"]
    result = expander.invoke({"article": article})
    state["article_state"] = result.content
    return state


def publisher(state: AgentState) -> AgentState:
    # print(f"publisher: Current state: {state}")
    # print("FINAL_STATE in publisher:", state)
    return state


def evaluator_router(state: AgentState) -> Literal["editor", "not_relevant"]:
    article = state["article_state"]

    MODEL = "gpt-4o-mini"
    TEMPERATURE = 0

    llm = ChatOpenAI(model=MODEL, temperature=TEMPERATURE)
    structured_llm_grader = llm.with_structured_output(TransferNewsGrader)

    system = """You are a grader assessing whether a news article concerns a football transfer. \n
        Check if the article explicitly mentions player transfers between clubs, potential transfers, or confirmed transfers. \n
        Provide a binary score 'yes' or 'no' to indicate whether the news is about a football transfer."""
    grade_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", "News Article:\n\n {article}")]
    )
    # should_write = grade_prompt | structured_llm_grader
    evaluator = grade_prompt | structured_llm_grader
    result = evaluator.invoke({"article": article})

    if result.binary_score == "yes":
        return "editor"
    else:
        return "not_relevant"


def editor_router(
    state: AgentState,
) -> Literal["translator", "publisher", "expander"]:
    TEMPERATURE = 0.5
    MODEL = "gpt-4o-mini"
    llm_postability = ChatOpenAI(model=MODEL, temperature=TEMPERATURE)
    structured_llm_postability_grader = llm_postability.with_structured_output(
        ArticlePostabilityGrader
    )

    postability_system = """You are a grader assessing whether a news article is ready to be posted, if it meets the minimum word count of 50 words, is not written in a sensationalistic style, and if it is in German. \n
        Evaluate the article for grammatical errors, completeness, appropriateness for publication, and EXAGERATED sensationalism. \n
        Also, confirm if the language used in the article is German and it meets the word count requirement. \n
        Provide four binary scores: one to indicate if the article can be posted ('yes' or 'no'), one for adequate word count ('yes' or 'no'), one for sensationalistic writing ('yes' or 'no'), and another if the language is German ('yes' or 'no')."""
    postability_grade_prompt = ChatPromptTemplate.from_messages(
        [("system", postability_system), ("human", "News Article:\n\n {article}")]
    )

    editor = postability_grade_prompt | structured_llm_postability_grader

    article = state["article_state"]
    result = editor.invoke({"article": article})
    # print(f"news_chef_router: Current state: {state}")
    # print("Editor result: ", result)
    if result.can_be_posted == "yes":
        return "publisher"
    elif result.is_language_german == "yes":
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

# Run examples...

print("\n======================================")
print("FIRST EXAMPLE...\n")
initial_state = {"article_state": "The Pope will visit Spain today"}
result = app.invoke(initial_state)
print("NOT relevant article example")
print("Final result:", result)

print("\n======================================")
print("SECOND EXAMPLE...\n")
initial_state = {"article_state": "Messi gonna switch from barca to real madrid"}
result = app.invoke(initial_state)
print("Article is RELEVANT and needs to be written\n")
print("Final article:\n\n", result["article_state"])
