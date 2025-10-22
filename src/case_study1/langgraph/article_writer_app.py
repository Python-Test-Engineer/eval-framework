# ### Team of Agents with a supervisor

# https://www.youtube.com/watch?v=9HhcFiSgLok&list=PLNVqeXDm5tIqUIPQHLk5Xw5mpisruvsac&index=7

import os
from datetime import datetime
import warnings
import time
from random import randint

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables.graph import MermaidDrawMethod
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


# NEW: Publisher Tool Functions
def get_article_price_randomly() -> int:
    """Tool that returns a random price for the article between 10-90 GBP"""
    price = randint(10, 90)
    console.print(f"[blue]📊 Article Price Tool: Generated price £{price}[/]")
    return price


def get_article_price_based_on_word_count() -> int:
    """Tool that returns a price for the article between 10-90 GBP based on word count and your own logic"""
    price = randint(50, 200) / 2.5
    console.print(f"[blue]📊 Article Price Tool: Generated price £{price}[/]")
    return price


def rate_article_price(price: int) -> str:
    """Tool that rates the price based on predefined ranges"""
    if 10 <= price <= 40:
        rating = "VERY_GOOD_VALUE"
    elif 41 <= price <= 70:
        rating = "GOOD_VALUE"
    else:  # price > 70
        rating = "EXPENSIVE"

    console.print(f"[yellow]💰 Price Rating Tool: £{price} is rated as {rating}[/]")
    return rating


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


class PublisherPricingResponse(BaseModel):
    """Response model for publisher pricing decisions"""

    article_price: int = Field(
        description="The determined price for the article in GBP (between 10-90)"
    )
    price_rating: str = Field(
        description="The cost-effectiveness rating: VERY_GOOD_VALUE, GOOD_VALUE, or EXPENSIVE"
    )
    pricing_justification: str = Field(
        description="Brief explanation of why this price and rating were chosen"
    )


class AgentState(TypedDict):
    article_state: str


def get_transfer_news_grade(state: AgentState) -> AgentState:
    return state


def evaluate_article(state: AgentState) -> AgentState:
    return state


def publisher(state: AgentState) -> AgentState:
    console.print(
        f"[magenta bold]📰 PUBLISHER NODE: Processing article for publication[/]"
    )
    print(f"publisher: Current state: {state}")

    # Set up the agentic publisher with LLM
    MODEL = "gpt-4o-mini"
    TEMPERATURE = 0.3
    llm_publisher = ChatOpenAI(model=MODEL, temperature=TEMPERATURE)
    structured_llm_publisher = llm_publisher.with_structured_output(
        PublisherPricingResponse
    )

    article = state["article_state"]
    word_count = len(article.split())

    # Randomly decide which pricing approach to use
    use_word_count_pricing = randint(0, 1) == 1  # 50/50 chance

    # Enhanced agentic publisher prompt with tool usage instructions
    if use_word_count_pricing:
        pricing_instruction = """You should use the get_article_price_based_on_word_count() tool to determine the base price. 
        This tool considers word count in its pricing logic. Call this tool and then you may adjust the price 
        slightly based on content quality, but stay within the 10-90 GBP range."""
        expected_tool = "get_article_price_based_on_word_count"
    else:
        pricing_instruction = """You should use the get_article_price_randomly() tool to get a random price between 10-90 GBP. 
        Call this tool and then you may adjust the price based on article quality and word count, 
        but stay within the 10-90 GBP range."""
        expected_tool = "get_article_price_randomly"
    #region PRICING
    publisher_system = f"""As a publisher you determine the price of the article and also rate its cost-effectiveness.

    PRICING STRATEGY:
    {pricing_instruction}
    
    After getting the tool result, you can make final adjustments but must stay within 10-90 GBP range.
    
    Rate the cost-effectiveness as:
    - VERY_GOOD_VALUE: 10-40 GBP (excellent value for money)
    - GOOD_VALUE: 41-70 GBP (reasonable pricing)  
    - EXPENSIVE: 71-90 GBP (premium pricing)
    
    Then use the rate_article_price() tool to get the official rating for your chosen price.
    
    Provide your final pricing decision with justification explaining:
    1. Which tool you used and why
    2. Any adjustments you made to the tool result
    3. How the final price reflects the article's value
    
    Available tools:
    - get_article_price_randomly(): Returns random price 10-90 GBP
    - get_article_price_based_on_word_count(): Returns price based on word count logic
    - rate_article_price(price): Returns rating for a given price"""
    #endregion PUBLISHER_AGENTIC_PROMPT
    publisher_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", publisher_system),
            ("human", "Article to price (Word count: {word_count}):\n\n{article}"),
        ]
    )

    # Simulate tool usage (since we can't actually bind tools to the structured output)
    start = time.perf_counter()

    # Call the appropriate pricing tool
    tool_calls = []

    if use_word_count_pricing:
        tool_price = get_article_price_based_on_word_count()
        tool_calls.append(
            {
                "tool_name": "get_article_price_based_on_word_count",
                "arguments": {},
                "result": tool_price,
            }
        )
    else:
        tool_price = get_article_price_randomly()
        tool_calls.append(
            {
                "tool_name": "get_article_price_randomly",
                "arguments": {},
                "result": tool_price,
            }
        )

    # Create enhanced prompt with tool result
    enhanced_prompt = f"""Article to price (Word count: {word_count}):

{article}

TOOL RESULT: The {expected_tool} tool returned: £{tool_price}

Based on this tool result, determine your final price (can be the same or adjusted within 10-90 GBP) and provide justification."""

    # Get pricing decision from the agent
    result = structured_llm_publisher.invoke(
        [
            {"role": "system", "content": publisher_system},
            {"role": "human", "content": enhanced_prompt},
        ]
    )

    # Get the rating for the final price
    price_rating = rate_article_price(result.article_price)
    tool_calls.append(
        {
            "tool_name": "rate_article_price",
            "arguments": {"price": result.article_price},
            "result": price_rating,
        }
    )

    end = time.perf_counter()
    time_taken = end - start

    console.print(
        f"[cyan]🤖 Publisher Agent Execution Time: {time_taken:.2f} seconds[/]"
    )

    # Extract results
    article_price = result.article_price
    justification = result.pricing_justification

    # Enhanced agent decision tracking
    tool_summary = " | ".join(
        [
            f"{tc['tool_name']}({', '.join([f'{k}={v}' for k, v in tc['arguments'].items()])})={tc['result']}"
            for tc in tool_calls
        ]
    )
    agent_decision = f"LLM_PRICING: strategy={expected_tool}, tools_used=[{tool_summary}], final_price={article_price}, rating={price_rating}"

    # Display results
    console.print(f"[green bold]✅ PUBLICATION SUMMARY:[/]")
    console.print(f"[white]   • Pricing Strategy: {expected_tool}[/]")
    console.print(f"[white]   • Tool Price: £{tool_price}[/]")
    console.print(f"[white]   • Final Article Price: £{article_price}[/]")
    console.print(f"[white]   • Price Rating: {price_rating}[/]")
    console.print(f"[white]   • Article Length: {word_count} words[/]")
    console.print(f"[white]   • Tools Called: {len(tool_calls)}[/]")
    console.print(f"[white]   • Pricing Justification: {justification}[/]")

    # Calculate tokens for cost tracking
    input_text = f"{publisher_system}\n{enhanced_prompt}"
    input_tokens = count_tokens(input_text, MODEL)
    output_tokens = count_tokens(str(result), MODEL)

    console.print(
        f"[cyan]   • Token Usage: Input={input_tokens}, Output={output_tokens}, Total={input_tokens + output_tokens}[/]"
    )

    # Enhanced logging with tool details
    tool_calls_log = "|".join(
        [f"{tc['tool_name']}({tc['arguments']})={tc['result']}" for tc in tool_calls]
    )
    # region EVALS05
    with open(
        "./src/case_study1/langgraph/05_article_writer_publisher_pricing.csv",
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            f"{get_report_date()}|ARTICLE_WRITER|PUBLISHER_AGENTIC|{MODEL}|{TEMPERATURE}|"
            f"{article_price}|{price_rating}|{word_count}|{input_tokens}|{output_tokens}|{time_taken:.2f}|"
            f"{justification.replace('|', ';')}|{agent_decision}|{expected_tool}|{tool_price}|"
            f"{tool_calls_log.replace('|', ';')}|{len(tool_calls)}\n"
        )
    # endregion EVALS05
    print("FINAL_STATE in publisher:", state)
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
    # region IS_AI
    system = f"""You are a researcher that determines the content type of an article.
        Check if the article refers to {SUBJECT} area.
        Provide a binary score 'yes' or 'no' to indicate whether the article is technical in nature."""

    grade_prompt = ChatPromptTemplate.from_messages(
        [("system", system), ("human", human_prompt)]
    )
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

    output_tokens = count_tokens(str(result), MODEL)
    print(f"01 Estimated output tokens: {output_tokens}")
    print(f"01 Estimated total tokens: {input_tokens + output_tokens}")
    # region EVALS01
    with open(
        "./src/case_study1/langgraph/01_article_writer_should_write.csv",
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            f"{get_report_date()}|ARTICLE_WRITER|EVALUATOR|{MODEL}|{TEMPERATURE}|{INPUT}|{OUTPUT}|{time_taken:.2f}\n"
        )
    # endregion EVALS01
    if result.binary_score == "yes":
        print("NEXT: EDITOR")
        return "editor"
    else:
        print("NEXT: END")
        return "not_relevant"


def translate_article(state: AgentState) -> AgentState:
    article = state["article_state"]
    llm_translation = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    # region TRANSLATE
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
    # region EVALS02
    with open(
        "./src/case_study1/langgraph/02_article_writer_translate.csv",
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            f"{get_report_date()}|ARTICLE_WRITER|TRANSLATE|{MODEL}|{TEMPERATURE}|{INPUT}|{time_taken:.2f}|{result_dict}\n"
        )
    # endregion EVALS02
    state["article_state"] = result.content
    return state


def expand_article(state: AgentState) -> AgentState:
    article = state["article_state"]
    llm_expansion = ChatOpenAI(model="gpt-4o-mini", temperature=0.5)
    # regiond EXPANDER
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
    # region EVALS03
    with open(
        "./src/case_study1/langgraph/03_article_writer_expand.csv",
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            f"{get_report_date()}|ARTICLE_WRITER|EXPANDER|{MODEL}|{TEMPERATURE}|{INPUT}|{time_taken:.2f}|{result_dict}\n"
        )
    # endregion EVALS03
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
    # region CAN_POST
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

    output_tokens = count_tokens(str(OUTPUT), MODEL)
    print(f"Estimated output tokens: {output_tokens}")
    print(f"Estimated total tokens: {input_tokens + output_tokens}")
    # region EVALS04
    with open(
        "./src/case_study1/langgraph/04_article_writer_publishable.csv",
        "a",
        encoding="utf-8",
    ) as f:
        f.write(
            f"{get_report_date()}|ARTICLE_WRITER|PUBLISHER|{MODEL}|{TEMPERATURE}|{INPUT[:75]}...|{OUTPUT}|{input_tokens}|{output_tokens}|{time_taken:.2f}\n"
        )
    # endregion EVALS04
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


img = app.get_graph().draw_mermaid_png(
    draw_method=MermaidDrawMethod.API,
)
with open("05_article_writer_publisher_pricing_workflow.png", "wb") as f:
    f.write(img)
# Run tests...

NUM_TITLES = len(blog_titles)
TITLE_LIMIT = randint(1, NUM_TITLES)  # Randomly choose a limit for testing

# run for all 30 title pairs
NUM_EXAMPLES = 6
for i in range(NUM_EXAMPLES):
    print("\n======================================")
    print("NON AI EXAMPLE...\n")
    non_ai_article = blog_titles[i][1]
    print(f"Non-AI article: {non_ai_article}")
    initial_state = {"article_state": blog_titles[i][1]}
    result = app.invoke(initial_state)

    print("\n======================================")
    print("AI EXAMPLE...\n")
    ai_article = blog_titles[i][0]
    print(f"AI article: {ai_article}")
    initial_state = {"article_state": blog_titles[i][0]}
    result = app.invoke(initial_state)
