import os
from langgraph.graph import Graph, END
from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
from typing import Dict, Any
from dotenv import load_dotenv, find_dotenv
from rich.console import Console
from openai import OpenAI

console = Console()

load_dotenv(find_dotenv(), override=True)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
MODEL = "gpt-4o-mini"


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

# Initialize OpenAI LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)


def remove_pii(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 1: Remove PII from user message using OpenAI LLM
    Uses LLM to identify and replace PII with asterisks
    """
    message = state["user_message"]

    # Use LLM to identify and replace PII
    pii_prompt = f"""
    You are a PII (Personally Identifiable Information) detection and removal system.
    
    Please identify and replace ALL PII in the following message with asterisks.
    Replace these types of information:
    - Names (first, last, full names) → ***
    - Email addresses → ***@***.***
    - Phone numbers → ***-***-****
    - Social Security Numbers → ***-**-****
    - Credit card numbers → ****-****-****-****
    - Addresses → *** *** Street
    - Other identifying information
    
    Return ONLY the message with PII replaced by asterisks. Do not add any explanations or additional text.
    
    Original message: "{message}"
    """

    response = llm.invoke([HumanMessage(content=pii_prompt)])
    cleaned_message = response.content.strip()

    print(f"PII Check: LLM processed message for PII removal")

    # Update state with cleaned message
    state["cleaned_message"] = cleaned_message
    return state


def check_self_destructive_sentiment(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 2: Check for self-destructive sentiment
    If detected, sets flag to end workflow with help message
    """
    cleaned_message = state["cleaned_message"]

    # Use LLM to analyze sentiment for self-destructive content
    sentiment_prompt = f"""
    Analyze this message for self-destructive sentiment, suicidal ideation, or self-harm content.
    Respond with only 'YES' if self-destructive content is detected, or 'NO' if it's safe.
    
    Message: "{cleaned_message}"
    """

    response = llm.invoke([HumanMessage(content=sentiment_prompt)])
    is_destructive = response.content.strip().upper() == "YES"

    print(f"Sentiment Check: Self-destructive content detected: {is_destructive}")

    # Set flags based on sentiment analysis
    state["is_self_destructive"] = is_destructive
    if is_destructive:
        state["final_response"] = "PLEASE GET HELP"
        state["should_end"] = True
    else:
        state["should_end"] = False

    return state


def provide_supportive_advice(state: Dict[str, Any]) -> Dict[str, Any]:
    """
    Node 3: Provide helpful supportive advice
    Only reached if no self-destructive content was detected
    """
    cleaned_message = state["cleaned_message"]

    # Generate supportive advice using LLM
    advice_prompt = f"""
    Provide helpful, supportive, and empathetic advice for this message.
    Be encouraging and constructive. Keep the response warm but professional.
    
    Message: "{cleaned_message}"
    """

    response = llm.invoke([HumanMessage(content=advice_prompt)])

    print(f"Advice Generation: Supportive response generated")

    # Set final response
    state["final_response"] = response.content.strip()
    return state


def should_end_early(state: Dict[str, Any]) -> str:
    """
    Conditional function to determine if workflow should end after sentiment check
    """
    if state.get("should_end", False):
        return END
    return "provide_advice"


# Create the workflow graph
def create_workflow():
    """
    Create and configure the LangGraph workflow
    """
    workflow = Graph()

    # Add nodes to the graph
    workflow.add_node("remove_pii", remove_pii)
    workflow.add_node("check_sentiment", check_self_destructive_sentiment)
    workflow.add_node("provide_advice", provide_supportive_advice)

    # Define the flow between nodes
    workflow.set_entry_point("remove_pii")
    workflow.add_edge("remove_pii", "check_sentiment")

    # Conditional edge: either end or continue to advice
    workflow.add_conditional_edges(
        "check_sentiment",
        should_end_early,
        {END: END, "provide_advice": "provide_advice"},
    )

    # Final edge from advice to end
    workflow.add_edge("provide_advice", END)

    return workflow.compile()


# Example usage
def process_user_message(user_message: str) -> str:
    """
    Process a user message through the complete workflow
    """
    print(f"\n=== Processing Message ===")
    print(f"Input: {user_message}")

    # Create workflow
    app = create_workflow()

    # Initial state
    initial_state = {
        "user_message": user_message,
        "cleaned_message": "",
        "is_self_destructive": False,
        "should_end": False,
        "final_response": "",
    }

    # Run the workflow
    result = app.invoke(initial_state)

    print(f"Output: {result['final_response']}")
    return result["final_response"]


# Test the workflow
if __name__ == "__main__":
    # Test cases
    test_messages = [
        "Hi, I'm John Doe and my email is john.doe@email.com. I'm feeling overwhelmed with work lately.",
        "I can't take it anymore, I want to end it all",
        "My phone number is 555-123-4567. Can you help me plan my career transition?",
    ]

    for msg in test_messages:
        response = process_user_message(msg)
        print("-" * 50)
