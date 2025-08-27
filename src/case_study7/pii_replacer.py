import os

import openai
from dotenv import load_dotenv, find_dotenv
from rich.console import Console

console = Console()
load_dotenv(find_dotenv(), override=True)  # Load environment variables from .env file

console = Console()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if OPENAI_API_KEY:
    console.print(
        f"[green]OpenAI API Key exists and begins {OPENAI_API_KEY[:14]}...[/]"
    )
else:
    console.print("[red]OpenAI API Key not set[/]")

MODEL = "gpt-4o-mini"
console.print(f"[dark_orange]Model selected: {MODEL}[/]")


def replace_pii(text, api_key):
    """
    Replace PII in text using GPT-4o-mini
    """
    client = openai.OpenAI(api_key=OPENAI_API_KEY)

    prompt = f"""Replace any personally identifiable information (PII) in the following text with asterisks (*). 
PII includes names, email addresses, phone numbers, addresses, social security numbers, credit card numbers, etc.
Return only the modified text with PII replaced by asterisks.

Text: {text}"""

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0,  # For consistent results
    )

    return response.choices[0].message.content


# Example usage
if __name__ == "__main__":

    # Example text with PII
    sample_text = "Hi, my name is John Smith and my email is john.smith@example.com. Please call me at 555-123-4567."

    # Replace PII
    cleaned_text = replace_pii(sample_text, OPENAI_API_KEY)
    print("Original:", sample_text)
    print("Cleaned: ", cleaned_text)
