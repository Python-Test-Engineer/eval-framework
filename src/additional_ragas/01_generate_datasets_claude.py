import os
from typing import List
from datasets import Dataset
from ragas.testset import TestsetGenerator
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document

from dotenv import load_dotenv, find_dotenv
from rich.console import Console

console = Console()

load_dotenv(find_dotenv(), override=True)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
MODEL = "gpt-4o-mini"

OUTPUT_FILE = "./src/ragas/01_generate_synthetic_langchain_agents_data.json"


def create_sample_documents() -> List[Document]:
    """Create sample documents about LangChain, OpenAI, and agents."""

    sample_texts = [
        """
        LangChain is a framework for developing applications powered by language models. 
        It provides a standard interface for chains, lots of integrations with other tools, 
        and end-to-end chains for common applications. LangChain enables developers to 
        build context-aware and reasoning applications by connecting language models to 
        other sources of computation or knowledge.
        """,
        """
        OpenAI agents are AI systems that can understand and execute complex tasks by 
        breaking them down into smaller steps. These agents use large language models 
        like GPT-4 to reason about problems, make decisions, and interact with various 
        tools and APIs. They can perform tasks like web browsing, code execution, 
        and data analysis autonomously.
        """,
        """
        LangChain agents are designed to use language models to choose a sequence of 
        actions to take. They have access to a suite of tools and can decide which 
        tool to use based on the user input. Popular agent types include ReAct agents, 
        which combine reasoning and acting, and conversational agents that maintain 
        context across interactions.
        """,
        """
        The integration between LangChain and OpenAI enables powerful agent architectures. 
        Developers can create agents that leverage OpenAI's language models while using 
        LangChain's tool ecosystem. This combination allows for sophisticated workflows 
        involving multiple AI services, databases, and external APIs.
        """,
        """
        Agent memory in LangChain allows agents to persist information across conversations. 
        This includes conversation buffer memory, summary memory, and vector store memory. 
        Memory enables agents to maintain context, learn from previous interactions, 
        and provide more personalized responses over time.
        """,
        """
        Tool calling is a fundamental capability of modern AI agents. OpenAI's function 
        calling feature allows models to generate structured outputs that can trigger 
        specific tools or functions. LangChain provides a framework to easily integrate 
        these tool calls into agent workflows, enabling complex multi-step reasoning.
        """,
    ]

    documents = []
    for i, text in enumerate(sample_texts):
        doc = Document(
            page_content=text.strip(),
            metadata={"source": f"doc_{i}", "topic": "langchain_openai_agents"},
        )
        documents.append(doc)

    return documents


def generate_synthetic_dataset_simple(documents: List[Document], num_samples: int = 10):
    """Generate synthetic test dataset using RAGAS - simplified approach."""

    # Initialize OpenAI models
    generator_llm = ChatOpenAI(model=MODEL, temperature=0.3)
    critic_llm = ChatOpenAI(model=MODEL, temperature=0)
    embeddings = OpenAIEmbeddings()

    # Create test generator
    generator = TestsetGenerator.from_langchain(
        generator_llm=generator_llm, critic_llm=critic_llm, embeddings=embeddings
    )

    # Simple generation without distributions (works with most RAGAS versions)
    print("Generating testset...")
    testset = generator.generate_with_langchain_docs(
        documents=documents, test_size=num_samples
    )

    return testset


def manual_synthetic_data():
    """Create synthetic data manually if RAGAS has issues."""

    questions = [
        {
            "question": "What is LangChain and what does it enable developers to build?",
            "contexts": [
                "LangChain is a framework for developing applications powered by language models. It enables developers to build context-aware and reasoning applications."
            ],
            "ground_truth": "LangChain is a framework for developing applications powered by language models that enables developers to build context-aware and reasoning applications by connecting language models to other sources of computation or knowledge.",
        },
        {
            "question": "How do OpenAI agents handle complex tasks?",
            "contexts": [
                "OpenAI agents are AI systems that can understand and execute complex tasks by breaking them down into smaller steps using large language models like GPT-4."
            ],
            "ground_truth": "OpenAI agents handle complex tasks by breaking them down into smaller steps, using large language models like GPT-4 to reason about problems, make decisions, and interact with various tools and APIs.",
        },
        {
            "question": "What are the popular types of LangChain agents?",
            "contexts": [
                "Popular agent types include ReAct agents, which combine reasoning and acting, and conversational agents that maintain context across interactions."
            ],
            "ground_truth": "Popular LangChain agent types include ReAct agents that combine reasoning and acting, and conversational agents that maintain context across interactions.",
        },
        {
            "question": "What types of memory can LangChain agents use?",
            "contexts": [
                "Agent memory in LangChain includes conversation buffer memory, summary memory, and vector store memory."
            ],
            "ground_truth": "LangChain agents can use conversation buffer memory, summary memory, and vector store memory to persist information across conversations.",
        },
        {
            "question": "How does tool calling work in AI agents?",
            "contexts": [
                "Tool calling allows OpenAI models to generate structured outputs that can trigger specific tools or functions, with LangChain providing framework integration."
            ],
            "ground_truth": "Tool calling allows AI models to generate structured outputs that trigger specific tools or functions, with LangChain providing a framework to integrate these tool calls into agent workflows.",
        },
    ]

    # Convert to Dataset format
    dataset_dict = {
        "question": [item["question"] for item in questions],
        "contexts": [item["contexts"] for item in questions],
        "ground_truth": [item["ground_truth"] for item in questions],
    }

    return Dataset.from_dict(dataset_dict)


def save_dataset(dataset: Dataset, filename: str = OUTPUT_FILE):
    """Save the generated dataset to a file."""

    # Convert to pandas DataFrame for easier viewing
    df = dataset.to_pandas()

    # Save as JSON
    df.to_json(filename, orient="records", indent=2)
    print(f"Dataset saved to {filename}")

    # Display sample questions
    print("\nGenerated questions:")
    print("-" * 50)
    for i, row in df.iterrows():
        print(f"Question {i+1}: {row['question']}")
        if isinstance(row["contexts"], list):
            context = row["contexts"][0] if row["contexts"] else "No context"
        else:
            context = str(row["contexts"])
        print(f"Context: {context[:150]}...")
        print(f"Ground Truth: {row['ground_truth']}")
        print("-" * 50)

    return df


def main():
    """Main function to generate synthetic data."""

    print("Starting synthetic data generation...")

    try:
        # Try RAGAS generation first
        print("Attempting RAGAS generation...")
        documents = create_sample_documents()
        dataset = generate_synthetic_dataset_simple(documents, num_samples=8)
        print("RAGAS generation successful!")

    except Exception as e:
        print(f"RAGAS generation failed: {e}")
        print("Falling back to manual synthetic data...")
        dataset = manual_synthetic_data()

    # Save and display results
    df = save_dataset(dataset)
    print(f"\nFinal dataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")


if __name__ == "__main__":
    # Install packages:
    # pip install ragas langchain langchain-openai datasets pandas

    main()
