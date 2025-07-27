# sample_dataset.py
"""
Sample dataset for RAG (Retrieval Augmented Generation) testing.
This dataset contains questions, contexts, and ground truth answers for evaluation.
"""

import pandas as pd
from typing import List, Dict


def create_sample_dataset() -> pd.DataFrame:
    """Create a sample dataset for RAG evaluation."""

    # Sample data about technology companies
    data = [
        {
            "question": "What is the mission of OpenAI?",
            "contexts": [
                "OpenAI is an AI research and deployment company. Our mission is to ensure that artificial general intelligence (AGI) benefits all of humanity.",
                "OpenAI conducts research in machine learning and artificial intelligence with the goal of promoting and developing friendly AI.",
                "The company was founded in 2015 by several tech entrepreneurs including Elon Musk and Sam Altman.",
            ],
            "ground_truth": "OpenAI's mission is to ensure that artificial general intelligence (AGI) benefits all of humanity.",
            "answer": "OpenAI's mission is to ensure that artificial general intelligence benefits all of humanity by conducting research and developing safe AI systems.",
        },
        {
            "question": "When was Microsoft founded?",
            "contexts": [
                "Microsoft Corporation was founded by Bill Gates and Paul Allen on April 4, 1975.",
                "The company started in Albuquerque, New Mexico, and later moved to Washington state.",
                "Microsoft became one of the world's largest software companies, known for Windows and Office products.",
            ],
            "ground_truth": "Microsoft was founded on April 4, 1975.",
            "answer": "Microsoft was founded on April 4, 1975, by Bill Gates and Paul Allen.",
        },
        {
            "question": "What programming language was created by Google?",
            "contexts": [
                "Google created the Go programming language, also known as Golang, which was announced in 2009.",
                "Go was designed by Robert Griesemer, Rob Pike, and Ken Thompson at Google.",
                "The language was created to address shortcomings in other languages used at Google for server-side development.",
            ],
            "ground_truth": "Google created the Go programming language (Golang).",
            "answer": "Google created the Go programming language, also known as Golang, which was announced in 2009.",
        },
        {
            "question": "What is the primary business model of Amazon?",
            "contexts": [
                "Amazon started as an online bookstore but has evolved into a massive e-commerce and cloud computing platform.",
                "Amazon's revenue comes from multiple sources including e-commerce, Amazon Web Services (AWS), advertising, and subscription services.",
                "AWS is Amazon's cloud computing division and has become one of the most profitable parts of the business.",
            ],
            "ground_truth": "Amazon's primary business model includes e-commerce, cloud computing (AWS), advertising, and subscription services.",
            "answer": "Amazon's primary business model is e-commerce, but they also generate significant revenue from cloud computing services through AWS, advertising, and subscription services.",
        },
        {
            "question": "Who is the current CEO of Tesla?",
            "contexts": [
                "Elon Musk is the CEO and product architect of Tesla, Inc.",
                "Tesla is an electric vehicle and clean energy company founded in 2003.",
                "Under Musk's leadership, Tesla has become one of the world's most valuable automakers.",
            ],
            "ground_truth": "Elon Musk is the current CEO of Tesla.",
            "answer": "Elon Musk is the current CEO of Tesla, Inc.",
        },
        {
            "question": "What is artificial intelligence?",
            "contexts": [
                "Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems.",
                "AI includes learning, reasoning, problem-solving, perception, and language understanding.",
                "Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.",
            ],
            "ground_truth": "Artificial Intelligence is the simulation of human intelligence processes by machines, including learning, reasoning, and problem-solving.",
            "answer": "Artificial Intelligence (AI) is the simulation of human intelligence in machines, enabling them to learn, reason, solve problems, and understand language.",
        },
    ]

    return pd.DataFrame(data)


def save_dataset(df: pd.DataFrame, filename: str = "./src/ragas/03_rag_test_dataset.csv"):
    """Save the dataset to a CSV file."""
    df.to_csv(filename, index=False)
    print(f"Dataset saved to {filename}")


if __name__ == "__main__":
    # Create and save the dataset
    dataset = create_sample_dataset()
    print("Sample Dataset:")
    print(dataset.head())
    save_dataset(dataset)
