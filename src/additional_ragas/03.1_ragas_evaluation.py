# ragas_evaluation_v2.py
"""
RAGAS evaluation using the latest API structure.
This version addresses the 'property' object error.
"""

import os
import json
from datasets import Dataset
import pandas as pd

# from create_sample_dataset import create_sample_dataset
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)


def setup_ragas_environment():
    """Set up the RAGAS environment properly."""

    # Set OpenAI API key if not already set
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠ OpenAI API key not found in environment")
        print("Please set it with: export OPENAI_API_KEY='your-key-here'")
        return False

    # Optional: Set other environment variables for RAGAS
    os.environ["RAGAS_DO_NOT_TRACK"] = "true"  # Disable tracking

    return True


def create_ragas_dataset(df: pd.DataFrame) -> Dataset:
    """Create a dataset in the correct format for RAGAS."""

    print("Creating RAGAS dataset...")

    # Ensure contexts are lists of strings
    df_copy = df.copy()
    df_copy["contexts"] = df_copy["contexts"].apply(
        lambda x: x if isinstance(x, list) else [str(x)]
    )

    # Create the dataset
    dataset_dict = {
        "question": df_copy["question"].tolist(),
        "contexts": df_copy["contexts"].tolist(),
        "answer": df_copy["answer"].tolist(),
        "ground_truth": df_copy["ground_truth"].tolist(),
    }

    return Dataset.from_dict(dataset_dict)


def run_ragas_evaluation_v2(dataset: Dataset):
    """Run RAGAS evaluation using the correct API."""

    print("Running RAGAS evaluation...")

    try:
        # Import required components
        from ragas import evaluate
        from ragas.metrics import (
            faithfulness,
            answer_relevancy,
            context_recall,
            context_precision,
        )

        # Create metrics list
        metrics = [faithfulness, answer_relevancy, context_recall, context_precision]

        print(f"Using {len(metrics)} metrics")

        # Run evaluation
        print("Starting evaluation (this may take a few minutes)...")
        result = evaluate(dataset, metrics=metrics)

        print("Evaluation completed!")

        # Convert result to dictionary if needed
        if hasattr(result, "to_pandas"):
            # If result has a to_pandas method
            result_df = result.to_pandas()
            return result_df.to_dict()
        elif hasattr(result, "__dict__"):
            # If result is an object with attributes
            return {k: v for k, v in result.__dict__.items() if not k.startswith("_")}
        else:
            # If result is already a dict-like object
            return dict(result)

    except Exception as e:
        print(f"Evaluation error: {str(e)}")
        print(f"Error type: {type(e)}")

        # Try alternative approach
        return run_alternative_evaluation(dataset)


def run_alternative_evaluation(dataset: Dataset):
    """Alternative evaluation approach if main method fails."""

    print("Trying alternative evaluation approach...")

    try:
        from ragas import evaluate
        from ragas.metrics import faithfulness

        # Run with just one metric to test
        result = evaluate(dataset, metrics=[faithfulness])

        # Extract results manually
        results = {}

        # Try different ways to access the results
        if hasattr(result, "faithfulness"):
            results["faithfulness"] = float(result.faithfulness)

        # Try accessing as dict
        try:
            if "faithfulness" in result:
                results["faithfulness"] = float(result["faithfulness"])
        except:
            pass

        # Try accessing result attributes
        for attr in dir(result):
            if not attr.startswith("_") and not callable(getattr(result, attr)):
                try:
                    value = getattr(result, attr)
                    if isinstance(value, (int, float)):
                        results[attr] = value
                except:
                    pass

        return results

    except Exception as e:
        print(f"Alternative evaluation also failed: {str(e)}")
        return {}


def display_results_v2(results: dict):
    """Display results in a formatted way."""

    if not results:
        print("No results to display.")
        return

    print("\n" + "=" * 60)
    print("RAGAS EVALUATION RESULTS")
    print("=" * 60)

    # Calculate average if we have multiple numeric results
    numeric_results = {k: v for k, v in results.items() if isinstance(v, (int, float))}

    if numeric_results:
        for metric, score in numeric_results.items():
            print(f"{metric.upper()}: {score:.4f}")

        if len(numeric_results) > 1:
            avg_score = sum(numeric_results.values()) / len(numeric_results)
            print(f"\nAVERAGE SCORE: {avg_score:.4f}")

    # Display non-numeric results
    non_numeric = {k: v for k, v in results.items() if not isinstance(v, (int, float))}
    if non_numeric:
        print("\nOTHER RESULTS:")
        for key, value in non_numeric.items():
            print(f"{key}: {value}")

    # Interpretation
    if numeric_results:
        avg_score = sum(numeric_results.values()) / len(numeric_results)
        print(f"\n{'='*60}")
        print("INTERPRETATION:")
        print("=" * 60)

        if avg_score >= 0.8:
            print("🟢 EXCELLENT: Your RAG system is performing very well!")
        elif avg_score >= 0.6:
            print(
                "🟡 GOOD: Your RAG system is performing well with room for improvement."
            )
        elif avg_score >= 0.4:
            print("🟠 MODERATE: Your RAG system needs significant improvements.")
        else:
            print("🔴 POOR: Your RAG system needs major improvements.")


def create_evaluation_summary(results: dict, dataset_size: int):
    """Create a summary of the evaluation."""

    summary = {
        "dataset_size": dataset_size,
        "metrics_evaluated": len(results),
        "results": results,
    }

    # Save to file
    with open("./src/ragas/03_ragas_summary.md", "w") as f:
        f.write("RAGAS EVALUATION SUMMARY\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Dataset Size: {dataset_size}\n")
        f.write(f"Metrics Evaluated: {len(results)}\n\n")
        f.write("Results:\n")
        for metric, score in results.items():
            f.write(f"  {metric}: {score}\n")

    print("Summary saved to ragas_summary.txt")
    return summary


def main():
    """Main execution function."""

    print("RAGAS Evaluation v2")
    print("=" * 50)

    # Setup environment
    if not setup_ragas_environment():
        return

    try:
        # Create sample dataset
        print("\n1. Creating sample dataset...")
        df = pd.read_csv(
            "./src/ragas/02_rag_test_dataset.csv", sep="|", encoding="utf-8"
        )
        print(f"   Created {len(df)} examples")

        # Prepare for RAGAS
        print("\n2. Preparing dataset for RAGAS...")
        dataset = create_ragas_dataset(df)
        print(f"   Dataset prepared with {len(dataset)} examples")

        # Run evaluation
        print("\n3. Running RAGAS evaluation...")
        results = run_ragas_evaluation_v2(dataset)
        # with open("./src/ragas/03_ragas_results.txt", "w") as f:
        #     json.dump(results, f, indent=4)

        # # Display results
        # print("\n4. Displaying results...")
        # display_results_v2(results)
        # with open("./src/ragas/03_ragas_results.txt", "w") as f:
        #     json.dump(results, f, indent=4)

        # Create summary
        print("\n5. Creating summary...")
        create_evaluation_summary(results, len(dataset))
        with open("./src/ragas/03_ragas_summary.json", "w") as f:
            json.dump(results, f, indent=4)
        print("\n✅ Evaluation completed successfully!")

    except Exception as e:
        print(f"\n❌ Error during evaluation: {str(e)}")
        import traceback

        print("\nFull traceback:")
        traceback.print_exc()

        print("\n🔧 Troubleshooting suggestions:")
        print("1. Make sure your OpenAI API key is set")
        print("2. Check your internet connection")
        print("3. Try running the simple test first: python ragas_simple_test.py")
        print("4. Update RAGAS: pip install --upgrade ragas")


if __name__ == "__main__":
    print("STARTING RAGAS Evaluation v2...")
    main()
