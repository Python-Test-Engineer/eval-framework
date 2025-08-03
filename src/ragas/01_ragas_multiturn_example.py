import os
from dotenv import load_dotenv, find_dotenv
from rich.console import Console


from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall,
)

console = Console()

load_dotenv(find_dotenv(), override=True)
OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]
GROQ_API_KEY = os.environ["GROQ_API_KEY"]
MODEL = "gpt-4o-mini"
# Try to import additional metrics with correct casing
try:
    from ragas.metrics import ContextUtilization

    context_utilization = ContextUtilization()
except ImportError:
    context_utilization = None

try:
    from ragas.metrics import answer_correctness
except ImportError:
    try:
        from ragas.metrics import AnswerCorrectness

        answer_correctness = AnswerCorrectness()
    except ImportError:
        answer_correctness = None

# Sample multiturn conversation data
sample_data = {
    "user_input": [
        # Conversation 1 - About Python programming
        "What is Python?",
        "How do I install Python packages?",
        "Can you show me an example of using pip?",
        # Conversation 2 - About machine learning
        "What is machine learning?",
        "What's the difference between supervised and unsupervised learning?",
        "Can you give me examples of each type?",
        # Conversation 3 - About data analysis
        "How do I analyze data with pandas?",
        "What are some common data cleaning operations?",
        "How do I handle missing values?",
    ],
    "response": [
        # Responses for each question
        "Python is a high-level, interpreted programming language known for its simplicity and readability. It's widely used for web development, data science, automation, and more.",
        "You can install Python packages using pip, which is Python's package installer. It comes pre-installed with Python 3.4+.",
        "Sure! Here's an example: `pip install pandas` to install the pandas library, or `pip install requests numpy` to install multiple packages at once.",
        "Machine learning is a subset of artificial intelligence that enables computers to learn and make decisions from data without being explicitly programmed for every task.",
        "Supervised learning uses labeled data to train models (like predicting house prices), while unsupervised learning finds patterns in unlabeled data (like customer segmentation).",
        "Supervised examples: email spam detection, image classification. Unsupervised examples: clustering customers, anomaly detection, dimensionality reduction.",
        "Pandas is a powerful Python library for data analysis. You can load data with pd.read_csv(), explore with .head(), .info(), and .describe() methods.",
        "Common operations include removing duplicates (.drop_duplicates()), filtering data, handling missing values, and data type conversions.",
        "For missing values, you can use .dropna() to remove them, .fillna() to fill with specific values, or .interpolate() for more sophisticated filling methods.",
    ],
    "contexts": [
        # Contexts for each response (note: 'contexts' not 'retrieved_contexts')
        [
            "Python is a programming language that lets you work quickly and integrate systems more effectively.",
            "Python is an interpreted, object-oriented, high-level programming language with dynamic semantics.",
        ],
        [
            "pip is the package installer for Python. You can use it to install packages from the Python Package Index.",
            "pip is a command line program. When you install pip, a pip command is added to your system.",
        ],
        [
            "The basic syntax for pip is: pip install package_name",
            "You can install multiple packages by listing them: pip install package1 package2",
            "Use pip list to see installed packages and pip show package_name for details.",
        ],
        [
            "Machine learning is a method of data analysis that automates analytical model building.",
            "ML is based on the idea that systems can learn from data and make decisions with minimal human intervention.",
        ],
        [
            "Supervised learning algorithms are trained using labeled examples.",
            "Unsupervised learning algorithms are used when the information used to train is neither classified nor labeled.",
        ],
        [
            "Common supervised learning algorithms include linear regression, decision trees, and neural networks.",
            "Unsupervised learning includes clustering algorithms like K-means and dimensionality reduction techniques.",
        ],
        [
            "Pandas provides data structures and operations for manipulating numerical tables and time series.",
            "The primary data structures in pandas are Series and DataFrame.",
        ],
        [
            "Data cleaning involves detecting and correcting corrupt or inaccurate records.",
            "Common data cleaning tasks include handling missing data, removing duplicates, and correcting data types.",
        ],
        [
            "Missing data can be handled by deletion, imputation, or interpolation methods.",
            "Pandas provides methods like dropna(), fillna(), and interpolate() for handling missing values.",
        ],
    ],
    "ground_truth": [
        # Ground truth answers for each question
        "Python is a high-level, interpreted programming language created by Guido van Rossum, known for its readable syntax and versatility.",
        "Python packages are installed using pip (Pip Installs Packages), which is the standard package manager for Python.",
        "Basic pip usage: 'pip install package_name' installs a package, 'pip list' shows installed packages, 'pip uninstall package_name' removes packages.",
        "Machine learning is a branch of AI that enables systems to automatically learn and improve from experience without explicit programming.",
        "Supervised learning uses labeled training data to learn a mapping function, while unsupervised learning discovers hidden patterns in data without labels.",
        "Supervised: classification (spam detection), regression (price prediction). Unsupervised: clustering (customer groups), association (market basket analysis).",
        "Pandas is a Python library providing high-performance data structures (DataFrame, Series) and analysis tools for structured data manipulation.",
        "Key data cleaning operations include handling missing values, removing duplicates, correcting data types, filtering outliers, and standardizing formats.",
        "Missing values can be handled by: dropping rows/columns (.dropna()), filling with values (.fillna()), or using interpolation methods (.interpolate()).",
    ],
}


def check_available_metrics():
    """Check what metrics are available in your ragas version"""
    import ragas.metrics as metrics

    available_metrics = [attr for attr in dir(metrics) if not attr.startswith("_")]

    print("Available metrics in your ragas version:")
    for metric in sorted(available_metrics):
        print(f"  - {metric}")

    # Show specifically which metrics we can use
    print("\nTrying to identify usable metrics:")
    basic_metrics = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    for metric in basic_metrics:
        if hasattr(metrics, metric):
            print(f"  ✅ {metric}")
        else:
            print(f"  ❌ {metric}")

    # Check PascalCase versions
    pascal_metrics = ["ContextUtilization", "AnswerCorrectness", "AnswerSimilarity"]
    for metric in pascal_metrics:
        if hasattr(metrics, metric):
            print(f"  ✅ {metric} (PascalCase)")
        else:
            print(f"  ❌ {metric} (PascalCase)")

    return available_metrics


def create_dataset_from_sample():
    """Convert sample data to ragas dataset format"""

    # Ensure all lists have the same length
    assert len(sample_data["user_input"]) == len(sample_data["response"])
    assert len(sample_data["user_input"]) == len(sample_data["contexts"])
    assert len(sample_data["user_input"]) == len(sample_data["ground_truth"])

    # Create dataset
    dataset = Dataset.from_dict(
        {
            "question": sample_data["user_input"],
            "answer": sample_data["response"],
            "contexts": sample_data["contexts"],
            "ground_truth": sample_data["ground_truth"],
        }
    )

    print(f"Dataset created with {len(dataset)} samples")
    print("Dataset columns:", dataset.column_names)

    return dataset


def run_basic_evaluation():
    """Run basic RAG evaluation using standard metrics"""

    print("Creating dataset...")
    dataset = create_dataset_from_sample()

    # Standard metrics that work with most ragas versions
    metrics = [
        faithfulness,  # How faithful answers are to context
        answer_relevancy,  # How relevant answers are to questions
        context_precision,  # Precision of retrieved contexts
        context_recall,  # Recall of retrieved contexts
    ]

    # Try to add more metrics if available
    if context_utilization is not None:
        metrics.append(context_utilization)
        print("✅ Added ContextUtilization metric")
    else:
        print("❌ ContextUtilization not available")

    if answer_correctness is not None:
        metrics.append(answer_correctness)
        print("✅ Added answer_correctness metric")
    else:
        print("❌ answer_correctness not available")

    print(f"\nUsing {len(metrics)} metrics for evaluation")

    print("\nRunning evaluation...")

    # Run evaluation
    try:
        results = evaluate(dataset, metrics=metrics, raise_exceptions=False)

        # Display results
        print("\n" + "=" * 50)
        print("EVALUATION RESULTS")
        print("=" * 50)

        print(f"Results type: {type(results)}")
        print(f"Results attributes: {dir(results)}")

        # Convert to DataFrame first
        try:
            results_df = results.to_pandas()
            print(f"Results DataFrame shape: {results_df.shape}")
            print(f"DataFrame columns: {results_df.columns.tolist()}")

            # Calculate and display overall scores from DataFrame
            metric_columns = [
                col
                for col in results_df.columns
                if col
                in [
                    "faithfulness",
                    "answer_relevancy",
                    "context_precision",
                    "context_recall",
                ]
            ]

            print("\nOverall Scores (averaged across all samples):")
            for col in metric_columns:
                if col in results_df.columns:
                    avg_score = results_df[col].mean()
                    print(f"{col}: {avg_score:.4f}")

            return results, results_df

        except Exception as e:
            print(f"Error converting to DataFrame: {e}")

            # Try alternative result access methods
            print("\nTrying alternative result access...")

            # Try accessing results as dictionary
            if hasattr(results, "__dict__"):
                print("Results dict:", results.__dict__)

            # Try accessing specific attributes
            for attr in ["scores", "metrics", "results", "data"]:
                if hasattr(results, attr):
                    print(f"Found attribute '{attr}': {getattr(results, attr)}")

            return results, None

    except Exception as e:
        print(f"Evaluation failed: {e}")
        return None, None


def simulate_multiturn_analysis(results_df):
    """Simulate multiturn analysis by grouping related questions"""

    if results_df is None:
        print("❌ Cannot perform multiturn analysis - results_df is None")
        return None

    print("\n" + "=" * 50)
    print("MULTITURN SIMULATION ANALYSIS")
    print("=" * 50)

    print(f"DataFrame shape: {results_df.shape}")
    print(f"DataFrame columns: {results_df.columns.tolist()}")

    # Show first few rows
    print("\nFirst few rows of results:")
    print(results_df.head())

    # Group questions by topic (every 3 questions = 1 conversation)
    conversation_topics = ["Python Basics", "Machine Learning", "Data Analysis"]

    results_df["conversation_id"] = results_df.index // 3
    results_df["conversation_topic"] = results_df["conversation_id"].map(
        lambda x: (
            conversation_topics[x] if x < len(conversation_topics) else f"Topic {x}"
        )
    )
    results_df["turn_number"] = results_df.index % 3 + 1

    # Find available metric columns
    metric_columns = [
        col
        for col in results_df.columns
        if col
        in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    ]

    print(f"\nFound metric columns: {metric_columns}")

    if metric_columns:
        print("\nConversation-level analysis:")

        # Calculate metrics by conversation
        conversation_metrics = (
            results_df.groupby("conversation_topic")[metric_columns].mean().round(4)
        )
        print(conversation_metrics)

        # Analyze turn patterns
        print("\nTurn-level analysis:")
        turn_metrics = results_df.groupby("turn_number")[metric_columns].mean().round(4)
        print(turn_metrics)
    else:
        print("❌ No recognized metric columns found for analysis")

    return results_df


def show_sample_conversation():
    """Display a sample conversation for reference"""
    print("\n" + "=" * 50)
    print("SAMPLE CONVERSATION")
    print("=" * 50)

    print("Topic: Python Basics")
    for i in range(3):  # First 3 questions
        print(f"\nTurn {i+1}:")
        print(f"User: {sample_data['user_input'][i]}")
        print(f"Assistant: {sample_data['response'][i][:100]}...")


def export_results(
    results_df, filename="./src/ragas/01_ragas_multiturn_style_evaluation.csv"
):
    """Export results to CSV"""
    if results_df is not None:
        results_df.to_csv(filename, index=False, sep="|")
        print(f"\nResults exported to: {filename}")


# Main execution
if __name__ == "__main__":
    print("RAGAS Multiturn-Style Evaluation")
    print("=" * 50)

    # Check available metrics
    available_metrics = check_available_metrics()

    # Show sample conversation
    show_sample_conversation()

    # Run evaluation
    try:
        results, results_df = run_basic_evaluation()

        if results_df is not None:
            # Simulate multiturn analysis
            enhanced_df = simulate_multiturn_analysis(results_df)

            # Export results
            export_results(enhanced_df)

            print("\n" + "=" * 50)
            print("EVALUATION COMPLETED!")
            print("=" * 50)

            # Show quick summary
            if "faithfulness" in results_df.columns:
                avg_faithfulness = results_df["faithfulness"].mean()
                print(f"Average faithfulness score: {avg_faithfulness:.4f}")

            if "answer_relevancy" in results_df.columns:
                avg_relevancy = results_df["answer_relevancy"].mean()
                print(f"Average answer relevancy: {avg_relevancy:.4f}")

    except Exception as e:
        print(f"Error: {e}")
        print("\nTroubleshooting tips:")
        print(
            "1. Make sure you have OpenAI API key set: export OPENAI_API_KEY='your-key'"
        )
        print("2. Try: pip install --upgrade ragas")
        print("3. Check your internet connection")


# Alternative simple test function
def quick_test():
    """Quick test with minimal data"""
    simple_data = {
        "question": ["What is Python?"],
        "answer": ["Python is a programming language."],
        "contexts": [["Python is a high-level programming language."]],
        "ground_truth": [
            "Python is a programming language used for various applications."
        ],
    }

    dataset = Dataset.from_dict(simple_data)

    try:
        results = evaluate(dataset, metrics=[faithfulness, answer_relevancy])
        print("Quick test results:", results)
        return True
    except Exception as e:
        print(f"Quick test failed: {e}")
        return False


# Uncomment to run quick test first
# print("Running quick test...")
# quick_test()
