# ragas_simple_test.py
"""
Simple test to verify RAGAS installation and basic functionality.
"""

import os
from dotenv import load_dotenv, find_dotenv

load_dotenv(find_dotenv(), override=True)


def test_ragas_installation():
    """Test basic RAGAS installation."""

    print("Testing RAGAS installation...")

    try:
        import ragas

        print(f"✓ RAGAS imported successfully (version: {ragas.__version__})")

        # Test metrics import
        from ragas.metrics import faithfulness, answer_relevancy

        print("✓ Basic metrics imported successfully")

        # Test dataset creation
        from datasets import Dataset

        simple_data = {
            "question": ["What is 2+2?"],
            "contexts": [["Two plus two equals four. This is basic arithmetic."]],
            "answer": ["2+2 equals 4"],
            "ground_truth": ["4"],
        }

        dataset = Dataset.from_dict(simple_data)
        print("✓ Dataset created successfully")

        # Test evaluation import
        from ragas import evaluate

        print("✓ Evaluate function imported successfully")

        # Simple evaluation test
        print("Running simple evaluation...")
        result = evaluate(dataset, metrics=[faithfulness])

        print(f"✓ Evaluation completed!")
        print(f"Result type: {type(result)}")

        # Try to access results
        if hasattr(result, "faithfulness"):
            print(f"Faithfulness score: {result.faithfulness}")
        else:
            print("Result structure:")
            print(result)

        return True

    except Exception as e:
        print(f"✗ Error: {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def test_openai_setup():
    """Test OpenAI setup."""

    print("\nTesting OpenAI setup...")

    try:
        import openai

        print("✓ OpenAI library imported")

        api_key = os.getenv("OPENAI_API_KEY")
        if api_key:
            print("✓ API key found in environment")
            print(f"API key starts with: {api_key[:14]}...")
        else:
            print("✗ No API key found in environment")
            return False

        return True

    except Exception as e:
        print(f"✗ OpenAI setup error: {str(e)}")
        return False


def main():
    """Run all tests."""

    print("RAGAS Simple Test")
    print("=" * 40)

    # Test OpenAI first
    openai_ok = test_openai_setup()

    # Test RAGAS
    ragas_ok = test_ragas_installation()

    print("\n" + "=" * 40)
    print("TEST SUMMARY")
    print("=" * 40)

    if openai_ok and ragas_ok:
        print("✓ All tests passed! RAGAS should work correctly.")
    else:
        print("✗ Some tests failed. Please address the issues above.")

        if not openai_ok:
            print("\nTo fix OpenAI issues:")
            print("1. Install OpenAI: pip install openai")
            print("2. Set API key: export OPENAI_API_KEY='your-key-here'")

        if not ragas_ok:
            print("\nTo fix RAGAS issues:")
            print("1. Install RAGAS: pip install ragas")
            print("2. Update dependencies: pip install --upgrade ragas datasets")


if __name__ == "__main__":
    main()
