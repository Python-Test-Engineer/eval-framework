import pandas as pd
import random
import json
import os
from typing import List, Dict

import openai
import time


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


# Create seed data for Python and Machine Learning Q&A
seed_data = {
    "question": [
        "What is the difference between a list and a tuple in Python?",
        "How do you handle exceptions in Python?",
        "What is overfitting in machine learning?",
        "Explain the concept of gradient descent.",
        "What is the purpose of __init__ method in Python classes?",
        "What is the difference between supervised and unsupervised learning?",
        "How do you create a virtual environment in Python?",
        "What is cross-validation in machine learning?",
        "What are Python decorators and how do you use them?",
        "Explain the bias-variance tradeoff in machine learning.",
    ],
    "answer": [
        "Lists are mutable (can be changed after creation) while tuples are immutable (cannot be changed). Lists use square brackets [] and tuples use parentheses (). Lists are better for data that changes, tuples for fixed data.",
        "Python uses try-except blocks for exception handling. You can catch specific exceptions with 'except ExceptionType:' or use a general 'except:' block. The 'finally' block executes regardless of exceptions.",
        "Overfitting occurs when a model learns the training data too well, including noise and random fluctuations. The model performs well on training data but poorly on new, unseen data. It can be prevented using regularization, cross-validation, or more training data.",
        "Gradient descent is an optimization algorithm used to minimize the cost function in machine learning. It iteratively adjusts model parameters in the direction of steepest descent (negative gradient) to find the minimum error.",
        "The __init__ method is a constructor in Python classes that initializes new objects. It's automatically called when creating an instance of a class and is used to set initial values for object attributes.",
        "Supervised learning uses labeled training data to learn patterns and make predictions on new data. Unsupervised learning finds hidden patterns in unlabeled data without target variables, like clustering or dimensionality reduction.",
        "Use 'python -m venv env_name' to create a virtual environment, then activate it with 'source env_name/bin/activate' (Linux/Mac) or 'env_name\\Scripts\\activate' (Windows). Install packages with pip in the activated environment.",
        "Cross-validation is a technique to evaluate model performance by splitting data into multiple folds. The model is trained on some folds and tested on others, rotating through all combinations to get a robust performance estimate.",
        "Decorators are functions that modify or extend the behavior of other functions without changing their code. They use the @decorator_name syntax and are commonly used for logging, timing, authentication, or caching functionality.",
        "Bias-variance tradeoff describes the balance between a model's ability to capture true patterns (bias) and its sensitivity to training data variations (variance). High bias leads to underfitting, high variance to overfitting.",
    ],
    "topic": [
        "python",
        "python",
        "machine_learning",
        "machine_learning",
        "python",
        "machine_learning",
        "python",
        "machine_learning",
        "python",
        "machine_learning",
    ],
}


class GPTSyntheticQAGenerator:
    def __init__(self, seed_csv_path: str = None):
        # Initialize OpenAI client
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY not found in environment variables. Please add it to your .env file"
            )

        openai.api_key = self.api_key
        self.client = openai.OpenAI(api_key=self.api_key)

        if seed_csv_path:
            self.seed_df = pd.read_csv(seed_csv_path)
        else:
            # Create seed data CSV
            self.seed_df = pd.DataFrame(seed_data)
            self.seed_df.to_csv("seed_qa_data.csv", index=False)
            print("Created seed_qa_data.csv with 10 seed questions")

        # Rate limiting
        self.request_count = 0
        self.last_request_time = 0
        self.min_request_interval = 1  # 1 second between requests

    def _rate_limit(self):
        """Simple rate limiting to avoid hitting API limits"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time

        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)

        self.last_request_time = time.time()

    def generate_variations_with_gpt(
        self, question: str, answer: str, topic: str, num_variations: int = 4
    ) -> List[Dict[str, str]]:
        """Generate question-answer variations using GPT-4o-mini"""
        self._rate_limit()

        prompt = f"""
        Given this seed question-answer pair about {topic}, generate {num_variations} variations that:
        1. Ask about the same or closely related concept
        2. Use different wording and question structures
        3. Maintain the same level of technical accuracy
        4. Keep answers informative but concise (2-4 sentences)

        Original:
        Question: {question}
        Answer: {answer}
        Topic: {topic}

        Please generate {num_variations} variations in JSON format:
        {{
            "variations": [
                {{
                    "question": "new question here",
                    "answer": "corresponding answer here",
                    "topic": "{topic}"
                }}
            ]
        }}

        Make sure each variation is unique and maintains educational value.
        """

        try:
            response = self.client.chat.completions.create(
                model=MODEL,
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in Python programming and machine learning. Generate high-quality educational Q&A variations.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1500,
                temperature=0.7,
            )

            content = response.choices[0].message.content.strip()

            # Try to parse JSON from the response
            try:
                # Sometimes GPT includes markdown code blocks
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_content = content[json_start:json_end].strip()
                else:
                    json_content = content

                data = json.loads(json_content)
                return data.get("variations", [])

            except json.JSONDecodeError:
                print(f"Failed to parse JSON response for question: {question[:50]}...")
                print(f"Response: {content[:200]}...")
                return []

        except Exception as e:
            print(f"Error generating variations for question: {question[:50]}...")
            print(f"Error: {str(e)}")
            return []

    def generate_new_questions_with_gpt(
        self, topic: str, num_questions: int = 5
    ) -> List[Dict[str, str]]:
        """Generate completely new questions using GPT-4o-mini"""
        self._rate_limit()

        if topic == "python":
            topic_description = "Python programming language, including syntax, data structures, OOP, libraries, best practices, and advanced concepts"
        else:
            topic_description = "Machine learning, including algorithms, model evaluation, data preprocessing, deep learning, and statistical concepts"

        prompt = f"""
        Generate {num_questions} new educational question-answer pairs about {topic_description}.
        
        Requirements:
        1. Questions should be at intermediate to advanced level
        2. Cover diverse subtopics within {topic}
        3. Answers should be informative, accurate, and 2-4 sentences long
        4. Suitable for technical interviews or educational content
        
        Format as JSON:
        {{
            "questions": [
                {{
                    "question": "question text here",
                    "answer": "detailed answer here",
                    "topic": "{topic}"
                }}
            ]
        }}
        
        Topics to consider for {topic}:
        """ + (
            """
        - Data structures (lists, dicts, sets, tuples)
        - Object-oriented programming
        - Functional programming features
        - Error handling and debugging
        - Performance optimization
        - Popular libraries (pandas, numpy, etc.)
        - Testing and best practices
        """
            if topic == "python"
            else """
        - Supervised vs unsupervised learning
        - Model evaluation and validation
        - Feature engineering and selection
        - Different algorithms (tree-based, neural networks, etc.)
        - Overfitting, underfitting, bias-variance
        - Data preprocessing and cleaning
        - Deep learning concepts
        """
        )

        try:
            response = self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert educator creating high-quality technical Q&A content.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=2000,
                temperature=0.8,
            )

            content = response.choices[0].message.content.strip()

            try:
                if "```json" in content:
                    json_start = content.find("```json") + 7
                    json_end = content.find("```", json_start)
                    json_content = content[json_start:json_end].strip()
                else:
                    json_content = content

                data = json.loads(json_content)
                return data.get("questions", [])

            except json.JSONDecodeError:
                print(f"Failed to parse JSON for new {topic} questions")
                print(f"Response: {content[:200]}...")
                return []

        except Exception as e:
            print(f"Error generating new {topic} questions: {str(e)}")
            return []

    def generate_synthetic_dataset(self, target_size: int = 50) -> pd.DataFrame:
        """Generate synthetic Q&A dataset using GPT-4o-mini"""
        synthetic_data = []

        print("Starting synthetic dataset generation with GPT-4o-mini...")
        print(f"Target size: {target_size} question-answer pairs")

        # Start with seed data
        print("\n1. Adding seed data...")
        for _, row in self.seed_df.iterrows():
            synthetic_data.append(
                {
                    "question": row["question"],
                    "answer": row["answer"],
                    "topic": row["topic"],
                    "source": "seed",
                }
            )

        print(f"Added {len(self.seed_df)} seed questions")

        # Generate variations from seed data
        print("\n2. Generating variations from seed data...")
        variations_per_seed = max(
            1, (target_size - len(self.seed_df)) // (len(self.seed_df) * 2)
        )

        for i, row in self.seed_df.iterrows():
            print(
                f"Generating variations for seed question {i+1}/{len(self.seed_df)}..."
            )

            variations = self.generate_variations_with_gpt(
                row["question"], row["answer"], row["topic"], variations_per_seed
            )

            for var in variations:
                if len(synthetic_data) >= target_size:
                    break

                synthetic_data.append(
                    {
                        "question": var.get("question", ""),
                        "answer": var.get("answer", ""),
                        "topic": var.get("topic", row["topic"]),
                        "source": "gpt_variation",
                    }
                )

            if len(synthetic_data) >= target_size:
                break

        print(
            f"Added {len([d for d in synthetic_data if d['source'] == 'gpt_variation'])} variations"
        )

        # Generate completely new questions if we need more
        remaining = target_size - len(synthetic_data)
        if remaining > 0:
            print(f"\n3. Generating {remaining} new questions...")

            # Split remaining questions between topics
            python_questions = remaining // 2
            ml_questions = remaining - python_questions

            # Generate Python questions
            if python_questions > 0:
                print(f"Generating {python_questions} new Python questions...")
                new_python = self.generate_new_questions_with_gpt(
                    "python", python_questions
                )
                for q in new_python:
                    if len(synthetic_data) >= target_size:
                        break
                    synthetic_data.append(
                        {
                            "question": q.get("question", ""),
                            "answer": q.get("answer", ""),
                            "topic": "python",
                            "source": "gpt_new",
                        }
                    )

            # Generate ML questions
            if ml_questions > 0 and len(synthetic_data) < target_size:
                print(f"Generating {ml_questions} new ML questions...")
                new_ml = self.generate_new_questions_with_gpt(
                    "machine_learning", ml_questions
                )
                for q in new_ml:
                    if len(synthetic_data) >= target_size:
                        break
                    synthetic_data.append(
                        {
                            "question": q.get("question", ""),
                            "answer": q.get("answer", ""),
                            "topic": "machine_learning",
                            "source": "gpt_new",
                        }
                    )

        # Convert to DataFrame and clean up
        synthetic_df = pd.DataFrame(synthetic_data[:target_size])

        # Remove any rows with empty questions or answers
        synthetic_df = synthetic_df[
            (synthetic_df["question"].str.strip() != "")
            & (synthetic_df["answer"].str.strip() != "")
        ].reset_index(drop=True)

        # Shuffle the dataset
        synthetic_df = synthetic_df.sample(frac=1).reset_index(drop=True)

        print(f"\n✅ Generated {len(synthetic_df)} total question-answer pairs")

        return synthetic_df

    def save_synthetic_dataset(
        self, df: pd.DataFrame, filename: str = "gpt_synthetic_qa_dataset.csv"
    ):
        """Save the synthetic dataset to CSV with statistics"""
        # Get current directory to ensure file is saved in same folder
        current_dir = os.getcwd()
        full_path = os.path.join(current_dir, filename)

        # Save to CSV
        df.to_csv(full_path, index=False)

        print(f"\n📁 Synthetic dataset saved to: {full_path}")
        print(f"📊 Dataset size: {len(df)} question-answer pairs")

        # Print statistics
        topic_counts = df["topic"].value_counts()
        source_counts = df["source"].value_counts()

        print(f"\n📈 Topic distribution:")
        for topic, count in topic_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {topic}: {count} pairs ({percentage:.1f}%)")

        print(f"\n🔍 Source distribution:")
        for source, count in source_counts.items():
            percentage = (count / len(df)) * 100
            print(f"  {source}: {count} pairs ({percentage:.1f}%)")

        # Check for quality
        avg_question_length = df["question"].str.len().mean()
        avg_answer_length = df["answer"].str.len().mean()

        print(f"\n📏 Quality metrics:")
        print(f"  Average question length: {avg_question_length:.1f} characters")
        print(f"  Average answer length: {avg_answer_length:.1f} characters")

        # Verify file was created
        if os.path.exists(full_path):
            file_size = os.path.getsize(full_path)
            print(f"✅ File successfully created: {file_size} bytes")
        else:
            print("❌ Warning: File may not have been created successfully")

        return df


def setup_environment():
    """Setup instructions and environment check"""
    print("=== GPT-4o-mini Synthetic Q&A Dataset Generator ===\n")

    # Check for .env file
    if not os.path.exists(".env"):
        print("⚠️  .env file not found!")
        print("Please create a .env file with your OpenAI API key:")
        print("OPENAI_API_KEY=your_api_key_here")
        print("\nTo get an API key:")
        print("1. Go to https://platform.openai.com/api-keys")
        print("2. Create a new API key")
        print("3. Add it to your .env file")
        return False

    # Check for API key
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  OPENAI_API_KEY not found in .env file!")
        print("Please add your OpenAI API key to the .env file:")
        print("OPENAI_API_KEY=your_api_key_here")
        return False

    print("✅ Environment setup complete!")
    print(f"🔑 API key found: {api_key[:8]}...{api_key[-4:]}")
    return True


def main():
    """Main function to generate synthetic dataset"""
    if not setup_environment():
        return None

    print("\n🚀 Starting dataset generation...")
    print(
        "📋 This will use OpenAI's GPT-4o-mini API to generate high-quality Q&A pairs"
    )
    print("💰 Estimated cost: ~$0.10-0.20 for 50 questions (very affordable!)")

    try:
        # Initialize generator
        generator = GPTSyntheticQAGenerator()

        # Generate synthetic dataset
        print(f"\n⏳ Generating synthetic dataset...")
        synthetic_df = generator.generate_synthetic_dataset(target_size=50)

        # Save dataset
        final_df = generator.save_synthetic_dataset(synthetic_df)

        # Display sample questions
        print(f"\n📋 Sample Generated Questions:")
        print("=" * 80)
        sample_questions = final_df.sample(min(5, len(final_df)))

        for i, (_, row) in enumerate(sample_questions.iterrows(), 1):
            print(f"\n🔹 Sample {i} [{row['topic'].upper()}] - {row['source']}:")
            print(f"❓ Q: {row['question']}")
            print(f"💡 A: {row['answer']}")
            if len(row["answer"]) > 150:
                print("   ...")

        print(f"\n🎉 Successfully generated {len(final_df)} question-answer pairs!")
        print("💾 Dataset saved and ready for use in training or fine-tuning!")

        return final_df

    except Exception as e:
        print(f"❌ Error during generation: {str(e)}")
        print("Please check your API key and internet connection.")
        return None


if __name__ == "__main__":
    # Installation reminder
    print("📦 Required packages: pip install openai python-dotenv pandas")
    print("💡 Make sure you have a .env file with OPENAI_API_KEY=your_key\n")

    dataset = main()
