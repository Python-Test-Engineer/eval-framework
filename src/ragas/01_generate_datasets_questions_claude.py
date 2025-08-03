import os
from typing import List, Dict
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

OUTPUT_FILE = "./src/ragas/01_generate_synthetic_qa_langchain_agents_data.json"

class SimpleSyntheticDataGenerator:
    """
    Generate synthetic datasets without relying on complex RAGAS APIs.
    Uses direct LLM calls for more reliable generation.
    """
    
    def __init__(self, openai_api_key: str = None):
        if openai_api_key:
            os.environ["OPENAI_API_KEY"] = openai_api_key
        
        # Initialize LLM for generation
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.3)
    
    def create_documents_from_content(self, content_sources: Dict[str, str]) -> List[Document]:
        """Convert your content into Document format."""
        documents = []
        
        for source_name, content in content_sources.items():
            # Split content into reasonable chunks
            chunks = self._split_content(content, chunk_size=800)
            
            for i, chunk in enumerate(chunks):
                doc = Document(
                    page_content=chunk.strip(),
                    metadata={
                        "source": source_name,
                        "chunk_id": i,
                        "total_chunks": len(chunks)
                    }
                )
                documents.append(doc)
        
        return documents
    
    def _split_content(self, content: str, chunk_size: int = 800) -> List[str]:
        """Split content into manageable chunks by sentences."""
        sentences = content.split('.')
        chunks = []
        current_chunk = ""
        
        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue
                
            if len(current_chunk + sentence) < chunk_size:
                current_chunk += sentence + ". "
            else:
                if current_chunk:
                    chunks.append(current_chunk.strip())
                current_chunk = sentence + ". "
        
        if current_chunk:
            chunks.append(current_chunk.strip())
        
        return chunks
    
    def generate_questions_from_documents(self, 
                                        documents: List[Document], 
                                        question_types: List[str] = None,
                                        num_samples: int = 20) -> Dataset:
        """Generate synthetic Q&A pairs from your documents."""
        
        if question_types is None:
            question_types = ["factual", "how_to", "conceptual", "troubleshooting"]
        
        questions = []
        contexts = []
        ground_truths = []
        question_type_labels = []
        
        print(f"Generating {num_samples} questions from {len(documents)} documents...")
        
        for i in range(num_samples):
            # Select a random document and question type
            doc = documents[i % len(documents)]
            q_type = question_types[i % len(question_types)]
            
            # Generate Q&A pair
            try:
                question, answer = self._generate_qa_with_llm(doc.page_content, q_type)
                
                questions.append(question)
                contexts.append([doc.page_content])
                ground_truths.append(answer)
                question_type_labels.append(q_type)
                
                if (i + 1) % 5 == 0:
                    print(f"Generated {i + 1}/{num_samples} questions...")
                    
            except Exception as e:
                print(f"Error generating question {i+1}: {e}")
                # Fallback to template-based generation
                question, answer = self._generate_qa_template(doc.page_content, q_type)
                questions.append(question)
                contexts.append([doc.page_content])
                ground_truths.append(answer)
                question_type_labels.append(q_type)
        
        # Create dataset
        dataset_dict = {
            "question": questions,
            "contexts": contexts,
            "ground_truth": ground_truths,
            "question_type": question_type_labels
        }
        
        return Dataset.from_dict(dataset_dict)
    
    def _generate_qa_with_llm(self, content: str, question_type: str) -> tuple:
        """Use LLM to generate question-answer pairs."""
        
        type_instructions = {
            "factual": "Generate a factual question that can be answered directly from the content. Focus on definitions, facts, or specific information.",
            "how_to": "Generate a 'how to' or procedural question about implementing or using something described in the content.",
            "conceptual": "Generate a conceptual question that requires understanding why something works or its importance/benefits.",
            "troubleshooting": "Generate a troubleshooting question about potential problems or issues related to the content.",
            "comparison": "Generate a question comparing different concepts, approaches, or options mentioned in the content."
        }
        
        instruction = type_instructions.get(question_type, type_instructions["factual"])
        
        prompt = f"""
Based on the following content, {instruction}

Content:
{content}

Generate:
1. A clear, specific question
2. A comprehensive answer based solely on the provided content

Format your response as:
Question: [your question here]
Answer: [your answer here]

Make sure the question is natural and the answer is accurate based on the content.
"""
        
        try:
            response = self.llm.invoke(prompt)
            result = response.content
            
            # Parse the response
            lines = result.split('\n')
            question_line = [line for line in lines if line.startswith('Question:')]
            answer_line = [line for line in lines if line.startswith('Answer:')]
            
            if question_line and answer_line:
                question = question_line[0].replace('Question:', '').strip()
                answer = answer_line[0].replace('Answer:', '').strip()
                return question, answer
            else:
                # Fallback parsing
                parts = result.split('Answer:')
                if len(parts) == 2:
                    question = parts[0].replace('Question:', '').strip()
                    answer = parts[1].strip()
                    return question, answer
                
        except Exception as e:
            print(f"LLM generation error: {e}")
        
        # Fallback to template-based generation
        return self._generate_qa_template(content, question_type)
    
    def _generate_qa_template(self, content: str, question_type: str) -> tuple:
        """Fallback template-based question generation."""
        
        # Extract key information
        sentences = [s.strip() for s in content.split('.') if s.strip()]
        first_sentence = sentences[0] if sentences else content[:100]
        
        templates = {
            "factual": {
                "question": f"What does this content explain about the main topic?",
                "answer": first_sentence
            },
            "how_to": {
                "question": f"How would you implement or use the concepts described?",
                "answer": f"Based on the content: {first_sentence}"
            },
            "conceptual": {
                "question": f"Why is the main concept described in this content important?",
                "answer": f"The content explains: {first_sentence}"
            },
            "troubleshooting": {
                "question": f"What should you consider when working with the concepts in this content?",
                "answer": f"Key considerations include: {first_sentence}"
            }
        }
        
        template = templates.get(question_type, templates["factual"])
        return template["question"], template["answer"]
    
    def enhance_existing_questions(self, existing_questions: List[str]) -> List[Dict[str, str]]:
        """Create variations of your existing questions."""
        
        enhanced = []
        
        for original_q in existing_questions:
            # Create variations
            variations = [
                original_q,  # Original
                self._rephrase_question(original_q, "formal"),
                self._rephrase_question(original_q, "casual"),
                self._rephrase_question(original_q, "technical"),
            ]
            
            for variation in variations:
                enhanced.append({
                    "question": variation,
                    "original": original_q,
                    "variation_type": "rephrase"
                })
        
        return enhanced
    
    def _rephrase_question(self, question: str, style: str) -> str:
        """Simple rephrasing logic."""
        
        style_transforms = {
            "formal": {
                "How do I": "What is the procedure for",
                "Can I": "Is it possible to",
                "What's": "What is"
            },
            "casual": {
                "What is the procedure for": "How do I",
                "Is it possible to": "Can I",
                "How does one": "How do you"
            },
            "technical": {
                "How do I": "What are the steps to programmatically",
                "What is": "What are the technical specifications for",
                "Can I": "What APIs or methods allow me to"
            }
        }
        
        transforms = style_transforms.get(style, {})
        result = question
        
        for old, new in transforms.items():
            if old in result:
                result = result.replace(old, new)
                break
        
        return result

def save_dataset(dataset: Dataset, filename: str = "synthetic_dataset.json"):
    """Save dataset and display sample questions."""
    
    df = dataset.to_pandas()
    df.to_json(filename, orient='records', indent=2)
    print(f"\nDataset saved to {filename}")
    
    print(f"\nDataset overview:")
    print(f"- Total questions: {len(df)}")
    print(f"- Question types: {df['question_type'].value_counts().to_dict()}")
    
    print(f"\nSample questions:")
    print("-" * 60)
    for i, row in df.head(3).iterrows():
        print(f"Type: {row['question_type']}")
        print(f"Q: {row['question']}")
        print(f"A: {row['ground_truth'][:150]}...")
        print("-" * 60)
    
    return df

# Example usage
def main():
    """Example of how to use the generator with your content."""
    
    # Your domain content
    your_content = {
        "api_documentation": """
        Our REST API provides endpoints for user authentication, data management, and analytics.
        Authentication uses JWT tokens with a 24-hour expiration. The base URL is https://api.example.com/v1.
        Rate limiting applies 1000 requests per hour per API key. All responses use JSON format.
        Error codes follow HTTP standards with detailed error messages in the response body.
        """,
        
        "user_guide": """
        To get started, create an account and generate your API key from the dashboard.
        Configure your HTTP client with the base URL and include your API key in the Authorization header.
        Common workflows include: 1) Authenticate to get a token, 2) Upload your data, 3) Process the data, 4) Retrieve results.
        Use pagination for large datasets and implement retry logic for network errors.
        """,
        
        "troubleshooting": """
        Common issues and solutions: 429 errors indicate rate limiting - implement exponential backoff.
        401 errors mean invalid or expired tokens - refresh your authentication.
        500 errors are server-side issues - retry after a delay.
        Network timeouts should trigger automatic retries with increasing delays.
        """
    }
    
    # Initialize generator
    generator = SimpleSyntheticDataGenerator()
    
    # Create documents
    print("Converting content to documents...")
    documents = generator.create_documents_from_content(your_content)
    print(f"Created {len(documents)} document chunks")
    
    # Generate synthetic Q&A pairs
    question_types = ["factual", "how_to", "troubleshooting", "conceptual"]
    dataset = generator.generate_questions_from_documents(
        documents=documents,
        question_types=question_types,
        num_samples=12
    )
    
    # Save and display results
    df = save_dataset(dataset, OUTPUT_FILE)
    
    # Example: Enhance existing questions
    existing_questions = [
        "How do I authenticate with the API?",
        "What are the rate limits?",
        "How do I handle errors?"
    ]
    
    enhanced = generator.enhance_existing_questions(existing_questions)
    print(f"\nEnhanced {len(existing_questions)} questions into {len(enhanced)} variations")

if __name__ == "__main__":
    main()
