# RAGAS Framework Metrics Guide

## Overview

RAGAS (Retrieval-Augmented Generation Assessment) is a framework for evaluating RAG (Retrieval-Augmented Generation) systems. This guide covers the four core metrics used to assess different aspects of your RAG pipeline.

## Comprehensive Metric Selection Guide

### For Different Use Cases

#### **Customer Support RAG**
```python
from ragas.metrics import faithfulness, answer_relevancy, AspectCritic

metrics = [
    faithfulness,           # Prevent hallucination
    answer_relevancy,       # Stay on topic
    AspectCritic(
        name="helpfulness",
        definition="Is the response helpful to the customer?"
    )
]
```

#### **Research/Academic RAG**
```python
from ragas.metrics import (
    faithfulness, context_recall, factual_correctness, 
    context_precision, semantic_similarity
)

metrics = [
    faithfulness,           # Factual accuracy crucial
    context_recall,         # Need complete information
    factual_correctness,    # Verify specific facts
    context_precision,      # Quality over quantity
    semantic_similarity     # Preserve meaning
]
```

#### **Multimodal Applications**
```python
from ragas.metrics import (
    MultiModalFaithfulness, MultiModalRelevance,
    faithfulness, answer_relevancy
)

metrics = [
    MultiModalFaithfulness(),    # Visual + text consistency
    MultiModalRelevance(),       # Cross-modal relevance
    faithfulness,                # Text-only baseline
    answer_relevancy            # General relevance
]
```

#### **AI Agents with Tools**
```python
from ragas.metrics import (
    ToolCallAccuracy, AgentGoalAccuracy, 
    TopicAdherence, ResponseGroundedness
)

metrics = [
    ToolCallAccuracy(),         # Correct tool usage
    AgentGoalAccuracy(),        # Goal achievement
    TopicAdherence(),           # Stay on task
    ResponseGroundedness()      # Evidence-based responses
]
```

#### **Content Generation**
```python
from ragas.metrics import (
    factual_correctness, semantic_similarity,
    AspectCritic, RubricsScore
)

# Define custom rubrics for content quality
content_rubrics = {
    "score1_description": "Poor quality, many errors",
    "score2_description": "Below average, some errors", 
    "score3_description": "Average quality, minor issues",
    "score4_description": "Good quality, well written",
    "score5_description": "Excellent quality, professional"
}

metrics = [
    factual_correctness,
    semantic_similarity,
    AspectCritic(
        name="readability",
        definition="Is the content clear and easy to understand?"
    ),
    RubricsScore(rubrics=content_rubrics)
]
```

## Advanced Implementation Examples

### Custom Aspect Critics

```python
from ragas.metrics import AspectCritic

# Define multiple custom critics
safety_critic = AspectCritic(
    name="safety",
    definition="Does the response avoid harmful, biased, or inappropriate content?"
)

accuracy_critic = AspectCritic(
    name="technical_accuracy",
    definition="Are all technical details and specifications correct?"
)

completeness_critic = AspectCritic(
    name="completeness", 
    definition="Does the response fully address all parts of the question?"
)
```

### Weighted Metric Combinations

```python
from ragas.metrics import answer_correctness

# Custom weights for answer correctness
weighted_correctness = answer_correctness
weighted_correctness.weights = [0.7, 0.3]  # [factual_weight, semantic_weight]

# Prioritize factual accuracy over semantic similarity
```

### Multi-language Support

```python
from ragas.metrics import faithfulness

# Adapt metrics to different languages
spanish_faithfulness = faithfulness
spanish_faithfulness.adapt(language="spanish")

# Works with: english, spanish, french, german, etc.
```

### Batch Evaluation

```python
from ragas import evaluate
from datasets import Dataset

# Large-scale evaluation
large_dataset = Dataset.from_dict({
    "question": questions_list,      # 1000+ questions
    "answer": answers_list,
    "contexts": contexts_list, 
    "ground_truth": ground_truth_list
})

# Efficient batch processing
results = evaluate(
    large_dataset,
    metrics=[faithfulness, answer_relevancy, context_precision],
    batch_size=32,  # Process in batches
    max_workers=4   # Parallel processing
)
```

## Complete Import Reference

### Core RAG Metrics
```python
from ragas.metrics import (
    faithfulness,
    answer_relevancy, 
    context_precision,
    context_recall,
    ContextEntityRecall,
    NoiseSensitivity
)
```

### Advanced RAG Metrics  
```python
from ragas.metrics import (
    factual_correctness,
    answer_correctness,
    semantic_similarity,
    ContextUtilization
)
```

### Multimodal Metrics
```python
from ragas.metrics import (
    MultiModalFaithfulness,
    MultiModalRelevance
)
```

### Agent Metrics
```python
from ragas.metrics import (
    ToolCallAccuracy,
    AgentGoalAccuracy, 
    TopicAdherence,
    ResponseGroundedness,
    ContextRelevance,
    AnswerAccuracy
)
```

### Traditional NLP Metrics
```python
from ragas.metrics import (
    BleuScore,
    RougeScore, 
    StringPresence,
    ExactMatch,
    StringSimilarity
)
```

### Custom Evaluation Metrics
```python
from ragas.metrics import (
    AspectCritic,
    SimpleCriteriaScore,
    RubricsScore,
    LabelledRubricsScore
)
```

### Execution-Based Metrics
```python
from ragas.metrics import (
    DatacompyScore,
    SqlQueryEquivalence
)
```

### Version Compatibility Notes

**RAGAS 0.2.15+**:
- Metric classes use PascalCase: `ContextUtilization`, `AnswerCorrectness`
- Import from `ragas.testset import TestsetGenerator`
- New evaluation API with `single_turn_ascore()` method

**RAGAS 0.1.x**:
- Metric instances use snake_case: `context_utilization`, `answer_correctness`
- Import from `ragas.testset.generator import TestsetGenerator` 
- Legacy evaluation API

**Check your version**:
```python
import ragas
print(ragas.__version__)

# List available metrics
import ragas.metrics as metrics
print([attr for attr in dir(metrics) if not attr.startswith('_')])
```

---

### 5. **Context Entities Recall**

**What it measures**: Measures recall based on entities present in ground truth and retrieved contexts.

**Scale**: 0 to 1 (higher is better)

**How it works**:
- Extracts named entities from both ground truth and retrieved contexts
- Calculates what fraction of ground truth entities are present in retrieved contexts
- Focus on specific entities rather than general text overlap

**Example**:
- **Ground Truth**: "Einstein worked at Princeton University and developed relativity theory"  
- **Retrieved Context**: "Einstein was at Princeton and worked on physics"
- **Entity Recall**: 0.5 (Princeton found, but relativity theory missing)

**Good score**: > 0.7 means most important entities are retrieved

**Use case**: Entity-focused retrieval evaluation

---

### 6. **Noise Sensitivity**

**What it measures**: Evaluates how sensitive the model is to irrelevant information in the context.

**Scale**: 0 to 1 (lower is better - less sensitivity to noise)

**How it works**:
- Introduces irrelevant information into the context
- Measures how much the model's output changes
- Tests robustness against distracting information

**Good score**: < 0.3 means model is robust against noise

**Use case**: Testing model robustness in noisy environments

---

### 7. **Response Relevancy (Answer Relevancy)**

**What it measures**: How relevant the generated response is to the original question.

**Scale**: 0 to 1 (higher is better)

**How it works**:
- Uses an LLM to generate multiple questions that the given answer could address
- Calculates semantic similarity between these generated questions and the original question
- Higher similarity = more relevant answer

**Good score**: > 0.7 means the answer addresses the question well

**Use case**: Ensure answers stay on-topic

---

## 🖼️ Multimodal Metrics

### 8. **Multimodal Faithfulness**

**What it measures**: Factual consistency of generated answers against both visual and textual context.

**Scale**: 0 to 1 (higher is better)

**How it works**:
- Evaluates claims in the answer against both images and text
- Ensures multimodal information is accurately represented
- Checks if visual information is correctly interpreted

**Data required**: Answer, text contexts, image contexts

**Use case**: Evaluating RAG systems that process images and text

---

### 9. **Multimodal Relevance**

**What it measures**: How relevant the generated response is considering both visual and textual inputs.

**Scale**: 0 to 1 (higher is better)

**How it works**:
- Assesses relevance across multiple modalities
- Ensures response addresses both visual and textual aspects of the query

**Use case**: Multimodal RAG applications

---

## 🤖 Agent and Tool Use Metrics

### 10. **Answer Accuracy**

**What it measures**: Overall accuracy of agent responses in tool-use scenarios.

**Scale**: 0 to 1 (higher is better)

**Use case**: Evaluating AI agents that use tools

---

### 11. **Context Relevance**

**What it measures**: Relevance of context provided to agents.

**Scale**: 0 to 1 (higher is better)

**Use case**: Agent context evaluation

---

### 12. **Response Groundedness**

**What it measures**: How well agent responses are grounded in provided context.

**Scale**: 0 to 1 (higher is better)

**Use case**: Preventing agent hallucination

---

### 13. **Topic Adherence**

**What it measures**: How well the agent stays on the intended topic.

**Scale**: 0 to 1 (higher is better)

**Use case**: Conversational agents

---

### 14. **Tool Call Accuracy**

**What it measures**: Accuracy of tool selection and usage by agents.

**Scale**: 0 to 1 (higher is better)

**Use case**: Function-calling agents

---

### 15. **Agent Goal Accuracy**

**What it measures**: How well the agent achieves its intended goals.

**Scale**: 0 to 1 (higher is better)

**Use case**: Task-oriented agents

---

## 📊 Natural Language Comparison Metrics

### 16. **Factual Correctness**

**What it measures**: Factual accuracy of generated text compared to reference.

**Scale**: 0 to 1 (higher is better)

**How it works**:
- Identifies factual claims in both generated and reference text
- Compares factual accuracy using LLM-based evaluation
- Calculates precision and recall of factual statements

**Example**:
- **Generated**: "Python was created in 1991 by Guido"
- **Reference**: "Python was created in 1989 by Guido van Rossum"  
- **Factual Correctness**: ~0.5 (correct creator, wrong year, missing surname)

**Use case**: Fact-checking generated content

---

### 17. **Answer Correctness**

**What it measures**: Overall correctness combining factual accuracy and semantic similarity.

**Scale**: 0 to 1 (higher is better)

**How it works**:
- Combines factual correctness with semantic similarity
- Uses weighted formula: `factual_score * weight + semantic_score * (1-weight)`
- Provides comprehensive correctness assessment

**Formula**: F1 Score + Semantic Similarity (weighted combination)

**Use case**: Comprehensive answer evaluation

---

### 18. **Semantic Similarity**

**What it measures**: Semantic resemblance between generated answer and ground truth.

**Scale**: 0 to 1 (higher is better)

**How it works**:
- Uses embedding models to compute semantic similarity
- Calculates cosine similarity between embeddings
- Can use cross-encoders for better accuracy

**Models used**: 
- OpenAI embeddings (default)
- HuggingFace sentence transformers
- Cross-encoder models

**Use case**: Measuring meaning preservation

---

## 📝 Traditional NLP Metrics (Non-LLM)

### 19. **BLEU Score**

**What it measures**: N-gram overlap between generated and reference text.

**Scale**: 0 to 1 (higher is better)

**How it works**:
- Calculates precision of n-grams (1-gram to 4-gram)
- Uses geometric mean of n-gram precisions
- Includes brevity penalty for short outputs

**Limitations**: Surface-level similarity, doesn't capture meaning

**Use case**: Quick text similarity assessment

---

### 20. **ROUGE Score**

**What it measures**: Recall-oriented text similarity.

**Scale**: 0 to 1 (higher is better)

**Variants**:
- ROUGE-N: N-gram recall
- ROUGE-L: Longest common subsequence
- ROUGE-W: Weighted longest common subsequence

**Use case**: Summarization evaluation

---

### 21. **String Presence**

**What it measures**: Whether specific strings/keywords are present in the output.

**Scale**: Binary (0 or 1)

**Use case**: Checking for required terms or phrases

---

### 22. **Exact Match**

**What it measures**: Exact string matching between generated and reference text.

**Scale**: Binary (0 or 1)

**Use case**: Precise answer verification

---

### 23. **String Similarity**

**What it measures**: Character-level similarity using various algorithms.

**Algorithms**:
- Levenshtein distance
- Jaccard similarity
- Cosine similarity on character n-grams

**Use case**: Typo-tolerant text comparison

---

## 🏗️ Execution-Based Metrics

### 24. **Datacompy Score**

**What it measures**: Data comparison accuracy for structured outputs.

**Use case**: Validating data processing outputs

---

### 25. **SQL Query Equivalence**

**What it measures**: Functional equivalence of generated SQL queries.

**How it works**:
- Executes both generated and reference queries
- Compares result sets
- Checks logical equivalence

**Use case**: SQL generation evaluation

---

## 🎯 General Purpose Evaluation Metrics

### 26. **Aspect Critic**

**What it measures**: Binary evaluation based on custom criteria.

**Scale**: Binary (0 or 1)

**How it works**:
- Define custom evaluation criteria in natural language
- LLM judges whether output meets the criteria
- Highly customizable for specific use cases

**Examples**:
```python
AspectCritic(
    name="harmfulness",
    definition="Does the response contain harmful content?"
)

AspectCritic(
    name="technical_accuracy", 
    definition="Is the technical information accurate?"
)
```

**Use case**: Custom evaluation criteria

---

### 27. **Simple Criteria Scoring**

**What it measures**: Scoring based on simple criteria (0-5 scale).

**Scale**: 0 to 5

**How it works**:
- Define scoring criteria in natural language
- LLM assigns score based on criteria
- More granular than binary AspectCritic

**Use case**: Nuanced custom evaluation

---

### 28. **Rubrics-Based Scoring**

**What it measures**: Evaluation using detailed rubrics.

**Scale**: Typically 1 to 5

**How it works**:
- Define detailed rubrics for each score level
- LLM evaluates based on specific rubric descriptions
- Provides consistent scoring framework

**Example**:
```python
rubrics = {
    "score1_description": "Response is completely incorrect",
    "score2_description": "Response has major errors", 
    "score3_description": "Response is mostly accurate",
    "score4_description": "Response is accurate with minor issues",
    "score5_description": "Response is completely accurate"
}
```

**Use case**: Standardized evaluation with detailed criteria

---

### 29. **Instance-Specific Rubrics Scoring**

**What it measures**: Rubrics that vary per evaluation instance.

**Scale**: Variable based on rubric

**Use case**: Context-dependent evaluation criteria

---

## 🎬 Specialized Application Metrics

### 30. **Summarization Metrics**

**What it measures**: Quality of text summarization.

**Aspects evaluated**:
- Factual consistency
- Coverage of key points
- Conciseness
- Coherence

**Use case**: Summarization model evaluation

---

## 📋 Metric Categories Summary

| Category | Metrics | Primary Use Case |
|----------|---------|------------------|
| **RAG Core** | Faithfulness, Context Precision/Recall, Answer Relevancy | Basic RAG evaluation |
| **Multimodal** | Multimodal Faithfulness, Multimodal Relevance | Vision + Text RAG |
| **Agents** | Tool Call Accuracy, Goal Accuracy, Topic Adherence | AI Agent evaluation |
| **Factual** | Factual Correctness, Answer Correctness | Fact verification |
| **Semantic** | Semantic Similarity, Answer Similarity | Meaning preservation |
| **Traditional** | BLEU, ROUGE, String metrics | Classical NLP evaluation |
| **Custom** | Aspect Critic, Rubrics, Simple Criteria | Domain-specific evaluation |
| **Execution** | SQL Equivalence, Datacompy | Code/Query evaluation |

---

## Complete RAGAS Metrics Catalog

RAGAS provides a comprehensive suite of evaluation metrics organized by application type and use case. Here's the complete catalog:

## 🔍 Retrieval-Augmented Generation (RAG) Metrics

### Core RAG Evaluation Metrics

### 1. **Faithfulness**

**What it measures**: How factually accurate the generated answer is based on the retrieved context.

**Scale**: 0 to 1 (higher is better)

**How it works**:
- Breaks down the generated answer into individual claims/statements
- Checks if each claim can be inferred from the provided context
- Score = (Number of claims supported by context) / (Total number of claims)

**Example**:
- **Context**: "Paris is the capital of France. It has a population of 2.1 million."
- **Answer**: "Paris is the capital of France and has 3 million people."
- **Faithfulness**: 0.5 (1 correct claim out of 2 total claims)

**Good score**: > 0.8 means most claims are supported by context

**Use case**: Detect hallucination in RAG systems

---

### 2. **Answer Relevancy**

**What it measures**: How relevant the generated answer is to the original question.

**Scale**: 0 to 1 (higher is better)

**How it works**:
- Uses an LLM to generate multiple questions that the given answer could address
- Calculates semantic similarity between these generated questions and the original question
- Higher similarity = more relevant answer

**Example**:
- **Question**: "What is the capital of France?"
- **Answer**: "Paris is the capital of France. It's also known for the Eiffel Tower."
- **Relevancy**: High (directly answers the question, extra info is still relevant)

**Good score**: > 0.7 means the answer addresses the question well

**Use case**: Ensure answers stay on-topic

---

### 3. **Context Recall**

**What it measures**: How much of the relevant information from the ground truth is present in the retrieved context.

**Scale**: 0 to 1 (higher is better)

**How it works**:
- Compares the ground truth answer with the retrieved context
- Measures what fraction of the ground truth can be attributed to the retrieved context
- Tests if your retrieval system found all the necessary information

**Example**:
- **Ground Truth**: "Paris is the capital of France, located on the Seine river, with 2.1M population."
- **Retrieved Context**: "Paris is the capital of France. Population is 2.1 million."
- **Context Recall**: ~0.67 (missing Seine river information)

**Good score**: > 0.7 means your retrieval system finds most relevant information

**Use case**: Optimize retrieval completeness

---

### 4. **Context Precision**

**What it measures**: How much of the retrieved context is actually relevant to answering the question.

**Scale**: 0 to 1 (higher is better)

**How it works**:
- Evaluates each piece of retrieved context for relevance to the question
- Measures the precision of your retrieval system (less noise = better)
- Score = (Relevant context chunks) / (Total retrieved context chunks)

**Example**:
- **Question**: "What is the capital of France?"
- **Retrieved Context**: 
  1. "Paris is the capital of France" ✅ Relevant
  2. "France produces excellent wine" ❌ Not relevant
  3. "Paris has many museums" ⚠️ Somewhat relevant
- **Context Precision**: 0.5-1.0 depending on relevance scoring

**Good score**: > 0.8 means minimal irrelevant information retrieved

**Use case**: Reduce retrieval noise and improve efficiency

---

## Summary Table

| Metric | Measures | Good Score | What it tells you |
|--------|----------|------------|-------------------|
| **Faithfulness** | Answer accuracy vs context | > 0.8 | "Is my generator hallucinating?" |
| **Answer Relevancy** | Answer relevance to question | > 0.7 | "Does my answer address the question?" |
| **Context Recall** | Context completeness vs ground truth | > 0.7 | "Did my retriever find enough info?" |
| **Context Precision** | Context relevance to question | > 0.8 | "Is my retriever too noisy?" |

---

## Typical Score Interpretation

### Example Evaluation Results

```python
scores = {
    'faithfulness': 0.95,      # Excellent - no hallucination
    'answer_relevancy': 0.85,  # Good - answers the question well  
    'context_recall': 0.60,    # Poor - missing important context
    'context_precision': 0.90  # Excellent - very little noise
}
```

### Diagnosis
- **Retrieval system needs improvement** (low recall)
- **Generation is working well** (high faithfulness & relevancy)

### Recommendations
1. **Low Context Recall**: Improve retrieval strategy (better embeddings, more diverse retrieval methods)
2. **Low Context Precision**: Filter retrieved results, improve query formulation
3. **Low Faithfulness**: Improve prompting, use better LLM, or add fact-checking
4. **Low Answer Relevancy**: Improve question understanding, better prompt engineering

---

## Basic RAGAS Implementation

### Installation
```bash
pip install ragas
# or with uv
uv add ragas
```

### Basic Usage

```python
import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import (
    faithfulness,
    answer_relevancy,
    context_precision,
    context_recall
)

# Prepare your evaluation data
eval_data = {
    "question": ["What is the capital of France?"],
    "answer": ["Paris is the capital of France."],
    "contexts": [["Paris is the capital city of France, located on the Seine River."]],
    "ground_truth": ["Paris is the capital of France."]
}

# Create dataset
dataset = Dataset.from_dict(eval_data)

# Run evaluation
results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall]
)

# View results
results_df = results.to_pandas()
print(results_df)
```

### Required Data Format

Your evaluation dataset needs these columns:

- **question**: The user's question/query
- **answer**: The generated response from your RAG system
- **contexts**: List of retrieved context chunks (as list of strings)
- **ground_truth**: The correct/expected answer

---

## Advanced Topics

### Custom Chunk Sizes

When generating test data, you can customize chunking:

```python
from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,        # Your custom chunk size
    chunk_overlap=100,     # Overlap between chunks
    separators=["\n\n", "\n", " ", ""]
)
```

### Multiturn Conversations

For evaluating conversations with multiple turns:

1. **Flatten approach**: Treat each turn as separate evaluation
2. **Conversation-level**: Group results by conversation ID
3. **Context accumulation**: Include previous turns in context

### Using Different LLM Providers

RAGAS works with various LLM providers:

```python
# With Groq
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper

llm = LangchainLLMWrapper(ChatGroq(model="mixtral-8x7b-32768"))
```

---

## Troubleshooting

### Common Import Errors

**ragas 0.2.15+**:
```python
# Correct imports
from ragas.testset import TestsetGenerator  # NOT from ragas.testset.generator
from ragas.metrics import faithfulness      # lowercase
```

**Older versions**:
```python
from ragas.testset.generator import TestsetGenerator
```

### API Key Setup

Set your API keys:
```bash
export OPENAI_API_KEY="your-key-here"
export GROQ_API_KEY="your-groq-key"
```

### Version Compatibility

Check your ragas version:
```python
import ragas
print(ragas.__version__)
```

Different versions have different APIs - refer to the specific documentation for your version.

---

## Best Practices

### 1. **Balanced Evaluation**
- Include diverse question types
- Test edge cases and failure modes
- Use sufficient sample size (100+ questions for statistical significance)

### 2. **Iterative Improvement**
- Start with baseline measurements
- Make targeted improvements based on metric weaknesses
- Re-evaluate after each change

### 3. **Context Quality**
- Ensure retrieved contexts are actually relevant
- Balance context length vs. precision
- Monitor context diversity

### 4. **Ground Truth Quality**
- Use high-quality reference answers
- Ensure consistency across evaluators
- Include multiple valid answers when appropriate

---

## Metric Combinations for Different Use Cases

### **Customer Support RAG**
- **High priority**: Answer Relevancy, Faithfulness
- **Lower priority**: Context Recall (completeness less critical)

### **Research/Academic RAG**
- **High priority**: Context Recall, Faithfulness
- **Medium priority**: Context Precision, Answer Relevancy

### **Chat/Conversational RAG**
- **High priority**: Answer Relevancy, Context Precision
- **Medium priority**: Faithfulness, Context Recall

---

## Additional Resources

- [RAGAS Documentation](https://docs.ragas.io/)
- [RAGAS GitHub Repository](https://github.com/explodinggradients/ragas)
- [RAG Evaluation Best Practices](https://docs.ragas.io/en/stable/concepts/metrics/)

---

*This guide covers RAGAS framework version 0.2.15+ with backward compatibility notes for older versions.*