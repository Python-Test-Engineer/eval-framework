# RAGAS Framework Metrics Guide

## Overview

RAGAS (Retrieval-Augmented Generation Assessment) is a framework for evaluating RAG (Retrieval-Augmented Generation) systems. This guide covers the four core metrics used to assess different aspects of your RAG pipeline.

---

## Core RAGAS Metrics

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