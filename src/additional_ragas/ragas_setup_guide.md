# RAGAS Testing Setup Guide

## Overview
This project includes a complete RAGAS (Retrieval Augmented Generation Assessment) evaluation setup for testing RAG applications.

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Set up Environment Variables
Create a `.env` file in your project root:
```bash
# .env file
OPENAI_API_KEY=your-openai-api-key-here
```

Or set the environment variable directly:
```bash
export OPENAI_API_KEY="your-openai-api-key-here"
```

## Project Structure
```
your-project/
├── sample_dataset.py      # Creates sample dataset
├── ragas_evaluation.py    # Main evaluation script
├── requirements.txt       # Dependencies
├── .env                   # Environment variables
└── README.md             # This file
```

## Usage

### 1. Create Sample Dataset
```bash
python sample_dataset.py
```
This creates a CSV file with sample questions, contexts, and answers.

### 2. Run RAGAS Evaluation
```bash
python ragas_evaluation.py
```

### 3. View Results
The script will output:
- Overall metric scores
- Individual example analysis
- Detailed evaluation report (`ragas_evaluation_report.txt`)

## RAGAS Metrics Explained

### Core Metrics
- **Faithfulness**: Measures if the answer is factually consistent with the retrieved context
- **Answer Relevancy**: Measures how relevant the answer is to the question asked
- **Context Recall**: Measures if all relevant information needed to answer the question is retrieved
- **Context Precision**: Measures the precision of the retrieval system
- **Context Relevancy**: Measures how relevant the retrieved contexts are to the question
- **Answer Correctness**: Measures the factual correctness of the generated answer
- **Answer Similarity**: Measures semantic similarity between generated answer and ground truth

### Score Interpretation
- **0.0 - 0.4**: Poor performance, significant improvements needed
- **0.4 - 0.7**: Moderate performance, some