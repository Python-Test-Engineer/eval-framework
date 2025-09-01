import pandas as pd
import json
import ast

# Your JSON data (you can also load this from a file)
json_data = {
    "user_input": {
        "0": "What is the mission of OpenAI?",
        "1": "When was Microsoft founded?",
        "2": "What programming language was created by Google?",
        "3": "What is the primary business model of Amazon?",
        "4": "Who is the current CEO of Tesla?",
        "5": "What is artificial intelligence?"
    },
    "retrieved_contexts": {
        "0": [
            "['OpenAI is an AI research and deployment company. Our mission is to ensure that artificial general intelligence (AGI) benefits all of humanity.', 'OpenAI conducts research in machine learning and artificial intelligence with the goal of promoting and developing friendly AI.', 'The company was founded in 2015 by several tech entrepreneurs including Elon Musk and Sam Altman.']"
        ],
        "1": [
            "['Microsoft Corporation was founded by Bill Gates and Paul Allen on April 4, 1975.', 'The company started in Albuquerque, New Mexico, and later moved to Washington state.', \"Microsoft became one of the world's largest software companies, known for Windows and Office products.\"]"
        ],
        "2": [
            "['Google created the Go programming language, also known as Golang, which was announced in 2009.', 'Go was designed by Robert Griesemer, Rob Pike, and Ken Thompson at Google.', 'The language was created to address shortcomings in other languages used at Google for server-side development.']"
        ],
        "3": [
            "['Amazon started as an online bookstore but has evolved into a massive e-commerce and cloud computing platform.', \"Amazon's revenue comes from multiple sources including e-commerce, Amazon Web Services (AWS), advertising, and subscription services.\", \"AWS is Amazon's cloud computing division and has become one of the most profitable parts of the business.\"]"
        ],
        "4": [
            "['Elon Musk is the CEO and product architect of Tesla, Inc.', 'Tesla is an electric vehicle and clean energy company founded in 2003.', \"Under Musk's leadership, Tesla has become one of the world's most valuable automakers.\"]"
        ],
        "5": [
            "['Artificial Intelligence (AI) is the simulation of human intelligence processes by machines, especially computer systems.', 'AI includes learning, reasoning, problem-solving, perception, and language understanding.', 'Machine learning is a subset of AI that enables computers to learn and improve from experience without being explicitly programmed.']"
        ]
    },
    "response": {
        "0": "OpenAI's mission is to ensure that artificial general intelligence benefits all of humanity by conducting research and developing safe AI systems.",
        "1": "Microsoft was founded on April 4, 1975, by Bill Gates and Paul Allen.",
        "2": "Google created the Go programming language, also known as Golang, which was announced in 2009.",
        "3": "Amazon's primary business model is e-commerce, but they also generate significant revenue from cloud computing services through AWS, advertising, and subscription services.",
        "4": "Elon Musk is the current CEO of Tesla, Inc.",
        "5": "Artificial Intelligence (AI) is the simulation of human intelligence in machines, enabling them to learn, reason, solve problems, and understand language."
    },
    "reference": {
        "0": "OpenAI's mission is to ensure that artificial general intelligence (AGI) benefits all of humanity.",
        "1": "Microsoft was founded on April 4, 1975.",
        "2": "Google created the Go programming language (Golang).",
        "3": "Amazon's primary business model includes e-commerce, cloud computing (AWS), advertising, and subscription services.",
        "4": "Elon Musk is the current CEO of Tesla.",
        "5": "Artificial Intelligence is the simulation of human intelligence processes by machines, including learning, reasoning, and problem-solving."
    },
    "faithfulness": {
        "0": 0.6666666666666666,
        "1": 1.0,
        "2": 1.0,
        "3": 1.0,
        "4": 1.0,
        "5": 1.0
    },
    "answer_relevancy": {
        "0": 0.9893007951252031,
        "1": 0.9544443098859813,
        "2": 0.8877759514141648,
        "3": 0.984300315826995,
        "4": 0.9908968554155795,
        "5": 0.9747489843693143
    },
    "context_recall": {
        "0": 1.0,
        "1": 1.0,
        "2": 1.0,
        "3": 1.0,
        "4": 1.0,
        "5": 1.0
    },
    "context_precision": {
        "0": 0.9999999999,
        "1": 0.9999999999,
        "2": 0.9999999999,
        "3": 0.9999999999,
        "4": 0.9999999999,
        "5": 0.9999999999
    }
}

def process_retrieved_contexts(contexts):
    """Process the retrieved_contexts field which contains stringified lists"""
    if isinstance(contexts, list) and len(contexts) > 0:
        try:
            # Parse the stringified list
            return ast.literal_eval(contexts[0])
        except:
            return contexts
    return contexts

def json_to_dataframe(data):
    """Convert the JSON data to a pandas DataFrame with each user input as a row"""
    
    # Get the number of rows based on user_input length
    num_rows = len(data['user_input'])
    
    # Create a list to store all rows
    rows = []
    
    for i in range(num_rows):
        row = {
            'user_input': data['user_input'][str(i)],
            'response': data['response'][str(i)],
            'reference': data['reference'][str(i)],
            'retrieved_contexts': process_retrieved_contexts(data['retrieved_contexts'][str(i)]),
            'faithfulness': data['faithfulness'][str(i)],
            'answer_relevancy': data['answer_relevancy'][str(i)],
            'context_recall': data['context_recall'][str(i)],
            'context_precision': data['context_precision'][str(i)]
        }
        rows.append(row)
    
    return pd.DataFrame(rows)

# Create the DataFrame
df = json_to_dataframe(json_data)

# Display the DataFrame
print("DataFrame Shape:", df.shape)
print("\nDataFrame Info:")
print(df.info())
print("\nFirst few rows:")
print(df.head())

# Save to CSV if needed
# df.to_csv('evaluation_results.csv', index=False)

# Alternative method if loading from a file:
def load_json_and_convert(file_path):
    """Load JSON from file and convert to DataFrame"""
    with open(file_path, 'r') as f:
        data = json.load(f)
    return json_to_dataframe(data)

# Example usage:
# df = load_json_and_convert('paste.txt')  # if your file is JSON format