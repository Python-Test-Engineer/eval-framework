import ast
import json
import re

CONTENT_LENGTH = 100  # Example constant, adjust as needed


def extract_content_key(pipe_delimited_line):
    """Extract the 'content' value from a pipe-delimited CSV line"""

    # Split the line by pipe delimiter
    fields = pipe_delimited_line.split("|")

    # The last field contains the JSON-like dictionary
    json_field = fields[-1]

    try:
        # First try to parse as JSON
        data_dict = json.loads(json_field)
    except json.JSONDecodeError:
        try:
            # If JSON fails, try ast.literal_eval (handles Python dict format)
            data_dict = ast.literal_eval(json_field)
        except (ValueError, SyntaxError) as e:
            print(f"Error parsing data: {e}")
            return None

    # Extract the content value
    content = data_dict.get("content", "")
    return content


def extract_content_key_safe(pipe_delimited_line):
    """Safer version with more error handling"""

    try:
        fields = pipe_delimited_line.split("|")
        if len(fields) < 8:  # Based on your example, should have 8 fields
            print(f"Warning: Expected at least 8 fields, got {len(fields)}")

        json_field = fields[7].strip()
        # Remove any surrounding quotes

        # Try both JSON and literal_eval
        for parser in [json.loads, ast.literal_eval]:
            try:
                data_dict = parser(json_field)
                content = data_dict.get("content", "")
                return content
            except:
                continue

        print("Could not parse the data dictionary")
        return None

    except Exception as e:
        print(f"Error processing line: {e}")
        return None


# Test with your example line
sample_line = """2025-07-22-17-42-45|ARTICLE_WRITER|EXPANDER|gpt-4o-mini|0|Pourquoi les agents d'IA remplaceront les logiciels traditionnels|4.67|{'content': "Les agents d'intelligence artificielle (IA) sont en passe de remplacer les logiciels traditionnels pour plusieurs raisons convaincantes. Tout d'abord, ces agents sont capables d'apprendre et de s'adapter en temps réel, ce qui leur permet d'améliorer \n\ncontinuellement leurs performances. Contrairement aux logiciels traditionnels qui nécessitent des mises à jour manuelles, les agents d'IA peuvent analyser de grandes quantités de données et en tirer des enseignements autonomes. De plus, ils offrent une personnalisation avancée, répondant ainsi aux besoins spécifiques des utilisateurs de manière plus efficace. En intégrant des capacités de traitement du langage naturel et d'analyse prédictive, les agents d'IA révolutionnent la manière dont nous interagissons avec la technologie, rendant les processus plus fluides et intuitifs. Cette transition vers des systèmes intelligents promet non seulement d'augmenter l'efficacité, mais aussi de transformer les modèles d'affaires traditionnels, ouvrant la voie à une nouvelle ère d'innovation numérique.", 'additional_kwargs': {'refusal': None}, 'response_metadata': {'token_usage': {'completion_tokens': 208, 'prompt_tokens': 60, 'total_tokens': 268, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': None, 'id': 'chatcmpl-BwAMZ5Lne3hZW7vV88Px0fDUltogl', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, 'type': 'ai', 'name': None, 'id': 'run--7f4b1fc6-3156-4c01-b63c-090e78b36c63-0', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 60, 'output_tokens': 208, 'total_tokens': 268, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}}"""

sample_line = sample_line.replace("\n", "")
print("Testing content extraction:")
print("=" * 50)

# Method 1: Basic extraction
content = extract_content_key(sample_line)
num_words = len(content.split())
print(f"Number of words in content: {num_words}")
if num_words > CONTENT_LENGTH:
    print(f"\nContent exceeds {CONTENT_LENGTH} words - PASS.\n")
else:
    print(f"\nContent does not exceed {CONTENT_LENGTH} words - FAIL.\n")

print(f"Extracted content: {content}")
