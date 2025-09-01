import ast


def extract_last_column_keys(csv_row: str) -> dict:
    """
    Extract all key-value pairs from the last column of a CSV row.
    Simple function that splits by pipe and parses the dictionary.
    """
    # Split by pipe and get the last column
    columns = csv_row.split("|")
    last_column = columns[-1]

    try:
        # Parse the dictionary string
        data = ast.literal_eval(last_column)

        # Simple recursive function to flatten all keys
        def get_all_keys(obj, prefix=""):
            result = {}
            if isinstance(obj, dict):
                for key, value in obj.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, dict):
                        result.update(get_all_keys(value, full_key))
                    else:
                        result[full_key] = value
            return result

        return get_all_keys(data)

    except Exception as e:
        print(f"Error parsing last column: {e}")
        return {}


# Test with your CSV row
csv_row02 = "2025-07-20-21-18-33|ARTICLE_WRITER|TRANSLATE|gpt-4o-mini|0|AI and Machine learning in modern business|0.56|{'content': \"L'IA et l'apprentissage automatique dans les affaires modernes\", 'additional_kwargs': {'refusal': None}, 'response_metadata': {'token_usage': {'completion_tokens': 12, 'prompt_tokens': 45, 'total_tokens': 57, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': None, 'id': 'chatcmpl-BvUmOydpjRTukoYH4KoeLWMZyKyCE', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, 'type': 'ai', 'name': None, 'id': 'run--ccdfc078-fb5b-4cbe-ba12-e55681af84af-0', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 45, 'output_tokens': 12, 'total_tokens': 57, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}}"
csv_row03 = """2025-07-20-21-18-36|ARTICLE_WRITER|EXPANDER|gpt-4o-mini|0|L'IA et l'apprentissage automatique dans les affaires modernes|2.29|{'content': "L'intelligence artificielle (IA) et l'apprentissage automatique (AA) jouent un rôle de plus en plus crucial dans les affaires modernes. Ces technologies permettent aux entreprises d'analyser d'énormes quantités de données pour en extraire des insights précieux, optimisant ainsi la prise de décision. Par exemple, les chatbots alimentés par l'IA améliorent le service client en offrant des réponses instantanées et personnalisées. De plus, l'apprentissage automatique aide à prédire les tendances du marché, permettant aux entreprises de s'adapter rapidement aux changements. En intégrant ces outils, les organisations peuvent non seulement accroître leur efficacité, mais aussi innover dans leurs produits et services, renforçant ainsi leur position sur le marché.", 'additional_kwargs': {'refusal': None}, 'response_metadata': {'token_usage': {'completion_tokens': 152, 'prompt_tokens': 60, 'total_tokens': 212, 'completion_tokens_details': {'accepted_prediction_tokens': 0, 'audio_tokens': 0, 'reasoning_tokens': 0, 'rejected_prediction_tokens': 0}, 'prompt_tokens_details': {'audio_tokens': 0, 'cached_tokens': 0}}, 'model_name': 'gpt-4o-mini-2024-07-18', 'system_fingerprint': None, 'id': 'chatcmpl-BvUmPAL1PopYjZYRe2rzTyVjCPVXm', 'service_tier': 'default', 'finish_reason': 'stop', 'logprobs': None}, 'type': 'ai', 'name': None, 'id': 'run--3dc7d24b-cf7b-4aa4-a9d3-983f21802838-0', 'example': False, 'tool_calls': [], 'invalid_tool_calls': [], 'usage_metadata': {'input_tokens': 60, 'output_tokens': 152, 'total_tokens': 212, 'input_token_details': {'audio': 0, 'cache_read': 0}, 'output_token_details': {'audio': 0, 'reasoning': 0}}}"""


# Extract keys from last column
keys = extract_last_column_keys(csv_row03)

# Print results
print("All keys from last column:")
print("-" * 40)
for key, value in keys.items():
    print(f"{key}: {value}")

print(f"\nTotal keys found: {len(keys)}")
