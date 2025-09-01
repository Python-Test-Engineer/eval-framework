def convert_string_to_dict(input_string):
    """
    Convert a space-separated string of key=value pairs to a dictionary.

    Args:
        input_string (str): String with key='value' pairs

    Returns:
        dict: Dictionary of key-value pairs
    """
    # Remove quotes and split by spaces
    result = {}

    # Split the string by spaces while preserving quoted values
    import re

    # Use regex to find key='value' pairs
    pairs = re.findall(r"(\w+)='([^']*)'", input_string)

    # Convert to dictionary
    result = dict(pairs)

    return result


# Example usage
if __name__ == "__main__":
    # Test string
    test_string = "can_be_posted='yes' meets_word_count='yes' is_not_sensationalistic='yes' is_language_french='yes'"

    # Convert to dictionary
    result_dict = convert_string_to_dict(test_string)
    print(result_dict)
    # Print the result
    print("Converted Dictionary:")
    for key, value in result_dict.items():
        print(f"{key}: {value}")
