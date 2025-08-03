import json
from pathlib import Path

import requests


def load_test_data(filename):
    project_directory = Path(__file__).parent.absolute()
    test_data_path = project_directory/"testdata"/filename
    with open(test_data_path) as f:
        return json.load(f)

#user_input = test_data["eval_sample"]["user_input"]
def get_llm_response(test_data):
    responseDict = requests.post("https://rahulshettyacademy.com/rag-llm/ask",
                                 json={
                                     "question": test_data["question"],
                                     "chat_history": [
                                     ]
                                 }).json()
    return responseDict
