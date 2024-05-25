from openai import OpenAI
import pandas as pd

client = OpenAI()

OPENAI_CHAT_MODELS = [
    "text-davinci-003",
]


def get_openai_api_key():
    return client.api_key


def eval_response(response):
    if response["status"] == 200:
        return response["choices"][0]["text"]
    else:
        return None


def request_response(model, prompt, system="You are a helpful assistant."):
    if model in OPENAI_CHAT_MODELS:
        response = client.chat.completions.create(
            engine=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=200,
        )
        response = response["choices"][0]["message"]["content"]
    return response


def save_result(result, filename="result.csv"):
    # Check if type is pandas df
    if isinstance(result, pd.DataFrame):
        result.to_csv(filename, index=False)
    else:
        ValueError("Result is not a pandas DataFrame")
        return None
