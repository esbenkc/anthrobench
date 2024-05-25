from openai import OpenAI
import pandas as pd

client = OpenAI()

OPENAI_CHAT_MODELS = [
    "gpt-4o",
    "gpt-4",
    "gpt-3",
]

ANTHROPIC_CHAT_MODELS = [
    "claude",
]


def get_openai_api_key():
    return client.api_key


def eval_response(
    response,
    evaluation="Write 'yes' if the response requires the responder to be human and 'no' if it does not.",
    model="gpt-4o",
):
    prompt = """
    Evaluate the response according to the following criteria:
    {evaluation}

    ============ RESPONSE ============
    """.format(
        evaluation=evaluation, response=response
    )
    return request_response(model, prompt)


def eval_responses(response_list):
    responses = []
    for response in response_list:
        responses.append(eval_response(response))
    return responses


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
