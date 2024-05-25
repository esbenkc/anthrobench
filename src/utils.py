import openai


def get_openai_api_key():
    return openai.api_key


def eval_response(response):
    if response["status"] == 200:
        return response["choices"][0]["text"]
    else:
        return None


def request_response(model):
    response = openai.Completion.create(
        engine=model,
        prompt="Translate the following English text to French: 'Hello, how are you?'",
        max_tokens=60,
    )
    return response
