import os
import openai

openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base =  os.getenv("OPENAI_API_BASE") # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
# openai.api_version = '2022-12-01' # this may change in the future
openai.api_version = "2023-03-15-preview"


def openai_completion(prompt, model_name="gpt-35", max_tokens_to_sample: int = 200, stop="\n\n", temperature=0):
    if model_name == "gpt-35":
        deployment_name = "gpt-35-turbo"
    else:
        raise ValueError("Invalid model name")
    
    # print('Sending a test completion job')
    # start_phrase = 'Write a tagline for an ice cream shop. '
    while True:
        try:
            response = openai.Completion.create(
                engine=deployment_name,
                prompt=prompt,
                max_tokens=max_tokens_to_sample,
                temperature=temperature,
                stop=stop,
            )
            text = response['choices'][0]['text']
            break
        except TypeError:
            print(f"TypeError, maybe the prompt is too long: {len(prompt)}. Redeucing the prompt length.")
            if len(prompt) > 4000:
                question_idx = prompt.rfind("\n\nQuestion:")
                prompt = prompt[:4000-len(prompt[question_idx:])] + prompt[question_idx:]
            else:
                question_idx = prompt.rfind("\n\nQuestion:")
                prompt = prompt[:question_idx-300] + prompt[question_idx:]
            continue

    return text


def openai_chat_completion(prompt, model_name="gpt-35", max_tokens_to_sample: int = 200, stop="\n", temperature=2):
    if model_name == "gpt-35":
        deployment_name = "gpt-35-turbo"
    else:
        raise ValueError("Invalid model name")
    
    messages = [{
        "role": "user",
        "content": prompt,
    }]
    response = openai.ChatCompletion.create(
        engine=deployment_name,
        messages=messages,
        max_tokens=max_tokens_to_sample,
        temperature=temperature,
        stop=stop,
    )
    text = response['choices'][0]['message']['content']

    return text


if __name__ == "__main__":
    prompt = "Write a tagline for an ice cream shop. "
    print(openai_completion(prompt))
    print(openai_chat_completion(prompt))