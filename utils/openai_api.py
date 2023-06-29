import os
import openai
import tiktoken
import datetime
from time import sleep
openai.api_key = os.getenv("OPENAI_API_KEY")
openai.api_base =  os.getenv("OPENAI_API_BASE") # your endpoint should look like the following https://YOUR_RESOURCE_NAME.openai.azure.com/
openai.api_type = 'azure'
# openai.api_version = '2022-12-01' # this may change in the future
# openai.api_version = "2023-03-15-preview"
openai.api_version = "2023-05-15"


def openai_completion(prompt, model_name="gpt-35", max_tokens_to_sample: int = 200, stop="\n\n", temperature=0):
    if model_name == "gpt-35":
        deployment_name = "gpt-35-turbo"
        enc = tiktoken.get_encoding("cl100k_base") 
        while len(enc.encode(prompt)) > 4096:
            if len(prompt) > 4000:
                question_idx = prompt.rfind("\n\nQuestion:")
                prompt = prompt[:4000-len(prompt[question_idx:])] + prompt[question_idx:]
            else:
                question_idx = prompt.rfind("\n\nQuestion:")
                prompt = prompt[:question_idx-300] + prompt[question_idx:]
    elif model_name == "text-davinci-003":
        deployment_name = "text-davinci-003"
        enc = tiktoken.get_encoding("p50k_base")
        while len(enc.encode(prompt)) > 4096:
            if len(prompt) > 4000:
                question_idx = prompt.rfind("\n\nQuestion:")
                prompt = prompt[:4000-len(prompt[question_idx:])] + prompt[question_idx:]
            else:
                question_idx = prompt.rfind("\n\nQuestion:")
                prompt = prompt[:question_idx-300] + prompt[question_idx:]
    else:
        raise ValueError("Invalid model name")
    
    # print('Sending a test completion job')
    # start_phrase = 'Write a tagline for an ice cream shop. '
    while True:
        sleep(1)
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
        # except TypeError:
        #     print(f"TypeError, maybe the prompt is too long: {len(prompt)}. Redeucing the prompt length.")
        #     if len(prompt) > 4000:
        #         question_idx = prompt.rfind("\n\nQuestion:")
        #         prompt = prompt[:4000-len(prompt[question_idx:])] + prompt[question_idx:]
        #     else:
        #         question_idx = prompt.rfind("\n\nQuestion:")
        #         prompt = prompt[:question_idx-300] + prompt[question_idx:]
        #     continue
        except openai.error.InvalidRequestError:
            print("Invalid request!!!")
            text = ""
            break
        except openai.error.Timeout:
            print("Timeout, retrying...")
            continue

    return text


def openai_chat_completion(prompt, model_name="gpt-35", max_tokens_to_sample: int = 200, stop="\n\n", temperature=0):
    if model_name == "gpt-35":
        deployment_name = "gpt-35-turbo"
    else:
        raise ValueError("Invalid model name")
    
    enc = tiktoken.get_encoding("cl100k_base") 

    while len(enc.encode(prompt)) > 4096:
        if len(prompt) > 4000:
            question_idx = prompt.rfind("\n\nQuestion:")
            prompt = prompt[:4000-len(prompt[question_idx:])] + prompt[question_idx:]
        else:
            question_idx = prompt.rfind("\n\nQuestion:")
            prompt = prompt[:question_idx-300] + prompt[question_idx:]
        continue
    current_date = str(datetime.date.today())
    messages = [
        # {"role": "system", "content": f"You are ChatGPT, a large language model trained by OpenAI.\nKnowledge cutoff: 2021-09\nCurrent Date: {current_date}"},
        {"role": "system", "content": "You are a helpful assistant responsible for extracting answers to given questions from specified documents. The answers must exist in the documents, and your replies should come directly from the documents without containing any other irrelevant information. You need to understand the document layout with the help of spaces and line breaks in the document."},
        # {"role": "system", "content": "You are an AI assistant that helps people find information."},
        {"role": "user","content": prompt}
    ]

    print(f"prompt_token_length: {len(enc.encode(prompt))}")
    while True:
        sleep(2)
        try:
            response = openai.ChatCompletion.create(
                engine=deployment_name,
                messages=messages,
                max_tokens=max_tokens_to_sample,
                temperature=temperature,
                stop=stop,
            )
            print(response['choices'][0]['message'])

            if 'content' in response['choices'][0]['message'].keys():
                text = response['choices'][0]['message']['content']
            else:
                text = ""
            break
        except openai.error.InvalidRequestError:
            print("Invalid request!!!")
            text = ""
            break
        except openai.error.Timeout:
            sleep(30)
            print("Timeout, retrying...")
            continue

    return text


if __name__ == "__main__":
    prompt = "Write a tagline for an ice cream shop. "
    print(openai_completion(prompt))
    print(openai_chat_completion(prompt))
