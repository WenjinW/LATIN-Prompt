import anthropic
import os
import io
import json

c = anthropic.Client(os.environ["ANTHROPIC_API_KEY"])


def cluade_completion(
    prompt,
    model_name="claude-v1.3",
    max_tokens_to_sample: int = 200,
    temperature=0.0,
):
    response = c.completion_stream(
        prompt=f"{anthropic.HUMAN_PROMPT} {prompt}{anthropic.AI_PROMPT}",
        stop_sequences=[anthropic.HUMAN_PROMPT],
        max_tokens_to_sample=max_tokens_to_sample,
        model=model_name,
        stream=True,
        temperature=temperature,
    )
    # for data in response:
    #     print(data)

    result = list(response)[-1]

    return result

