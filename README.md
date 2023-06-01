# LATIN-Promot

## Prepare the environment
`pip install -r requirements.txt`

## Prepare environment variable for Cluade and OpenAI
Set the `ANTHROPIC_API_KEY` for Cluade. Please refer to the script `./utils/claude.py`.

Set the `OPENAI_API_KEY` and `OPENAI_API_BASE` for Azure OpenAI. Please refer to the script `./utils/openai_api.py`

## Prepare the dataset
Download and put the DocVQA dataset into the `./datas/docvqa`

## Example: Claude + LATIN-Prompt on DocVQA
```bash
bash script/claude_eval.sh 0 claude docvqa task_instruction_space
```

## Example: Claude + Plain Prompt on DocVQA
```bash
bash script/claude_eval.sh 0 claude docvqa plain
```

## Example: Claude + Task Description on DocVQA
```bash
bash script/claude_eval.sh 0 claude docvqa task_instruction
```

## Example: Claude + Layout-aware Document on DocVQA
```bash
bash script/claude_eval.sh 0 claude docvqa space
```