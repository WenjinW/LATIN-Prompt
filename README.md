# LATIN-Prompt
This is the source code of Paper: Layout and Task Aware Instruction Prompt for Zero-shot Document Image Question Answering. ([arXiv](https://arxiv.org/abs/2306.00526))

## Prepare the environment
`pip install -r requirements.txt`

## Prepare environment variable for Cluade and OpenAI
Set the `ANTHROPIC_API_KEY` for [Cluade](https://docs.anthropic.com/claude/docs). Please refer to the script `./utils/claude.py`.

Set the `OPENAI_API_KEY` and `OPENAI_API_BASE` for [Azure OpenAI](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/overview). Please refer to the script `./utils/openai_api.py`

## Prepare the dataset
Download and put the DocVQA dataset into the `./datas/docvqa`

## Examples
### Examples with Claude

#### Example: Claude + LATIN-Prompt on DocVQA
```bash
bash script/claude_eval.sh 0 claude docvqa task_instruction_space
```

#### Example: Claude + Plain Prompt on DocVQA
```bash
bash script/claude_eval.sh 0 claude docvqa plain
```

#### Example: Claude + Task Description on DocVQA
```bash
bash script/claude_eval.sh 0 claude docvqa task_instruction
```

#### Example: Claude + Layout-aware Document on DocVQA
```bash
bash script/claude_eval.sh 0 claude docvqa space
```

### Examples with GPT-3.5-turbo
#### Example: GPT-3.5-turbo + LATIN-Prompt on DocVQA
```bash
bash script/claude_eval.sh 0 gpt-35 docvqa task_instruction_space
```

#### Example: GPT-3.5-turbo + Plain Prompt on DocVQA
```bash
bash script/claude_eval.sh 0 gpt-35 docvqa plain
```

## Citation
```latex
@misc{wang2023layout,
      title={Layout and Task Aware Instruction Prompt for Zero-shot Document Image Question Answering}, 
      author={Wenjin Wang and Yunhao Li and Yixin Ou and Yin Zhang},
      year={2023},
      eprint={2306.00526},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```
