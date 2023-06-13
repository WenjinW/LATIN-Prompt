# LATIN-Prompt
This is the source code of Paper: Layout and Task Aware Instruction Prompt for Zero-shot Document Image Question Answering. ([arXiv](https://arxiv.org/abs/2306.00526))

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
