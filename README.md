<h1 align="center">
LATIN-Prompt
</h1>

<h3 align="center">
Layout and Task Aware Instruction Prompt for Zero-shot Document Image Question Answering
</h3>

<p align="center">
  <a href="https://arxiv.org/abs/2306.00526"><img alt="Static Badge" src="https://img.shields.io/badge/arXiv-Paper-<COLOR>.svg"></a>
</p>

## News
- **2022.06**: We now provide implementations based on [Alpaca-7B](https://github.com/tatsu-lab/stanford_alpaca) and [Vicuna-13B](https://github.com/vllm-project/vllm). 

## Roadmap
- [x] DUE OCR results
- [x] Alpaca 7B
- [x] Vicuna 13B
- [x] Azure gpt-3.5-turbo + Completion
- [ ] Azure gpt-3.5-turbo + ChatCompletion (doing)
- [ ] Azure text-davinci-003 + Completion (doing)
- [ ] MPT-30B-Chat (todo)
- [ ] Orca (todo)
- [ ] gpt-4 (todo, there is currently no available API)


## Preparation
#### Prepare the environment
`pip install -r requirements.txt`

#### Prepare environment variable for Cluade and OpenAI
Set the `ANTHROPIC_API_KEY` for [Cluade](https://docs.anthropic.com/claude/docs). Please refer to the script `./utils/claude.py`.

Set the `OPENAI_API_KEY` and `OPENAI_API_BASE` for [Azure OpenAI](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/overview). Please refer to the script `./utils/openai_api.py`

#### Prepare the dataset
Download and put the DocVQA dataset into the `./datas/docvqa`

## Examples
### Examples with Claude

#### Example: Claude + LATIN-Prompt on DocVQA (Azure OCR)
```bash
bash script/claude_eval.sh 0 claude docvqa_due_azure task_instruction_space
```

#### Example: Claude + Plain Prompt on DocVQA (Azure OCR)
```bash
bash script/claude_eval.sh 0 claude docvqa_due_azure plain
```

#### Example: Claude + Task Description on DocVQA (Azure OCR)
```bash
bash script/claude_eval.sh 0 claude docvqa_due_azure task_instruction
```

#### Example: Claude + Layout-aware Document on DocVQA (Azure OCR)
```bash
bash script/claude_eval.sh 0 claude docvqa_due_azure space
```

### Examples with Azure GPT-3.5-turbo (ChatGPT) Completion
#### Example: GPT-3.5-turbo + LATIN-Prompt on DocVQA (Azure OCR)
```bash
bash script/claude_eval.sh 0 gpt-35 docvqa_due_azure task_instruction_space
```

#### Example: GPT-3.5-turbo + Plain Prompt on DocVQA (Azure OCR)
```bash
bash script/claude_eval.sh 0 gpt-35 docvqa_due_azure plain
```

### Examples with Azure GPT-3.5-turbo (ChatGPT) ChatCompletion
#### Example: GPT-3.5-turbo + LATIN-Prompt on DocVQA
```bash
bash script/claude_eval.sh 0 gpt-35-chat docvqa task_instruction_space
```

### Examples with Alpaca and Vicuna
#### Example: Alpaca + LATIN-Prompt on DocVQA (Azure OCR)
```bash
bash script/llama_eval.sh 0 alpaca-7b docvqa_due_azure task_instruction_space
```

#### Example: Vicuna + LATIN-Prompt on DocVQA (Azure OCR)
```bash
bash script/vllm_eval.sh 0 vicuna-13b docvqa_due_azure task_instruction_space
```

## Performance
### DocVQA (Azure OCR, DUE)
The performance in this table is based on the Azure OCR results provided in [DUE Benchmark](https://github.com/due-benchmark) by default.
The Official OCR represents the performance is based on the OCR results provided in [Robust Reading Competition](https://rrc.cvc.uab.es/?com=introduction)
<table style="undefined;table-layout: fixed; width: 1204px">
<colgroup>
<col style="width: 319.2px">
<col style="width: 109.2px">
<col style="width: 112.2px">
<col style="width: 170.2px">
<col style="width: 264.2px">
<col style="width: 229.2px">
</colgroup>
<thead>
  <tr>
    <th rowspan="2">Model</th>
    <th rowspan="2">Prompt</th>
    <th colspan="2">Test Data</th>
    <th colspan="2">Val Data</th>
  </tr>
  <tr>
    <th>ANLS</th>
    <th>↑</th>
    <th>ANLS</th>
    <th>↑</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td rowspan="2">Claude</td>
    <td>Plain</td>
    <td>0.2298</td>
    <td>-</td>
    <td>0.2144</td>
    <td>-</td>
  </tr>
  <tr>
    <td>LATIN</td>
    <td>0.8366</td>
    <td>+0.6038</td>
    <td>0.8311</td>
    <td>+0.6167</td>
  </tr>
  <tr>
    <td rowspan="2">Azure ChatGPT (Completion)</td>
    <td>Plain</td>
    <td>0.6866</td>
    <td>-</td>
    <td>0.6795</td>
    <td>-</td>
  </tr>
  <tr>
    <td>LATIN</td>
    <td>0.8255</td>
    <td>+0.1389</td>
    <td>0.8135</td>
    <td>+0.1340</td>
  </tr>
  <tr>
    <td rowspan="2">Azure ChatGPT (ChatCompletion)</td>
    <td>Plain</td>
    <td>TODO</td>
    <td>-</td>
    <td>TODO</td>
    <td>-</td>
  </tr>
  <tr>
    <td>LATIN</td>
    <td>TODO</td>
    <td>TODO</td>
    <td>0.5954 (Official OCR)</td>
    <td>TODO</td>
  </tr>
  <tr>
    <td rowspan="2">Alpaca (7B)</td>
    <td>Plain</td>
    <td>0.3567</td>
    <td>-</td>
    <td>0.3506</td>
    <td>-</td>
  </tr>
  <tr>
    <td>LATIN</td>
    <td>0.4200</td>
    <td>+0.0633</td>
    <td>0.4304</td>
    <td>+0.0798 </td>
  </tr>
  <tr>
    <td rowspan="2">Vicuna (13B)</td>
    <td>Plain</td>
    <td>DOING</td>
    <td>-</td>
    <td>DOING</td>
    <td>-</td>
  </tr>
  <tr>
    <td>LATIN</td>
    <td>0.4725 </td>
    <td>DOING</td>
    <td>0.4597 </td>
    <td>DOING</td>
  </tr>
</tbody>
</table>

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
