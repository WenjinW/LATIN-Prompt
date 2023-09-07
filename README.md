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
- **2023.09.07**: We support the Llama2 and Llama2-chat models.
- **2023.09.06**: We introduce LATIN-Tuning for Alpaca, which enhances its zero-shot performance on DocVQA from 0.3567 to 0.6697.
- **2023.06.30**: We now provide implementations based on Azure OpenAI text-davinci-003 Completion. 
- **2023.06.29**: We now provide implementations based on [Alpaca-7B](https://github.com/tatsu-lab/stanford_alpaca) and [Vicuna-13B](https://github.com/vllm-project/vllm). 

## Roadmap
- [x] DUE OCR results
- [x] Alpaca 7B
- [x] Vicuna 13B
- [x] Azure OpenAI gpt-3.5-turbo + Completion
- [ ] Azure OpenAI gpt-3.5-turbo + ChatCompletion (doing)
- [x] Azure OpenAI text-davinci-003 + Completion
- [ ] MPT-30B-Chat (todo)
- [ ] Orca (todo)
- [ ] GPT-4 and offical OpenAI API (todo, we are working hard to seek access to official OpenAI API)
- [x] LLaMA2-chat

## Preparation
#### Prepare the environment
```bash
pip install -r requirements.txt
```

#### Set environment variable for Cluade and OpenAI
Set the `ANTHROPIC_API_KEY` for [Cluade](https://docs.anthropic.com/claude/docs). Please refer to the script `./utils/claude.py`.
```bash
export ANTHROPIC_API_KEY="Your API key"
```

Set the `OPENAI_API_KEY` and `OPENAI_API_BASE` for [Azure OpenAI](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/overview). Please refer to the script `./utils/openai_api.py`. For the differences between Azure OpenAI and OpenAI, see [here](https://learn.microsoft.com/en-us/azure/cognitive-services/openai/overview#comparing-azure-openai-and-openai).
```bash
export OPENAI_API_KEY="Your API key"
export OPENAI_API_BASE="Your base url"
```

**Note**: Currently, due to resource constraints, our experiments are all based on the Azure OpenAI API. At the same time, we are working hard to seek access to official OpenAI API. If you can provide relevant resources, please contact the author and we will be very grateful.

#### Set environment variable for data directory
```bash
export DATAS_DIR="Your data directory"
```

#### Prepare the dataset
- Download DocVQA dataset with Azure OCR results from [DUE Benchmark](https://github.com/due-benchmark) and put it into the `DATAS_DIR`.
- Download DocVQA dataset with Official OCR results from [Robust Reading Competition](https://rrc.cvc.uab.es/?com=introduction) and put it into the `DATAS_DIR`.

#### Set the path of model checkpoints
Refer to the script `./utils/model_path_config.py`.

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

### Examples with Azure OpenAI GPT-3.5-turbo (ChatGPT) Completion
#### Example: GPT-3.5-turbo + LATIN-Prompt on DocVQA (Azure OCR)
```bash
bash script/claude_eval.sh 0 gpt-35 docvqa_due_azure task_instruction_space
```

#### Example: GPT-3.5-turbo + Plain Prompt on DocVQA (Azure OCR)
```bash
bash script/claude_eval.sh 0 gpt-35 docvqa_due_azure plain
```

### Examples with Azure OpenAI GPT-3.5-turbo (ChatGPT) ChatCompletion
#### Example: GPT-3.5-turbo + LATIN-Prompt on DocVQA
```bash
bash script/claude_eval.sh 0 gpt-35-chat docvqa task_instruction_space
```

### Examples with Azure OpenAI text-davinci-003 Completion
#### Example: text-davinci-003 + LATIN-Prompt on DocVQA (Azure OCR)
```bash
bash script/claude_eval.sh 0 text-davinci-003 docvqa_due_azure task_instruction_space
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

### Examples with LLaMA2-chat
#### Example: LLaMA2-chat + LATIN-Prompt on DocVQA (Azure OCR)
```bash
bash script/vllm_eval.sh 0 llama2-13b-chat docvqa_due_azure task_instruction_space
```

## Performance
### DocVQA (Azure OCR, DUE)
The performance in this table is based on the Azure OCR results provided in [DUE Benchmark](https://github.com/due-benchmark) by default.
The Official OCR represents the performance is based on the OCR results provided in [Robust Reading Competition](https://rrc.cvc.uab.es/?com=introduction)


<style type="text/css">
.tg  {border-collapse:collapse;border-spacing:0;}
.tg td{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  overflow:hidden;padding:10px 5px;word-break:normal;}
.tg th{border-color:black;border-style:solid;border-width:1px;font-family:Arial, sans-serif;font-size:14px;
  font-weight:normal;overflow:hidden;padding:10px 5px;word-break:normal;}
.tg .tg-lboi{border-color:inherit;text-align:left;vertical-align:middle}
.tg .tg-9wq8{border-color:inherit;text-align:center;vertical-align:middle}
.tg .tg-c3ow{border-color:inherit;text-align:center;vertical-align:top}
</style>
<table class="tg">
<thead>
  <tr>
    <th class="tg-9wq8" rowspan="2">Model</th>
    <th class="tg-9wq8" rowspan="2">Prompt</th>
    <th class="tg-c3ow" colspan="2">Test Data</th>
    <th class="tg-c3ow" colspan="2">Val Data</th>
  </tr>
  <tr>
    <th class="tg-c3ow">ANLS</th>
    <th class="tg-c3ow">⬆</th>
    <th class="tg-c3ow">ANLS</th>
    <th class="tg-c3ow">⬆</th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-lboi" rowspan="2">Claude</td>
    <td class="tg-c3ow">Plain</td>
    <td class="tg-c3ow">0.2298</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">0.2144</td>
    <td class="tg-c3ow">-</td>
  </tr>
  <tr>
    <td class="tg-c3ow">LATIN</td>
    <td class="tg-c3ow">0.8366</td>
    <td class="tg-c3ow">+0.6038</td>
    <td class="tg-c3ow">0.8311</td>
    <td class="tg-c3ow">+0.6167</td>
  </tr>
  <tr>
    <td class="tg-lboi" rowspan="2">Azure OpenAI ChatGPT (Completion)</td>
    <td class="tg-c3ow">Plain</td>
    <td class="tg-c3ow">0.6866</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">0.6795</td>
    <td class="tg-c3ow">-</td>
  </tr>
  <tr>
    <td class="tg-c3ow">LATIN</td>
    <td class="tg-c3ow">0.8255</td>
    <td class="tg-c3ow">+0.1389</td>
    <td class="tg-c3ow">0.8135</td>
    <td class="tg-c3ow">+0.1340</td>
  </tr>
  <tr>
    <td class="tg-lboi" rowspan="2">Azure OpenAI ChatGPT (ChatCompletion)</td>
    <td class="tg-c3ow">Plain</td>
    <td class="tg-c3ow">TODO</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">TODO</td>
    <td class="tg-c3ow">-</td>
  </tr>
  <tr>
    <td class="tg-c3ow">LATIN</td>
    <td class="tg-c3ow">TODO</td>
    <td class="tg-c3ow">TODO</td>
    <td class="tg-c3ow">0.5954 (Official OCR)</td>
    <td class="tg-c3ow">TODO</td>
  </tr>
  <tr>
    <td class="tg-lboi">Azure OpenAI text-davinci-003 (Completion)</td>
    <td class="tg-c3ow">LATIN</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">0.8188</td>
    <td class="tg-c3ow">-</td>
  </tr>
  <tr>
    <td class="tg-lboi" rowspan="3">Alpaca (7B)</td>
    <td class="tg-c3ow">Plain</td>
    <td class="tg-c3ow">0.3567</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">0.3506</td>
    <td class="tg-c3ow">-</td>
  </tr>
  <tr>
    <td class="tg-c3ow">LATIN</td>
    <td class="tg-c3ow">0.4200</td>
    <td class="tg-c3ow">+0.0633</td>
    <td class="tg-c3ow">0.4304</td>
    <td class="tg-c3ow">+0.0798</td>
  </tr>
  <tr>
    <td class="tg-c3ow">LATIN-Tuning + LATIN-Prompt</td>
    <td class="tg-c3ow">0.6697</td>
    <td class="tg-c3ow">+0.3130</td>
    <td class="tg-c3ow">0.6668</td>
    <td class="tg-c3ow">+0.3162</td>
  </tr>
  <tr>
    <td class="tg-lboi" rowspan="2">Vicuna (13B)</td>
    <td class="tg-c3ow">Plain</td>
    <td class="tg-c3ow">0.0710</td>
    <td class="tg-c3ow">-</td>
    <td class="tg-c3ow">0.0688</td>
    <td class="tg-c3ow">-</td>
  </tr>
  <tr>
    <td class="tg-c3ow">LATIN</td>
    <td class="tg-c3ow">0.4725</td>
    <td class="tg-c3ow">+0.4015</td>
    <td class="tg-c3ow">0.4597</td>
    <td class="tg-c3ow">+0.3909</td>
  </tr>
  <tr>
    <td class="tg-lboi" rowspan="2">Llama2-13b-chat</td>
    <td class="tg-9wq8">Plain</td>
    <td class="tg-9wq8">0.1783</td>
    <td class="tg-9wq8">-</td>
    <td class="tg-9wq8">0.1863</td>
    <td class="tg-9wq8">-</td>
  </tr>
  <tr>
    <td class="tg-9wq8">LATIN</td>
    <td class="tg-9wq8">0.4283</td>
    <td class="tg-9wq8">+0.2500</td>
    <td class="tg-9wq8">0.4435</td>
    <td class="tg-9wq8">+0.2572</td>
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
