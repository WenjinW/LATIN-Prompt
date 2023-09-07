import sys
sys.path.append('./')
import os
from dataclasses import dataclass, field
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, LlamaForCausalLM
from transformers import (
    set_seed,
    TrainingArguments,
    HfArgumentParser,
)
from transformers.data.data_collator import DataCollatorMixin
import datasets
import wandb

from metric.anls import ANLS
from utils.model_path_config import model_path_config

from utils.util import is_same_line, union_box, is_adj_line
from utils import space_layout


PROMPT_DICT = {
    "prompt_task": (
        "You are asked to answer questions asked on a document image.\n"
        "The answers to questions are short text spans taken verbatim from the document. "
        "This means that the answers comprise a set of contiguous text tokens present in the document.\n"
        "Document:\n{document}\n\n"
        "Question: {question}\n\n"
        "Directly extract the answer of the question from the document with as few words as possible .\n\n"
        "Answer:"
    ),
    "prompt_plain": (
        "Document:\n{document}\n\n"
        "Question: {question}\n\n"
        "Directly extract the answer of the question from the document.\n\n"
        "Answer:"
    ),
}


@dataclass
class CustomArguments:
    model_name_or_path: str = field(
        default="llama-7b",
        metadata={"help": "Path to pretrained model or model identifier\
                  from huggingface.co/models"}
    )
    dataset_name: str = field(
        default="docvqa",
        metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    results_dir: str = field(
        default="results",
        metadata={"help": "The directory to save the results."}
    )
    datas_dir: str = field(
        default="",
        metadata={"help": "The directory to save the datas."}
    )
    wandb_project: str = field(
        default="Layout",
        metadata={"help": "The name of the wandb project."}
    )
    prompt: str = field(
        default="plain",
        metadata={"help": "The prompt type. (plain, alpaca, layout)"}
    )

    def __post_init__(self):
        self.model_name_or_path = model_path_config[self.model_name_or_path]
        self.datas_dir = os.path.expanduser(self.datas_dir)


class DataCollatorForDocVQA(DataCollatorMixin):
    def __init__(self, prompt_type, two_stage=False):
        super().__init__()
        self.prompt_type = prompt_type
        self.two_stage = two_stage

    def space_layout(self, texts, boxes):

        return space_layout.space_layout(texts, boxes)

    def __call__(self, features):
        batch = {
            "text": [],
            "image": [],
            "question": [],
            "answers": [],
            "questionId": [],
            "word_boxes": [],
        }
        for example in features:
            question = example["question"]
            batch["question"].append(question)

            if self.prompt_type == "plain":
                doc = " ".join(example["texts"])
                text = PROMPT_DICT["prompt_plain"].format_map({
                    "document": doc,
                    "question": question
                })
            elif self.prompt_type == "task_instruction_space":
                space_line_texts = self.space_layout(
                    example["texts"],
                    example["text_boxes"],
                )
                doc = "\n".join(space_line_texts)
                text = PROMPT_DICT["prompt_task"].format_map({
                    "document": doc,
                    "question": question
                })
            elif self.prompt_type == "task_instruction":
                doc = " ".join(example["texts"])
                text = PROMPT_DICT["prompt_task"].format_map({
                    "document": doc,
                    "question": question
                })
            elif self.prompt_type == "space":
                space_line_texts = self.space_layout(
                    example["texts"],
                    example["text_boxes"],
                )
                doc = "\n".join(space_line_texts)
                text = PROMPT_DICT["prompt_plain"].format_map({
                    "document": doc,
                    "question": question
                })
            else:
                raise ValueError("Invalid prompt type.")
            
            batch["text"].append(text)
            batch["image"].append(example["image"])
            batch["answers"].append(example["answers"])
            batch["questionId"].append(example["questionId"])
            batch["word_boxes"].append(example["text_boxes"])
        
        return batch


def main():
    parser = HfArgumentParser((CustomArguments, TrainingArguments))
    custom_args, training_args = parser.parse_args_into_dataclasses()
    for k, v in custom_args.__dict__.items():
        print(k, v)

    logged_config = {
        "dataset_name": custom_args.dataset_name,
    }

    wandb.init(
        project=custom_args.wandb_project,
        name=training_args.run_name,
        config=logged_config,
    )

    set_seed(training_args.seed)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    processor = AutoTokenizer.from_pretrained(custom_args.model_name_or_path)
    model = LlamaForCausalLM.from_pretrained(
        custom_args.model_name_or_path, torch_dtype=torch.float16,
    )
    model.to(device)
    # model = LlamaForCausalLM.from_pretrained(
    #     custom_args.model_name_or_path, device_map="auto", load_in_8bit=True,
    # )

    if custom_args.dataset_name == "docvqa":
        data = datasets.load_dataset("utils/docvqa.py")

    anls_metric = ANLS(
        result_dir=custom_args.results_dir,
        exp_name=training_args.run_name,
        dataset_name=custom_args.dataset_name
    )

    collate_fn = DataCollatorForDocVQA(
        prompt_type=custom_args.prompt,
    )

    # evaluate on the validation dataset
    all_preds = []
    all_answers = []
    all_questions = []
    all_question_ids = []

    count = 0
    print(f"Begin from the {count+1}-th example.")
    for i in tqdm(range(count, len(data["validation"])), desc='Processing'):
        example = data["validation"][i]
        print("="*30)
        batch = collate_fn([example])
        print(batch["text"][0])

        inputs = processor(
            text=batch["text"],
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        print(batch["answers"][0])
        generated_ids = model.generate(inputs.input_ids, max_new_tokens=100)
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        generated_text = [
            t.strip()[len(batch["text"][i]):] for i, t in enumerate(generated_text)
        ]
        
        print("Outputs:")
        print(generated_text)
        
        all_preds.extend(generated_text)
        all_answers.extend(batch["answers"])
        all_questions.extend(batch["question"])
        all_question_ids.extend(batch["questionId"])
    
    val_anls = anls_metric.compute_and_save(
        qids=all_question_ids,
        questions=all_questions,
        predictions=all_preds,
        references=all_answers,
        split="val"
    )

    wandb.log({"val_anls": val_anls})
    print({"val_anls": val_anls})

    # evaluate on the test dataset
    all_preds = []
    all_questions = []
    all_question_ids = []
    count = 0
    for i in tqdm(range(count, len(data["test"])), desc='Processing'):
        example = data["test"][i]
        print("="*30)
        batch = collate_fn([example])
        print(batch["text"][0])

        inputs = processor(
            text=batch["text"],
            return_tensors="pt",
        )
        inputs = inputs.to(device)
        print(batch["answers"][0])
        generated_ids = model.generate(inputs.input_ids, max_new_tokens=100)
        generated_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        
        # generated_text = [t.strip() for t in generated_text]
        generated_text = [
            t.strip()[len(batch["text"][i]):] for i, t in enumerate(generated_text)
        ]

        print("Outputs:")
        print(generated_text)
        
        all_preds.extend(generated_text)
        all_questions.extend(batch["question"])
        all_question_ids.extend(batch["questionId"])

    test_anls = anls_metric.compute_and_save(
        qids=all_question_ids,
        questions=all_questions,
        predictions=all_preds,
        split="test"
    )
    print({"test_anls": test_anls})


if __name__ == "__main__":
    main()