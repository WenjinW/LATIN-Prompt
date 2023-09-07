import sys
sys.path.append('./')
import os
from dataclasses import dataclass, field
from tqdm import tqdm
import math

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
from utils import space_layout



PROMPT_DICT = {
    "prompt_task": (
        "You are asked to answer a question asked on a document image.\n"
        "The answer for a question in this can be any of the following types:\n"
        "1. Answer is a piece contiguous text from the document.\n"
        '2. Answer is a list of "items" , where each item is a piece of text from the document (multiple spans). '
        "In such cases your model/method is expected to output an answer where each item is separated by a comma and a space. "
        'For example if the question is "What are the three common symptoms of COVID-19?" Answer must be in the format "fever, dry cough, tiredness". '
        'In such cases "and" should not be used to connect last item and the penultimate item and a space after the comma is required so that your answer match exactly with the ground truth.\n'
        "3. Answer is a contiguous piece of text from the question itself (a span from the question)\n"
        '4. Answer is a number ( for example "2", "2.5", "2%", " 2/3" etc..). '
        'For example there are questions asking for count of something or cases where answer is sum of two values given in the image.\n'
        "Document:\n{document}\n\n"
        "Question: {question}\n\n"
        "Directly answer the question from the document with as few words as possible .\n\n"
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
    two_stage: bool = field(
        default=False,
        metadata={"help": "Whether to use two-stage inference."}
    )

    def __post_init__(self):
        self.datas_dir = os.path.expanduser(self.datas_dir)


class DataCollatorForDocVQA(DataCollatorMixin):
    def __init__(self, prompt_type, two_stage=False):
        super().__init__()
        self.prompt_type = prompt_type
        self.two_stage = two_stage

    def space_layout(self, texts, boxes):

        return space_layout.space_layout(texts, boxes)

    def __call__(self, example):
        new_example = {
            "question": example["question"],
            "questionId": example["questionId"],
            "answers": example["answers"],
            "image": example["image"],
        }

        question = example["question"]

        if self.prompt_type == "plain":
            doc = " ".join(example["texts"])
            text = PROMPT_DICT["prompt_plain"].format_map({
                "document": doc,
                "question": question
            })
            new_example["text_boxes"] = example["text_boxes"]
            new_example["texts"] = text
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
            new_example["text_boxes"] = example["text_boxes"]
            new_example["texts"] = text
        elif self.prompt_type == "task_instruction":
            doc = " ".join(example["texts"])
            text = PROMPT_DICT["prompt_task"].format_map({
                "document": doc,
                "question": question
            })
            new_example["text_boxes"] = example["text_boxes"]
            new_example["texts"] = text
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
            new_example["text_boxes"] = example["text_boxes"]
            new_example["texts"] = text
        else:
            raise ValueError("Invalid prompt type.")
            
        return new_example


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

    if custom_args.dataset_name == "infographicvqa":
        data = datasets.load_dataset("utils/infographicVQA.py")

    if custom_args.model_name_or_path == "claude":
        from utils import claude
    elif custom_args.model_name_or_path == "gpt-35":
        from utils import openai_api

    anls_metric = ANLS(
        result_dir=custom_args.results_dir,
        exp_name=training_args.run_name,
        dataset_name=custom_args.dataset_name
    )

    collate_fn = DataCollatorForDocVQA(
        prompt_type=custom_args.prompt,
        two_stage=custom_args.two_stage,
    )

    all_preds = []
    all_answers = []
    all_questions = []
    all_question_ids = []
    save_interval = 100

    count = anls_metric.load_and_count("val")
    print(f"Begin from the {count+1}-th example.")
    for i in tqdm(range(count, len(data["validation"])), desc='Processing'):
        example = data["validation"][i]
        print("="*30)
        example = collate_fn(example)
        print(example["texts"])

        if custom_args.model_name_or_path == "claude":
            response = claude.cluade_completion(example["texts"])["completion"].strip()
        elif custom_args.model_name_or_path == "gpt-35":
            response = openai_api.openai_completion(example["texts"]).strip()
        # response = claude.cluade_completion(example["texts"])
        
        # delete the space at the beginning and end
        # generated_text = response["completion"].strip()
        generated_text = response
        print(example["answers"])
        print("Outputs:")
        print(generated_text)
        
        all_preds.append(generated_text)
        all_answers.append(example["answers"])
        all_questions.append(example["question"])
        all_question_ids.append(example["questionId"])

        if (i + 1) % save_interval == 0:
            val_anls = anls_metric.load_and_save(
                qids=all_question_ids,
                questions=all_questions,
                predictions=all_preds,
                references=all_answers,
                split="val"
            )
            wandb.log({"val_anls": val_anls})
            print({"val_anls": val_anls})
            all_preds = []
            all_answers = []
            all_questions = []
            all_question_ids = []
    
    val_anls = anls_metric.load_and_save(
        qids=all_question_ids,
        questions=all_questions,
        predictions=all_preds,
        references=all_answers,
        split="val"
    )

    wandb.log({"val_anls": val_anls})
    print({"val_anls": val_anls})

    # evaluate on the test dataset
    count = anls_metric.load_and_count("test")
    print(f"Begin from the {count+1}-th example.")
    all_preds = []
    all_question_ids = []
    all_questions = []
    for i in tqdm(range(count, len(data["test"])), desc='Processing'):
        example = data["test"][i]
        print("="*30)
        example = collate_fn(example)
        print(example["texts"])

        # response = claude.cluade_completion(example["texts"])
        if custom_args.model_name_or_path == "claude":
            response = claude.cluade_completion(example["texts"])["completion"].strip()
        elif custom_args.model_name_or_path == "gpt-35":
            response = openai_api.openai_completion(example["texts"]).strip()
        
        # delete the space at the beginning and end
        # generated_text = response["completion"].strip()
        generated_text = response
        print(example["answers"])
        print("Outputs:")
        print(generated_text)
        
        all_preds.append(generated_text)
        all_questions.append(example["question"])
        all_question_ids.append(example["questionId"])

        if (i + 1) % save_interval == 0:
            test_anls = anls_metric.load_and_save(
                qids=all_question_ids,
                questions=all_questions,
                predictions=all_preds,
                split="test"
            )
            print({"test_anls": test_anls})
            all_preds = []
            all_question_ids = []
            all_questions = []


    test_anls = anls_metric.load_and_save(
        qids=all_question_ids,
        questions=all_questions,
        predictions=all_preds,
        split="test"
    )
    print({"test_anls": test_anls})


if __name__ == "__main__":
    main()