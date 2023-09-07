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
        "You are asked to answer questions asked on a document.\n"
        "The answers to questions are short text spans taken verbatim from the document. "
        "This means that the answers comprise a set of contiguous text tokens present in the document.\n"
        "Document:\n{document}\n\n"
        "Question: {question}\n\n"
        "Directly extract the answer of the question from the document with as few words as possible .\n\n"
        "You also need to output your confidence in the answer, which must be an integer between 0-100.\n"
        "The output format is as follows, where [] indicates a placeholder and does not need to be actually output:\n"
        "[Confidence score], [Extracted Answer]\n"
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

            if self.prompt_type == "task_instruction_space":
                all_page_texts = []
                for page_id, page_texts in enumerate(example["texts"]):
                    space_line_texts = self.space_layout(
                        page_texts,
                        example["text_boxes"][page_id],
                    )
                    all_page_texts.append("\n".join(space_line_texts))

                text = []
                for page_texts in all_page_texts:
                    text.append(PROMPT_DICT["prompt_task"].format_map({
                        "document": page_texts,
                        "question": question
                    }))
            else:
                raise ValueError("Invalid prompt type.")
            
            batch["text"].append(text)
            batch["answers"].append(example["answers"])
            batch["question"].append(example["question"])
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

    if custom_args.dataset_name == "mpdocvqa":
        data = datasets.load_dataset("utils/mpdocvqa.py")

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
        batch = collate_fn([example])
        max_score = -1
        generated_text = ""
        for page_id, text in enumerate(batch["text"][0]):
            response = claude.cluade_completion(text)
            page_generated_text = response["completion"].strip()
            comma_pos = page_generated_text.find(",")
            score = page_generated_text[:comma_pos].strip()
            if score.isdigit():
                score = float(score)
            else:
                continue
            page_generated_text = page_generated_text[comma_pos+1:].strip()

            print(f"Score: {score}\t Text: {page_generated_text}")
            if score > max_score:
                max_score = score
                generated_text = page_generated_text
            if max_score == 100:
                break
        
        generated_text = [
            generated_text.strip(),  # delete the space at the beginning and end
        ]
        # print(generated_text)
        
        print(batch["answers"][0])
        print("Outputs:")
        print(generated_text)

        all_preds.extend(generated_text)
        all_answers.extend(batch["answers"])
        all_questions.extend(batch["question"])
        all_question_ids.extend(batch["questionId"])

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
        batch = collate_fn([example])
        max_score = -1
        generated_text = ""
        for page_id, text in enumerate(batch["text"][0]):
            response = claude.cluade_completion(text)
            page_generated_text = response["completion"].strip()
            comma_pos = page_generated_text.find(",")
            score = page_generated_text[:comma_pos].strip()
            if score.isdigit():
                score = float(score)
            else:
                continue
            page_generated_text = page_generated_text[comma_pos+1:].strip()

            print(f"Score: {score}\t Text: {page_generated_text}")
            if score > max_score:
                max_score = score
                generated_text = page_generated_text
            if max_score == 100:
                break
        generated_text = [
            generated_text.strip(),  # delete the space at the beginning and end
        ]
        
        print("Outputs:")
        print(generated_text)

        all_preds.extend(generated_text)
        all_questions.extend(batch["question"])
        all_question_ids.extend(batch["questionId"])

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