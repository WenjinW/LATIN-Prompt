import sys
sys.path.append('./')
import os
from dataclasses import dataclass, field
from tqdm import tqdm

from transformers import (
    set_seed,
    TrainingArguments,
    HfArgumentParser,
)
from transformers.data.data_collator import DataCollatorMixin
import datasets
import wandb
import openai

from metric.anls import ANLS
from utils import claude, space_layout, openai_api


PROMPT_DICT = {
    "prompt_input": (
        "Below is an instruction that describes a task, paired with an input that provides further context. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Input:\n{input}\n\n### Response:"
    ),
    "prompt_no_input": (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request.\n\n"
        "### Instruction:\n{instruction}\n\n### Response:"
    ),
    "prompt_input_demonstration": (
        "You are asked to extract the answer of a question from the given document. "
        "The extraction means you must answer the question directly using the original words from the document.\n"
        "Here is an example of extracting the answer from a document:\n"
        "### Example\n"    
        "Document: In meteorology, precipitation is any product of the condensation of atmospheric water vapor that falls under gravity.\n"
        "The main forms of precipitation include drizzle, rain, sleet, snow, graupel and hail...\n"
        "Precipitation forms as smaller droplets coalesce via collision with other rain drops or ice crystals within a cloud.\n"
        "Short, intense periods of rain in scattered locations are called “showers”.\n\n"
        "Question: Where do water droplets collide with ice crystals to form precipitation?\n\n"
        "Directly extract the answer of the question from the document.\n\n"
        "Answer:\n"
        "within a cloud\n"
        "### Example End\n\n"
        "Now, please finish the next task.\n"
        "Document: {document}\n\n"
        "Question: {question}\n\n"
        "Directly extract the answer of the question from the document.\n\n"
        "Answer:"
    ),
    "prompt_input_demonstration_short": (
        "You are asked to extract the answer of a question from the given document. "
        "The extraction means you must answer the question directly using the original words from the document.\n"
        "Here is an example of extracting the answer from a document:\n"
        "### Example\n"    
        "Document: In meteorology, precipitation is any product of the condensation of atmospheric water vapor that falls under gravity.\n"
        "The main forms of precipitation include drizzle, rain, sleet, snow, graupel and hail...\n"
        "Precipitation forms as smaller droplets coalesce via collision with other rain drops or ice crystals within a cloud.\n"
        "Short, intense periods of rain in scattered locations are called “showers”.\n\n"
        "Question: Where do water droplets collide with ice crystals to form precipitation?\n\n"
        "Directly extract the answer of the question from the document with as few words as possible .\n\n"
        "Answer:\n"
        "within a cloud\n"
        "### Example End\n\n"
        "Now, please finish the next task.\n"
        "Document: {document}\n\n"
        "Question: {question}\n\n"
        "Directly extract the answer of the question from the document with as few words as possible .\n\n"
        "Answer:"
    ),
    "prompt_task": (
        "You are asked to answer questions asked on a document image.\n"
        "The answers to questions are short text spans taken verbatim from the document. "
        "This means that the answers comprise a set of contiguous text tokens present in the document.\n"
        # "Document: {document}\n\n"
        "Document:\n{document}\n\n"
        "Question: {question}\n\n"
        "Directly extract the answer of the question from the document with as few words as possible.\n\n"
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
    split_name: str = field(
        default="val_test",
        metadata={"help": "The split name. (val, test)"}
    )


    def __post_init__(self):
        self.datas_dir = os.path.expanduser(self.datas_dir)


class DataCollatorForDocVQA(DataCollatorMixin):
    def __init__(self, prompt_type):
        super().__init__()
        self.prompt_type = prompt_type

    def space_layout(self, texts, boxes):
        # ids = boxes_sort(boxes)
        # new_texts = [texts[i] for i in ids]
        # new_boxes = [boxes[i] for i in ids]
        # return space_layout.space_layout(new_texts, new_boxes)

        # note: do not sort the boxes
        return space_layout.space_layout(texts, boxes)

    def __call__(self, features):
        batch = {
            "text": [],
            "question": [],
            "answers": [],
            "questionId": [],
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
            batch["answers"].append(example["answers"])
            batch["questionId"].append(example["questionId"])

        
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

    if custom_args.dataset_name == "docvqa_due_azure":
        data = datasets.load_dataset("utils/docvqa_due_azure.py")

    anls_metric = ANLS(
        result_dir=custom_args.results_dir,
        exp_name=training_args.run_name,
        dataset_name=custom_args.dataset_name
    )

    collate_fn = DataCollatorForDocVQA(
        prompt_type=custom_args.prompt,
    )

    save_interval = 20

    if "val" in custom_args.split_name:
        print("Process the val dataset.")

        all_preds = []
        all_answers = []
        all_questions = []
        all_question_ids = []
        

        count = anls_metric.load_and_count("val")
        print(f"Begin from the {count+1}-th example.")
        invalid_request_ids_val = []
        for i in tqdm(range(count, len(data["validation"])), desc='Processing'):
            example = data["validation"][i]
            print("="*30)
            batch = collate_fn([example])
            print(batch["text"][0])

            if custom_args.model_name_or_path == "claude":
                response = claude.cluade_completion(batch["text"][0])["completion"].strip()
            elif custom_args.model_name_or_path in ["gpt-35", "text-davinci-003"]:
                try:
                    response = openai_api.openai_completion(batch["text"][0]).strip()
                except openai.error.InvalidRequestError:
                    print("InvalidRequestError!!!")
                    response = ""
                    invalid_request_ids_val.append(i)

            generated_text = [
                response
            ]

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
    
    if "test" in custom_args.split_name:
        print("Process the test dataset.")
        # evaluate on the test dataset
        count = anls_metric.load_and_count("test")
        print(f"Begin from the {count+1}-th example.")
        all_preds = []
        all_question_ids = []
        all_questions = []
        invalid_request_ids_test = []
        for i in tqdm(range(count, len(data["test"])), desc='Processing'):
            example = data["test"][i]
            print("="*30)
            batch = collate_fn([example])
            print(batch["text"][0])

            if custom_args.model_name_or_path == "claude":
                response = claude.cluade_completion(batch["text"][0])["completion"].strip()
            elif custom_args.model_name_or_path == "gpt-35":
                try:
                    response = openai_api.openai_completion(batch["text"][0]).strip()
                except openai.error.InvalidRequestError:
                    print("InvalidRequestError!!!")
                    response = ""
                    invalid_request_ids_test.append(i)
            
            generated_text = [
                response
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
        print("invalid_request_ids_val:", invalid_request_ids_val)
        print("invalid_request_ids_test:", invalid_request_ids_test)


if __name__ == "__main__":
    main()