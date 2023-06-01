import json
import os
from pathlib import Path

import datasets

from PIL import Image

logger = datasets.logging.get_logger(__name__)


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")
    w, h = image.size
    return image, (w, h)


def normalize_bbox(bbox, size):
    return [
        int(1000 * bbox[0] / size[0]),
        int(1000 * bbox[1] / size[1]),
        int(1000 * bbox[2] / size[0]),
        int(1000 * bbox[3] / size[1]),
    ]


def quad_to_box(quad):
    box = (
        max(0, quad["x1"]),
        max(0, quad["y1"]),
        quad["x3"],
        quad["y3"]
    )
    if box[3] < box[1]:
        bbox = list(box)
        tmp = bbox[3]
        bbox[3] = bbox[1]
        bbox[1] = tmp
        box = tuple(bbox)
    if box[2] < box[0]:
        bbox = list(box)
        tmp = bbox[2]
        bbox[2] = bbox[0]
        bbox[0] = tmp
        box = tuple(bbox)
    return box


def box4point_to_box2point(box4point):
    # bounding box = [x0, y0, x1, y1, x2, y2, x3, y3]
    all_x = [box4point[2 * i] for i in range(4)]
    all_y = [box4point[2 * i + 1] for i in range(4)]
    box2point = [min(all_x), min(all_y), max(all_x), max(all_y)]
    return box2point


class DocVQAConfig(datasets.BuilderConfig):
    """BuilderConfig for DocVQA"""
    def __init__(self, **kwargs):
        """BuilderConfig for DocVQA.
        Args:
          **kwargs: keyword arguments forwarded to super.
        """
        super(DocVQAConfig, self).__init__(**kwargs)


class DocVQA(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        DocVQAConfig(name="docvqa", version=datasets.Version("1.0.0"), description="DocVQA dataset"),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features(
                {
                    "questionId": datasets.Value("string"),
                    "question": datasets.Value("string"),
                    "answers": datasets.Sequence(datasets.Value("string")),
                    "words": datasets.Sequence(datasets.Value("string")),
                    "word_boxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "texts": datasets.Sequence(datasets.Value("string")),
                    "text_boxes": datasets.Sequence(datasets.Sequence(datasets.Value("int64"))),
                    "image": datasets.features.Image(),
                }
            ),
            supervised_keys=None,
            homepage="xxx",
        )

    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""
        """Uses local files located with data_dir"""
        dest = Path("./datas/docvqa")
        return [
            datasets.SplitGenerator(
                name=datasets.Split.TRAIN, gen_kwargs={
                    "dirpath": dest/"train",
                    "filepath": dest/"train/train_v1.0.json",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.VALIDATION, gen_kwargs={
                    "dirpath": dest/"val",
                    "filepath": dest/"val/val_v1.0.json",
                }
            ),
            datasets.SplitGenerator(
                name=datasets.Split.TEST, gen_kwargs={
                    "dirpath": dest/"test",
                    "filepath": dest/"test/test_v1.0.json",
                }
            ),
        ]

    def get_line_bbox(self, bboxs):
        x = [bboxs[i][j] for i in range(len(bboxs)) for j in range(0, len(bboxs[i]), 2)]
        y = [bboxs[i][j] for i in range(len(bboxs)) for j in range(1, len(bboxs[i]), 2)]

        x0, y0, x1, y1 = min(x), min(y), max(x), max(y)

        assert x1 >= x0 and y1 >= y0
        bbox = [[x0, y0, x1, y1] for _ in range(len(bboxs))]
        return bbox

    def _generate_examples(self, dirpath, filepath):
        logger.info("⏳ Generating examples from = %s", filepath)
        ann_dir = os.path.join(dirpath, "ocr_results") 
        img_dir = os.path.join(dirpath, "documents")

        with open(filepath, 'r') as f:
            data = json.load(f)
        is_test = True if data["dataset_split"] == "test" else False
        
        for i, example in enumerate(data["data"][:10]):
        # for i, example in enumerate(data["data"]):
            name = example["ucsf_document_id"] + "_" + example["ucsf_document_page_no"]
            img_path = os.path.join(img_dir, name+".png")
            ann_path = os.path.join(ann_dir, name+".json")
            question = example["question"]
            question_id = example["questionId"]
            answers = example["answers"] if not is_test else None
            image, size = load_image(img_path)
            with open(ann_path, 'r') as f:
                ann_data = json.load(f)["recognitionResults"][0]["lines"]
            words = []
            word_boxes = []
            texts = []
            text_boxes = []
            for line in ann_data:
                texts.append(line["text"])
                text_boxes.append(box4point_to_box2point(line["boundingBox"]))
                for word in line["words"]:
                    words.append(word["text"])
                    word_boxes.append(box4point_to_box2point(word["boundingBox"]))

            item = {
                "questionId": question_id,
                "question": question,
                "answers": answers,
                "words": words,
                "word_boxes": word_boxes,
                "texts": texts,
                "text_boxes": text_boxes,
                "image": image,
                }

            yield i, item

if __name__ == "__main__":
    dataset = datasets.load_dataset(
        os.path.abspath(__file__),
    )