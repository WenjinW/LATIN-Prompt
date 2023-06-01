import os
import json
import sys
sys.path.append(".")
from utils.util import is_same_line, union_box, boxes_sort


def space_layout(texts, boxes):
    line_boxes = []
    line_texts = []
    max_line_char_num = 0
    line_width = 0
    # print(f"len_boxes: {len(boxes)}")
    while len(boxes) > 0:
        line_box = [boxes.pop(0)]
        line_text = [texts.pop(0)]
        char_num = len(line_text[-1])
        line_union_box = line_box[-1]
        while len(boxes) > 0 and is_same_line(line_box[-1], boxes[0]):
            line_box.append(boxes.pop(0))
            line_text.append(texts.pop(0))
            char_num += len(line_text[-1])
            line_union_box = union_box(line_union_box, line_box[-1])
        line_boxes.append(line_box)
        line_texts.append(line_text)
        if char_num >= max_line_char_num:
            max_line_char_num = char_num
            line_width = line_union_box[2] - line_union_box[0]
    
    # print(line_width)

    char_width = line_width / max_line_char_num
    print(char_width)
    if char_width == 0:
        char_width = 1

    space_line_texts = []
    for i, line_box in enumerate(line_boxes):
        space_line_text = ""
        for j, box in enumerate(line_box):
            left_char_num = int(box[0] / char_width)
            space_line_text += " " * (left_char_num - len(space_line_text))
            space_line_text += line_texts[i][j]
        space_line_texts.append(space_line_text)

    return space_line_texts


if __name__ == "__main__":

    filepath = "/home/xxx/workspace/VrDU/datas/funsd/testing_data/annotations/82092117.json"
    with open(filepath, "r") as f:
        data = json.load(f)
    texts = []
    text_boxes = []

    for i, item in enumerate(data["form"]):
        texts.append(item["text"])
        # texts.append("{" + f'{i}-{item["text"]}' + "}")
        text_boxes.append(item["box"])
    ids = boxes_sort(text_boxes)
    texts = ["{" + f'{count}-{texts[i]}' + "}" for count, i in enumerate(ids)]
    text_boxes = [text_boxes[i] for i in ids]
    space_line_texts = space_layout(texts=texts, boxes=text_boxes)
    with open("82092117.txt", "w") as f:
        f.write("\n".join(space_line_texts))