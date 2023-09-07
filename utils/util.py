import copy
import logging
import random
import requests
import time

import spacy
import datasets
import torch
from faker import Faker

logger = logging.getLogger(__name__)

# from torch_geometric.data import Data


def compute_l2_dist(q, s):
    """compute l2 dist

    Params:
        q: N1xd
        L: N2xd
    """

    q = torch.unsqueeze(q, dim=1) # N1 x 1 x d
    print(f"q shape: {q.shape}")
    s = torch.unsqueeze(s, dim=0) # 1 x N2 x d
    print(f"s shape: {s.shape}")

    all_dist = []
    for i in range(q.shape[0]):
        # print(f"q[i].shape: {q[i].shape}")
        qi = torch.unsqueeze(q[i], dim=1) # 1 x 1 x d
        # print(f"qi.shape: {qi.shape}")
        # print(f"s shape: {s.shape}")
        dist = (qi - s) ** 2 # 1 x 1 x d - 1 x N2 x d
        # print(f"dist shape: {dist.shape}")
        dist = torch.sum(dist, dim=-1) # 1 x 1 x d - 1 x N2 x d
        # print(f"dist shape: {dist.shape}")
        dist = torch.sqrt(dist)
        all_dist.append(dist)

    all_dist = torch.cat(all_dist, dim=0)
    return all_dist


def cos_sim(feat1, feat2):
    """
    Params:
        feat1: P1 x D, the features of current documents
        feat2: P2 x D, the features of candidate reference documents
    """
    feat1 = feat1 / torch.sqrt(torch.sum(feat1 ** 2, dim=-1, keepdim=True))
    feat2 = feat2 / torch.sqrt(torch.sum(feat2 ** 2, dim=-1, keepdim=True))
    
    sim = feat1 @ feat2.T # P1 x P2
    sim = 0.5 * sim + 0.5  # scale [-1, 1] to [0, 1], we use the 0 to mask !!!

    return sim


def document_sim(feat1, feat2, mask1, mask2):
    """
    Params:
        feat1: B1 x L1 x D, the features of current documents
        feat2: B2 x L2 x D, the features of candidate reference documents
        mask1: B1 x L1, the masks of current documents
        mask2: B2 x L2, the masks of candidate reference documents
    """
    feat1 = feat1 / torch.sqrt(torch.sum(feat1 ** 2, dim=-1, keepdim=True))
    feat2 = feat2 / torch.sqrt(torch.sum(feat2 ** 2, dim=-1, keepdim=True))

    mask = torch.einsum("ij,pq->ijqp", mask1, mask2) # B1 x L1 x L2 x B2
    # print(f"mask.shape: {mask.shape}")
    mask = torch.permute(mask, dims=[0, 3, 1, 2]) # B1 x B2 x L1 x L2
    # print(f"mask.shape: {mask.shape}")

    # B1 x L1 x D, B2 x L2 x D -> B1 x L1 x L2 x B2
    # if B1 = 1 and B2 = 1, sim = feat1 @ feat2.T
    sim = torch.einsum("ijl,lqp->ijqp", feat1, feat2.T)  # B1 x L1 x L2 x B2
    sim = 0.5 * sim + 0.5  # scale [-1, 1] to [0, 1], we use the 0 to mask !!!

    sim = torch.permute(sim, dims=[0, 3, 1, 2]) # B1 x B2 x L1 x L2
    sim = sim * mask
    sim, _ = torch.max(sim, dim=-1) # B1 x B2 x L1
    sim = sim * torch.unsqueeze(mask1, dim=1)
    # print(f"sim.shape: {sim.shape}")
    sim = torch.sum(sim, dim=-1) # B1 x B2
    # print(f"sim.shape: {sim.shape}")
    sim = sim / torch.sum(torch.unsqueeze(mask1, dim=1), dim=-1)
    # print(f"sim.shape: {sim.shape}")
    # print(f"sim: {sim}")

    return sim


def token_sim(feat1, feat2, mask1, mask2):
    """
    Params:
        feat1: P1 x L1 x D, the features of current tokens
        feat2: P1 x L2 x D, the features of candidate reference tokens
        mask1: P1 x L1, the masks of current documents
        mask2: P2 x L2, the masks of candidate reference documents
    """
    P1, L1, D = feat1.shape
    P2, L2, D = feat2.shape

    feat1 = feat1 / torch.sqrt(torch.sum(feat1 ** 2, dim=-1, keepdim=True))
    feat2 = feat2 / torch.sqrt(torch.sum(feat2 ** 2, dim=-1, keepdim=True))
    # feat1 = torch.reshape(feat1, [-1, L1, D])  # P1 x L1 x D
    # feat2 = torch.reshape(feat2, [-1, L2, D])  # P2 x L2 x D

    # mask1 = torch.reshape(mask1, [-1, L1])  # P1 x L1
    # mask2 = torch.reshape(mask2, [-1, L2])  # P2 x L2
    print(f"mask1.shape: {mask1.shape}")

    mask = torch.einsum("ij,pq->ijqp", mask1, mask2) # P1 x L1 x L2 x P2

    # print(f"mask.shape: {mask.shape}")
    mask = torch.permute(mask, dims=[0, 3, 1, 2]) # P1 x P2 x L1 x L2
    # print(f"mask.shape: {mask.shape}")

    # P1 x L1 x D, P2 x L2 x D -> P1 x L1 x L2 x P2
    # if P1 = 1 and P2 = 1, sim = feat1 @ feat2.T
    sim = torch.einsum("ijl,lqp->ijqp", feat1, feat2.T)  # P1 x L1 x L2 x P2
    sim = 0.5 * sim + 0.5  # scale [-1, 1] to [0, 1], we use the 0 to mask !!!

    sim = torch.permute(sim, dims=[0, 3, 1, 2]) # P1 x P2 x L1 x L2
    sim = sim * mask
    # obtain the max for each row
    sim, _ = torch.max(sim, dim=-1) # P1 x P2 x L1

    # mean (ignore the unvalid position by mask1)
    sim = sim * torch.unsqueeze(mask1, dim=1)
    # print(f"sim.shape: {sim.shape}")
    sim = torch.sum(sim, dim=-1) # P1 x P2
    # print(f"sim.shape: {sim.shape}")
    sim = sim / torch.sum(torch.unsqueeze(mask1, dim=1), dim=-1)
    # print(f"sim.shape: {sim.shape}")
    # print(f"sim: {sim}")

    return sim


def get_token_layout_masks(rel_2d_pos, labels):
    """
    Params:
        rel 2d pos: B x H x L x L
        labels: B x L
    """
    B, H, L1, L2 = rel_2d_pos.shape
    layout_masks = labels != -100  # B x L
    print(f"layout_masks.shape: {layout_masks.shape}")
    layout_masks = torch.unsqueeze(layout_masks, dim=1) * torch.unsqueeze(layout_masks, dim=2) # BxLxL
    print(f"layout_masks.shape: {layout_masks.shape}")

    rel_2d_pos = torch.permute(rel_2d_pos, [0, 2, 3, 1]) # B x L1 x L2 x H
    print(f"rel_2d_pos.shape: {rel_2d_pos.shape}")

    rel_2d_pos = torch.reshape(rel_2d_pos, [-1, L2, H]) # (B x L1) x L2 x H
    # print(f"rel_2d_pos.shape: {rel_2d_pos.shape}")
    layout_masks = torch.reshape(layout_masks, [-1, L2])  # P2 x L2

    return rel_2d_pos, layout_masks


# select the reference label for each token
def search_reference_label(feat1, feat2, mask1, mask2):
    """
    Params:
        feat1: B x L1 x D, the features of current documents
        feat2: B x L2 x D, the features of candidate reference documents
        mask1: B x L1, the masks of current documents
        mask2: B x L2, the masks of candidate reference documents
    """
    feat1 = feat1 / torch.sqrt(torch.sum(feat1 ** 2, dim=-1, keepdim=True))
    feat2 = feat2 / torch.sqrt(torch.sum(feat2 ** 2, dim=-1, keepdim=True))

    # B x L1 x 1, B x 1 x L2 -> B x L1 x L2
    mask = torch.unsqueeze(mask1, dim=-1) * torch.unsqueeze(mask2, dim=-2)
    # print(f"mask.shape: {mask.shape}")
    
    # B x L1 x D * B x D x L2 -> B x L1 x L2, batched matrix multiply
    sim = feat1 @ torch.transpose(feat2, dim0=1, dim1=2)
    sim = 0.5 * sim + 0.5  # scale [-1, 1] to [0, 1], we use the 0 to mask !!!
    # print(f"sim.shape: {sim.shape}")

    sim = sim * mask  # B x L1 x L2
    
    """
    After mask, the reference token id of the special token is always the zero, which
    corresponds to the CLS of the reference document.
    The reason is that the similarity betweem it and each token is set to 0 
    """
    reference_token_ids = torch.argmax(sim, dim=-1) # B x L1
    # print(f"reference_token_id.shape: {reference_token_ids.shape}")
    # print(f"reference_token_id:\n {reference_token_ids}")

    return reference_token_ids


def conceptnet_get(token="WHERE", num=10):
    r = requests.get(f"https://api.conceptnet.io/c/en/{token.lower()}?offset=0&limit=1000")
    content = r.json()
    candidate_tokens = []
    # print(len(content['edges']))
    for edge in content["edges"]:
        # "RelatedTo"
        if edge["rel"]["label"] in ["Synonym"]:
            candidate_token = edge["end"]["term"].split("/")[-1]
            lan = edge["end"]["language"]
            if lan == "en" and candidate_token != token.lower():
                candidate_tokens.append(candidate_token)
            candidate_token = edge["start"]["term"].split("/")[-1]
            lan = edge["start"]["language"]
            if lan == "en" and candidate_token != token.lower():
                candidate_tokens.append(candidate_token)
            if len(candidate_tokens) >= num:
                break                         

    # print(f"token: {token} {candidate_tokens}")
    return candidate_tokens


def create_aug_data(
    train_dataset,
    text_column_name,
    num=10,
    org_nums=0, 
    id2label=None,
    features=None,
    ):
    new_examples = []
    count = org_nums
    for example in train_dataset:
        time1 = time.time()
        new_examples_tokens = [[] for _ in range(num)]
        new_examples_diff = [0 for _ in range(num)]
        for i, token in enumerate(example[text_column_name]):
            # print(token)
            candidate_tokens = conceptnet_get(token, num=num)
            for j, new_tokens in enumerate(new_examples_tokens):
                if len(candidate_tokens) > 0:
                    new_tokens.append(random.choice(candidate_tokens))
                    new_examples_diff[j] += 1
                else:
                    new_tokens.append(token)

        # logger.info(example[text_column_name])
        for new_tokens in new_examples_tokens:
            new_example = copy.deepcopy(example)
            # print(type(new_example))
            new_example[text_column_name] = new_tokens
            new_example["id"] = str(count)
            new_example["ner_tags"] = [id2label[id] for id in example["ner_tags"]]
            count += 1
            logger.info(new_example)
            new_examples.append(new_example)
            # logger.info(new_tokens)
        # logger.info(new_examples_diff)
        time2 = time.time()
        logger.info(f"Process time: {time2-time1}s")
        # logger.info("="*100)

    def gen(examples):
        for example in examples:
            yield example

    new_datas = datasets.arrow_dataset.Dataset.from_generator(
        gen,
        gen_kwargs={"examples": new_examples},
        features=features,
        )

    return new_datas

"""
CARDINAL, DATE, EVENT, FAC, GPE, LANGUAGE, LAW,
LOC, MONEY, NORP, ORDINAL, ORG, PERCENT, PERSON, PRODUCT, QUANTITY, TIME, WORK_OF_ART
"""

def create_new_ent(ent_tokens, ent_label, ent_box, fake: Faker):
    if ent_label == "DATE": 
        new_tokens = str(fake.date()).split(" ")
    elif ent_label == "TIME": 
        new_tokens = str(fake.time()).split(" ")
    elif ent_label == "CARDINAL": 
        new_tokens = str(random.randint(0, 1000)).split(" ")
    elif ent_label == "QUANTITY": 
        new_tokens = str(random.randint(0, 1000)).split(" ")
    elif ent_label == "FAC": 
        new_tokens = str(fake.phone_number()).split(" ")
    elif ent_label == "PERSON": 
        new_tokens = str(fake.name()).split(" ")
    elif ent_label == "GPE": 
        new_tokens = str(fake.city()).split(" ")
    elif ent_label == "ORG": 
        new_tokens = str(fake.company()).split(" ")
    else:
        new_tokens = ent_tokens

    new_boxes = [[p for p in ent_box] for _ in new_tokens]
    

    # print(f"ent_tokens: {ent_tokens}")
    # print(f"ent_label: {ent_label}")
    # print(f"new_tokens: {new_tokens}")
    # print(f"new_boxes: {new_boxes}")
    # print("="*100)

    return new_tokens, new_boxes

def create_aug_data_new(
    train_dataset,
    text_column_name,
    num=10,
    org_nums=0, 
    id2label=None,
    features=None,
    seed=0,
    ):
    nlp = spacy.load("en_core_web_sm")
    new_examples = []
    count = org_nums
    fake = Faker()
    Faker.seed(seed)
    for example in train_dataset:
        time1 = time.time()
        sentence = " ".join(example[text_column_name])
        boxes = example["bboxes"]
        labels = example["ner_tags"]
        doc = nlp(sentence)
        # print(doc)
        # print(f"num_ent: {len(doc.ents)}")
        for _ in range(num):
            i = 0
            new_sentence = []
            org_token_idx = 0
            new_sentence_boxes = []
            new_sentence_labels = []
            begin_idx = 0
            
            # print(f"num org tokens: {len(boxes)}")
            # print(f"num sentence_tokens: {len(sentence.split(' '))}")
            for ent in doc.ents:
                # the entity obtained by the spaCy may ignore the first punctuation !!!
                ent_start_pos = ent.start_char
                while ent_start_pos > 0 and sentence[ent_start_pos-1] != " ":
                    ent_start_pos -= 1

                # the entity obtained by the spaCy may ignore the last punctuation !!!
                ent_end_pos = ent.end_char
                while(ent_end_pos < len(sentence) and sentence[ent_end_pos] != " "):
                    ent_end_pos += 1
                
                if sentence[begin_idx:ent_start_pos].strip() != "":
                    not_ent_tokens = sentence[begin_idx:ent_start_pos].strip().split(" ")
                    for t in not_ent_tokens:
                        new_sentence.append(t)
                        new_sentence_labels.append(id2label[labels[org_token_idx]])
                        # print(org_token_idx)
                        new_sentence_boxes.append([p for p in boxes[org_token_idx]])
                        org_token_idx += 1
                
                ent_tokens = sentence[ent_start_pos:ent_end_pos].strip().split(" ")
                # print(f"ent_tokens: {ent_tokens}")
                ent_label = ent.label_
                ent_box = [p for p in boxes[org_token_idx]]
                ent_ner_label = id2label[labels[org_token_idx]]
                new_tokens, new_boxes = create_new_ent(ent_tokens, ent_label, ent_box, fake)
                
                org_token_idx += len(ent_tokens)

                for j in range(len(new_tokens)):
                    new_sentence.append(new_tokens[j])
                    new_sentence_boxes.append(new_boxes[j])
                    if j == 0:
                        new_sentence_labels.append(ent_ner_label)
                    else:
                        if ent_ner_label[0] == "B":
                            new_sentence_labels.append("I"+ent_ner_label[1:])
                        else:
                            new_sentence_labels.append(ent_ner_label)

                begin_idx = ent_end_pos

            # process the remained tokens after the last entity
            if sentence[begin_idx:].strip() != "":
                not_ent_tokens = sentence[begin_idx:].strip().split(" ")
                # if "" in not_ent_tokens:
                #     not_ent_tokens.remove("")
                # print(f"not_ent_tokens: {not_ent_tokens}")
                if not_ent_tokens:
                    for t in not_ent_tokens:
                        new_sentence.append(t)
                        new_sentence_labels.append(id2label[labels[org_token_idx]])
                        # print(f"org_token_idx: {org_token_idx}")
                        new_sentence_boxes.append([p for p in boxes[org_token_idx]])
                        org_token_idx += 1
            
            print(f"num new_tokens: {len(new_sentence)}")
            print(f"num new_tokens_boxes: {len(new_sentence_boxes)}")

            new_example = copy.deepcopy(example)
            new_example[text_column_name] = new_sentence
            new_example["bboxes"] = new_sentence_boxes
            new_example["id"] = str(count)
            new_example["ner_tags"] = new_sentence_labels
            count += 1
            logger.info(new_example)
            new_examples.append(new_example)

        time2 = time.time()
        logger.info(f"Process time: {time2-time1}s")

    def gen(examples):
        for example in examples:
            yield example

    new_datas = datasets.arrow_dataset.Dataset.from_generator(
        gen,
        gen_kwargs={"examples": new_examples},
        features=features,
        )

    return new_datas


def union_box(box1, box2):
    """
    Params:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    """
    x1 = min(box1[0], box2[0])
    y1 = min(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])

    return [x1, y1, x2, y2]


def is_same_line(box1, box2):
    """
    Params:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    """
    
    box1_midy = (box1[1] + box1[3]) / 2
    box2_midy = (box2[1] + box2[3]) / 2

    if box1_midy < box2[3] and box1_midy > box2[1] and box2_midy < box1[3] and box2_midy > box1[1]:
        return True
    else:
        return False


def is_adj_line(box1, box2):
    """
    Params:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    """
    h1 = box1[3] - box1[1]
    h2 = box2[3] - box2[1]
    h_dist = max(box1[1], box2[1]) - min(box1[3], box2[3])

    box1_midx = (box1[0] + box1[2]) / 2
    box2_midx = (box2[0] + box2[2]) / 2

    # if h_dist <= min(h1, h2) and box1_midx < box2[2] and box1_midx > box2[0] and box2_midx < box1[2] and box2_midx > box1[0]:
    if h_dist <= min(h1, h2): # v2
        return True
    else:
        return False


def boxes_sort(boxes):
    """ From left top to right bottom
    Params:
        boxes: [[x1, y1, x2, y2], [x1, y1, x2, y2], ...]
    """
    sorted_id = sorted(range(len(boxes)), key=lambda x: (boxes[x][1], boxes[x][0]))

    # sorted_boxes = [boxes[id] for id in sorted_id]


    return sorted_id



if __name__ == "__main__":
    boxes_sort([[1, 2, 3, 4], [1, 1, 3, 3], [1, 3, 3, 5], [1, 4, 3, 6]])