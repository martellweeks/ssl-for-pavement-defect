import ast
import json
import os
import random

import numpy as np
import pandas as pd
from pycocotools.coco import COCO

from config import paths


def move_labels_to_train(
    image_list: list,
    output_path: str,
    output_file_tag: str = None,
    train_anns_file: str = None,
    test_anns_file: str = None,
):
    train_json_file = open(
        train_anns_file if train_anns_file is not None else paths.train_anns_path
    )
    train_json = json.load(train_json_file)

    test_json_file = open(
        test_anns_file if test_anns_file is not None else paths.test_anns_path
    )
    test_json = json.load(test_json_file)

    image_name_list = list()
    for img_name in image_list:
        image_name_list.append(img_name.split("/")[-1])

    image_list = image_name_list

    image_id_list = list()
    new_images = list()
    new_annotations = list()

    for img in test_json["images"]:
        if img["file_name"] in image_list:
            image_id_list.append(img["id"])
            new_images.append(img)
            test_json["images"].remove(img)

    for label in test_json["annotations"]:
        if label["image_id"] in image_id_list:
            new_annotations.append(label)
            test_json["annotations"].remove(label)

    train_json["images"].extend(new_images)
    train_json["annotations"].extend(new_annotations)

    os.makedirs(output_path, exist_ok=True)

    if output_file_tag is not None:
        with open(os.path.join(output_path, f"train_{output_file_tag}.json"), "w") as f:
            json.dump(train_json, f)
        with open(os.path.join(output_path, f"test_{output_file_tag}.json"), "w") as f:
            json.dump(test_json, f)
    else:
        with open(os.path.join(output_path, "train.json"), "w") as f:
            json.dump(train_json, f)
        with open(os.path.join(output_path, "test.json"), "w") as f:
            json.dump(test_json, f)


def get_random_n_images(test_anns_file: str = None, no_img: int = 10):
    test_json_file = open(
        test_anns_file if test_anns_file is not None else paths.test_anns_path
    )
    test_json = json.load(test_json_file)

    image_list = list()
    rand_idx = random.sample(range(len(test_json["images"])), no_img)

    for idx in rand_idx:
        image_list.append(test_json["images"][idx]["file_name"])

    return image_list
