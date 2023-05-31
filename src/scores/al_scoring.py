import ast
import json
import os

import numpy as np
import pandas as pd
from pycocotools.coco import COCO

from config import paths


def get_per_category_abundance():
    coco_json_file = open(paths.train_anns_path)
    coco_json = json.load(coco_json_file)

    cat_counts = np.zeros(len(coco_json["categories"]))
    for anno in coco_json["annotations"]:
        cat_counts[anno["category_id"] - 1] += 1

    cat_abundance = 1 - (cat_counts / np.sum(cat_counts))

    return cat_abundance


def calculate_al_score(file_path: str, output_path: str = "./output/"):
    data = pd.read_csv(file_path)

    cat_abundance = get_per_category_abundance()
    data["total_score"] = (
        0.5 * data["box loss"]
        + 0.5 * data["mask loss"]
        + (1.0 - data["class avg score"])
        + (0.1 * data["no of predictions"])
    )

    print(data.head)

    cat_ab_val_list = list()
    for cats in data["categories"]:
        cats = ast.literal_eval(cats)
        cat_ab_val = (
            np.average([cat_abundance[cat] for cat in cats]) if len(cats) != 0 else 0
        )
        cat_ab_val_list.append(cat_ab_val)

    data["cat_abundance_val"] = cat_ab_val_list
    print(data["cat_abundance_val"][5], type(data["cat_abundance_val"][5]))
    data["total_score"] = data["total_score"] + data["cat_abundance_val"]

    print(data.head)

    data.drop("cat_abundance_val", axis=1, inplace=True)

    output_data = data.sort_values(
        by=["total_score"], ascending=False, na_position="last"
    )
    output_data.to_csv(os.path.join(output_path, "al_score.csv"))


def get_top_n_images(score_file_path: str, no_img: int = 10):
    data = pd.read_csv(score_file_path)

    image_list = data.iloc[:no_img, 2]
    return image_list.values.tolist()


def get_alpl_n_images(score_file_path: str, no_img: int = 10):
    data = pd.read_csv(score_file_path)
    data["box loss"].replace("", np.nan, inplace=True)
    data.dropna(subset=["box loss"], inplace=True)

    half_nos = int(no_img / 2)
    al_list = data.iloc[:half_nos, 2]
    pl_list = data.iloc[-(no_img - half_nos) :, 2]
    return al_list.values.tolist() + pl_list.values.tolist()
