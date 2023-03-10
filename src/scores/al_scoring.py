import os

import numpy as np
import pandas as pd


def al_score_calculation(file_path: str, output_path: str = "./output/"):
    data = pd.read_csv(file_path)

    data["total_score"] = (
        data["box loss"]
        + data["mask loss"]
        + (1.0 - data["class avg score"])
        + (0.05 * data["no of predictions"])
    )
    output_data = data.sort_values(
        by=["total_score"], ascending=False, na_position="last"
    )
    output_data.to_csv(os.path.join(output_path, "al_score.csv"))


def get_top_n_images(score_file_path: str, no_img: int = 10):
    data = pd.read_csv(score_file_path)

    image_list = data.iloc[:no_img, 2]
    return image_list.values.tolist()
