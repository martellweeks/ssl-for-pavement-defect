import ast
import json
import os
import random

import numpy as np
import pandas as pd
from pycocotools.coco import COCO

from config import paths


def remove_labels(
    output_path: str,
    output_file_tag: str = None,
    test_anns_file: str = None,
):
    test_json_file = open(
        test_anns_file if test_anns_file is not None else paths.test_anns_path
    )
    test_json = json.load(test_json_file)

    test_json["annotations"] = []

    if output_file_tag is not None:
        with open(
            os.path.join(output_path, f"test_unlabeled_{output_file_tag}.json"), "w"
        ) as f:
            json.dump(test_json, f)
    else:
        with open(os.path.join(output_path, "test_unlabeled.json"), "w") as f:
            json.dump(test_json, f)
