import datetime
import os

project_path = ""

raw_data_path = os.path.join(project_path, "data/raw")
train_data_path = os.path.join(project_path, "data/train")
val_data_path = os.path.join(project_path, "data/val")
test_data_path = os.path.join(project_path, "data/test")

raw_anns_filename = "data/annotations/raw.json"
train_anns_filename = "data/annotations/train_cats_3411.json"
val_anns_filename = "data/annotations/val_cats_3411.json"
test_anns_filename = "data/annotations/test_cats_3411.json"
metadata_filename = "data/annotations/metadata.json"

raw_anns_path = os.path.join(project_path, raw_anns_filename)
train_anns_path = os.path.join(project_path, train_anns_filename)
val_anns_path = os.path.join(project_path, val_anns_filename)
test_anns_path = os.path.join(project_path, test_anns_filename)

output_path = os.path.join(project_path, "output")
output_model_filename = f"model_{datetime.date.today()}"
