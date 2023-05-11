import datetime
import os

project_path = ""

raw_data_path = os.path.join(project_path, "data/A12")
train_data_path = os.path.join(project_path, "data/A12")
val_data_path = os.path.join(project_path, "data/A12")
test_data_path = os.path.join(project_path, "data/A12")

raw_anns_filename = "data/annotations/A12/raw.json"
train_anns_filename = "data/annotations/A12/vanilla/train_patchonly.json"
val_anns_filename = "data/annotations/A12/vanilla/val_patchonly.json"
test_anns_filename = "data/annotations/A12/vanilla/test_patchonly.json"
metadata_filename = "data/annotations/A14_L2/metadata.json"

raw_anns_path = os.path.join(project_path, raw_anns_filename)
train_anns_path = os.path.join(project_path, train_anns_filename)
val_anns_path = os.path.join(project_path, val_anns_filename)
test_anns_path = os.path.join(project_path, test_anns_filename)

output_path = os.path.join(project_path, "output/")
final_model_filename = "output"
final_model_path = os.path.join(project_path, "models")
final_model_full_path = os.path.join(project_path, "models/output.pth")
