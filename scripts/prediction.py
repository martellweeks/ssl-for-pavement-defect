import torch, detectron2
import os
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.data import MetadataCatalog
from detectron2.data.datasets import load_coco_json
from detectron2.engine import DefaultPredictor
from detectron2.utils.visualizer import Visualizer

import glob
import cv2
import argparse
import warnings

warnings.filterwarnings(action='ignore', category=UserWarning)

parser = argparse.ArgumentParser(description='Detectron2 Model Prediction Labeler')
parser.add_argument('--image_folder', default='data/predictions/raw', type=str)
parser.add_argument('--metadata_file', default='data/annotations/metadata.json', type=str)
parser.add_argument('--weights_file',   default='models/A14_20221119.pth', type=str)
parser.add_argument('--thresh_score', default=0.5, type=float)
args = parser.parse_args()


metadata_json = load_coco_json(args.metadata_file, '', 'metadata')
metadata = MetadataCatalog.get('metadata')

cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"))
cfg.DATALOADER.NUM_WORKERS = 2
cfg.MODEL.WEIGHTS = args.weights_file
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 14
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.thresh_score
cfg.MODEL.MASK_ON = True
cfg.OUTPUT_DIR = "data/predictions/result"
cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if not os.path.exists(cfg.OUTPUT_DIR):
    os.makedirs(cfg.OUTPUT_DIR)


predictor = DefaultPredictor(cfg)

prediction_imgs = glob.glob(os.path.join('data/predictions/raw/*'))
pred_output_dir = cfg.OUTPUT_DIR

for d in prediction_imgs:  
    print(d)
    im = cv2.imread(d)
    outputs = predictor(im)
    v = Visualizer(im[:, :, ::-1],
        metadata=metadata,
    )
    out_pred = v.draw_instance_predictions(outputs["instances"].to("cpu"))
    cv2.imwrite(
        os.path.join(
            pred_output_dir,
            f"{d.split('/', -1)[-1][:-4]}_pred{d.split('/', -1)[-1][-4:]}"
        ),
        cv2.cvtColor(out_pred.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB)
    )