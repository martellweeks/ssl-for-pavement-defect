import glob
import logging
import os
from datetime import datetime

import cv2
import detectron2
import pandas as pd
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, DatasetMapper, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from tqdm import tqdm

from config import paths
from config.config import cfg as cfg
from src.model import hooks
from src.model.al_scoring_head import ALScoringROIHeads


def startup():
    logger = setup_logger()
    fileHandler = logging.FileHandler(f"log_{datetime.now()}")
    logger.addHandler(fileHandler)
    logger.setLevel(level=logging.DEBUG)

    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    logger.info("Check versions...")
    logger.info(f"torch:  {TORCH_VERSION} ; cuda:  {CUDA_VERSION}")
    logger.info(f"detectron2: {detectron2.__version__}")

    register_coco_instances("train", {}, paths.train_anns_path, paths.train_data_path)
    register_coco_instances("val", {}, paths.val_anns_path, paths.val_data_path)
    register_coco_instances("test", {}, paths.test_anns_path, paths.test_data_path)

    return logger


def train():
    logger = startup()

    cfg.MODEL.ROI_HEADS.NAME = "ALScoringROIHeads"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = hooks.MyTrainer(cfg)
    trainer.resume_or_load(resume=False)
    logger.info("Start training...")
    trainer.train()

    logger.info("Training completed")
    checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.save(paths.output_model_filename)
    logger.info("Final model saved")


def predict():
    logger = startup()

    test_ds = DatasetCatalog.get("test")

    cfg.MODEL.ROI_HEADS.NAME = "ALScoringROIHeads"
    cfg.MODEL.WEIGHTS = "./models/20230302_pvmt_al_allcat_2000it.pth"

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    predictor = DefaultPredictor(cfg)

    score_predictions = list()
    prediction_imgs = glob.glob(os.path.join("data/test/*"))
    pred_output_dir = cfg.OUTPUT_DIR
    for d in tqdm(prediction_imgs, desc="Running prediction on images"):
        im = cv2.imread(d)
        predictions = predictor(im)

        score_predictions.append(
            [
                d,
                predictions["instances"].box_score_prediction[0].item()
                if len(predictions["instances"].box_score_prediction) != 0
                else "NA",
                predictions["instances"].mask_score_prediction[0].item()
                if len(predictions["instances"].mask_score_prediction) != 0
                else "NA",
            ]
        )

    predictionsDF = pd.DataFrame(
        data=score_predictions, columns=["img", "box loss", "mask loss"]
    )
    predictionsDF.to_csv(os.path.join(cfg.OUTPUT_DIR, "loss_predictions.csv"))

    logger.info(score_predictions)
