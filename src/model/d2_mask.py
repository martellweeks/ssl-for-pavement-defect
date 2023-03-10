import glob
import logging
import os
from datetime import datetime
from typing import List

import cv2
import detectron2
import pandas as pd
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetCatalog, DatasetMapper, MetadataCatalog
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm

from config import paths
from config.config import cfg as cfg
from src.model import hooks
from src.model.al_scoring_head import ALScoringROIHeads


def startup(regist_instances: bool = True):
    logger = setup_logger()
    fileHandler = logging.FileHandler(f"log_{datetime.now()}")
    logger.addHandler(fileHandler)
    logger.setLevel(level=logging.DEBUG)

    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    logger.info("Check versions...")
    logger.info(f"torch:  {TORCH_VERSION} ; cuda:  {CUDA_VERSION}")
    logger.info(f"detectron2: {detectron2.__version__}")

    if regist_instances:
        register_coco_instances(
            "train", {}, paths.train_anns_path, paths.train_data_path
        )
        register_coco_instances("val", {}, paths.val_anns_path, paths.val_data_path)
        register_coco_instances("test", {}, paths.test_anns_path, paths.test_data_path)

    return logger


def train(output_folder: str = None):
    logger = startup()

    cfg.MODEL.ROI_HEADS.NAME = "ALScoringROIHeads"

    if output_folder is not None:
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, output_folder)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = hooks.MyTrainer(cfg)
    trainer.resume_or_load(resume=False)

    logger.info("Start training...")
    trainer.train()

    logger.info("Training completed")
    checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.save(paths.output_model_filename)
    logger.info("Final model saved")


def train_model_only(output_folder: str = None, regist_instances: bool = True):
    logger = startup(regist_instances=regist_instances)

    cfg.MODEL.ROI_HEADS.NAME = "ALScoringROIHeads"

    if output_folder is not None:
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, output_folder)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    trainer = hooks.MyTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # Freeze weights and biases of score prediction module

    trainer.model.roi_heads.box_scorer.conv1.bias.requires_grad = False
    trainer.model.roi_heads.box_scorer.conv1.weight.requires_grad = False
    trainer.model.roi_heads.box_scorer.fc1.bias.requires_grad = False
    trainer.model.roi_heads.box_scorer.fc1.weight.requires_grad = False
    trainer.model.roi_heads.box_scorer.fc2.bias.requires_grad = False
    trainer.model.roi_heads.box_scorer.fc2.weight.requires_grad = False
    trainer.model.roi_heads.mask_scorer.conv1.bias.requires_grad = False
    trainer.model.roi_heads.mask_scorer.conv1.weight.requires_grad = False
    trainer.model.roi_heads.mask_scorer.fc1.bias.requires_grad = False
    trainer.model.roi_heads.mask_scorer.fc1.weight.requires_grad = False
    trainer.model.roi_heads.mask_scorer.fc2.bias.requires_grad = False
    trainer.model.roi_heads.mask_scorer.fc2.weight.requires_grad = False

    logger.info("Start training...")
    trainer.train()

    logger.info("Training completed")
    checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.save(paths.output_model_filename)
    logger.info("Final model saved")


def train_scores_only(
    output_folder: str = None, model_weights: str = None, regist_instances: bool = True
):
    logger = startup(regist_instances=regist_instances)

    cfg.MODEL.ROI_HEADS.NAME = "ALScoringROIHeads"

    if output_folder is not None:
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, output_folder)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if model_weights is not None:
        cfg.MODEL.WEIGHTS = model_weights

    trainer = hooks.MyTrainer(cfg)
    trainer.resume_or_load(resume=False)

    # Freeze weights and biases of all model except score predictors

    trainer.model.backbone.fpn_lateral2.weight.requires_grad = False
    trainer.model.backbone.fpn_lateral3.weight.requires_grad = False
    trainer.model.backbone.fpn_lateral4.weight.requires_grad = False
    trainer.model.backbone.fpn_lateral5.weight.requires_grad = False
    trainer.model.backbone.fpn_lateral2.bias.requires_grad = False
    trainer.model.backbone.fpn_lateral3.bias.requires_grad = False
    trainer.model.backbone.fpn_lateral4.bias.requires_grad = False
    trainer.model.backbone.fpn_lateral5.bias.requires_grad = False
    trainer.model.backbone.fpn_output2.weight.requires_grad = False
    trainer.model.backbone.fpn_output3.weight.requires_grad = False
    trainer.model.backbone.fpn_output4.weight.requires_grad = False
    trainer.model.backbone.fpn_output5.weight.requires_grad = False
    trainer.model.backbone.fpn_output2.bias.requires_grad = False
    trainer.model.backbone.fpn_output3.bias.requires_grad = False
    trainer.model.backbone.fpn_output4.bias.requires_grad = False
    trainer.model.backbone.fpn_output5.bias.requires_grad = False
    for conv in trainer.model.backbone.lateral_convs:
        conv.weight.requires_grad = False
        conv.bias.requires_grad = False
    for conv in trainer.model.backbone.output_convs:
        conv.weight.requires_grad = False
        conv.bias.requires_grad = False

    trainer.model.proposal_generator.rpn_head.anchor_deltas.weight.requires_grad = False
    trainer.model.proposal_generator.rpn_head.anchor_deltas.bias.requires_grad = False
    trainer.model.proposal_generator.rpn_head.conv.weight.requires_grad = False
    trainer.model.proposal_generator.rpn_head.conv.bias.requires_grad = False
    trainer.model.proposal_generator.rpn_head.objectness_logits.weight.requires_grad = (
        False
    )
    trainer.model.proposal_generator.rpn_head.objectness_logits.bias.requires_grad = (
        False
    )

    trainer.model.roi_heads.box_head.fc1.weight.requires_grad = False
    trainer.model.roi_heads.box_head.fc2.weight.requires_grad = False
    trainer.model.roi_heads.box_head.fc1.bias.requires_grad = False
    trainer.model.roi_heads.box_head.fc2.bias.requires_grad = False
    for i in trainer.model.roi_heads.box_head.fcs:
        i.weight.requires_grad = False
        i.bias.requires_grad = False
    trainer.model.roi_heads.box_predictor.bbox_pred.weight.requires_grad = False
    trainer.model.roi_heads.box_predictor.cls_score.weight.requires_grad = False
    trainer.model.roi_heads.box_predictor.bbox_pred.bias.requires_grad = False
    trainer.model.roi_heads.box_predictor.cls_score.bias.requires_grad = False

    for i in trainer.model.roi_heads.mask_head.conv_norm_relus:
        i.weight.requires_grad = False
        i.bias.requires_grad = False
    trainer.model.roi_heads.mask_head.deconv.weight.requires_grad = False
    trainer.model.roi_heads.mask_head.deconv.bias.requires_grad = False
    trainer.model.roi_heads.mask_head.mask_fcn1.weight.requires_grad = False
    trainer.model.roi_heads.mask_head.mask_fcn2.weight.requires_grad = False
    trainer.model.roi_heads.mask_head.mask_fcn3.weight.requires_grad = False
    trainer.model.roi_heads.mask_head.mask_fcn4.weight.requires_grad = False
    trainer.model.roi_heads.mask_head.predictor.weight.requires_grad = False
    trainer.model.roi_heads.mask_head.mask_fcn1.bias.requires_grad = False
    trainer.model.roi_heads.mask_head.mask_fcn2.bias.requires_grad = False
    trainer.model.roi_heads.mask_head.mask_fcn3.bias.requires_grad = False
    trainer.model.roi_heads.mask_head.mask_fcn4.bias.requires_grad = False
    trainer.model.roi_heads.mask_head.predictor.bias.requires_grad = False

    logger.info("Start training...")
    trainer.train()

    logger.info("Training completed")
    checkpointer = DetectionCheckpointer(trainer.model, save_dir=cfg.OUTPUT_DIR)
    checkpointer.save(paths.output_model_filename)
    logger.info("Final model saved")


def freeze_model_parts(model):
    for component in model:
        if hasattr(component, "bias"):
            component.bias = False
            component.weight = False
        else:
            freeze_model_parts(component)


def predict(model_weights: str, regist_instances: bool = True, output_path: str = None):
    logger = startup(regist_instances=regist_instances)

    test_ds = DatasetCatalog.get("test")

    cfg.MODEL.ROI_HEADS.NAME = "ALScoringROIHeads"
    cfg.MODEL.WEIGHTS = model_weights
    cfg.SOLVER.IMS_PER_BATCH = 1

    if output_path is not None:
        pred_output_dir = output_path
    else:
        pred_output_dir = cfg.OUTPUT_DIR
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    predictor = DefaultPredictor(cfg)

    score_predictions = list()
    prediction_imgs = glob.glob(os.path.join("data/test/*"))
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
                torch.mean(predictions["instances"].scores).item()
                if len(predictions["instances"].scores) != 0
                else "NA",
                torch.min(predictions["instances"].scores).item()
                if len(predictions["instances"].scores) != 0
                else "NA",
                len(predictions["instances"].scores),
            ]
        )

    predictionsDF = pd.DataFrame(
        data=score_predictions,
        columns=[
            "img",
            "box loss",
            "mask loss",
            "class avg score",
            "class min score",
            "no of predictions",
        ],
    )
    predictionsDF.to_csv(os.path.join(pred_output_dir, "loss_predictions.csv"))

    logger.info(score_predictions)


def label_predictions_on_images(
    image_list: List[str], regist_instances: bool = True, output_path: str = "./output/"
):
    logger = startup(regist_instances=regist_instances)

    test_ds = DatasetCatalog.get("test")
    metadata_json = load_coco_json(paths.metadata_filename, "", "metadata")
    metadata = MetadataCatalog.get("metadata")

    cfg.MODEL.ROI_HEADS.NAME = "ALScoringROIHeads"
    cfg.MODEL.WEIGHTS = "./output/0310_score/model_final.pth"

    os.makedirs(output_path, exist_ok=True)

    predictor = DefaultPredictor(cfg)

    prediction_imgs = image_list
    pred_output_dir = output_path
    for idx, d in enumerate(tqdm(prediction_imgs, desc="Running prediction on images")):
        im = cv2.imread(d)
        predictions = predictor(im)

        v = Visualizer(
            im[:, :, ::-1],
            metadata=metadata,
        )
        out_pred = v.draw_instance_predictions(predictions["instances"].to("cpu"))
        cv2.imwrite(
            os.path.join(
                pred_output_dir,
                f"{1+idx}_{d.split('/', -1)[-1][:-4]}_pred{d.split('/', -1)[-1][-4:]}",
            ),
            cv2.cvtColor(out_pred.get_image()[:, :, ::-1], cv2.COLOR_BGR2RGB),
        )
