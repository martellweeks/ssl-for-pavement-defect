import ast
import glob
import json
import logging
import os
from datetime import datetime
from typing import List

import cv2
import detectron2
import pandas as pd
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import CfgNode
from detectron2.data import DatasetCatalog, MetadataCatalog, build_detection_test_loader
from detectron2.data.datasets import load_coco_json, register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.modeling import build_model
from detectron2.utils.logger import setup_logger
from detectron2.utils.visualizer import Visualizer
from tqdm import tqdm

from config import config, paths
from src.model import hooks
from src.model.al_scoring_head import ALScoringROIHeads


def startup(regist_instances: bool = True, cfg: CfgNode = None):
    logger = setup_logger(output="./log_0414_al.log")

    TORCH_VERSION = ".".join(torch.__version__.split(".")[:2])
    CUDA_VERSION = torch.__version__.split("+")[-1]
    logger.info("Check versions...")
    logger.info(f"torch:  {TORCH_VERSION} ; cuda:  {CUDA_VERSION}")
    logger.info(f"detectron2: {detectron2.__version__}")

    if regist_instances:
        register_coco_instances("train", {}, paths.train_anns_path, paths.raw_data_path)
        register_coco_instances("val", {}, paths.val_anns_path, paths.raw_data_path)
        register_coco_instances("test", {}, paths.test_anns_path, paths.raw_data_path)

    if cfg is None:
        cfg = config.get_default_cfg()

    return logger, cfg


def register_new_coco_instance(annotation_path: str, data_path: str, tag: str):
    return register_coco_instances(tag, {}, annotation_path, data_path)


def train(output_folder: str = None, cfg: CfgNode = None):
    logger, cfg = startup(cfg=cfg)

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
    checkpointer.save(paths.final_model_filename)
    logger.info("Final model saved")


def train_model_only(
    output_folder: str = None,
    model_weights: str = None,
    regist_instances: bool = True,
    cfg: CfgNode = None,
):
    logger, cfg = startup(regist_instances=regist_instances, cfg=cfg)

    cfg.MODEL.ROI_HEADS.NAME = "ALScoringROIHeads"

    if output_folder is not None:
        cfg.OUTPUT_DIR = os.path.join(cfg.OUTPUT_DIR, output_folder)

    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

    if model_weights is not None:
        cfg.MODEL.WEIGHTS = model_weights

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
    trainer.model.roi_heads.mask_scorer.conv2.bias.requires_grad = False
    trainer.model.roi_heads.mask_scorer.conv2.weight.requires_grad = False
    trainer.model.roi_heads.mask_scorer.batchnorm.bias.requires_grad = False
    trainer.model.roi_heads.mask_scorer.batchnorm.weight.requires_grad = False

    logger.info("Start training...")
    trainer.train()

    logger.info("Training completed")
    if os.path.exists(paths.final_model_full_path):
        os.remove(paths.final_model_full_path)
    checkpointer = DetectionCheckpointer(trainer.model, save_dir=paths.final_model_path)
    checkpointer.save(paths.final_model_filename)
    logger.info("Final model saved")


def train_scores_only(
    output_folder: str = None,
    model_weights: str = None,
    regist_instances: bool = True,
    cfg: CfgNode = None,
):
    logger, cfg = startup(regist_instances=regist_instances, cfg=cfg)

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
    if os.path.exists(paths.final_model_full_path):
        os.remove(paths.final_model_full_path)
    checkpointer = DetectionCheckpointer(trainer.model, save_dir=paths.final_model_path)
    checkpointer.save(paths.final_model_filename)
    logger.info("Final model saved")


def train_mask_score_only(
    output_folder: str = None,
    model_weights: str = None,
    regist_instances: bool = True,
    cfg: CfgNode = None,
):
    logger, cfg = startup(regist_instances=regist_instances, cfg=cfg)

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

    trainer.model.roi_heads.box_scorer.conv1.bias.requires_grad = False
    trainer.model.roi_heads.box_scorer.conv1.weight.requires_grad = False
    trainer.model.roi_heads.box_scorer.fc1.bias.requires_grad = False
    trainer.model.roi_heads.box_scorer.fc1.weight.requires_grad = False
    trainer.model.roi_heads.box_scorer.fc2.bias.requires_grad = False
    trainer.model.roi_heads.box_scorer.fc2.weight.requires_grad = False

    logger.info("Start training...")
    trainer.train()

    logger.info("Training completed")
    if os.path.exists(paths.final_model_full_path):
        os.remove(paths.final_model_full_path)
    checkpointer = DetectionCheckpointer(trainer.model, save_dir=paths.final_model_path)
    checkpointer.save(paths.final_model_filename)
    logger.info("Final model saved")


def train_box_score_only(
    output_folder: str = None,
    model_weights: str = None,
    regist_instances: bool = True,
    cfg: CfgNode = None,
):
    logger, cfg = startup(regist_instances=regist_instances, cfg=cfg)

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

    trainer.model.roi_heads.mask_scorer.conv1.bias.requires_grad = False
    trainer.model.roi_heads.mask_scorer.conv1.weight.requires_grad = False
    trainer.model.roi_heads.mask_scorer.fc1.bias.requires_grad = False
    trainer.model.roi_heads.mask_scorer.fc1.weight.requires_grad = False
    trainer.model.roi_heads.mask_scorer.fc2.bias.requires_grad = False
    trainer.model.roi_heads.mask_scorer.fc2.weight.requires_grad = False
    trainer.model.roi_heads.mask_scorer.conv2.bias.requires_grad = False
    trainer.model.roi_heads.mask_scorer.conv2.weight.requires_grad = False
    trainer.model.roi_heads.mask_scorer.batchnorm.bias.requires_grad = False
    trainer.model.roi_heads.mask_scorer.batchnorm.weight.requires_grad = False

    logger.info("Start training...")
    trainer.train()

    logger.info("Training completed")
    if os.path.exists(paths.final_model_full_path):
        os.remove(paths.final_model_full_path)
    checkpointer = DetectionCheckpointer(trainer.model, save_dir=paths.final_model_path)
    checkpointer.save(paths.final_model_filename)
    logger.info("Final model saved")


def predict_scores(
    model_weights: str,
    regist_instances: bool = True,
    output_path: str = None,
    cfg: CfgNode = None,
    test_anns_file: str = None,
):
    logger, cfg = startup(regist_instances=regist_instances, cfg=cfg)

    test_ds = DatasetCatalog.get("test")

    cfg.MODEL.ROI_HEADS.NAME = "ALScoringROIHeads"
    cfg.MODEL.WEIGHTS = model_weights
    cfg.SOLVER.IMS_PER_BATCH = 1

    if output_path is not None:
        pred_output_dir = output_path
    else:
        pred_output_dir = cfg.OUTPUT_DIR
    os.makedirs(pred_output_dir, exist_ok=True)

    predictor = DefaultPredictor(cfg)

    test_json_file = open(
        test_anns_file if test_anns_file is not None else paths.test_anns_path
    )
    test_json = json.load(test_json_file)

    prediction_imgs = list()

    for img in test_json["images"]:
        prediction_imgs.append(os.path.join("data/A14_L2", img["file_name"]))

    score_predictions = list()
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
                predictions["instances"].pred_classes.tolist()
                if type(predictions["instances"].pred_classes.tolist()) == list
                else ast.literal_eval(predictions["instances"].pred_classes.tolist()),
            ]
        )

    predictions_df = pd.DataFrame(
        data=score_predictions,
        columns=[
            "img",
            "box loss",
            "mask loss",
            "class avg score",
            "class min score",
            "no of predictions",
            "categories",
        ],
        dtype="object",
    )
    predictions_df.to_csv(os.path.join(pred_output_dir, "loss_predictions.csv"))


def label_predictions_on_images(
    image_list: List[str],
    regist_instances: bool = True,
    output_path: str = "./output/",
    cfg: CfgNode = None,
):
    logger, cfg = startup(regist_instances=regist_instances, cfg=cfg)

    test_ds = DatasetCatalog.get("test")
    metadata_json = load_coco_json(paths.metadata_filename, "", "metadata")
    metadata = MetadataCatalog.get("metadata")

    cfg.MODEL.ROI_HEADS.NAME = "ALScoringROIHeads"
    cfg.MODEL.WEIGHTS = "./output/A14_L2/0322_score_1/model_final.pth"

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


def get_coco_eval_results(
    model_weights: str,
    regist_instances: bool = True,
    test_data_tag: str = "test",
    output_path: str = "./output/",
    cfg: CfgNode = None,
):
    logger, cfg = startup(regist_instances=regist_instances, cfg=cfg)

    cfg.MODEL.ROI_HEADS.NAME = "ALScoringROIHeads"
    cfg.MODEL.WEIGHTS = model_weights
    cfg.DATASETS.TEST = (test_data_tag,)

    os.makedirs(output_path, exist_ok=True)

    predictor = DefaultPredictor(cfg)

    evaluator = COCOEvaluator("test", cfg, False, output_dir=output_path)
    val_loader = build_detection_test_loader(cfg, "test")

    logger.info("Running COCO evaluation on test dataset with current model")

    with open(os.path.join(output_path, "COCO_metrics_evaluation.json"), "w") as f:
        json.dump(inference_on_dataset(predictor.model, val_loader, evaluator), f)

    logger.info("File saved")
