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

from config import config, paths, setup
from src.controller import d2_mask
from src.data import al_label_transfer
from src.engine.trainer import BaseTrainer, CNSTrainerManager
from src.models.al_roi_heads import ALScoringROIHeads
from src.models.cns_roi_heads import CNSALROIHeads, CNSStandardROIHeads
from src.models.rcnn import CNSGeneralizedRCNN
from src.scores import al_scoring


def train_model(
    output_folder: str = None,
    model_weights: str = None,
    regist_instances: bool = True,
    logfile: str = "cns",
    cfg: CfgNode = None,
):
    d2_mask.train_model_only_cns(
        output_folder=output_folder,
        model_weights=model_weights,
        regist_instances=regist_instances,
        logfile=logfile,
        cfg=cfg,
    )


def coco_eval(
    model_weights: str = None,
    regist_instances: bool = True,
    output_path: str = None,
    cfg: CfgNode = None,
):
    d2_mask.get_coco_eval_results_cns(
        model_weights=model_weights,
        regist_instances=regist_instances,
        output_path=output_path,
        cfg=cfg,
    )


def train_score(
    output_folder: str = None,
    model_weights: str = None,
    regist_instances: bool = True,
    logfile: str = "cns",
    cfg: CfgNode = None,
):
    d2_mask.train_scores_only_cns(
        output_folder=output_folder,
        model_weights=model_weights,
        regist_instances=regist_instances,
        logfile=logfile,
        cfg=cfg,
    )


def sample_al_sets(
    model_weights: str = None,
    regist_instances: bool = True,
    output_path: str = None,
    test_anns_file: str = None,
    no_img: int = 40,
) -> list:
    d2_mask.predict_scores(
        model_weights=model_weights,
        output_path=output_path,
        regist_instances=regist_instances,
        test_anns_file=test_anns_file,
    )

    al_scoring.calculate_al_score(
        file_path=os.path.join(output_path, "loss_predictions.csv"),
        output_path=output_path,
    )

    return al_scoring.get_top_n_images(
        score_file_path=os.path.join(output_path, "al_score.csv"),
        no_img=no_img,
    )


def sample_rand_sets(test_anns_file: str = None, no_img: int = 40) -> list:
    return al_label_transfer.get_random_n_images(
        test_anns_file=test_anns_file, no_img=no_img
    )


def transfer_labels(
    train_anns_file: str = None,
    test_anns_file: str = None,
    image_list: list = None,
    output_path: str = None,
    output_file_tag: str = None,
):
    al_label_transfer.move_labels_to_train(
        image_list=image_list,
        output_path=output_path,
        output_file_tag=output_file_tag,
        train_anns_file=train_anns_file,
        test_anns_file=test_anns_file,
    )


def register_new_labels(
    train_anns_file: str = None,
    test_anns_file: str = None,
    iter_tag: int = None,
):
    d2_mask.register_new_coco_instance(
        annotation_path=train_anns_file,
        data_path=paths.raw_data_path,
        tag=f"train_{iter_tag}",
    )

    d2_mask.register_new_coco_instance(
        annotation_path=test_anns_file,
        data_path=paths.raw_data_path,
        tag=f"test_{iter_tag}",
    )
