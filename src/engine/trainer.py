import logging
import os
import time
import weakref
from typing import Dict

import numpy as np
import torch
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data import DatasetMapper, MetadataCatalog, build_detection_test_loader
from detectron2.engine import (
    DefaultTrainer,
    SimpleTrainer,
    TrainerBase,
    create_ddp_model,
)
from detectron2.evaluation import COCOEvaluator, PascalVOCDetectionEvaluator
from detectron2.utils import comm
from detectron2.utils.events import (
    CommonMetricPrinter,
    JSONWriter,
    TensorboardXWriter,
    get_event_storage,
)
from detectron2.utils.logger import setup_logger

from src.data.build import build_ss_train_loader
from src.data.mapper import CNSDatasetMapper, TestDatasetMapper
from src.engine.hooks import LossEvalHook


class BaseTrainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, cfg, True, output_folder)

    def build_hooks(self):
        hooks = super().build_hooks()
        hooks.insert(
            -1,
            LossEvalHook(
                self.cfg.TEST.EVAL_PERIOD,
                self.model,
                build_detection_test_loader(
                    self.cfg, self.cfg.DATASETS.TEST[0], DatasetMapper(self.cfg, True)
                ),
            ),
        )
        return hooks


class CNSTrainerManager(DefaultTrainer):
    """A trainer manager for the semi-supervised learning task using consistency loss based on x-flips.

    Modifications are minimal comparing to D2's `DefaultTrainer`, so see its documentation for more
    details. The only differences are injection of a different trainer `CNSTrainer` along with weight scheduling
    parameters, and a CNS-specific semi-supervised data loader defined in `build_train_loader`.


    @martellweeks: Code taken from https://github.com/vlfom/CSD-detectron2 and modified
    """

    def __init__(self, cfg):
        """Initializes the CNSTrainer.

        Most of the code is from `super.__init__()`, the only change is that for `self._trainer`
        the `CNSTrainer` is used and weight scheduling parameters are injected into it, look for
        "CNS: ... " comments.
        """
        TrainerBase.__init__(
            self
        )  # CNS: don't call `super`'s init as we are overriding it
        logger = logging.getLogger("detectron2")
        if not logger.isEnabledFor(logging.INFO):  # setup_logger is not called for d2
            setup_logger()
        cfg = DefaultTrainer.auto_scale_workers(cfg, comm.get_world_size())

        # Assume these objects must be constructed in this order.
        model = self.build_model(cfg)
        optimizer = self.build_optimizer(cfg, model)
        data_loader = self.build_train_loader(cfg)

        model = create_ddp_model(model, broadcast_buffers=False)
        self._trainer = CNSTrainer(
            model, data_loader, optimizer
        )  # CNS: use a CNS-specific trainer
        # CNS: inject weight scheduling parameters into trainer
        (
            self._trainer.solver_cns_beta,
            self._trainer.solver_cns_t0,
            self._trainer.solver_cns_t1,
            self._trainer.solver_cns_t2,
            self._trainer.solver_cns_t,
            self._trainer.solver_train_al_scoring_modules,
        ) = (
            cfg.SOLVER.CNS_WEIGHT_SCHEDULE_RAMP_BETA,
            cfg.SOLVER.CNS_WEIGHT_SCHEDULE_RAMP_T0,
            cfg.SOLVER.CNS_WEIGHT_SCHEDULE_RAMP_T1,
            cfg.SOLVER.CNS_WEIGHT_SCHEDULE_RAMP_T2,
            cfg.SOLVER.CNS_WEIGHT_SCHEDULE_RAMP_T,
            cfg.SOLVER.TRAIN_AL_SCORING_MODULES,
        )

        self.scheduler = self.build_lr_scheduler(cfg, optimizer)
        self.start_iter = 0
        self.max_iter = cfg.SOLVER.MAX_ITER
        self.cfg = cfg
        self.checkpointer = DetectionCheckpointer(
            # Assume you want to save checkpoints together with logs/statistics
            model,
            cfg.OUTPUT_DIR,
            trainer=weakref.proxy(self),
        )

        self.register_hooks(self.build_hooks())

    def build_train_loader(self, cfg):
        """Defines a data loader to use in the training loop."""
        dataset_mapper = CNSDatasetMapper(cfg, True)

        # In addition to dataloader, fetch a list of labeled and unlabeled dicts
        (labeled_dicts, unlabeled_dicts), train_loader = build_ss_train_loader(
            cfg, dataset_mapper
        )

        # If in random_split mode, store the filenames of images for future reference
        if cfg.DATASETS.MODE == "RANDOM_SPLIT":
            labeled_ids = [d["image_id"] for d in labeled_dicts]
            unlabeled_ids = [d["image_id"] for d in unlabeled_dicts]
            self._data_split_ids = (labeled_ids, unlabeled_ids)

        return train_loader

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        """Defines a data loader to use in the testing loop."""
        dataset_mapper = TestDatasetMapper(cfg, False)
        return build_detection_test_loader(cfg, dataset_name, mapper=dataset_mapper)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        """Create evaluator(s) for a given dataset.

        Modified from D2's example `tools/train_net.py`"""
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")

        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type

        if evaluator_type == "coco":
            return COCOEvaluator(dataset_name, output_dir=output_folder)
        if evaluator_type == "pascal_voc":
            return PascalVOCDetectionEvaluator(dataset_name)

        raise NotImplementedError(
            "No evaluator implementation for the dataset {} with the type {}".format(
                dataset_name, evaluator_type
            )
        )


class CNSTrainer(SimpleTrainer):
    """The actual trainer that runs the forward and backward passes

    @martellweeks: Code taken from https://github.com/vlfom/CSD-detectron2 and modified
    """

    def run_step(self):
        """Implements a training iteration for the CNS method."""

        assert self.model.training, "The model must be in the training mode"

        # Get a tuple of labeled and unlabeled instances (with their x-flipped versions)
        # Format: ([labeled_img, labeled_img_xflip], [unlabeled_im, unlabeled_img_xflip])
        # where first list (batch) is of size `cfg.SOLVER.IMS_PER_BATCH_LABELED` and the latter
        # is of size `cfg.SOLVER.IMS_PER_BATCH_UNLABELED`
        start = time.perf_counter()
        data_labeled, data_unlabeled = next(self._data_loader_iter)
        data_time = time.perf_counter() - start

        # A boolean that indicates whether CNS loss should be calculated at this iteration, see
        # `config.SOLVER.CNS_WEIGHT_SCHEDULE_RAMP_T0`
        use_cns = self.iter >= self.solver_cns_t0
        self.solver_cns_loss_weight = 0
        if use_cns:
            self._update_cns_loss_weight()  # Call CNS weight scheduling (could be a hook though)

        # Get losses, format (from :meth:`CNSGeneralizedRCNN.forward`):
        # - "loss_cls", "loss_rpn_cls": bbox roi and rpn classification loss
        # - "loss_box_reg", "loss_rpn_loc": bbox roi and rpn localization loss (see :meth:`FastRCNNOutputLayers.losses`)
        # - "sup_cns_loss_cls": CNS consistency loss for classification on labeled data
        # - "sup_cns_loss_box_reg": CNS consistency loss for localization on labeled data
        # - "unsup_cns_loss_cls", "unsup_cns_loss_box_reg": CNS losses on unlabeled data
        loss_dict = self.model(data_labeled, data_unlabeled, use_cns=use_cns)

        losses_sup = (  # Sum up the supervised losses
            loss_dict["loss_rpn_cls"]
            + loss_dict["loss_rpn_loc"]
            + loss_dict["loss_cls"]
            + loss_dict["loss_box_reg"]
            + loss_dict["loss_mask"]
        )
        losses_al = loss_dict["loss_box_score"] + loss_dict["loss_mask_score"]
        losses_cns = (  # Sum up the CNS losses
            loss_dict["sup_cns_loss_cls"]
            + loss_dict["sup_cns_loss_box_reg"]
            + loss_dict["unsup_cns_loss_cls"]
            + loss_dict["unsup_cns_loss_box_reg"]
        )
        losses = (
            losses_sup + self.solver_cns_loss_weight * losses_cns
        )  # Calculate the total loss
        if self.solver_train_al_scoring_modules:
            losses += losses_al

        self.optimizer.zero_grad()
        losses.backward()

        # Log metrics
        self._write_metrics(
            loss_dict, data_time, train_al=self.solver_train_al_scoring_modules
        )

        # Backprop
        self.optimizer.step()

    def _update_cns_loss_weight(self):
        """Controls weight scheduling for the CNS loss: updates the weight coefficient at each iteration.

        See CNS paper abstract for more details.
        """

        if self.iter < self.solver_cns_t0:
            self.solver_cns_loss_weight = 0
        elif self.iter < self.solver_cns_t1:
            self.solver_cns_loss_weight = (
                np.exp(
                    -5
                    * np.power(
                        (
                            1
                            - (self.iter - self.solver_cns_t0)
                            / (self.solver_cns_t1 - self.solver_cns_t0)
                        ),
                        2,
                    )
                )
                * self.solver_cns_beta
            )
        elif self.iter < self.solver_cns_t2:
            self.solver_cns_loss_weight = self.solver_cns_beta
        else:
            self.solver_cns_loss_weight = (
                np.exp(
                    -12.5
                    * np.power(
                        (
                            1
                            - (self.solver_cns_t - self.iter)
                            / (self.solver_cns_t - self.solver_cns_t2)
                        ),
                        2,
                    )
                )
                * self.solver_cns_beta
            )

    def _write_metrics(
        self,
        loss_dict: Dict[str, torch.Tensor],
        data_time: float,
        prefix: str = "",
        train_al: bool = True,
    ):
        """
        Args:
            loss_dict (dict): dict of scalar losses
            data_time (float): time taken by the dataloader iteration
        """
        metrics_dict = {k: v.detach().cpu().item() for k, v in loss_dict.items()}
        metrics_dict["data_time"] = data_time

        # Gather metrics among all workers for logging
        # This assumes we do DDP-style training, which is currently the only
        # supported method in detectron2.
        all_metrics_dict = comm.gather(metrics_dict)

        if comm.is_main_process():
            storage = get_event_storage()

            # data_time among workers can have high variance. The actual latency
            # caused by data_time is the maximum among workers.
            data_time = np.max([x.pop("data_time") for x in all_metrics_dict])
            storage.put_scalar("data_time", data_time)

            # average the rest metrics
            metrics_dict = {
                k: np.mean([x[k] for x in all_metrics_dict])
                for k in all_metrics_dict[0].keys()
            }
            total_sup_loss = sum(
                metrics_dict[k]
                for k in ["loss_rpn_cls", "loss_rpn_loc", "loss_cls", "loss_box_reg"]
            )
            if not np.isfinite(total_sup_loss):
                raise FloatingPointError(
                    f"Loss became infinite or NaN at iteration={self.iter}!\n"
                    f"loss_dict = {metrics_dict}"
                )

            storage.put_scalar("{}total_loss".format(prefix), total_sup_loss)
            if len(metrics_dict) > 1:
                storage.put_scalars(**metrics_dict)

            storage.put_scalar(
                "cns_weight", self.solver_cns_loss_weight
            )  # CNS loss weight

            # Store aggregates
            cns_loss = (
                metrics_dict["sup_cns_loss_cls"]
                + metrics_dict["sup_cns_loss_box_reg"]
                + metrics_dict["unsup_cns_loss_cls"]
                + metrics_dict["unsup_cns_loss_box_reg"]
            ) * self.solver_cns_loss_weight
            storage.put_scalar("total_cns_loss", cns_loss)  # Sum of the CNS losses
            al_loss = 0
            if train_al:
                al_loss = (
                    metrics_dict["loss_box_score"] + metrics_dict["loss_mask_score"]
                )
                storage.put_scalar("total_score_loss", al_loss)
            storage.put_scalar(  # Sum of all losses
                "total_all_loss",
                total_sup_loss + cns_loss + al_loss,
            )
