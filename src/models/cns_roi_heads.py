"""
@martellweeks: Code taken from https://github.com/vlfom/CSD-detectron2 and modified
"""

import inspect
import os
from typing import Any, Dict, List, Optional, Tuple

import torch
from detectron2.config import configurable
from detectron2.layers import ShapeSpec
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.proposal_generator.proposal_utils import (
    add_ground_truth_to_proposals,
)
from detectron2.modeling.roi_heads import (
    ROI_HEADS_REGISTRY,
    ROIHeads,
    StandardROIHeads,
    select_foreground_proposals,
)
from detectron2.modeling.roi_heads.box_head import build_box_head
from detectron2.modeling.roi_heads.fast_rcnn import FastRCNNOutputLayers
from detectron2.modeling.roi_heads.keypoint_head import build_keypoint_head
from detectron2.modeling.roi_heads.mask_head import build_mask_head
from detectron2.modeling.roi_heads.roi_heads import (
    select_proposals_with_visible_keypoints,
)
from detectron2.structures import Boxes, ImageList, Instances, pairwise_iou
from torch import nn


@ROI_HEADS_REGISTRY.register()
class CNSStandardROIHeads(StandardROIHeads):
    """Extends `StandardROIHeads`'s with support for disabling HNM during training and returning raw predictions.

    Both features are required for the CNS loss. Code is largely taken from `StandardROIHeads`.
    """

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        supervised: Any = True,
    ):
        """Applies RoI using given features and proposals.

        Args:
            supervised: defines the type of forward pass during training, if True - a standard
            forward pass is performed, if False - no GT matching (HNM) is performed, and
            RoI raw predictions are returned

        Returns:
            If `self.training=True`, returns Tuple[Tuple[Tensor, Tensor], Dict], where Dict is a dictionary
            of losses, and a tuple of Tensors are:
            - First tensor: shape (N,K+1), scores for each of the N box. Each row contains the
            scores for K object categories and 1 background class.
            - Second tensor: bounding box regression deltas for each box. Shape is shape (N,Kx4),
            or (N,4) for class-agnostic regression.
            (see `FastRCNNOutputLayers.forward` for more details)

            If `self.training=False`, returns Tuple[List[Instances], Dict], where Instances is a
            list of predicted instances per image, and Dict is an empty dictionary (kept for
            compatibility).

        The code is largely taken from :meth:`StandardROIHeads.forward`. The only modified lines
        are noted by "CNS: ..." comments.
        """
        del images
        if (
            self.training and supervised
        ):  # CNS: if self.supervised = False, we don't need HNM
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            # CNS: get raw predictions along with losses
            predictions, losses = self._forward_box(features, proposals, supervised)
            if (
                supervised
            ):  # CNS: calculate losses only for the standard supervised pass
                # Usually the original proposals used by the box head are used by the mask, keypoint
                # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
                # predicted by the box head.
                losses.update(self._forward_mask(features, proposals))
                losses.update(self._forward_keypoint(features, proposals))
            return predictions, losses  # CNS: return both predictions and losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        supervised: Any = True,
    ):
        """Forward logic of the bbox prediction head.

        The code is taken from :meth:`StandardROIHeads._forward_box`. Additional `supervised` arg is added.
        Look for "CNS: ..." comments to find modified lines.
        """

        features = [features[f] for f in self.box_in_features]
        box_features = self.box_pooler(features, [x.proposal_boxes for x in proposals])
        box_features = self.box_head(box_features)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            if (
                supervised
            ):  # CNS: calculate predictions and losses for standard supervised pass
                losses = self.box_predictor.losses(predictions, proposals)
                # proposals is modified in-place below, so losses must be computed first.
                if self.train_on_pred_boxes:
                    with torch.no_grad():
                        pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                            predictions, proposals
                        )
                        for proposals_per_image, pred_boxes_per_image in zip(
                            proposals, pred_boxes
                        ):
                            proposals_per_image.proposal_boxes = Boxes(
                                pred_boxes_per_image
                            )
            else:  # CNS: for unsupervised CNS passes no losses can exist (we don't have GTs)
                losses = None
            return predictions, losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            return pred_instances


@ROI_HEADS_REGISTRY.register()
class CNSALROIHeads(ROIHeads):
    """
    It's "standard" in a sense that there is no ROI transform sharing
    or feature sharing between tasks.
    Each head independently processes the input features by each head's
    own pooler and head.

    This class is used by most models, such as FPN and C5.
    To implement more models, you can subclass it and implement a different
    :meth:`forward()` or a head.
    """

    @configurable
    def __init__(
        self,
        *,
        box_in_features: List[str],
        box_pooler: ROIPooler,
        box_head: nn.Module,
        box_predictor: nn.Module,
        box_scorer: nn.Module,
        mask_in_features: Optional[List[str]] = None,
        mask_pooler: Optional[ROIPooler] = None,
        mask_head: Optional[nn.Module] = None,
        mask_scorer: Optional[nn.Module] = None,
        keypoint_in_features: Optional[List[str]] = None,
        keypoint_pooler: Optional[ROIPooler] = None,
        keypoint_head: Optional[nn.Module] = None,
        train_on_pred_boxes: bool = False,
        **kwargs,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            box_in_features (list[str]): list of feature names to use for the box head.
            box_pooler (ROIPooler): pooler to extra region features for box head
            box_head (nn.Module): transform features to make box predictions
            box_predictor (nn.Module): make box predictions from the feature.
                Should have the same interface as :class:`FastRCNNOutputLayers`.
            mask_in_features (list[str]): list of feature names to use for the mask
                pooler or mask head. None if not using mask head.
            mask_pooler (ROIPooler): pooler to extract region features from image features.
                The mask head will then take region features to make predictions.
                If None, the mask head will directly take the dict of image features
                defined by `mask_in_features`
            mask_head (nn.Module): transform features to make mask predictions
            keypoint_in_features, keypoint_pooler, keypoint_head: similar to ``mask_*``.
            train_on_pred_boxes (bool): whether to use proposal boxes or
                predicted boxes from the box head to train other heads.
        """
        super().__init__(**kwargs)
        # keep self.in_features for backward compatibility
        self.in_features = self.box_in_features = box_in_features
        self.box_pooler = box_pooler
        self.box_head = box_head
        self.box_predictor = box_predictor
        self.box_scorer = box_scorer

        self.mask_on = mask_in_features is not None
        if self.mask_on:
            self.mask_in_features = mask_in_features
            self.mask_pooler = mask_pooler
            self.mask_head = mask_head
            self.mask_scorer = mask_scorer

        self.keypoint_on = keypoint_in_features is not None
        if self.keypoint_on:
            self.keypoint_in_features = keypoint_in_features
            self.keypoint_pooler = keypoint_pooler
            self.keypoint_head = keypoint_head

        self.train_on_pred_boxes = train_on_pred_boxes

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg)
        ret["train_on_pred_boxes"] = cfg.MODEL.ROI_BOX_HEAD.TRAIN_ON_PRED_BOXES
        # Subclasses that have not been updated to use from_config style construction
        # may have overridden _init_*_head methods. In this case, those overridden methods
        # will not be classmethods and we need to avoid trying to call them here.
        # We test for this with ismethod which only returns True for bound methods of cls.
        # Such subclasses will need to handle calling their overridden _init_*_head methods.
        if inspect.ismethod(cls._init_box_head):
            ret.update(cls._init_box_head(cfg, input_shape))
        if inspect.ismethod(cls._init_mask_head):
            ret.update(cls._init_mask_head(cfg, input_shape))
        if inspect.ismethod(cls._init_keypoint_head):
            ret.update(cls._init_keypoint_head(cfg, input_shape))
        return ret

    @classmethod
    def _init_box_head(cls, cfg, input_shape):
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_BOX_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_BOX_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_BOX_HEAD.POOLER_TYPE
        # fmt: on

        # If StandardROIHeads is applied on multiple feature maps (as in FPN),
        # then we share the same predictors and therefore the channel counts must be the same
        in_channels = [input_shape[f].channels for f in in_features]
        # Check all channel counts are equal
        assert len(set(in_channels)) == 1, in_channels
        in_channels = in_channels[0]

        box_pooler = ROIPooler(
            output_size=pooler_resolution,
            scales=pooler_scales,
            sampling_ratio=sampling_ratio,
            pooler_type=pooler_type,
        )
        # Here we split "box head" and "box predictor", which is mainly due to historical reasons.
        # They are used together so the "box predictor" layers should be part of the "box head".
        # New subclasses of ROIHeads do not need "box predictor"s.
        box_head = build_box_head(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )
        box_predictor = FastRCNNOutputLayers(cfg, box_head.output_shape)
        box_scorer = BoxScorePredictionLayers(
            cfg,
            ShapeSpec(
                channels=in_channels, height=pooler_resolution, width=pooler_resolution
            ),
        )
        return {
            "box_in_features": in_features,
            "box_pooler": box_pooler,
            "box_head": box_head,
            "box_predictor": box_predictor,
            "box_scorer": box_scorer,
        }

    @classmethod
    def _init_mask_head(cls, cfg, input_shape):
        if not cfg.MODEL.MASK_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)
        sampling_ratio    = cfg.MODEL.ROI_MASK_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_MASK_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"mask_in_features": in_features}
        ret["mask_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["mask_head"] = build_mask_head(cfg, shape)
        ret["mask_scorer"] = MaskScorePredictionLayers(cfg, input_shape=shape)
        return ret

    @classmethod
    def _init_keypoint_head(cls, cfg, input_shape):
        if not cfg.MODEL.KEYPOINT_ON:
            return {}
        # fmt: off
        in_features       = cfg.MODEL.ROI_HEADS.IN_FEATURES
        pooler_resolution = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_RESOLUTION
        pooler_scales     = tuple(1.0 / input_shape[k].stride for k in in_features)  # noqa
        sampling_ratio    = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_SAMPLING_RATIO
        pooler_type       = cfg.MODEL.ROI_KEYPOINT_HEAD.POOLER_TYPE
        # fmt: on

        in_channels = [input_shape[f].channels for f in in_features][0]

        ret = {"keypoint_in_features": in_features}
        ret["keypoint_pooler"] = (
            ROIPooler(
                output_size=pooler_resolution,
                scales=pooler_scales,
                sampling_ratio=sampling_ratio,
                pooler_type=pooler_type,
            )
            if pooler_type
            else None
        )
        if pooler_type:
            shape = ShapeSpec(
                channels=in_channels, width=pooler_resolution, height=pooler_resolution
            )
        else:
            shape = {f: input_shape[f] for f in in_features}
        ret["keypoint_head"] = build_keypoint_head(cfg, shape)
        return ret

    def forward(
        self,
        images: ImageList,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        targets: Optional[List[Instances]] = None,
        supervised: Any = True,
    ) -> Tuple[List[Instances], Dict[str, torch.Tensor]]:
        """
        See :class:`ROIHeads.forward`.
        """
        del images
        if self.training and supervised:
            assert targets, "'targets' argument is required during training"
            proposals = self.label_and_sample_proposals(proposals, targets)
        del targets

        if self.training:
            predictions, losses = self._forward_box(features, proposals, supervised)
            # Usually the original proposals used by the box head are used by the mask, keypoint
            # heads. But when `self.train_on_pred_boxes is True`, proposals will contain boxes
            # predicted by the box head.
            if supervised:
                losses.update(self._forward_mask(features, proposals))
                losses.update(self._forward_keypoint(features, proposals))
            return predictions, losses
        else:
            pred_instances = self._forward_box(features, proposals)
            # During inference cascaded prediction is used: the mask and keypoints heads are only
            # applied to the top scoring box detections.
            pred_instances = self.forward_with_given_boxes(features, pred_instances)
            return pred_instances, {}

    def forward_with_given_boxes(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ) -> List[Instances]:
        """
        Use the given boxes in `instances` to produce other (non-box) per-ROI outputs.

        This is useful for downstream tasks where a box is known, but need to obtain
        other attributes (outputs of other heads).
        Test-time augmentation also uses this.

        Args:
            features: same as in `forward()`
            instances (list[Instances]): instances to predict other outputs. Expect the keys
                "pred_boxes" and "pred_classes" to exist.

        Returns:
            list[Instances]:
                the same `Instances` objects, with extra
                fields such as `pred_masks` or `pred_keypoints`.
        """
        assert not self.training
        assert instances[0].has("pred_boxes") and instances[0].has("pred_classes")

        instances = self._forward_mask(features, instances)
        instances = self._forward_keypoint(features, instances)
        return instances

    def _forward_box(
        self,
        features: Dict[str, torch.Tensor],
        proposals: List[Instances],
        supervised: Any = True,
    ):
        """
        Forward logic of the box prediction branch. If `self.train_on_pred_boxes is True`,
            the function puts predicted boxes in the `proposal_boxes` field of `proposals` argument.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            proposals (list[Instances]): the per-image object proposals with
                their matching ground truth.
                Each has fields "proposal_boxes", and "objectness_logits",
                "gt_classes", "gt_boxes".

        Returns:
            In training, a dict of losses.
            In inference, a list of `Instances`, the predicted instances.
        """
        features = [features[f] for f in self.box_in_features]
        box_features_after_pool = self.box_pooler(
            features, [x.proposal_boxes for x in proposals]
        )
        box_features = self.box_head(box_features_after_pool)
        predictions = self.box_predictor(box_features)
        del box_features

        if self.training:
            if supervised:
                losses = self.box_predictor.losses(predictions, proposals)
                # proposals is modified in-place below, so losses must be computed first.
                if self.train_on_pred_boxes:
                    with torch.no_grad():
                        pred_boxes = self.box_predictor.predict_boxes_for_gt_classes(
                            predictions, proposals
                        )
                        for proposals_per_image, pred_boxes_per_image in zip(
                            proposals, pred_boxes
                        ):
                            proposals_per_image.proposal_boxes = Boxes(
                                pred_boxes_per_image
                            )
                score_predictions = self.box_scorer(
                    box_features_after_pool, self.training
                )
                del box_features_after_pool
                losses.update(
                    self.box_scorer.losses(
                        score_predictions, 10 * losses.get("loss_box_reg")
                    )
                )
            else:  # CNS: for unsupervised CNS passes no losses can exist (we don't have GTs)
                losses = None
            return predictions, losses
        else:
            pred_instances, _ = self.box_predictor.inference(predictions, proposals)
            score_predictions = self.box_scorer(box_features_after_pool, self.training)
            for instance in pred_instances:
                instance.set(
                    "box_score_prediction", score_predictions.expand(len(instance))
                )
            return pred_instances

    def _forward_mask(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ):
        """
        Forward logic of the mask prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict masks.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_masks" and return it.
        """
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals.
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            features = [features[f] for f in self.mask_in_features]
            boxes = [
                x.proposal_boxes if self.training else x.pred_boxes for x in instances
            ]
            features = self.mask_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.mask_in_features}

        if self.training:
            loss = self.mask_head(features, instances)
            score_predictions = self.mask_scorer(features)
            loss.update(
                self.mask_scorer.losses(score_predictions, 10 * loss.get("loss_mask"))
            )
            return loss
        else:
            instances = self.mask_head(features, instances)
            score_predictions = self.mask_scorer(features)
            for instance in instances:
                instance.set(
                    "mask_score_prediction", score_predictions.expand(len(instance))
                )
            return instances

    def _forward_keypoint(
        self, features: Dict[str, torch.Tensor], instances: List[Instances]
    ):
        """
        Forward logic of the keypoint prediction branch.

        Args:
            features (dict[str, Tensor]): mapping from feature map names to tensor.
                Same as in :meth:`ROIHeads.forward`.
            instances (list[Instances]): the per-image instances to train/predict keypoints.
                In training, they can be the proposals.
                In inference, they can be the boxes predicted by R-CNN box head.

        Returns:
            In training, a dict of losses.
            In inference, update `instances` with new fields "pred_keypoints" and return it.
        """
        if not self.keypoint_on:
            return {} if self.training else instances

        if self.training:
            # head is only trained on positive proposals with >=1 visible keypoints.
            instances, _ = select_foreground_proposals(instances, self.num_classes)
            instances = select_proposals_with_visible_keypoints(instances)

        if self.keypoint_pooler is not None:
            features = [features[f] for f in self.keypoint_in_features]
            boxes = [
                x.proposal_boxes if self.training else x.pred_boxes for x in instances
            ]
            features = self.keypoint_pooler(features, boxes)
        else:
            features = {f: features[f] for f in self.keypoint_in_features}
        return self.keypoint_head(features, instances)


class BoxScorePredictionLayers(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.conv1 = nn.Conv2d(
            input_shape.channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten(start_dim=0)
        self.fc1 = nn.Linear(64 * input_shape.height * input_shape.width, 1)
        self.fc2 = nn.Linear(1280, 1)

    def forward(self, features, training: bool = True):
        res = self.conv1(features)
        res = self.relu(res)
        res = self.flatten1(res)
        res = self.fc1(res)
        res = self.relu(res)
        res = self.flatten2(res)
        if res.shape[0] != 1280:
            res = nn.functional.pad(res, pad=(0, 1280 - res.shape[0]))
        res = self.fc2(res)
        return res

    def losses(self, predictions, proposals):
        loss = nn.functional.mse_loss(
            predictions, torch.tensor([proposals], device="cuda:0")
        )
        return {"loss_box_score": loss}

    def predict(self, features):
        return self.forward(features)


class MaskScorePredictionLayers(nn.Module):
    def __init__(self, cfg, input_shape: ShapeSpec, **kwargs):
        super().__init__(**kwargs)
        self.input_shape = input_shape
        self.conv1 = nn.Conv2d(
            input_shape.channels,
            64,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.conv2 = nn.Conv2d(
            64,
            32,
            kernel_size=3,
            stride=1,
            padding=1,
        )
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.batchnorm = nn.BatchNorm2d(num_features=64)
        self.dropout = nn.Dropout()
        self.flatten1 = nn.Flatten()
        self.flatten2 = nn.Flatten(start_dim=0)
        self.fc1 = nn.Linear(
            32 * int(input_shape.height / 2) * int(input_shape.width / 2), 1
        )
        self.fc2 = nn.Linear(128, 1)

    def forward(self, features):
        res = self.conv1(features)
        res = self.relu(res)
        res = self.pool(res)
        res = self.batchnorm(res)
        res = self.conv2(res)
        res = self.relu(res)
        res = self.flatten1(res)
        res = self.fc1(res)
        res = self.relu(res)
        res = self.flatten2(res)
        if res.shape[0] != 128:
            res = nn.functional.pad(res, pad=(0, 128 - res.shape[0]))
        res = self.dropout(res)
        res = self.fc2(res)
        return res

    def losses(self, predictions, proposals):
        loss = nn.functional.huber_loss(
            predictions, torch.tensor([proposals], device="cuda:0")
        )
        return {"loss_mask_score": loss}
