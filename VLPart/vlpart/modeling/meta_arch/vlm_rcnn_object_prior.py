# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict, List, Optional

import torch
from torch import nn
import torch.nn.functional as F
from torch.cuda.amp import autocast

from detectron2.config import configurable
from detectron2.modeling import META_ARCH_REGISTRY
from detectron2.modeling.poolers import ROIPooler
from detectron2.structures import ImageList, Instances

from .vlm_rcnn import VLMRCNN


@META_ARCH_REGISTRY.register()
class VLMRCNNObjectPrior(VLMRCNN):
    @configurable
    def __init__(
        self,
        *,
        object_prior_enabled=False,
        object_prior_mode="none",
        object_prior_fpn=False,
        object_prior_rpn_roi=False,
        object_prior_fpn_alpha_init=0.0,
        object_prior_rpn_roi_beta_init=0.1,
        object_prior_rpn_rerank=True,
        object_prior_fpn_features=(),
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.object_prior_enabled = object_prior_enabled
        self.object_prior_mode = object_prior_mode
        self.object_prior_fpn = object_prior_fpn
        self.object_prior_rpn_roi = object_prior_rpn_roi
        self.object_prior_rpn_rerank = object_prior_rpn_rerank
        self.object_prior_fpn_features = list(object_prior_fpn_features)

        if self.object_prior_enabled and self.object_prior_fpn:
            self.object_prior_fpn_alphas = nn.Parameter(
                torch.full(
                    (len(self.object_prior_fpn_features),),
                    float(object_prior_fpn_alpha_init),
                    dtype=torch.float32,
                )
            )
        else:
            self.register_parameter("object_prior_fpn_alphas", None)

        if self.object_prior_enabled and self.object_prior_rpn_roi:
            self.object_prior_rpn_beta = nn.Parameter(
                torch.tensor(float(object_prior_rpn_roi_beta_init), dtype=torch.float32)
            )
            self.object_prior_score_pooler = ROIPooler(
                output_size=7,
                scales=(1.0,),
                sampling_ratio=0,
                pooler_type="ROIAlignV2",
            )
        else:
            self.register_parameter("object_prior_rpn_beta", None)
            self.object_prior_score_pooler = None

    @classmethod
    def from_config(cls, cfg):
        ret = super().from_config(cfg)
        object_prior_cfg = cfg.MODEL.OBJECT_PRIOR
        ret.update(
            {
                "object_prior_enabled": object_prior_cfg.ENABLED,
                "object_prior_mode": object_prior_cfg.MODE,
                "object_prior_fpn": object_prior_cfg.FPN,
                "object_prior_rpn_roi": object_prior_cfg.RPN_ROI,
                "object_prior_fpn_alpha_init": object_prior_cfg.FPN_ALPHA_INIT,
                "object_prior_rpn_roi_beta_init": object_prior_cfg.RPN_ROI_BETA_INIT,
                "object_prior_rpn_rerank": object_prior_cfg.RPN_RERANK,
                "object_prior_fpn_features": tuple(cfg.MODEL.RPN.IN_FEATURES),
            }
        )
        return ret

    def preprocess_object_prior(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.object_prior_enabled:
            return None
        if self.object_prior_mode != "gt":
            raise ValueError(
                f"Unsupported MODEL.OBJECT_PRIOR.MODE='{self.object_prior_mode}'. "
                "This experiment only implements MODE='gt'."
            )
        if any("object_prior" not in x for x in batched_inputs):
            raise KeyError("MODEL.OBJECT_PRIOR.ENABLED=True but batched input misses object_prior")

        priors = []
        for x in batched_inputs:
            prior = self._move_to_current_device(x["object_prior"]).float().clamp_(0.0, 1.0)
            if prior.dim() == 2:
                prior = prior.unsqueeze(0)
            if prior.shape[0] != 1:
                raise ValueError(f"object_prior must have shape [1,H,W], got {tuple(prior.shape)}")
            priors.append(prior)

        return ImageList.from_tensors(
            priors,
            self.backbone.size_divisibility,
            padding_constraints=self.backbone.padding_constraints,
        )

    def apply_object_prior_to_fpn(self, features, object_priors):
        if (
            not self.object_prior_enabled
            or not self.object_prior_fpn
            or object_priors is None
            or self.object_prior_fpn_alphas is None
        ):
            return features

        guided = dict(features)
        prior_tensor = object_priors.tensor
        for idx, name in enumerate(self.object_prior_fpn_features):
            if name not in guided:
                continue
            feature = guided[name]
            prior = F.interpolate(
                prior_tensor.to(dtype=feature.dtype),
                size=feature.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            alpha = self.object_prior_fpn_alphas[idx].to(dtype=feature.dtype)
            guided[name] = feature * (1.0 + alpha.view(1, 1, 1, 1) * prior)
        return guided

    def apply_object_prior_to_proposals(self, proposals, object_priors):
        if (
            not self.object_prior_enabled
            or not self.object_prior_rpn_roi
            or object_priors is None
            or self.object_prior_score_pooler is None
            or len(proposals) == 0
        ):
            return proposals

        boxes = [p.proposal_boxes for p in proposals]
        num_boxes = [len(p) for p in proposals]
        if sum(num_boxes) == 0:
            return proposals

        pooled = self.object_prior_score_pooler([object_priors.tensor.float()], boxes)
        prior_scores = pooled.mean(dim=(1, 2, 3)).split(num_boxes, dim=0)
        beta = self.object_prior_rpn_beta
        guided_proposals = []
        for proposals_per_image, scores_per_image in zip(proposals, prior_scores):
            if len(proposals_per_image) == 0:
                guided_proposals.append(proposals_per_image)
                continue

            scores_per_image = scores_per_image.to(
                device=proposals_per_image.proposal_boxes.tensor.device,
                dtype=proposals_per_image.proposal_boxes.tensor.dtype,
            )
            proposals_per_image.object_prior_scores = scores_per_image
            if proposals_per_image.has("objectness_logits"):
                proposals_per_image.objectness_logits = (
                    proposals_per_image.objectness_logits
                    + beta.to(dtype=proposals_per_image.objectness_logits.dtype) * scores_per_image
                )
                if self.object_prior_rpn_rerank:
                    order = torch.argsort(proposals_per_image.objectness_logits, descending=True)
                    proposals_per_image = proposals_per_image[order]
            elif proposals_per_image.has("scores"):
                proposals_per_image.scores = (
                    proposals_per_image.scores
                    + beta.to(dtype=proposals_per_image.scores.dtype) * scores_per_image
                )
                if self.object_prior_rpn_rerank:
                    order = torch.argsort(proposals_per_image.scores, descending=True)
                    proposals_per_image = proposals_per_image[order]
            guided_proposals.append(proposals_per_image)
        return guided_proposals

    def forward(self, batched_inputs: List[Dict[str, torch.Tensor]]):
        if not self.training:
            return self.inference(batched_inputs)

        gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
        for inst, x in zip(gt_instances, batched_inputs):
            inst._ann_type = x["ann_type"] if "ann_type" in x else "box"
            if "pos_category_ids" in x and len(x["pos_category_ids"]) > 0:
                inst._pos_category_ids = x["pos_category_ids"]
            else:
                inst._pos_category_ids = inst.gt_classes.unique()
            inst._dataset_source = x["dataset_source"] if "dataset_source" in x else 0
        ann_types = [inst._ann_type for inst in gt_instances]
        assert len(set(ann_types)) == 1
        ann_type = ann_types[0]

        dataset_sources = [inst._dataset_source for inst in gt_instances]
        assert len(set(dataset_sources)) == 1
        dataset_source = dataset_sources[0]

        images = self.preprocess_image(batched_inputs)
        object_priors = self.preprocess_object_prior(batched_inputs)

        if self.fp16:
            with autocast():
                features = self.backbone(images.tensor.half())
            features = {k: v.float() for k, v in features.items()}
        else:
            features = self.backbone(images.tensor)
        features = self.apply_object_prior_to_fpn(features, object_priors)

        proposals, proposal_losses = self.proposal_generator(images, features, gt_instances)
        proposals = self.apply_object_prior_to_proposals(proposals, object_priors)

        proposals, detector_losses = self.roi_heads(
            images,
            features,
            proposals,
            gt_instances,
            ann_type=ann_type,
            dataset_source=dataset_source,
            object_priors=object_priors,
        )

        losses = {}
        losses.update(detector_losses)
        if ann_type in ["box", "part"]:
            losses.update(proposal_losses)
        else:
            losses.update({k: v * 0 for k, v in proposal_losses.items()})

        return losses

    def inference(
        self,
        batched_inputs: List[Dict[str, torch.Tensor]],
        detected_instances: Optional[List[Instances]] = None,
        do_postprocess: bool = True,
    ):
        assert not self.training

        images = self.preprocess_image(batched_inputs)
        object_priors = self.preprocess_object_prior(batched_inputs)
        features = self.backbone(images.tensor)
        features = self.apply_object_prior_to_fpn(features, object_priors)
        proposals, _ = self.proposal_generator(images, features)
        proposals = self.apply_object_prior_to_proposals(proposals, object_priors)

        if self.eval_proposal:
            results = self.proposals_to_instances(proposals)
        else:
            results, _ = self.roi_heads(images, features, proposals, object_priors=object_priors)

        if do_postprocess:
            max_shape = images.tensor.shape[2:]
            return VLMRCNN._postprocess(results, batched_inputs, images.image_sizes, max_shape)
        return results
