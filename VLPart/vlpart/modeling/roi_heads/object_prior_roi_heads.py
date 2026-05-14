# Copyright (c) Facebook, Inc. and its affiliates.
import torch

from detectron2.config import configurable
from detectron2.modeling import ROI_HEADS_REGISTRY
from detectron2.modeling.poolers import ROIPooler
from detectron2.modeling.roi_heads.cascade_rcnn import _ScaleGradient
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference
from detectron2.modeling.roi_heads.roi_heads import select_foreground_proposals

from .roi_heads_vlm import CascadeVLMROIHeads, VLMROIHeads


class ObjectPriorROIHeadsMixin:
    def _init_object_prior(
        self,
        *,
        object_prior_enabled=False,
        object_prior_rpn_roi=False,
        object_prior_mask_head=False,
        object_prior_roi_beta_init=0.1,
        object_prior_mask_alpha_init=0.0,
        object_prior_score_size=7,
        object_prior_mask_size=14,
    ):
        self.object_prior_enabled = object_prior_enabled
        self.object_prior_rpn_roi = object_prior_rpn_roi
        self.object_prior_mask_head = object_prior_mask_head

        if self.object_prior_enabled and self.object_prior_rpn_roi:
            self.object_prior_roi_beta = torch.nn.Parameter(
                torch.tensor(float(object_prior_roi_beta_init), dtype=torch.float32)
            )
            self.object_prior_score_pooler = ROIPooler(
                output_size=object_prior_score_size,
                scales=(1.0,),
                sampling_ratio=0,
                pooler_type="ROIAlignV2",
            )
        else:
            self.register_parameter("object_prior_roi_beta", None)
            self.object_prior_score_pooler = None

        if self.object_prior_enabled and self.object_prior_mask_head:
            self.object_prior_mask_alpha = torch.nn.Parameter(
                torch.tensor(float(object_prior_mask_alpha_init), dtype=torch.float32)
            )
            self.object_prior_mask_pooler = ROIPooler(
                output_size=object_prior_mask_size,
                scales=(1.0,),
                sampling_ratio=0,
                pooler_type="ROIAlignV2",
            )
        else:
            self.register_parameter("object_prior_mask_alpha", None)
            self.object_prior_mask_pooler = None

    @classmethod
    def _object_prior_from_config(cls, cfg):
        object_prior_cfg = cfg.MODEL.OBJECT_PRIOR
        return {
            "object_prior_enabled": object_prior_cfg.ENABLED,
            "object_prior_rpn_roi": object_prior_cfg.RPN_ROI,
            "object_prior_mask_head": object_prior_cfg.MASK_HEAD,
            "object_prior_roi_beta_init": object_prior_cfg.RPN_ROI_BETA_INIT,
            "object_prior_mask_alpha_init": object_prior_cfg.MASK_ALPHA_INIT,
            "object_prior_score_size": 7,
            "object_prior_mask_size": cfg.MODEL.ROI_MASK_HEAD.POOLER_RESOLUTION,
        }

    def _pool_prior_scores(self, object_priors, boxes):
        if (
            not self.object_prior_enabled
            or not self.object_prior_rpn_roi
            or object_priors is None
            or self.object_prior_score_pooler is None
        ):
            return None
        num_boxes = [len(x) for x in boxes]
        if sum(num_boxes) == 0:
            return object_priors.tensor.new_zeros((0,))
        pooled = self.object_prior_score_pooler([object_priors.tensor.float()], boxes)
        return pooled.mean(dim=(1, 2, 3))

    def _apply_prior_to_logits(self, predictions, proposals, object_priors):
        prior_scores = self._pool_prior_scores(object_priors, [x.proposal_boxes for x in proposals])
        if prior_scores is None or prior_scores.numel() == 0:
            return predictions

        cls_scores, proposal_deltas = predictions
        prior_scores = prior_scores.to(device=cls_scores.device, dtype=cls_scores.dtype)
        beta = self.object_prior_roi_beta.to(dtype=cls_scores.dtype)
        cls_scores = cls_scores.clone()
        if cls_scores.shape[1] > 1:
            cls_scores[:, :-1] = cls_scores[:, :-1] + beta * prior_scores[:, None]
        else:
            cls_scores = cls_scores + beta * prior_scores[:, None]
        return cls_scores, proposal_deltas

    def _apply_prior_to_mask_features(self, mask_features, object_priors, boxes):
        if (
            not self.object_prior_enabled
            or not self.object_prior_mask_head
            or object_priors is None
            or self.object_prior_mask_pooler is None
            or mask_features.numel() == 0
        ):
            return mask_features
        prior_roi = self.object_prior_mask_pooler([object_priors.tensor.float()], boxes)
        prior_roi = prior_roi.to(device=mask_features.device, dtype=mask_features.dtype)
        alpha = self.object_prior_mask_alpha.to(dtype=mask_features.dtype)
        return mask_features * (1.0 + alpha.view(1, 1, 1, 1) * prior_roi)

    def _forward_mask(self, features, instances, object_priors=None):
        if not self.mask_on:
            return {} if self.training else instances

        if self.training:
            instances, _ = select_foreground_proposals(instances, self.num_classes)

        if self.mask_pooler is not None:
            mask_features = [features[f] for f in self.mask_in_features]
            boxes = [x.proposal_boxes if self.training else x.pred_boxes for x in instances]
            mask_features = self.mask_pooler(mask_features, boxes)
            mask_features = self._apply_prior_to_mask_features(mask_features, object_priors, boxes)
        else:
            mask_features = {f: features[f] for f in self.mask_in_features}
        return self.mask_head(mask_features, instances)


@ROI_HEADS_REGISTRY.register()
class VLMROIHeadsObjectPrior(ObjectPriorROIHeadsMixin, VLMROIHeads):
    @configurable
    def __init__(self, *, object_prior_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self._init_object_prior(**(object_prior_kwargs or {}))

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["object_prior_kwargs"] = cls._object_prior_from_config(cfg)
        return ret

    def forward(
        self,
        images,
        img_features,
        proposals,
        targets=None,
        ann_type="box",
        classifier_info=(None, None, None),
        dataset_source=None,
        object_priors=None,
    ):
        del images
        features = [img_features[f] for f in self.box_in_features]
        image_sizes = [x.image_size for x in proposals]

        if self.training:
            if ann_type in ["box", "part", "ppart"]:
                proposals = self.label_and_sample_proposals(proposals, targets)
            else:
                proposals = self.get_top_proposals(proposals)

            pool_boxes = [x.proposal_boxes for x in proposals]
            pool_features = self.box_pooler(features, pool_boxes)

            box_features = self.box_head(pool_features)
            box_predictions = self.box_predictor(box_features, dataset_source=dataset_source)
            box_predictions = self._apply_prior_to_logits(
                box_predictions, proposals, object_priors
            )

            if ann_type in ["box", "part"]:
                loss = self.box_predictor.losses(
                    box_predictions, proposals, dataset_source=dataset_source
                )
            elif ann_type in ["ppart"]:
                loss = self.box_predictor.part_classification_losses(
                    box_predictions, proposals, targets
                )
            else:
                loss = self.box_predictor.image_label_losses(
                    box_predictions, proposals, targets, classifier_info=classifier_info
                )

            if ann_type in ["box", "part"] and targets[0].has("gt_masks"):
                mask_loss = self._forward_mask(img_features, proposals, object_priors=object_priors)
            else:
                mask_loss = self._get_empty_mask_loss(
                    img_features, proposals, device=proposals[0].objectness_logits.device
                )

            losses = {}
            losses.update(loss)
            losses.update({k: v * self.mask_weight for k, v in mask_loss.items()})
            return proposals, losses

        pool_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.box_pooler(features, pool_boxes)
        box_features = self.box_head(box_features)

        box_predictions = self.box_predictor(box_features)
        box_predictions = self._apply_prior_to_logits(box_predictions, proposals, object_priors)
        boxes = self.box_predictor.predict_boxes(box_predictions, proposals)
        category_scores = self.box_predictor.predict_probs(box_predictions, proposals)

        if self.mult_object_score:
            if len(proposals) > 0 and proposals[0].has("scores"):
                proposal_scores = [p.get("scores") for p in proposals]
            else:
                proposal_scores = [p.get("objectness_logits").sigmoid() for p in proposals]
            scores = [(cs * ps[:, None]) ** 0.5 for cs, ps in zip(category_scores, proposal_scores)]
        else:
            scores = category_scores

        predictor = self.box_predictor
        pred_instances, _ = fast_rcnn_inference(
            boxes,
            scores,
            image_sizes,
            predictor.test_score_thresh,
            predictor.test_nms_thresh,
            predictor.test_topk_per_image,
        )
        self._forward_mask(img_features, pred_instances, object_priors=object_priors)
        return pred_instances, {}


@ROI_HEADS_REGISTRY.register()
class CascadeVLMROIHeadsObjectPrior(ObjectPriorROIHeadsMixin, CascadeVLMROIHeads):
    @configurable
    def __init__(self, *, object_prior_kwargs=None, **kwargs):
        super().__init__(**kwargs)
        self._init_object_prior(**(object_prior_kwargs or {}))

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = super().from_config(cfg, input_shape)
        ret["object_prior_kwargs"] = cls._object_prior_from_config(cfg)
        return ret

    def _forward_box(
        self,
        features,
        proposals,
        targets=None,
        ann_type="box",
        classifier_info=(None, None, None),
        dataset_source=None,
        object_priors=None,
    ):
        if (not self.training) and self.mult_proposal_score:
            proposal_scores = [p.get("objectness_logits").sigmoid() for p in proposals]

        features = [features[f] for f in self.box_in_features]
        head_outputs = []
        prev_pred_boxes = None
        image_sizes = [x.image_size for x in proposals]

        for k in range(self.num_cascade_stages):
            if k > 0:
                proposals = self._create_proposals_from_boxes(prev_pred_boxes, image_sizes)
                if self.training and ann_type in ["box", "part"]:
                    proposals = self._match_and_label_boxes(proposals, k, targets)
            predictions = self._run_stage(
                features,
                proposals,
                k,
                classifier_info=classifier_info,
                dataset_source=dataset_source,
            )
            predictions = self._apply_prior_to_logits(predictions, proposals, object_priors)
            prev_pred_boxes = self.box_predictor[k].predict_boxes(
                (predictions[0], predictions[1]), proposals
            )
            head_outputs.append((self.box_predictor[k], predictions, proposals))

        if self.training:
            losses = {}
            for stage, (predictor, predictions, proposals) in enumerate(head_outputs):
                if ann_type in ["box", "part"]:
                    loss = predictor.losses(
                        predictions, proposals, dataset_source=dataset_source
                    )
                elif ann_type in ["ppart"]:
                    loss = predictor.part_classification_losses_x(
                        predictions, proposals, targets
                    )
                else:
                    loss = predictor.image_label_losses(
                        predictions, proposals, targets, classifier_info=classifier_info
                    )
                losses.update({k + f"_stage{stage}": v for k, v in loss.items()})
            return losses

        scores_per_stage = [h[0].predict_probs(h[1], h[2]) for h in head_outputs]
        scores = [
            sum(list(scores_per_image)) * (1.0 / self.num_cascade_stages)
            for scores_per_image in zip(*scores_per_stage)
        ]
        if self.mult_proposal_score:
            scores = [(s * ps[:, None]) ** 0.5 for s, ps in zip(scores, proposal_scores)]
        predictor, predictions, proposals = head_outputs[-1]
        boxes = predictor.predict_boxes((predictions[0], predictions[1]), proposals)
        pred_instances, _ = fast_rcnn_inference(
            boxes,
            scores,
            image_sizes,
            predictor.test_score_thresh,
            predictor.test_nms_thresh,
            predictor.test_topk_per_image,
        )
        return pred_instances

    def forward(
        self,
        images,
        features,
        proposals,
        targets=None,
        ann_type="box",
        classifier_info=(None, None, None, None),
        dataset_source=None,
        object_priors=None,
    ):
        del images
        if self.training:
            if ann_type in ["box", "part"]:
                proposals = self.label_and_sample_proposals(proposals, targets)
            else:
                proposals = self.get_top_proposals(proposals)

            losses = self._forward_box(
                features,
                proposals,
                targets,
                ann_type=ann_type,
                classifier_info=classifier_info,
                dataset_source=dataset_source,
                object_priors=object_priors,
            )

            if ann_type in ["box", "part"] and targets[0].has("gt_masks"):
                mask_losses = self._forward_mask(features, proposals, object_priors=object_priors)
                losses.update({k: v * self.mask_weight for k, v in mask_losses.items()})
            else:
                losses.update(
                    self._get_empty_mask_loss(
                        features, proposals, device=proposals[0].objectness_logits.device
                    )
                )
            return proposals, losses

        pred_instances = self._forward_box(
            features,
            proposals,
            classifier_info=classifier_info,
            object_priors=object_priors,
        )
        self._forward_mask(features, pred_instances, object_priors=object_priors)
        return pred_instances, {}

    def _run_stage(
        self,
        features,
        proposals,
        stage,
        classifier_info=(None, None, None),
        dataset_source=None,
    ):
        pool_boxes = [x.proposal_boxes for x in proposals]
        box_features = self.box_pooler(features, pool_boxes)
        box_features = _ScaleGradient.apply(box_features, 1.0 / self.num_cascade_stages)
        box_features = self.box_head[stage](box_features)
        return self.box_predictor[stage](
            box_features,
            dataset_source=dataset_source,
            classifier_info=classifier_info,
        )
