import copy
import logging

import cv2
import numpy as np
from PIL import Image
import torch

import detectron2.data.detection_utils as utils
import detectron2.data.transforms as T

from .dataset_mapper_filterbybox import filter_empty_instances_by_box
from .dataset_mapper_withimage import DatasetMapperWithImage


logger = logging.getLogger("detectron2.vlpart.data.dataset_mapper_vigor_noisy")


class DatasetMapperWithImageVigorNoisy(DatasetMapperWithImage):
    def _read_mask(self, mask_path, image_shape):
        mask = utils.read_image(mask_path, "L").squeeze(2)
        if mask.shape != image_shape:
            h, w = image_shape
            mask = cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)
        return mask == 0

    def _read_gt_nearby_mix_image(self, dataset_dict):
        image = utils.read_image(dataset_dict["vigor_scene_file_name"], format=self.image_format)
        target_fg = self._read_mask(dataset_dict["gt_object_mask_path"], image.shape[:2])
        nearby_fg = self._read_mask(dataset_dict["vigor_secondary_object_mask_path"], image.shape[:2])
        fg = np.logical_or(target_fg, nearby_fg)
        masked = np.zeros_like(image)
        masked[fg] = image[fg]
        return masked

    def _read_input_image(self, dataset_dict):
        input_type = dataset_dict.get("vigor_input_type")
        if input_type == "gt_nearby_object_mix":
            return self._read_gt_nearby_mix_image(dataset_dict)
        if "file_name" in dataset_dict:
            return utils.read_image(dataset_dict["file_name"], format=self.image_format)

        ori_image, _, _ = self.tar_dataset[dataset_dict["tar_index"]]
        ori_image = utils._apply_exif_orientation(ori_image)
        return utils.convert_PIL_to_numpy(ori_image, self.image_format)

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        ori_image = self._read_input_image(dataset_dict)

        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(copy.deepcopy(ori_image), sem_seg=sem_seg_gt)
        if self.use_diff_bs_size and self.is_train:
            transforms = self.dataset_augs[dataset_dict["dataset_source"]](aug_input)
        else:
            transforms = self.augmentations(aug_input)
        image_weak_aug, sem_seg_gt = aug_input.image, aug_input.sem_seg
        image_shape = image_weak_aug.shape[:2]

        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict,
                image_shape,
                transforms,
                proposal_topk=self.proposal_topk,
            )

        if not self.is_train:
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            for anno in dataset_dict["annotations"]:
                if not self.use_instance_mask:
                    anno.pop("segmentation", None)
                if not self.use_keypoint:
                    anno.pop("keypoints", None)

            all_annos = [
                (
                    utils.transform_instance_annotations(
                        obj,
                        transforms,
                        image_shape,
                        keypoint_hflip_indices=self.keypoint_hflip_indices,
                    ),
                    obj.get("iscrowd", 0),
                )
                for obj in dataset_dict.pop("annotations")
            ]
            annos = [ann[0] for ann in all_annos if ann[1] == 0]
            instances = utils.annotations_to_instances(
                annos,
                image_shape,
                mask_format=self.instance_mask_format,
            )

            del all_annos
            if self.recompute_boxes:
                instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
            dataset_dict["instances"] = filter_empty_instances_by_box(instances)

        if (
            self.with_ann_type
            and self.strong_aug_on_parsed
            and self.dataset_ann[dataset_dict["dataset_source"]] == "ppart"
        ):
            image_pil = Image.fromarray(image_weak_aug.astype("uint8"), "RGB")
            image_strong_aug = np.array(self.strong_augmentation(image_pil))
            dataset_dict["image"] = torch.as_tensor(
                np.ascontiguousarray(image_strong_aug.transpose(2, 0, 1))
            )
        else:
            dataset_dict["image"] = torch.as_tensor(
                np.ascontiguousarray(image_weak_aug.transpose(2, 0, 1))
            )

        if self.with_ann_type:
            dataset_dict["pos_category_ids"] = dataset_dict.get("pos_category_ids", [])
            dataset_dict["ann_type"] = self.dataset_ann[dataset_dict["dataset_source"]]
        return dataset_dict
