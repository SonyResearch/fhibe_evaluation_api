# SPDX-License-Identifier: Apache-2.0
"""Module containing constants used in FHIBE metrics.

This module contains lookup dicts for which metrics are used
for each task, which metrics include thresholds, and text descriptions
of the metrics themselves.
"""

import numpy as np

TASK_METRIC_LOOKUP_DICT = {
    "person_localization": ["AR_IOU"],
    "person_parsing": ["AR_MASK"],
    "keypoint_estimation": ["PCK", "AR_OKS"],
    "face_localization": ["AR_IOU"],
    "body_parts_detection": ["AR_DET", "ACC_DET"],
    "face_parsing": ["F1"],
    "face_encoding": ["CURRICULAR_FACE", "LPIPS", "PSNR"],
    "face_verification": ["VAL"],
    "face_super_resolution": ["LPIPS"],
}

METRIC_THRESHOLDS_DEFAULTS = {
    "AR_IOU": np.arange(0.5, 1, 0.05),
    "AR_OKS": np.arange(0.5, 1, 0.05),
    "AR_MASK": np.arange(0.5, 1, 0.05),
    "AR_DET": np.arange(0.5, 1, 0.05),
    "ACC_DET": [0.5],
    "PCK": np.arange(0.1, 1, 0.1),
}

METRIC_DESCRIPTION_DICT = {
    "AR_IOU": {
        "title": "Average recall over intersection over union.",
        "description": (
            "For each ground truth bounding box, "
            "the best IoU out of all predicted bounding boxes is obtained. "
            "At each value in a list of IoU thresholds between 0 and 1, "
            "each image is given a value of 1 (correct) or 0 (incorrect) "
            "based on whether IoU > threshold. Using these binary outcomes, "
            "the recall is calculated. "
            "The average recall over all thresholds is reported."
        ),
    },
    "AR_OKS": {
        "title": "Average recall over object keypoint similarity.",
        "description": (
            "The object keypoint similarity (OKS) is calculated as in the "
            "COCO evaluation dataset: https://cocodataset.org/#keypoints-eval. "
            "It has a minimum value (worst) of 0 and a maximum value (best) of 1. "
            "At each value in a list of OKS thresholds between 0 and 1, "
            "each image is given a value of 1 (correct) or 0 (incorrect) "
            "based on whether OKS &#x2265; threshold. Using these binary outcomes, "
            "the recall is calculated. "
            "The average recall over all thresholds is reported."
        ),
    },
    "AR_MASK": {
        "title": "Average recall over IoU of the person mask.",
        "description": (
            "For each ground truth person mask, the best IoU out of all "
            "predicted masks is obtained. Then, at each value in a list "
            "of IoU thresholds between 0 and 1, the recall is calculated. "
            "The average recall over all thresholds is reported."
        ),
    },
    "AR_DET": {
        "title": "Average recall of the detection of body parts.",
        "description": (
            "At each value in a list "
            "of thresholds between 0 and 1, the recall is calculated "
            "for each body part predicted by the model, "
            "where a positive prediction (existence of body part) "
            "is determined if prob(body_part) >= threshold. "
            "The average recall over all thresholds is reported "
            "for each body part and averaged over all body parts."
        ),
    },
    "ACC_DET": {
        "title": "Accuracy of the detection of body parts.",
        "description": (
            "At each value in a list "
            "of thresholds between 0 and 1, the accuracy is calculated "
            "for each body part predicted by the model, "
            "where a positive prediction (existence of body part) "
            "is determined if prob(body_part) >= threshold. "
            "The accuracy reported is the average over all thresholds and "
            "reported for each body part as well as the average over all body parts."
        ),
    },
    "PCK": {
        "title": "Percentage correct keypoints.",
        "description": (
            "The distance between each ground truth "
            "keypoint and the closest predicted keypoint is compared to the product "
            "`thresh*face_bbox_diag`, where `thresh` "
            "is a threshold value and `face_bbox_diag` "
            "is the length of the diagonal of the ground truth face bounding box. "
            "If the distance is less than the product, "
            "a keypoint is considered correct. "
            "The fraction of correct keypoints in the set of ground truth keypoints "
            "in an image is the PCK at a single threshold for a single image. "
            "This is repeated for each threshold in a list of thresholds, "
            "and the mean over all thresholds is reported. "
            "PCK has a minimum value (worst) of 0 and maximum value (best) of 1. "
        ),
    },
    "F1": {
        "title": "F1 score, averaged over each face part mask.",
        "description": (
            "For each face part, "
            "true positives are calculated as the number of pixels in the "
            "intersection of the ground truth and predicted masks, "
            "false positives are the intersection of the predicted mask "
            "and the logical not of the ground truth mask, false negatives "
            "are the intersection of the ground truth mask and the logical "
            "not of the predicted mask. The F1 score for each face part "
            "is calculated as the harmonic mean of precision and recall. "
            "The average F1 score over all face parts is reported."
        ),
    },
    "CURRICULAR_FACE": {
        "title": "CurricularFace embedding similarity score.",
        "description": (
            "This calculates the dot product between the embedding of an encoded image "
            "and the embedding its corresponding original image, "
            "using the CurricularFace backbone model to embed both images."
        ),
    },
    "VAL": {
        "title": "Validation rate.",
        "description": (
            "Calculates the validation rate at a "
            "false acceptance rate of 0.001 using k-fold cross-validation."
        ),
    },
    "LPIPS": {
        "title": "Learned perceptual image patch similarity.",
        "description": (
            "This is calculated for each individual image using "
            "the Pytorch Image Quality (PIQ) library: "
            "https://piq.readthedocs.io/en/latest/modules.html#learned-perceptual-image-patch-similarity-lpips."
        ),
    },
    "PSNR": {
        "title": "Peak signal-to-noise ratio.",
        "description": (
            "This is calculated for each "
            "individual image using the Pytorch Image Quality (PIQ) library: "
            "https://piq.readthedocs.io/en/latest/functions.html#peak-signal-to-noise-ratio-psnr."
        ),
    },
}
