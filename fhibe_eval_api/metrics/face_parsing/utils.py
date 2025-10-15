# Copyright (c) Sony AI Inc.
# All rights reserved.
"""Module containing metric utilties specific to the face parsing task."""
import os
from typing import Any, Dict, List

import cv2
import numpy as np
from tqdm import tqdm

from fhibe_eval_api.common.metrics import f1_score
from fhibe_eval_api.common.utils import save_json_file

# masks in CelebAMask-HQ and dml_csr
CELEBA_MASK_LABELS = [
    "background",
    "skin",
    "nose",
    "eye_g",
    "l_eye",
    "r_eye",
    "l_brow",
    "r_brow",
    "l_ear",
    "r_ear",
    "mouth",
    "u_lip",
    "l_lip",
    "hair",
    "hat",
    "ear_r",
    "neck_l",
    "neck",
    "cloth",
]

CELEBA_MASK_HQ_LABELS_DICT = {
    label: idx for idx, label in enumerate(CELEBA_MASK_LABELS)
}
IDX_TO_LABELS = {idx: label for idx, label in enumerate(CELEBA_MASK_LABELS)}


def face_parsing_results(
    filepaths: List[str],
    mask_filepaths: List[str],
    model_outputs: Dict[str, Any],
    save_json_filepath: str,
) -> Dict[str, Any]:
    """Calculate f1 score between predicted and gt face mask.

    Args:
        filepaths (List[str]): List of image filepaths.
        mask_filepaths (List[str]): List of mask image filepaths.
        model_outputs (Dict[str, Any]): Contains the masks
        save_json_filepath (str): Filepath to save the results to.
    """
    results: Dict[str, Any] = dict()
    with tqdm(total=len(filepaths), desc="Face Parsing - Calculate F1 scores") as pbar:
        for filepath, mask_filepath in zip(filepaths, mask_filepaths):
            # pred is in the form of combined mask
            pred = np.array(model_outputs[filepath]["detections"])
            if not os.path.isfile(mask_filepath):
                raise RuntimeError(f"Face parsing mask file not found: {mask_filepath}")
            gt_combined = cv2.imread(mask_filepath, cv2.IMREAD_GRAYSCALE)
            gt_combined = np.asarray(gt_combined, dtype=np.int32)

            img_results = {}
            for gt_label in np.unique(gt_combined):
                pred_masked = pred == gt_label
                gt_masked = gt_combined == gt_label

                # harmonic mean (F1) score for segmentation masks
                f1_value = f1_score(pred_masked, gt_masked)
                img_results[IDX_TO_LABELS[gt_label]] = f1_value

            results[filepath] = img_results
            pbar.update(1)

    save_json_file(filepath=save_json_filepath, data=results, indent=True)
    return results
