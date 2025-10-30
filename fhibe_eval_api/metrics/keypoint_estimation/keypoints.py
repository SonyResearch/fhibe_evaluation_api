# SPDX-License-Identifier: Apache-2.0
"""Module containing metrics for the keypoint estimation task."""
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from tqdm import tqdm

from fhibe_eval_api.common.metrics import (
    best_oks_scores_for_gt_keypoints,
    percentage_correct_keypoints_score,
)
from fhibe_eval_api.common.utils import save_json_file


def calc_percentage_correct_keypoints(
    thresholds: List[float],
    filepaths: List[str],
    model_outputs: Dict[str, Any],
    annotations_dataframe: pd.DataFrame,
    gt_keypoint_column_name: str,
    gt_face_bbox_column_name: str,
    save_json_filepath: str,
) -> Dict[str, Dict[str, Any]]:
    """Calculate percentage correct keypoints (pck) for a list of images.

    Args:
        thresholds: List of fractions of face bbox diagonal.
            If the distance between predicted and ground truth keypoint
            is below this threshold, a prediction is considered "correct".
        filepaths (List[str]): List of image filepaths.
        model_outputs (Dict[str, Any]): Dictionary of model outputs containing
            bbox predictions and bbox scores.
        annotations_dataframe (pd.DataFrame): Dataframe containing information such as
            ground-truth bboxes.
        gt_keypoint_column_name (str): The column name in the dataframe for containing
            the ground-truth keypoints.
        gt_face_bbox_column_name (str): The column name in the dataframe for containing
            the ground-truth face bounding boxes.
        save_json_filepath (str): Filepath to save the results to.

    Returns:
        Dictionary containing the PCK scores for each image and threshold.
    """
    if gt_keypoint_column_name not in annotations_dataframe.columns:
        raise ValueError(
            f"{gt_keypoint_column_name} is not a column in the provided dataframe."
        )
    if gt_face_bbox_column_name not in annotations_dataframe.columns:
        raise ValueError(
            f"{gt_face_bbox_column_name} is not a column in the provided dataframe."
        )

    all_pck_scores = np.zeros(
        (len(annotations_dataframe), len(thresholds)), dtype=float
    )
    with tqdm(
        total=len(filepaths),
        desc="Keypoint estimation - Computing scores (PCK, OKS)",
    ) as pbar:
        for filepath in filepaths:
            pred_keypoints = model_outputs[filepath]["detections"]

            annotations_dataframe_subset = annotations_dataframe[
                annotations_dataframe["filepath"] == filepath
            ]
            gt_keypoints = [
                gt_keypoint
                for gt_keypoint in annotations_dataframe_subset.loc[
                    :, gt_keypoint_column_name
                ].tolist()
            ]
            # PCK score
            gt_face_bboxes = [
                gt_face_bbox
                for gt_face_bbox in annotations_dataframe_subset.loc[
                    :, gt_face_bbox_column_name
                ].tolist()
            ]
            pcks = []
            for pred_keypoints_n, gt_keypoints_n, gt_face_bboxes_n in zip(
                pred_keypoints, gt_keypoints, gt_face_bboxes
            ):
                pck_list = percentage_correct_keypoints_score(
                    pred_keypoints=pred_keypoints_n,
                    gt_keypoints=gt_keypoints_n[:, :2],
                    face_bbox=gt_face_bboxes_n,
                    thresholds=thresholds,
                    visible=gt_keypoints_n[:, 2] == 2,
                )
                pcks.append(pck_list)
            all_pck_scores[annotations_dataframe_subset.index.tolist()] = pcks

            pbar.update(1)

    results = {"thresholds": list(thresholds), "results": {}}
    for idx, pcks in zip(annotations_dataframe.index.tolist(), all_pck_scores):
        results["results"][str(idx)] = {
            "mean_percentage_correct_keypoints": np.nanmean(pcks),
            "pcks@thresholds": list(pcks),
        }
        for key in (
            "image_id",  # fhibe
            "subject_id",  # fhibe and coco
            "filepath",  # all
        ):
            if key in annotations_dataframe.columns:
                results["results"][str(idx)][key] = annotations_dataframe.iloc[idx][key]
    save_json_file(filepath=save_json_filepath, data=results, indent=4)
    return results


def calc_object_keypoint_similarity(
    filepaths: List[str],
    model_outputs: Dict[str, Any],
    annotations_dataframe: pd.DataFrame,
    gt_keypoint_column_name: str,
    gt_face_bbox_column_name: str,
    gt_person_segments_area_column_names: str,
    kpt_oks_sigmas: List[float],
    save_json_filepath: str,
) -> List[float]:
    """Calculate object keypoint similarity (OKS) for a list of images.

    Args:
        filepaths: List of image filepaths.
        model_outputs: Dictionary of model outputs containing
            bbox predictions and bbox scores.
        annotations_dataframe: Dataframe containing information such as
            ground-truth bboxes.
        gt_keypoint_column_name: The column name in the dataframe for containing
            the ground-truth keypoints.
        gt_face_bbox_column_name: The column name in the dataframe for containing
            the ground-truth face bounding boxes.
        gt_person_segments_area_column_names: The column name in the datafrmae
            for containing the segment areas.
        kpt_oks_sigmas: Array/List of uncertainties for each keypoint
            to be used in OKS metric calculation.
        save_json_filepath: Filepath to save the results to.

    Returns:
        List of OKS scores, one for each image.
    """
    if gt_keypoint_column_name not in annotations_dataframe.columns:
        raise ValueError(
            f"{gt_keypoint_column_name} is not a column in the provided dataframe."
        )
    if gt_face_bbox_column_name not in annotations_dataframe.columns:
        raise ValueError(
            f"{gt_face_bbox_column_name} is not a column in the provided dataframe."
        )
    all_oks_scores = np.zeros((len(annotations_dataframe)), dtype=float)
    with tqdm(
        total=len(filepaths),
        desc="Keypoint estimation - Computing scores (PCK, OKS)",
    ) as pbar:
        for filepath in filepaths:
            pred_keypoints = model_outputs[filepath]["detections"]

            annotations_dataframe_subset = annotations_dataframe[
                annotations_dataframe["filepath"] == filepath
            ]
            gt_keypoints = [
                gt_keypoint
                for gt_keypoint in annotations_dataframe_subset.loc[
                    :, gt_keypoint_column_name
                ].tolist()
            ]
            # OKS score
            gt_person_segments_areas = [
                gt_person_segments_area
                for gt_person_segments_area in annotations_dataframe_subset.loc[
                    :, gt_person_segments_area_column_names
                ].tolist()
            ]

            okss = best_oks_scores_for_gt_keypoints(
                pred_keypoints=pred_keypoints,
                gt_keypoints=gt_keypoints,
                gt_persona_segment_areas=gt_person_segments_areas,
                kpt_oks_sigmas=kpt_oks_sigmas,
            )
            all_oks_scores[annotations_dataframe_subset.index.tolist()] = okss

            pbar.update(1)

    save_json_file(filepath=save_json_filepath, data=all_oks_scores.tolist())
    return all_oks_scores.tolist()
