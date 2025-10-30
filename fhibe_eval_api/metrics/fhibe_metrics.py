# SPDX-License-Identifier: Apache-2.0
"""Module containing metrics used in the evaluation.

This module contains functions that calculate metrics on
model outputs and aggregate the metric values over attributes
and intersections of attributes. 
"""

import logging
import os
from typing import Any, Dict, List, cast

import pandas as pd
from tqdm import tqdm

from fhibe_eval_api.common.loggers import setup_logging
from fhibe_eval_api.common.metrics import curricular_face_score as cfs_builtin
from fhibe_eval_api.common.metrics import (
    learned_perceptual_image_patch_similarity_score,
    peak_signal_to_noise_ratio_score,
)
from fhibe_eval_api.common.utils import read_json_file, save_json_file
from fhibe_eval_api.evaluate.utils import _decode_mask
from fhibe_eval_api.metrics.face_parsing.utils import face_parsing_results
from fhibe_eval_api.metrics.keypoint_estimation.keypoints import (
    calc_object_keypoint_similarity,
    calc_percentage_correct_keypoints,
)
from fhibe_eval_api.metrics.utils import (
    compute_gt_bbox_iou_scores,
    compute_gt_mask_iou_scores,
    gather_body_part_detection_scores,
    group_f1_results,
    group_face_metric_scores,
    group_thresholded_body_part_results,
    group_thresholded_metric_results,
    group_val_scores,
    save_thresholded_body_part_results,
    save_thresholded_results_per_file,
)

setup_logging("info")


def average_recall_bbox(
    task_name: str,
    intersectional_groups: List[str],
    filepaths: List[str],
    model_outputs: dict[str, Any],
    annotations_dataframe: pd.DataFrame,
    thresholds: List[float],
    current_results_dir: str,
    **kwargs: Dict[str, Any],
) -> pd.DataFrame:
    """Compute average recall for bounding box tasks.

    Saves the intersectional results to disk.

    Args:
        task_name: The name of the model task to evaluation
        intersectional_groups: A list of the demographic attributes to split
            the data on.
        filepaths: A list of the filepaths of the images to evaluate.
        model_outputs: Results from running inference on the model
        annotations_dataframe: Contains the annotation metadata
        thresholds: A list of metric thresholds
        current_results_dir: The directory in which to save the results.
        **kwargs: Additional keyword arguments containing task-specific parameters

    Return:
        A dataframe containing the metric results aggregated in demographic
        groups.
    """
    gt_column_name = cast(str, kwargs.get("gt_column_name"))

    # first check if bbox iou scores are already calculated
    iou_filename = os.path.join(current_results_dir, "gt_bbox_iou_scores.json")
    if not os.path.isfile(iou_filename):
        gt_bbox_iou_scores = compute_gt_bbox_iou_scores(
            filepaths=filepaths,
            model_outputs=model_outputs,
            annotations_dataframe=annotations_dataframe,
            gt_column_name=gt_column_name,
        )
        save_json_file(iou_filename, gt_bbox_iou_scores)
    else:
        gt_bbox_iou_scores = read_json_file(iou_filename)

    # Get thresholded results by file
    results_path = os.path.join(current_results_dir, "results_iou_threshold.json")
    detailed_results_path = os.path.join(
        current_results_dir, "detailed_results_iou_threshold.json"
    )
    metric_threshold_dict = save_thresholded_results_per_file(
        annotations_dataframe=annotations_dataframe,
        metric_vals=gt_bbox_iou_scores,
        results_path=results_path,
        detailed_results_path=detailed_results_path,
        thresholds=thresholds,
    )
    grouped_results_dict = group_thresholded_metric_results(
        threshold_dict=metric_threshold_dict,
        metric_name="AR_IOU",
        annotations_dataframe=annotations_dataframe,
        intersectional_groups=intersectional_groups,
        save_json_filepath=os.path.join(
            current_results_dir, "intersectional_results_AR_IOU.json"
        ),
    )
    return grouped_results_dict


def average_recall_mask(
    task_name: str,
    intersectional_groups: List[str],
    filepaths: List[str],
    model_outputs: dict[str, Any],
    annotations_dataframe: pd.DataFrame,
    thresholds: List[float],
    current_results_dir: str,
    **kwargs: Dict[str, Any],
) -> pd.DataFrame:
    """Compute average recall for segmentation mask tasks.

    Saves the intersectional results to disk.

    Args:
        task_name: The name of the model task to evaluation
        intersectional_groups: A list of the demographic attributes to split
            the data on.
        filepaths: A list of the filepaths of the images to evaluate.
        model_outputs: Results from running inference on the model
        annotations_dataframe: Contains the annotation metadata
        thresholds: A list of metric thresholds
        current_results_dir: The directory in which to save the results.
        **kwargs: Additional keyword arguments containing task-specific parameters

    Return:
        A dataframe containing the metric results aggregated in demographic
        groups.
    """
    gt_mask_column_name = cast(str, kwargs.get("gt_column_name"))
    # first check if bbox iou scores are already calculated
    iou_filename = os.path.join(current_results_dir, "gt_mask_iou_scores.json")
    if not os.path.isfile(iou_filename):
        gt_mask_iou_scores = compute_gt_mask_iou_scores(
            filepaths=filepaths,
            model_outputs=model_outputs,
            annotations_dataframe=annotations_dataframe,
            gt_mask_column_name=gt_mask_column_name,
            to_rle=True,
        )
        save_json_file(iou_filename, gt_mask_iou_scores)
    else:
        gt_mask_iou_scores = read_json_file(iou_filename)

    # Get thresholded results by file
    results_path = os.path.join(current_results_dir, "results_iou_threshold.json")
    detailed_results_path = os.path.join(
        current_results_dir, "detailed_results_iou_threshold.json"
    )
    metric_threshold_dict = save_thresholded_results_per_file(
        annotations_dataframe=annotations_dataframe,
        metric_vals=gt_mask_iou_scores,
        results_path=results_path,
        detailed_results_path=detailed_results_path,
        thresholds=thresholds,
    )
    grouped_results_dict = group_thresholded_metric_results(
        threshold_dict=metric_threshold_dict,
        metric_name="AR_MASK",
        annotations_dataframe=annotations_dataframe,
        intersectional_groups=intersectional_groups,
        save_json_filepath=os.path.join(
            current_results_dir, "intersectional_results_AR_MASK.json"
        ),
    )
    return grouped_results_dict


def average_recall_body_part_detection(
    task_name: str,
    intersectional_groups: List[str],
    filepaths: List[str],
    model_outputs: dict[str, Any],
    annotations_dataframe: pd.DataFrame,
    thresholds: List[float],
    current_results_dir: str,
    **kwargs: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, int | float]]]:
    """Compute average recall for the body parts detection task.

    Saves the intersectional results to disk.

    Args:
        task_name: The name of the model task to evaluation
        intersectional_groups: A list of the demographic attributes to split
            the data on.
        filepaths: A list of the filepaths of the images to evaluate.
        model_outputs: Results from running inference on the model
        annotations_dataframe: Contains the annotation metadata
        thresholds: A list of metric thresholds
        current_results_dir: The directory in which to save the results.
        **kwargs: Additional keyword arguments containing task-specific parameters

    Return:
        Dictionary containing intersectional results,
            broken down by body part within each group.
    """
    # Model outputs map unique filepath to detections,
    # but for metrics we need detections for each annotations_dataframe entry
    # first check if bp detections are already reorganized
    bp_det_filename = os.path.join(current_results_dir, "body_part_detections.json")
    if not os.path.isfile(bp_det_filename):
        bp_det_dict = gather_body_part_detection_scores(
            filepaths=filepaths,
            model_outputs=model_outputs,
            annotations_dataframe=annotations_dataframe,
        )
        save_json_file(bp_det_filename, bp_det_dict)
    else:
        bp_det_dict = read_json_file(bp_det_filename)

    # Get thresholded results by file
    results_path = os.path.join(
        current_results_dir, "results_body_parts_AR_DET_threshold.json"
    )
    detailed_results_path = os.path.join(
        current_results_dir, "detailed_results_body_parts_AR_DET_threshold.json"
    )
    metric_threshold_dict = save_thresholded_body_part_results(
        annotations_dataframe=annotations_dataframe,
        body_parts_dict_list=bp_det_dict,
        results_path=results_path,
        detailed_results_path=detailed_results_path,
        thresholds=thresholds,
    )
    grouped_results = group_thresholded_body_part_results(
        threshold_dict=metric_threshold_dict,
        metric_name="AR_DET",
        annotations_dataframe=annotations_dataframe,
        intersectional_groups=intersectional_groups,
        save_json_filepath=os.path.join(
            current_results_dir, "intersectional_results_AR_DET.json"
        ),
    )
    return grouped_results


def accuracy_body_part_detection(
    task_name: str,
    intersectional_groups: List[str],
    filepaths: List[str],
    model_outputs: dict[str, Any],
    annotations_dataframe: pd.DataFrame,
    thresholds: List[float],
    current_results_dir: str,
    **kwargs: Dict[str, Any],
) -> Dict[str, Dict[str, Dict[str, int | float]]]:
    """Compute accuracy for the body parts detection task.

    Note: The accuracy will be averaged over thresholds,
    where the threshold is used to determine what is a positive
    prediction. The default is to use a threshold of [0.5], but
    in priniciple one could use a list of any lenght.

    Saves the intersectional results to disk.

    Args:
        task_name: The name of the model task to evaluation
        intersectional_groups: A list of the demographic attributes to split
            the data on.
        filepaths: A list of the filepaths of the images to evaluate.
        model_outputs: Results from running inference on the model
        annotations_dataframe: Contains the annotation metadata
        thresholds: A list of metric thresholds
        current_results_dir: The directory in which to save the results.
        **kwargs: Additional keyword arguments containing task-specific parameters

    Return:
        Dictionary containing intersectional results,
            broken down by body part within each group.
    """
    # Model outputs map unique filepath to detections,
    # but for metrics we need detections for each annotations_dataframe entry
    # first check if bp detections are already reorganized
    bp_det_filename = os.path.join(current_results_dir, "body_part_detections.json")
    if not os.path.isfile(bp_det_filename):
        bp_det_dict = gather_body_part_detection_scores(
            filepaths=filepaths,
            model_outputs=model_outputs,
            annotations_dataframe=annotations_dataframe,
        )
        save_json_file(bp_det_filename, bp_det_dict)
    else:
        bp_det_dict = read_json_file(bp_det_filename)

    # Get thresholded results by file
    results_path = os.path.join(
        current_results_dir, "results_body_parts_ACC_DET_threshold.json"
    )
    detailed_results_path = os.path.join(
        current_results_dir, "detailed_results_body_parts_ACC_DET_threshold.json"
    )
    metric_threshold_dict = save_thresholded_body_part_results(
        annotations_dataframe=annotations_dataframe,
        body_parts_dict_list=bp_det_dict,
        results_path=results_path,
        detailed_results_path=detailed_results_path,
        thresholds=thresholds,
    )
    grouped_results = group_thresholded_body_part_results(
        threshold_dict=metric_threshold_dict,
        metric_name="ACC_DET",
        annotations_dataframe=annotations_dataframe,
        intersectional_groups=intersectional_groups,
        save_json_filepath=os.path.join(
            current_results_dir, "intersectional_results_ACC_DET.json"
        ),
    )
    return grouped_results


def percentage_correct_keypoints(
    task_name: str,
    intersectional_groups: List[str],
    filepaths: List[str],
    model_outputs: dict[str, Any],
    annotations_dataframe: pd.DataFrame,
    thresholds: List[float],
    current_results_dir: str,
    **kwargs: Dict[str, Any],
) -> pd.DataFrame:
    """Compute percentage correct keypoints (PCK) for the keypoint task.

    For each ground truth person bounding box, the PCK is computed as the fraction of
    keypoints within that bounding box that are deemed "correct". Correctness
    is defined as distance(predicted,truth) < thresh * face_bbox_diag,
    where face_bbox_diag is the length of the diagonal of the face bounding box
    for the subject.

    Saves the intersectional results to disk.

    Args:
        task_name: The name of the model task to evaluation
        intersectional_groups: A list of the demographic attributes to split
            the data on.
        filepaths: A list of the filepaths of the images to evaluate.
        model_outputs: Results from running inference on the model
        annotations_dataframe: Contains the annotation metadata
        thresholds: A list of metric thresholds
        current_results_dir: The directory in which to save the results.
        **kwargs: Additional keyword arguments containing task-specific parameters

    Return:
        A dataframe containing the intersectional results.
    """
    # Check if pck file has been generated already
    pck_filename = os.path.join(current_results_dir, "pck_scores_threshold.json")
    if not os.path.isfile(pck_filename):
        gt_keypoint_column_name = "keypoints_coco_fmt"
        gt_face_bbox_column_name = "face_bbox"
        # Next line saves the JSON file
        pck_results_dict = calc_percentage_correct_keypoints(
            thresholds=thresholds,
            filepaths=filepaths,
            model_outputs=model_outputs,
            annotations_dataframe=annotations_dataframe,
            gt_keypoint_column_name=gt_keypoint_column_name,
            gt_face_bbox_column_name=gt_face_bbox_column_name,
            save_json_filepath=pck_filename,
        )
    else:
        pck_results_dict = read_json_file(pck_filename)

    grouped_results_dict = group_thresholded_metric_results(
        threshold_dict=pck_results_dict,
        metric_name="PCK",
        annotations_dataframe=annotations_dataframe,
        intersectional_groups=intersectional_groups,
        save_json_filepath=os.path.join(
            current_results_dir, "intersectional_results_PCK.json"
        ),
        **kwargs,  # Pass custom keypoints if provided
    )
    return grouped_results_dict


def object_keypoint_similarity(
    task_name: str,
    intersectional_groups: List[str],
    filepaths: List[str],
    model_outputs: dict[str, Any],
    annotations_dataframe: pd.DataFrame,
    thresholds: List[float],
    current_results_dir: str,
    **kwargs: Dict[str, Any],
) -> pd.DataFrame:
    """Compute object keypoint similarity for the keypoint task.

    Saves the intersectional results to disk.

    Args:
        task_name: The name of the model task to evaluation
        intersectional_groups: A list of the demographic attributes to split
            the data on.
        filepaths: A list of the filepaths of the images to evaluate.
        model_outputs: Results from running inference on the model
        annotations_dataframe: Contains the annotation metadata
        thresholds: A list of metric thresholds
        current_results_dir: The directory in which to save the results.
        **kwargs: Additional keyword arguments containing task-specific parameters

    Return:
        A dataframe containing the intersectional results.
    """
    # Check if oks file has been generated already
    oks_filename = os.path.join(current_results_dir, "oks_scores.json")
    if not os.path.isfile(oks_filename):
        gt_keypoint_column_name = "keypoints_coco_fmt"
        gt_face_bbox_column_name = "face_bbox"
        gt_person_segments_area_column_names = "area"
        kpt_oks_sigmas = cast(List[float], kwargs["kpt_oks_sigmas"])
        # Next line saves the JSON file
        oks_ious = calc_object_keypoint_similarity(
            filepaths=filepaths,
            model_outputs=model_outputs,
            annotations_dataframe=annotations_dataframe,
            gt_keypoint_column_name=gt_keypoint_column_name,
            gt_face_bbox_column_name=gt_face_bbox_column_name,
            gt_person_segments_area_column_names=gt_person_segments_area_column_names,
            kpt_oks_sigmas=kpt_oks_sigmas,
            save_json_filepath=oks_filename,
        )
    else:
        oks_ious = read_json_file(oks_filename)

    results_path = os.path.join(current_results_dir, "results_oks_threshold.json")
    detailed_results_path = os.path.join(
        current_results_dir, "detailed_results_oks_threshold.json"
    )
    metric_threshold_dict = save_thresholded_results_per_file(
        annotations_dataframe=annotations_dataframe,
        metric_vals=oks_ious,
        results_path=results_path,
        detailed_results_path=detailed_results_path,
        thresholds=thresholds,
    )
    grouped_results_dict = group_thresholded_metric_results(
        threshold_dict=metric_threshold_dict,
        metric_name="AR_OKS",
        annotations_dataframe=annotations_dataframe,
        intersectional_groups=intersectional_groups,
        save_json_filepath=os.path.join(
            current_results_dir, "intersectional_results_AR_OKS.json"
        ),
        **kwargs,  # Pass custom keypoints if provided
    )
    return grouped_results_dict


def f1_scores_parsing(
    task_name: str,
    intersectional_groups: List[str],
    filepaths: List[str],
    model_outputs: dict[str, Any],
    annotations_dataframe: pd.DataFrame,
    thresholds: List[float] | None,
    current_results_dir: str,
    **kwargs: Dict[str, Any],
) -> pd.DataFrame:
    """Compute mean f1 score over all face mask categories for the face parsing task.

    Saves the intersectional results to disk.

    Args:
        task_name: The name of the model task to evaluation
        intersectional_groups: A list of the demographic attributes to split
            the data on.
        filepaths: A list of the filepaths of the images to evaluate.
        model_outputs: Results from running inference on the model
        annotations_dataframe: Contains the annotation metadata
        thresholds: Not used for this metric
        current_results_dir: The directory in which to save the results.
        **kwargs: Additional keyword arguments containing task-specific parameters

    Return:
        A dataframe containing the intersectional results.
    """
    for key in tqdm(model_outputs, desc="Decoding detection masks:"):
        model_outputs[key]["detections"] = _decode_mask(
            model_outputs[key]["detections_rle"]
        )
    f1_results_filepath = os.path.join(current_results_dir, "F1_scores.json")
    if not os.path.isfile(f1_results_filepath):
        all_f1_scores = face_parsing_results(
            filepaths=filepaths,
            mask_filepaths=cast(list[str], kwargs["mask_filepaths"]),
            model_outputs=model_outputs,
            save_json_filepath=f1_results_filepath,
        )
        logging.info(f"Saved F1 scores to: {f1_results_filepath}")
    else:
        all_f1_scores = read_json_file(f1_results_filepath)

    grouped_results_dict = group_f1_results(
        all_f1_scores=all_f1_scores,
        annotations_dataframe=annotations_dataframe,
        intersectional_groups=intersectional_groups,
        save_json_filepath=os.path.join(
            current_results_dir, "intersectional_results_F1.json"
        ),
    )
    return grouped_results_dict


def validation_rate_face_verification(
    task_name: str,
    intersectional_groups: List[str],
    filepaths: List[str],
    model_outputs: dict[str, Any],
    annotations_dataframe: pd.DataFrame,
    thresholds: List[float],
    current_results_dir: str,
    **kwargs: Dict[str, Any],
) -> pd.DataFrame:  # pragma: no cover
    """Compute validation rate as false acceptance rate=0.001.

    This is used for the face verification task.

    Saves the intersectional results to disk.

    Args:
        task_name: The name of the model task to evaluation
        intersectional_groups: A list of the demographic attributes to split
            the data on.
        filepaths: A list of the filepaths of the images to evaluate.
        model_outputs: Results from running inference on the model
        annotations_dataframe: Contains the annotation metadata
        thresholds: A list of metric thresholds
        current_results_dir: The directory in which to save the results.
        **kwargs: Additional keyword arguments containing task-specific parameters


    Return:
        A dataframe containing the intersectional results.
    """
    aligned_filepaths = cast(List[str], kwargs["aligned_img_filepaths"])
    save_json_filepath = os.path.join(
        current_results_dir, "intersectional_results_VAL.json"
    )
    grouped_results_dict = group_val_scores(
        model_outputs=model_outputs,
        aligned_filepaths=aligned_filepaths,
        annotations_dataframe=annotations_dataframe,
        intersectional_groups=intersectional_groups,
        save_json_filepath=save_json_filepath,
        person_col="person",
        n_pairs=3000,
        multiplicity=2,
        seed=7789,
    )
    return grouped_results_dict


def learned_perceptual_image_patch_similarity(
    task_name: str,
    intersectional_groups: List[str],
    filepaths: List[str],
    model_outputs: dict[str, Any],
    annotations_dataframe: pd.DataFrame,
    thresholds: None,
    current_results_dir: str,
    **kwargs: Dict[str, Any],
) -> pd.DataFrame:
    """Compute LPIPS metric for face encoding task.

    Saves the intersectional results to disk.

    Args:
        task_name: The name of the model task to evaluation
        intersectional_groups: A list of the demographic attributes to split
            the data on.
        filepaths: A list of the filepaths of the images to evaluate.
        model_outputs: Results from running inference on the model
        annotations_dataframe: Contains the annotation metadata
        thresholds: Not used for this metric, but a necessary parameter
        current_results_dir: The directory in which to save the results.
        **kwargs: Additional keyword arguments containing task-specific parameters

    Return:
        A dataframe containing the intersectional results.
    """
    lpips_filename = os.path.join(current_results_dir, "lpips_scores.json")
    if not os.path.isfile(lpips_filename):
        if task_name == "face_encoding":
            lpips_scores = learned_perceptual_image_patch_similarity_score(
                image_paths_1=cast(List[str], kwargs["encoded_filepaths"]),
                image_paths_2=cast(List[str], kwargs["aligned_filepaths"]),
                batch_size=256,
            )
            lpips_result_dict = {f: float(s) for f, s in zip(filepaths, lpips_scores)}
        elif task_name == "face_super_resolution":
            origin_image_filepath = list(model_outputs.keys())
            super_image_filepath = list(
                x["super_res_filename"] for x in model_outputs.values()
            )
            lpips_values = learned_perceptual_image_patch_similarity_score(
                origin_image_filepath, super_image_filepath
            )

            lpips_result_dict = dict(zip(origin_image_filepath, lpips_values))
        save_json_file(lpips_filename, lpips_result_dict, indent=4)
    else:
        lpips_result_dict = read_json_file(lpips_filename)

    save_json_filepath = os.path.join(
        current_results_dir, "intersectional_results_LPIPS.json"
    )
    if task_name == "face_encoding":
        grouped_results_dict = group_face_metric_scores(
            scores=lpips_result_dict,
            metric_name="LPIPS",
            annotations_dataframe=annotations_dataframe,
            intersectional_groups=intersectional_groups,
            save_json_filepath=save_json_filepath,
        )
    elif task_name == "face_super_resolution":
        grouped_results_dict = group_face_metric_scores(
            scores=lpips_result_dict,
            metric_name="LPIPS",
            annotations_dataframe=annotations_dataframe,
            intersectional_groups=intersectional_groups,
            save_json_filepath=save_json_filepath,
        )

    return grouped_results_dict


def curricular_face_score(
    task_name: str,
    intersectional_groups: List[str],
    filepaths: List[str],
    model_outputs: dict[str, Any],
    annotations_dataframe: pd.DataFrame,
    thresholds: List[float],
    current_results_dir: str,
    **kwargs: Dict[str, Any],
) -> pd.DataFrame:  # pragma: no cover
    """Compute embedding similarity score using the curricular face model.

    This is use for the face encoding task.

    Saves the intersectional results to disk.

    Args:
        task_name: The name of the model task to evaluation
        intersectional_groups: A list of the demographic attributes to split
            the data on.
        filepaths: A list of the filepaths of the images to evaluate.
        model_outputs: Results from running inference on the model
        annotations_dataframe: Contains the annotation metadata
        thresholds: A list of metric thresholds
        current_results_dir: The directory in which to save the results.
        **kwargs: Additional keyword arguments containing task-specific parameters

    Return:
        A dataframe containing the intersectional results.
    """
    cfs_filename = os.path.join(current_results_dir, "curricular_face_scores.json")
    if not os.path.isfile(cfs_filename):
        cfs_scores = cfs_builtin(
            image_paths_1=kwargs["encoded_filepaths"],
            image_paths_2=kwargs["aligned_filepaths"],
            batch_size=256,
            images_aligned=False,
        )
        cfs_result_dict = {f: float(s) for f, s in zip(filepaths, cfs_scores)}
        save_json_file(cfs_filename, cfs_result_dict, indent=4)
    else:
        cfs_result_dict = read_json_file(cfs_filename)

    save_json_filepath = os.path.join(
        current_results_dir, "intersectional_results_CURRICULAR_FACE.json"
    )
    grouped_results_dict = group_face_metric_scores(
        scores=cfs_result_dict,
        metric_name="CURRICULAR_FACE",
        annotations_dataframe=annotations_dataframe,
        intersectional_groups=intersectional_groups,
        save_json_filepath=save_json_filepath,
    )

    return grouped_results_dict


def peak_signal_to_noise_ratio(
    task_name: str,
    intersectional_groups: List[str],
    filepaths: List[str],
    model_outputs: dict[str, Any],
    annotations_dataframe: pd.DataFrame,
    thresholds: List[float],
    current_results_dir: str,
    **kwargs: Dict[str, Any],
) -> pd.DataFrame:  # pragma: no cover
    """Compute LPIPS metric for face encoding task.

    Saves the intersectional results to disk.

    Args:
        task_name: The name of the model task to evaluation
        intersectional_groups: A list of the demographic attributes to split
            the data on.
        filepaths: A list of the filepaths of the images to evaluate.
        model_outputs: Results from running inference on the model
        annotations_dataframe: Contains the annotation metadata
        thresholds: A list of metric thresholds
        current_results_dir: The directory in which to save the results.
        **kwargs: Additional keyword arguments containing task-specific parameters

    Return:
        A dataframe containing the intersectional results.
    """
    psnr_filename = os.path.join(current_results_dir, "psnr_scores.json")
    if not os.path.isfile(psnr_filename):
        psnr_scores = peak_signal_to_noise_ratio_score(
            image_paths_1=cast(List[str], kwargs["encoded_filepaths"]),
            image_paths_2=cast(List[str], kwargs["aligned_filepaths"]),
            batch_size=256,
        )
        psnr_result_dict = {f: float(s) for f, s in zip(filepaths, psnr_scores)}
        save_json_file(psnr_filename, psnr_result_dict, indent=4)
    else:
        psnr_result_dict = read_json_file(psnr_filename)

    save_json_filepath = os.path.join(
        current_results_dir, "intersectional_results_PSNR.json"
    )
    grouped_results_dict = group_face_metric_scores(
        scores=psnr_result_dict,
        metric_name="PSNR",
        annotations_dataframe=annotations_dataframe,
        intersectional_groups=intersectional_groups,
        save_json_filepath=save_json_filepath,
    )

    return grouped_results_dict


METRIC_FUNCTION_MAPPER = {
    "AR_IOU": average_recall_bbox,
    "F1": f1_scores_parsing,
    "PCK": percentage_correct_keypoints,
    "AR_OKS": object_keypoint_similarity,
    "AR_MASK": average_recall_mask,
    "AR_DET": average_recall_body_part_detection,
    "ACC_DET": accuracy_body_part_detection,
    "VAL": validation_rate_face_verification,
    "LPIPS": learned_perceptual_image_patch_similarity,
    "CURRICULAR_FACE": curricular_face_score,
    "PSNR": peak_signal_to_noise_ratio,
}
