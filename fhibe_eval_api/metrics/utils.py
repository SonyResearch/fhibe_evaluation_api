# SPDX-License-Identifier: Apache-2.0
"""Module containing utilities for metrics.

This module contains helper functions for metrics.
"""

import itertools
import random
from typing import Any, Dict, List

import numpy as np
import pandas as pd
from pycocotools.mask import decode
from tqdm import tqdm

from fhibe_eval_api.common.metrics import (
    best_iou_scores_for_gt_boxes,
    best_iou_scores_for_gt_masks,
)
from fhibe_eval_api.common.utils import eval_custom, save_json_file
from fhibe_eval_api.evaluate.constants import (
    ATTRIBUTE_CONSOLIDATION_DICT,
    MULTI_SELECTION_ATTRIBUTES,
)
from fhibe_eval_api.metrics.face_verification.utils import (
    evaluate,
    get_negative_pairs,
    get_positive_pairs,
)


def compute_gt_bbox_iou_scores(
    filepaths: list[str],
    model_outputs: dict[str, Any],
    annotations_dataframe: pd.DataFrame,
    gt_column_name: str,
) -> List[float]:
    """Calculate the iou scores for each ground-truth bbox.

    Args:
        filepaths (List[str]): List of unique image filepaths.
        model_outputs: Dictionary of model outputs containing
            bbox predictions and bbox scores.
        annotations_dataframe: Dataframe containing information such as
            ground-truth bboxes.
        gt_column_name: The column name in the dataframe for containing the
            ground-truth bboxes.
        save_json_filepath: Filepath to save the results to.

    Return:
        List of best iou values for each ground truth bbox
        against all predicted bboxes.
    """
    gt_iou_scores = np.array([0] * len(annotations_dataframe), dtype=float)
    with tqdm(
        total=len(filepaths), desc="[compute_gt_bbox_iou_scores] Computing IoU"
    ) as pbar:
        for filepath in filepaths:
            pred_bboxes = model_outputs[filepath]["detections"]
            if pred_bboxes is not None:
                annotations_dataframe_subset = annotations_dataframe[
                    annotations_dataframe["filepath"] == filepath
                ]
                gt_bboxes = [
                    gt_bbox
                    for gt_bbox in annotations_dataframe_subset.loc[
                        :, gt_column_name
                    ].tolist()
                ]
                for gt_bbox in gt_bboxes:
                    assert len(gt_bbox) == 4 and isinstance(gt_bbox, list)

                best_iou_scores = best_iou_scores_for_gt_boxes(pred_bboxes, gt_bboxes)
                gt_iou_scores[annotations_dataframe_subset.index.tolist()] = (
                    best_iou_scores
                )

            pbar.update(1)
    return list(gt_iou_scores)


def compute_gt_mask_iou_scores(
    filepaths: list[str],
    model_outputs: dict[str, Any],
    annotations_dataframe: pd.DataFrame,
    gt_mask_column_name: str,
    to_rle: bool,
) -> List[float]:
    """Calculate the iou scores for each ground-truth segmentation mask.

    Args:
        filepaths: List of unique image filepaths.
        model_outputs: Dictionary of model outputs containing
            bbox predictions and bbox scores.
        annotations_dataframe: Dataframe containing information such as
            ground-truth bboxes.
        gt_mask_column_name: The column name in the dataframe for containing the
            ground-truth segmentation masks.
        to_rle: Whether the masks are run-length encoded and need to be decoded

    Return:
        List of best iou values for each ground truth bbox
        against all predicted bboxes.
    """
    if gt_mask_column_name not in annotations_dataframe.columns:
        raise ValueError(
            f"{gt_mask_column_name} is not a column in the provided dataframe."
        )

    # iou scores should be in the order of the dataframe,
    # which is potentially different from the order in filepaths (the unique list)
    gt_mask_iou_scores = np.array([0] * len(annotations_dataframe), dtype=float)

    with tqdm(
        total=len(filepaths), desc="[compute_gt_mask_iou_scores] Computing IoU"
    ) as pbar:
        for filepath in filepaths:
            pred_masks = model_outputs[filepath]["detections"]
            if pred_masks is not None:
                annotations_dataframe_subset = annotations_dataframe[
                    annotations_dataframe["filepath"] == filepath
                ]
                if to_rle:
                    pred_masks = [decode(pm) for pm in pred_masks]

                    gt_masks = [
                        decode(gtm)
                        for gtm in annotations_dataframe_subset.loc[
                            :, gt_mask_column_name
                        ].tolist()
                    ]

                else:
                    gt_masks = [
                        gt_mask
                        for gt_mask in annotations_dataframe_subset.loc[
                            :, gt_mask_column_name
                        ].tolist()
                    ]
                best_iou_scores: List[float] = best_iou_scores_for_gt_masks(
                    pred_masks, gt_masks
                )
                assert 1 <= len(best_iou_scores) <= 2
                if len(best_iou_scores) == 1:
                    gt_mask_iou_scores[annotations_dataframe_subset.index.tolist()] = (
                        best_iou_scores
                    )
                elif len(best_iou_scores) == 2:
                    if len(annotations_dataframe_subset.index.tolist()) == 2:
                        gt_mask_iou_scores[
                            annotations_dataframe_subset.index.tolist()
                        ] = best_iou_scores
                    else:
                        # When using mini dataset and an image contains two people,
                        # it is possible that the annotation dataframe has a single row
                        # for the image, so we choose the best bbox iou of the two.
                        gt_mask_iou_scores[
                            annotations_dataframe_subset.index.tolist()
                        ] = [max(best_iou_scores)]

                pbar.update(1)
    return list(gt_mask_iou_scores)


def gather_body_part_detection_scores(
    filepaths: list[str],
    model_outputs: dict[str, Any],
    annotations_dataframe: pd.DataFrame,
) -> List[Dict[str, float]]:
    """Gather the body part detections for each entry in annotations_dataframe.

    Args:
        filepaths (List[str]): List of unique image filepaths.
        model_outputs: Dictionary of model outputs containing
            bbox predictions and bbox scores.
        annotations_dataframe: Dataframe containing information such as
            ground-truth bboxes.
        save_json_filepath: Filepath to save the results to.

    Return:
        List of body part detection dictionaries
    """
    bp_det_dict_list: List[Dict[str, float]] = [
        {} for _ in range(len(annotations_dataframe))
    ]
    with tqdm(
        total=len(filepaths),
        desc="[gather_body_part_detection_scores] Assigning detections to subjects",
    ) as pbar:
        for filepath in filepaths:
            # detections is a list of dicts, one for each
            # ground truth person bbox
            detections = model_outputs[filepath]["detections"]
            if detections is not None:
                annotations_dataframe_subset = annotations_dataframe[
                    annotations_dataframe["filepath"] == filepath
                ]
                assert 1 <= len(detections) <= 2
                index_list = annotations_dataframe_subset.index.tolist()
                for ii, index in enumerate(index_list):
                    bp_det_dict_list[index] = detections[ii]

            pbar.update(1)
    return bp_det_dict_list


def save_thresholded_results_per_file(
    annotations_dataframe: pd.DataFrame,
    metric_vals: List[float],
    results_path: str,
    detailed_results_path: str,
    thresholds: List[float],
) -> Dict[str, Any]:
    """Save thresholded metric results for each image.

    Saves summarized results to results_path and detailed results
    to detailed_results_path.

    Args:
        annotations_dataframe: Dataframe containing information such as
            ground-truth bboxes.
        metric_vals: A list of the metric values for each corresponding image in the
            dataframe.
        results_path: Filepath to save the summarized per-image results.
        detailed_results_path: Filepath to save the detailed per-image results.
        thresholds: List of metric thresholds

    Return:
        The summarized results dictionary.
    """
    results: Dict[str, Any] = {}
    detailed_results: Dict[str, Dict[str, Any]] = {}
    for thr in thresholds:
        threshold_key = f"{thr:.2f}"
        results[threshold_key] = {
            str(k): int(v >= thr)
            for k, v in zip(annotations_dataframe.index.tolist(), metric_vals)
        }

        detailed_results[threshold_key] = {
            "summary": "0.0",
            "individual_results": [],
        }
        correct = 0
        for idx, val in zip(annotations_dataframe.index.tolist(), metric_vals):
            current_gt = {
                "idx": idx,
                "val": val,
                "accept": int(val >= thr),
            }
            if val >= thr:
                correct += 1
            for key in (
                "image_id",  # fhibe
                "subject_id",  # fhibe and coco
                "filepath",  # all
                "person_id",  # facet
                "filename",  # facet
                "annId",  # coco
                "uid_ori",  # coco
                "ImageID",  # open_images_miap
            ):
                if key in annotations_dataframe.columns:
                    if isinstance(annotations_dataframe.iloc[idx][key], np.int64):
                        current_gt[key] = int(annotations_dataframe.iloc[idx][key])
                    else:
                        current_gt[key] = annotations_dataframe.iloc[idx][key]

            detailed_results[threshold_key]["individual_results"].append(current_gt)

        detailed_results[threshold_key][
            "summary"
        ] = f"{(float(correct) / len(annotations_dataframe)):.2f}"

    save_json_file(filepath=results_path, data=results, indent=4)

    # different file for compatibility with existing code.
    save_json_file(filepath=detailed_results_path, data=detailed_results, indent=4)
    return results


def save_thresholded_body_part_results(
    annotations_dataframe: pd.DataFrame,
    body_parts_dict_list: List[Dict[str, float]],
    results_path: str,
    detailed_results_path: str,
    thresholds: List[float],
) -> Dict[str, Dict[int, Dict[str, int]]]:
    """Save thresholded metric results for the body parts detection task.

    Saves summarized results to results_path and detailed results
    to detailed_results_path.

    Args:
        annotations_dataframe: Dataframe containing information such as
            ground-truth bboxes.
        body_parts_dict_list: A list of dictionaries, where each dict maps
            the string body part to the detection probability. The list indices
            match the annotation_dataframe order.
        results_path: Filepath to save the summarized per-image results.
        detailed_results_path: Filepath to save the detailed per-image results.
        thresholds: List of metric thresholds

    Return:
        The summarized results dictionary.
    """
    results: Dict[str, Dict[int, Dict[str, int]]] = {}
    detailed_results: Dict[str, Dict[str, Any]] = {}
    for thr in thresholds:
        threshold_key = f"{thr:.2f}"
        results[threshold_key] = {}
        for k, bp_dict in zip(
            annotations_dataframe.index.tolist(), body_parts_dict_list
        ):
            results[threshold_key][str(k)] = {
                bp: int(v >= thr) for bp, v in bp_dict.items()
            }

        detailed_results[threshold_key] = {
            "summary": {},
            "individual_results": [],
        }
        bp_above_threshold_dict = {
            bp: 0 for bp in bp_dict.keys()
        }  # keep track of # detected by body part
        for idx, bp_dict in zip(
            annotations_dataframe.index.tolist(), body_parts_dict_list
        ):
            current_gt = {
                "idx": idx,
            }
            for bp, v in bp_dict.items():
                current_gt[bp] = {"val": v, f"accept_{bp}": int(v >= thr)}
                if int(v >= thr) == 1:
                    bp_above_threshold_dict[bp] += 1
            for key in (
                "image_id",  # fhibe
                "subject_id",  # fhibe and coco
                "filepath",  # all
                "person_id",  # facet
                "filename",  # facet
                "annId",  # coco
                "uid_ori",  # coco
                "ImageID",  # open_images_miap
            ):
                if key in annotations_dataframe.columns:
                    if isinstance(annotations_dataframe.iloc[idx][key], np.int64):
                        current_gt[key] = int(annotations_dataframe.iloc[idx][key])
                    else:
                        current_gt[key] = annotations_dataframe.iloc[idx][key]

            detailed_results[threshold_key]["individual_results"].append(current_gt)

        for bp in bp_dict:
            detailed_results[threshold_key]["summary"][
                bp
            ] = f"{(bp_above_threshold_dict[bp] / len(annotations_dataframe)):.2f}"

    save_json_file(filepath=results_path, data=results, indent=4)

    # different file for compatibility with existing code.
    save_json_file(filepath=detailed_results_path, data=detailed_results, indent=4)
    return results


def group_thresholded_metric_results(
    threshold_dict,
    metric_name,
    annotations_dataframe,
    intersectional_groups,
    save_json_filepath,
    **kwargs,
) -> Dict[str, Dict[str, Dict[str, List | float | int]]]:
    """Calculate metrics in intersectional groups.

    This function is used to group metrics that are thresholded,
    such as the average recall metrics and percentage correct keypoints.
    Metrics are aggregated over each attribute individually as well as
    all combinations of attributes. Attributes with multiple selections
    are separated into individual selections.

    Args:
        threshold_dict: Dictionary mapping threshold value to list of
            values indicating whether metric value was above threshold
        metric_name: The name of the metric
        annotations_dataframe: Dataframe containing annotation info
        intersectional_groups: List of attributes to intersect and aggregate over.
        save_json_filepath: Filepath to save the results to.
        kwargs: Additional keyword arguments, such as custom keypoints list

    Return:
        Dictionary of intersectional results
    """
    result_dict = {}  # Maps attribute combo to dict of

    # attribute_val: {"scores": list, metric_name: float, class_size: int}
    # Get individual selections from all multi-selection attrs
    indiv_selections_dict = {}
    for attr in intersectional_groups:
        if attr in MULTI_SELECTION_ATTRIBUTES:
            class_names = get_individual_selections(annotations_dataframe, attr)
            indiv_selections_dict[attr] = class_names

    combinations = []
    for r in range(1, len(intersectional_groups) + 1):
        combinations.extend(itertools.combinations(intersectional_groups, r))

    for column_combination in combinations:
        column_combination_list = list(column_combination)
        attr_key = str(column_combination_list)
        result_dict[attr_key] = {}  # Top level key is attribute combination

        # Make a column that is a list of all possible combinations of attributes
        annotations_dataframe["group_label_list"] = annotations_dataframe[
            column_combination_list
        ].apply(
            lambda x: make_group_labels(
                x, column_combination_list, indiv_selections_dict
            ),
            axis=1,
        )

        # Store a list of scores for each attribute as well as mean score and class size
        for ix, attr_val_list in enumerate(
            annotations_dataframe["group_label_list"].tolist()
        ):
            if metric_name in ["AR_IOU", "AR_OKS", "AR_MASK"]:
                # Calculate the score, # of acceptances / # of thresholds
                n_accept = sum([threshold_dict[thr][str(ix)] for thr in threshold_dict])
                score = n_accept / len(threshold_dict)
            elif metric_name == "PCK":
                score = threshold_dict["results"][str(ix)][
                    "mean_percentage_correct_keypoints"
                ]
            else:
                raise NotImplementedError(
                    f"This function is not implemented for metric: {metric_name}"
                )

            # Loop through all combinations of attributes for this entry
            # There could be multiple because some attributes have multiple selections
            for class_name in attr_val_list:
                class_name_key = str(class_name)
                if class_name_key in result_dict[attr_key]:
                    if ~np.isnan(score):
                        result_dict[attr_key][class_name_key]["scores"].append(score)
                else:
                    if ~np.isnan(score):
                        result_dict[attr_key][class_name_key] = {
                            "scores": [score],
                        }
        # Calculate mean score and class size for each attribute combo
        # and add them to the dict
        for class_name, class_dict in result_dict[attr_key].items():
            class_dict[metric_name] = np.mean(class_dict["scores"])
            class_dict["Class_Size"] = len(class_dict["scores"])

    # Finally, sort the subdictionaries so the attribute values are ordered
    for attr_key in result_dict:
        sorted_keys = sorted(
            result_dict[attr_key], key=lambda x: sorting_function(attr_key, x)
        )
        sorted_sub_dict = {key: result_dict[attr_key][key] for key in sorted_keys}
        result_dict[attr_key] = sorted_sub_dict
    if kwargs.get("custom_keypoints"):
        result_dict["custom_keypoints"] = kwargs["custom_keypoints"]
    save_json_file(save_json_filepath, result_dict, indent=4)
    return result_dict


def group_thresholded_body_part_results(
    threshold_dict: Dict[str, Any],
    metric_name: str,
    annotations_dataframe: pd.DataFrame,
    intersectional_groups: List[str],
    save_json_filepath: str,
) -> Dict[str, Dict[str, Dict[str, int | float]]]:
    """Calculate body part metrics in intersectional groups.

    This function is used to group metrics for the body_parts_detection task
    that are thresholded, such as the average recall and average precision.
    The metrics are calculated for each body part separately and
    also averaged over each body part.

    Args:
        threshold_dict: Dictionary mapping threshold value to list of
            values indicating whether metric value was above threshold,
            for each body part.
        metric_name: The name of the metric
        annotations_dataframe: Dataframe containing annotation info
        intersectional_groups: List of attributes to split the data on.
        save_json_filepath: Filepath to save the results to.

    Return:
        Dictionary containing intersectional results,
            broken down by body parts within each group.
    """
    result_dict = {}

    # Get individual selections from all multi-selection attrs
    indiv_selections_dict = {}
    for attr in intersectional_groups:
        if attr in MULTI_SELECTION_ATTRIBUTES:
            class_names = get_individual_selections(annotations_dataframe, attr)
            indiv_selections_dict[attr] = class_names

    thresholds = list(threshold_dict.keys())
    # Get list of body parts predicted by this model
    first_key = list(threshold_dict[thresholds[0]].keys())[0]
    all_predicted_body_parts = list(threshold_dict[thresholds[0]][first_key].keys())

    # Make a new column of the visible ground truth body parts
    # that are also body parts predicted by this model
    # this could differ by image, so it needs its own column.
    annotations_dataframe["visible_body_parts_predicted"] = annotations_dataframe[
        "visible_body_parts"
    ].apply(lambda y: [x for x in y if x in all_predicted_body_parts])

    combinations = []
    for r in range(1, len(intersectional_groups) + 1):
        combinations.extend(itertools.combinations(intersectional_groups, r))

    for column_combination in combinations:
        column_combination_list = list(column_combination)
        attr_key = str(column_combination_list)
        result_dict[attr_key] = {}  # Top level key is attribute combination

        # Make a column that is a list of all possible combinations of attributes
        annotations_dataframe["group_label_list"] = annotations_dataframe[
            column_combination_list
        ].apply(
            lambda x: make_group_labels(
                x, column_combination_list, indiv_selections_dict
            ),
            axis=1,
        )

        # Store a list of scores for each attribute as well as mean score and class size
        for ix, attr_val_list in enumerate(
            annotations_dataframe["group_label_list"].tolist()
        ):
            visible_body_parts = annotations_dataframe[
                "visible_body_parts_predicted"
            ].iloc[ix]

            if metric_name == "AR_DET":
                # Recall = TP / (TP + FN) so only consider examples
                # where the body part was present in the ground truth
                score_dict = {
                    bp: sum(
                        [threshold_dict[thr][str(ix)][bp] for thr in threshold_dict]
                    )
                    / len(threshold_dict)
                    for bp in visible_body_parts
                }

            elif metric_name == "ACC_DET":
                score_dict = {}
                for bp in all_predicted_body_parts:
                    if bp in visible_body_parts:
                        score_dict[bp] = sum(
                            [threshold_dict[thr][str(ix)][bp] for thr in threshold_dict]
                        ) / len(threshold_dict)
                    else:
                        score_dict[bp] = 1.0 - sum(
                            [threshold_dict[thr][str(ix)][bp] for thr in threshold_dict]
                        ) / len(threshold_dict)

            # Loop through all combinations of attributes for this entry
            # There could be multiple because some attributes have multiple selections
            for class_name in attr_val_list:
                class_name_key = str(class_name)
                if class_name_key in result_dict[attr_key]:
                    for bp, score in score_dict.items():
                        result_dict[attr_key][class_name_key][bp]["scores"].append(
                            score
                        )
                else:
                    result_dict[attr_key][class_name_key] = {}
                    for bp in all_predicted_body_parts:
                        if bp in score_dict:
                            result_dict[attr_key][class_name_key][bp] = {
                                "scores": [score_dict[bp]]
                            }
                        else:
                            result_dict[attr_key][class_name_key][bp] = {"scores": []}

        # Calculate mean score and class size for each attribute combo
        # and add them to the dict for each body part
        for class_name, class_dict in result_dict[attr_key].items():
            for bp in class_dict:
                class_dict[bp][metric_name] = np.mean(class_dict[bp]["scores"])
                class_dict[bp]["Class_Size"] = len(class_dict[bp]["scores"])

    # Finally, sort the subdictionaries so the attribute values are ordered
    for attr_key in result_dict:
        sorted_keys = sorted(
            result_dict[attr_key], key=lambda x: sorting_function(attr_key, x)
        )
        sorted_sub_dict = {key: result_dict[attr_key][key] for key in sorted_keys}
        result_dict[attr_key] = sorted_sub_dict

    save_json_file(save_json_filepath, result_dict, indent=4)
    return result_dict


def group_f1_results(
    all_f1_scores: Dict[str, Any],
    annotations_dataframe: pd.DataFrame,
    intersectional_groups: List[str],
    save_json_filepath: str,
) -> pd.DataFrame:
    """Create face parsing results table.

    Args:
        all_f1_scores: Dictionary containing f1 score results for
            each ground-truth mask.
        annotations_dataframe: Dataframe containing information such as
            ground-truth bboxes.
        intersectional_groups: List of attributes to split the data on.
        save_json_filepath: Filepath to save the results to.

    Return:
        Dictionary containing intersectional results
    """
    metric_name = "F1"
    result_dict = {}  # Maps attribute combo to dict of
    # attribute_val: {"scores": list, metric_name: float, class_size: int}
    # Get individual selections from all multi-selection attrs
    indiv_selections_dict = {}
    for attr in intersectional_groups:
        if attr in MULTI_SELECTION_ATTRIBUTES:
            class_names = get_individual_selections(annotations_dataframe, attr)
            indiv_selections_dict[attr] = class_names

    combinations = []
    for r in range(1, len(intersectional_groups) + 1):
        combinations.extend(itertools.combinations(intersectional_groups, r))

    for column_combination in combinations:
        column_combination_list = list(column_combination)
        attr_key = str(column_combination_list)
        result_dict[attr_key] = {}  # Top level key is attribute combination

        # Make a column that is a list of all possible combinations of attributes
        annotations_dataframe["group_label_list"] = annotations_dataframe[
            column_combination_list
        ].apply(
            lambda x: make_group_labels(
                x, column_combination_list, indiv_selections_dict
            ),
            axis=1,
        )

        # Store a list of scores for each attribute as well as mean score and class size
        for f1_dict, attr_val_list in zip(
            all_f1_scores.values(),
            annotations_dataframe["group_label_list"].tolist(),
        ):
            # The score for this subject is the mean f1 score over the body parts
            score = np.mean(list(f1_dict.values()))

            # Loop through all combinations of attributes for this entry
            # There could be multiple because some attributes have multiple selections
            for class_name in attr_val_list:
                class_name_key = str(class_name)
                if class_name_key in result_dict[attr_key]:
                    result_dict[attr_key][class_name_key]["scores"].append(score)
                else:
                    result_dict[attr_key][class_name_key] = {
                        "scores": [score],
                    }

        # Calculate mean score and class size for each attribute combo
        # and add them to the dict
        for class_name, class_dict in result_dict[attr_key].items():
            class_dict[metric_name] = np.mean(class_dict["scores"])
            class_dict["Class_Size"] = len(class_dict["scores"])

    # Finally, sort the subdictionaries so the attribute values are ordered
    for attr_key in result_dict:
        sorted_keys = sorted(
            result_dict[attr_key], key=lambda x: sorting_function(attr_key, x)
        )
        sorted_sub_dict = {key: result_dict[attr_key][key] for key in sorted_keys}
        result_dict[attr_key] = sorted_sub_dict
    save_json_file(save_json_filepath, result_dict, indent=4)
    return result_dict


def group_val_scores(
    model_outputs: Dict[str, Any],
    aligned_filepaths: List[str],
    annotations_dataframe: pd.DataFrame,
    intersectional_groups: List[str],
    save_json_filepath: str,
    person_col: str = "person",
    n_pairs: int = 3000,
    multiplicity: int = 2,
    seed: int = 7789,
) -> pd.DataFrame:  # pragma: no cover
    """Create face verification results table.

    Args:
        model_outputs: Dictionary containing results of whether a
            ground-truth bbox was detected.
        aligned_filepaths: List of filepaths to the aligned images
        annotations_dataframe: Dataframe containing information such as
            ground-truth bboxes.
        intersectional_groups: List of attributes to split the data on.
        save_json_filepath: Filepath to save the results to.
        person_col: The column in the dataframe containing the subject id
        n_pairs: The max number of pairs to use
        multiplicity: If n_pairs not provided, applies this multiplier to
            the number of positive pairs to obtain the number of negative pairs
        seed: Random seed.

    Return:
        Dictionary containing intersectional results
    """
    annotations_dataframe["aligned_filepath"] = aligned_filepaths
    result_dict = {}  # Maps attribute combo to dict of
    # attribute_val: {"scores": list, metric_name: float, class_size: int}
    # Get individual selections from all multi-selection attrs
    indiv_selections_dict = {}
    for attr in intersectional_groups:
        if attr in MULTI_SELECTION_ATTRIBUTES:
            class_names = get_individual_selections(annotations_dataframe, attr)
            indiv_selections_dict[attr] = class_names

    combinations = []
    for r in range(1, len(intersectional_groups) + 1):
        combinations.extend(itertools.combinations(intersectional_groups, r))

    for column_combination in combinations:
        column_combination_list = list(column_combination)
        attr_key = str(column_combination_list)
        result_dict[attr_key] = {}  # Top level key is attribute combination

        # Make a column that is a list of all possible combinations of attributes
        annotations_dataframe["group_label_list"] = annotations_dataframe[
            column_combination_list
        ].apply(
            lambda x: make_group_labels(
                x, column_combination_list, indiv_selections_dict
            ),
            axis=1,
        )

        # For each unique combination of attributes, find the subset of the dataframe
        # possessing this attribute combo and calculate val
        unique_class_names = set()
        for attr_val_list in annotations_dataframe["group_label_list"].tolist():
            for attr_val in attr_val_list:
                unique_class_names.add(tuple(attr_val))

        for class_name in unique_class_names:
            subset_df = annotations_dataframe.loc[
                annotations_dataframe["group_label_list"].apply(
                    lambda clist: list(class_name) in clist
                )
            ]
            # Make sure that we keep only entries for which we have model predictions
            subj_img_paths = [x.split("/")[-2:] for x in list(model_outputs.keys())]
            subset_df = subset_df[
                subset_df["aligned_filepath"]
                .apply(lambda x: x.split("/")[-2:])
                .isin(subj_img_paths)
            ]
            subset_df.reset_index(inplace=True, drop=True)
            # Ensure data have at least 2 images per subject
            subset_df = subset_df.groupby(person_col).filter(lambda x: len(x) >= 2)
            people = subset_df[person_col].unique().tolist()
            if (
                not len(people) >= 2
            ):  # Ensure that data have at least 2 different subjects
                continue

            people_im_count = {x: 0 for x in people}
            people_im_id = {x: [] for x in people}  # type: ignore

            for i, (fpath, name) in enumerate(
                zip(subset_df["aligned_filepath"], subset_df[person_col])
            ):
                people_im_count[name] += 1
                people_im_id[name].append(fpath)

            random.seed(seed)

            same_pairs = get_positive_pairs(
                subset_df,
                filepath_col="aligned_filepath",
                person_col=person_col,
                n_pairs=n_pairs,
            )  # type: ignore

            if n_pairs is not None:
                n_pairs_neg = n_pairs
            else:
                n_pairs_neg = multiplicity * len(same_pairs)

            diff_pairs = get_negative_pairs(
                subset_df,
                filepath_col="aligned_filepath",
                person_col=person_col,
                n_pairs=n_pairs_neg,
            )  # type: ignore

            same_pairs = [list(x) for x in same_pairs]
            diff_pairs = [list(x) for x in diff_pairs]
            # if len(same_pairs) < 3000 or len(diff_pairs) < 3000:
            #     continue
            # NOTE: in case we cannot find same/diff pairs for the particular
            # intersectional combination, skip
            if not len(same_pairs) or not len(diff_pairs):
                print(
                    f"Failed to find pairs.\nSame pairs: {len(same_pairs)}\n"
                    f"Diff pairs: {len(diff_pairs)}"
                )
                continue

            actual_issame = [1] * len(same_pairs) + [0] * len(diff_pairs)
            embeddings1 = np.zeros(
                (
                    len(same_pairs) + len(diff_pairs),
                    len(model_outputs[same_pairs[0][0]]["detections"]),
                )
            )
            embeddings2 = np.zeros(
                (
                    len(same_pairs) + len(diff_pairs),
                    len(model_outputs[same_pairs[0][0]]["detections"]),
                )
            )

            all_pairs = same_pairs + diff_pairs
            for pair_i, paired_paths in enumerate(all_pairs):
                embeddings1[pair_i] = np.array(
                    model_outputs[paired_paths[0]]["detections"]
                )
                embeddings2[pair_i] = np.array(
                    model_outputs[paired_paths[1]]["detections"]
                )

            dist = (embeddings1 * embeddings2).sum(1)
            dist = dist * (-1)

            random.seed(11)
            arr_idxs = list(range(len(actual_issame)))
            random.shuffle(arr_idxs)
            dist = dist[arr_idxs]
            embeddings1 = embeddings1[arr_idxs]
            embeddings2 = embeddings2[arr_idxs]
            actual_issame = [actual_issame[x] for x in arr_idxs]

            nrof_folds = min(embeddings1.shape[0], 10)
            tpr, fpr, accuracy, val, val_std, far, fp, fn = evaluate(
                embeddings1, embeddings2, dist, actual_issame, nrof_folds=nrof_folds
            )  # type: ignore

            result_dict[attr_key][str(list(class_name))] = {
                "VAL": np.around(val, 2),
                "VAL_Error": np.around(val_std, 2),
                "Class_Size": len(subset_df),
                "Identities": len(people),
                "Same / Different Pairs": str(len(same_pairs))
                + "/"
                + str(len(diff_pairs)),
            }

    # Finally, sort the subdictionaries so the attribute values are ordered
    for attr_key in result_dict:
        sorted_keys = sorted(
            result_dict[attr_key], key=lambda x: sorting_function(attr_key, x)
        )
        sorted_sub_dict = {key: result_dict[attr_key][key] for key in sorted_keys}
        result_dict[attr_key] = sorted_sub_dict

    save_json_file(save_json_filepath, result_dict, indent=4)
    return result_dict


def group_face_metric_scores(
    scores: Dict[str, Any],
    metric_name: str,
    annotations_dataframe: pd.DataFrame,
    intersectional_groups: List[str],
    save_json_filepath: str,
) -> pd.DataFrame:
    """Aggregate metrics from several FHIBE-face tasks.

    This function works for the metrics which provide a single metric per image,
    i.e., LPIPS, PSNR, CURRICULAR_FACE.

    Args:
        scores: Dictionary containing metric scores,
            e.g., filename: lpips_val
        metric_name: The name of the metric used to evaluate the task.
        annotations_dataframe: Dataframe containing information such as
            ground-truth bboxes.
        intersectional_groups: List of attributes to split the data on.
        save_json_filepath: Filepath to save the results to.

    Return:
        Dictionary containing intersectional results
    """
    result_dict = {}  # Maps attribute combo to dict of
    # attribute_val: {"scores": list, metric_name: float, class_size: int}
    # Get individual selections from all multi-selection attrs
    indiv_selections_dict = {}
    for attr in intersectional_groups:
        if attr in MULTI_SELECTION_ATTRIBUTES:
            class_names = get_individual_selections(annotations_dataframe, attr)
            indiv_selections_dict[attr] = class_names

    combinations = []
    for r in range(1, len(intersectional_groups) + 1):
        combinations.extend(itertools.combinations(intersectional_groups, r))

    for column_combination in combinations:
        column_combination_list = list(column_combination)
        attr_key = str(column_combination_list)
        result_dict[attr_key] = {}  # Top level key is attribute combination

        # Make a column that is a list of all possible combinations of attributes
        annotations_dataframe["group_label_list"] = annotations_dataframe[
            column_combination_list
        ].apply(
            lambda x: make_group_labels(
                x, column_combination_list, indiv_selections_dict
            ),
            axis=1,
        )

        # Store a list of scores for each attribute as well as mean score and class size
        for filepath, attr_val_list in annotations_dataframe[
            ["filepath", "group_label_list"]
        ].values:
            score = scores[filepath]

            # Loop through all combinations of attributes for this entry
            # There could be multiple because some attributes have multiple selections
            for class_name in attr_val_list:
                class_name_key = str(class_name)
                if class_name_key in result_dict[attr_key]:
                    result_dict[attr_key][class_name_key]["scores"].append(score)
                else:
                    result_dict[attr_key][class_name_key] = {
                        "scores": [score],
                    }
        # Calculate mean score and class size for each attribute combo
        # and add them to the dict
        for class_name, class_dict in result_dict[attr_key].items():
            class_dict[metric_name] = np.mean(class_dict["scores"])
            class_dict["Class_Size"] = len(class_dict["scores"])

    # Finally, sort the subdictionaries so the attribute values are ordered
    for attr_key in result_dict:
        sorted_keys = sorted(
            result_dict[attr_key], key=lambda x: sorting_function(attr_key, x)
        )
        sorted_sub_dict = {key: result_dict[attr_key][key] for key in sorted_keys}
        result_dict[attr_key] = sorted_sub_dict
    save_json_file(save_json_filepath, result_dict, indent=4)
    return result_dict


def get_individual_selections(
    annotations_dataframe: pd.DataFrame, attr: str
) -> List[str]:
    """Get a list of all individual selections for a single attribute in the dataframe.

    Also get larger groupings for ancestry and nationality.

    Args:
        annotations_dataframe: Dataframe containing annotation information
        attr: The name of the attribute from which to get individual selections

    Return:
        List of selections sorted by the attribute number, e.g.,
    "0" in "0. She/her/hers".
    """
    all_selections = []
    for sel_str in annotations_dataframe[attr]:
        sel_list = eval_custom(sel_str)
        assert isinstance(sel_list, list)
        for sel in sel_list:
            if sel not in all_selections:
                all_selections.append(sel)

    return sorted(all_selections, key=lambda x: int(x.split(".")[0]))


def make_group_labels(
    attr_val_list, attribute_names, indiv_selections_dict
) -> List[Any]:
    """Make a list of all possible combinations of attributes.

    Some attributes may be multi-selection attributes,
    so we need to account for them.

    Args:
        attr_val_list: List of attribute values
        attribute_names: List of attribute names
        indiv_selections_dict: Dictionary mapping attribute name to all
            possible single select values of the attribute.

    Return:
        List of attribute value combinations, sorted by original attribute order.
    """
    sgl_attr_vals = []  # List of single-attribute values in this attr val
    # list of lists, where each sublist is the list of
    # indiv selections in this attr val
    multi_attr_list = []
    sgl_attr_indices = []  # Indices of single-attribute values
    multi_attr_indices = []  # Indices of multi-selection attributes

    for ix, attr in enumerate(attribute_names):
        attr_val = attr_val_list.iloc[ix]
        if attr not in MULTI_SELECTION_ATTRIBUTES:
            sgl_attr_vals.append(attr_val)
            sgl_attr_indices.append(ix)
        else:
            multi_attr_vals = []
            # Add larger groupings for nationality
            if attr == "nationality":
                region_dict = ATTRIBUTE_CONSOLIDATION_DICT[attr]
                for continent, countries in region_dict.items():
                    if continent in multi_attr_vals:
                        continue
                    if any([country in attr_val for country in countries]):
                        multi_attr_vals.append(continent)
            # Loop through all possible individual selections
            # and see if they are in the attribute string.
            for indiv_sel in indiv_selections_dict[attr]:
                if indiv_sel in attr_val:
                    multi_attr_vals.append(indiv_sel)

            multi_attr_list.append(multi_attr_vals)
            multi_attr_indices.append(ix)
    # Generate all combinations of elements from multi_attr_list
    combinations = list(itertools.product(*multi_attr_list))

    # Sort each combination by original attribute order
    result = []
    for comb in combinations:
        combined_attrs = [None] * len(attribute_names)
        for i, val in enumerate(sgl_attr_vals):
            combined_attrs[sgl_attr_indices[i]] = val
        for i, val in enumerate(comb):
            combined_attrs[multi_attr_indices[i]] = val
        result.append(combined_attrs)
    return result


def sorting_function(attr_key: str, attr_val_str: str) -> List[Any]:
    """A sorting key for the attribute subdicts of intersectional results dict.

    Args:
        attr_key: The attribute name key from the intersectional results dict
        attr_val_str: The attribute value key from the intersectional results dict

    Return:
        List where each element is the sorting key for the
        corresponding attribute. All attributes are sorted
        by the attribute number, e.g., "0" in "0. She/her/hers",
        except age, which can be sorted by the entire string itself.
    """
    attr_key_list = eval(attr_key)
    attr_val_list = eval(attr_val_str)
    sort_key_list = []
    for ix, attr_val in enumerate(attr_val_list):
        attr_name = attr_key_list[ix]
        if attr_name in [
            "age",
            "user_hour_captured",
            "location_country",
            "apparent_skin_color_hue_lum",
        ]:
            sort_key_list.append(attr_val)
        else:
            sort_key_list.append(int(attr_val.split(".")[0].split("['")[-1]))
    return sort_key_list
