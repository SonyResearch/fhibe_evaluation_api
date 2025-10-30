# SPDX-License-Identifier: Apache-2.0
"""Utility functions for model evaluation.

This module contains helper functions for performing the model evaluation.
"""

import logging
import os
from multiprocessing import cpu_count
from typing import Any, Dict, List, Tuple, Union

import numpy as np
import numpy.typing as npt
import pandas as pd
from pycocotools.mask import decode, encode
from tqdm import tqdm

from fhibe_eval_api.common.loggers import setup_logging
from fhibe_eval_api.datasets.fhibe import FHIBEPublicEval
from fhibe_eval_api.datasets.fhibe_face import FHIBEFacePublicEval
from fhibe_eval_api.evaluate.constants import (
    DEFAULT_RANDOM_STATE,
    FHIBE_ATTRIBUTE_LIST,
    FHIBE_FACE_ATTRIBUTE_LIST,
    TASK_DICT,
    VALID_DATASET_NAMES,
)
from fhibe_eval_api.metrics.constants import TASK_METRIC_LOOKUP_DICT
from fhibe_eval_api.models.base_model import BaseModelWrapper

setup_logging("info")


def get_eval_api(
    dataset_name: str,
    dataset_base: str,
    data_dir: str,
    processed_data_dir: str,
    intersectional_column_names: List[str] | None = None,
    use_age_buckets: bool = True,
    use_mini_dataset: bool = False,
    mini_dataset_size: int = 50,
) -> Union[FHIBEPublicEval, FHIBEFacePublicEval]:
    """Obtain the evaluation API class instance for preparing the task.

    Args:
        dataset_name: "fhibe", "fhibe_downsampled", or "fhibe_face_crop_align"
        dataset_base: "fhibe" or "fhibe_face" (strips off qualifiers)
        data_dir: The path to the raw/ data directory
        processed_data_dir: The path to the processed/ data directory
        intersectional_column_names: List of demographic groups
            for aggregation.
        use_age_buckets: Whether to use age buckets.
        use_mini_dataset: Whether to use the mini dataset.
        mini_dataset_size: The # of datapoints to sample if use_mini_dataset=True

    Return:
        The evaluation API class instance.
    """
    print("processed_data_dir: ", processed_data_dir)
    annotations_csv_fp = os.path.join(
        processed_data_dir,
        dataset_name,
        f"{dataset_name}.csv",
    )
    if not os.path.isfile(annotations_csv_fp):
        raise RuntimeError(
            f"Combined annotation file not found: {annotations_csv_fp}. "
        )
    dataframe = pd.read_csv(annotations_csv_fp)

    # Update the filepath (image) and json_path (annotation) columns to absolute paths

    dataframe["filepath"] = dataframe["filepath"].apply(
        lambda x: process_filepaths(x, data_dir)
    )
    dataframe["json_path"] = dataframe["json_path"].apply(
        lambda x: process_filepaths(x, data_dir)
    )

    if use_mini_dataset:
        dataframe = dataframe.sample(
            n=mini_dataset_size if mini_dataset_size else 50,
            random_state=DEFAULT_RANDOM_STATE,
        ).reset_index()
    eval_api: Union[FHIBEPublicEval, FHIBEFacePublicEval]
    if dataset_base == "fhibe":
        eval_api = FHIBEPublicEval(
            dataframe=dataframe,
            age_buckets=use_age_buckets,
            intersectional_column_names=intersectional_column_names,
        )
    elif dataset_base == "fhibe_face":
        eval_api = FHIBEFacePublicEval(
            dataframe=dataframe,
            aligned="align" in dataset_name,
            data_dir=data_dir,
            processed_data_dir=processed_data_dir,
            age_buckets=use_age_buckets,
            intersectional_column_names=intersectional_column_names,
        )
    return eval_api


def process_filepaths(filepath: str, data_dir: str) -> str:
    """Convert relative filepaths to absolute paths.

    Args:
        filepath: The relative filepath from the metadata dataframe.
        data_dir: The root data directory to prepend to relative paths.

    Return:
        The processed filepath.
    """
    return os.path.join(data_dir, filepath)


def prepare_evaluation(
    eval_api: Union[FHIBEPublicEval, FHIBEFacePublicEval],
    task_name: str,
    dataset_name: str,
    model: BaseModelWrapper,
    model_name: str,
    current_results_dir: str,
    precomputed_masks: List[str] | None = None,
    cuda: bool = True,
    **task_kwargs: Dict[str, Any],
) -> Tuple[pd.DataFrame, list[str], Dict[str, Any]]:
    """Prepare data for evaluation.

    Extracts the dataframe, list of image filepaths
    and other task-specific kwargs needed to run
    the evaluation.

    Args:
        eval_api: An evaluation API object
        task_name: The name of the task
        dataset_name: The name of the dataset
        model: The model instance
        model_name: The name of the model
        current_results_dir: The name of the directory where results are stored
            for this model task.
        precomputed_masks: A list of precomputed masks to speed up
            the task preparation.
        cuda: Whether to use the GPU for task preparation.
            Only relevant for some tasks.
        **task_kwargs: Additional task-specific parameters.

    Return:
        None
    """
    if task_name == "person_localization":
        annotations_dataframe, img_filepaths, gt_column_name = (
            eval_api.prepare_person_localization()  # type: ignore
        )
        kwargs = {"gt_column_name": gt_column_name}
    elif task_name == "person_parsing":
        annotations_dataframe, img_filepaths, gt_column_name = (
            eval_api.prepare_person_parsing(  # type: ignore
                to_rle=True, precomputed_masks=precomputed_masks
            )
        )
        kwargs = {
            "to_rle": True,
            "gt_column_name": gt_column_name,
        }
    elif task_name == "keypoint_estimation":
        custom_keypoints = task_kwargs.get("custom_keypoints", None)
        (
            annotations_dataframe,
            img_filepaths,
            img_filepath_gt_bboxes,
            gt_keypoint_column_name,
            gt_face_bbox_column_name,
            kpt_oks_sigmas,
        ) = eval_api.prepare_keypoint_estimation(  # type: ignore
            precomputed_areas=precomputed_masks, custom_keypoints=custom_keypoints
        )
        kwargs = {
            "img_filepath_gt_bboxes": img_filepath_gt_bboxes,
            "gt_keypoint_column_name": gt_keypoint_column_name,
            "gt_face_bbox_column_name": gt_face_bbox_column_name,
            "kpt_oks_sigmas": kpt_oks_sigmas,
            "custom_keypoints": custom_keypoints,
        }

    elif task_name == "face_localization":
        annotations_dataframe, img_filepaths, gt_column_name = (
            eval_api.prepare_face_localization()  # type: ignore
        )
        kwargs = {"gt_column_name": gt_column_name}

    elif task_name == "body_parts_detection":
        (
            annotations_dataframe,
            img_filepaths,
            img_filepath_gt_bboxes,
            gt_column_name,
        ) = eval_api.prepare_body_parts_detection()  # type: ignore
        kwargs = {
            "gt_column_name": gt_column_name,
            "img_filepath_gt_bboxes": img_filepath_gt_bboxes,
        }

    elif task_name == "face_parsing":
        (
            annotations_dataframe,
            img_filepaths,
            mask_filepaths,
            prediction_change_map,
        ) = eval_api.prepare_face_parsing()  # type: ignore
        # prediction_change_map maps left and right ear into skin
        # since FHIBE does not have explicit ear annotations

        kwargs = {
            "mask_filepaths": mask_filepaths,
            "prediction_change_map": prediction_change_map,
        }

    elif task_name == "face_encoding":
        annotations_dataframe, img_filepaths = eval_api.prepare_face_encoding()

        encodings_dir = os.path.join(current_results_dir, "face_encodings")
        os.makedirs(encodings_dir, exist_ok=True)
        encoded_filepaths = [
            os.path.join(encodings_dir, os.path.basename(original_path))
            for original_path in annotations_dataframe["filepath"]
        ]
        annotations_dataframe["encoded_filepath"] = encoded_filepaths

        align_method = "ffhq"
        if eval_api._aligned:
            aligned_filepaths = img_filepaths
        else:
            annotations_dataframe["aligned_filepath"] = [
                os.path.join(eval_api.data_dir, fp)
                for fp in annotations_dataframe["aligned_filepath"]
            ]
            aligned_filepaths = [
                aligned_filepath.replace("/raw/", "/processed/").replace(
                    f"/{dataset_name}/",
                    f"/{dataset_name}/aligned_{task_name}_{align_method}/",
                )
                for aligned_filepath in annotations_dataframe["aligned_filepath"]
            ]
            print(
                f"aligned_filepaths are converted from {dataset_name}/ to "
                "{dataset_name}/aligned_{task_name}_{align_method}/"
            )

        kwargs = {
            "aligned": eval_api.is_aligned,  # type: ignore
            "aligned_filepaths": aligned_filepaths,
            "encoded_filepaths": encoded_filepaths,
        }
    elif task_name == "face_verification":
        # Temporarily hardcode to _aligned=False
        _aligned = eval_api._aligned  # type: ignore
        eval_api._aligned = False  # type: ignore

        annotations_dataframe, img_filepaths = eval_api.prepare_face_verification()  # type: ignore
        annotations_dataframe["aligned_filepath"] = [
            os.path.join(eval_api.data_dir, fp)
            for fp in annotations_dataframe["aligned_filepath"]
        ]
        aligned_img_filepaths = annotations_dataframe["aligned_filepath"].tolist()
        max_workers = min(8, cpu_count() - 1)
        img_filepaths_full = [
            os.path.join(eval_api.data_dir, fp) for fp in img_filepaths
        ]
        success = model.align_faces(  # type:ignore
            img_filepaths=img_filepaths_full,
            aligned_img_filepaths=aligned_img_filepaths,
            batch_size=1,
            num_workers=max_workers,
            cuda=cuda,
        )

        success_aligned_img_filepaths = [
            filepath
            for exists, filepath in zip(success, aligned_img_filepaths)
            if exists
        ]

        # Revert hardcoding
        eval_api._aligned = _aligned  # type: ignore

        kwargs = {
            "aligned": eval_api.is_aligned,  # type: ignore
            "aligned_img_filepaths": aligned_img_filepaths,
            "success": success,
            "success_aligned_img_filepaths": success_aligned_img_filepaths,
        }

    elif task_name == "face_super_resolution":
        assert isinstance(eval_api, FHIBEFacePublicEval)
        annotations_dataframe, img_filepaths = eval_api.prepare_face_super_resolution()
        kwargs = {
            "aligned": eval_api.is_aligned,
        }
    else:
        logging.error("Task not supported.")

    return annotations_dataframe, img_filepaths, kwargs


def validate_dataset_and_task(
    task_name: str, dataset_name: str, dataset_base: str
) -> None:
    """Validate that dataset can be used for a given task.

    Args:
        task_name: The name of the task
        dataset_name: The name of the dataset
        dataset_base: The base name of the dataset, e.g., "fhibe"
            or "fhibe_face" without additional suffixes.

    Return:
        None
    """
    if dataset_name not in VALID_DATASET_NAMES:
        raise ValueError(
            f"dataset: {dataset_name} must be one of {VALID_DATASET_NAMES}"
        )
    if task_name not in TASK_DICT[dataset_base]:
        raise KeyError(
            f"Task name: {task_name} not found in list of "
            f"available tasks for dataset: {dataset_name}: {TASK_DICT[dataset_base]}"
        )
    return None


def validate_metrics(
    task_name: str,
    metrics: List[str] | Dict[str, Dict[str, Any]] | None = None,
) -> List[str] | Dict[str, Dict[str, Any]]:
    """Validate the format of user-inputted metrics.

    Args:
        task_name: The name of the task
        metrics: The user-provided metrics to be validated.

    Return:
        The metrics in either list or dictionary format.
    """
    if not metrics:
        return TASK_METRIC_LOOKUP_DICT[task_name]

    if not (isinstance(metrics, dict) or isinstance(metrics, list)):
        raise ValueError("metrics must be a dictionary, list, or None")
    if isinstance(metrics, dict):
        for metric_name in metrics:
            if not isinstance(metrics[metric_name], dict):
                raise ValueError(f"Metric key: {metric_name} must map to a dict.")
            if "thresholds" not in metrics[metric_name]:
                raise KeyError(
                    f"metrics['{metric_name}'] dict must have a 'thresholds' key."
                )
            if not isinstance(metrics[metric_name]["thresholds"], list):
                raise ValueError(
                    f"metrics['{metric_name}']['thresholds'] must map to a list."
                )
            for key in metrics[metric_name]:
                if key != "thresholds":
                    raise KeyError(
                        f"Key: {key} should not be present in "
                        f"metrics['{metric_name}'] dict."
                    )

    for metric_name in metrics:
        if metric_name.upper() not in TASK_METRIC_LOOKUP_DICT[task_name]:
            raise ValueError(
                f"Metric: {metric_name} is not implemented for task: {task_name}."
            )

    if isinstance(metrics, list):
        return {m.upper(): {"thresholds": None} for m in metrics}

    if isinstance(metrics, dict):
        return {m.upper(): metrics[m] for m in metrics}


def validate_attributes(
    dataset_base: str,
    attributes: List[str],
) -> List[str]:
    """Validate the format of user-inputted attribute name list.

    Args:
        dataset_base: "fhibe" or "fhibe_face"
        attributes: List of attribute names supplied by user
    Return:
        List of attribute names sorted in the order of
        column appearance in the metadata JSON file.
    """
    if dataset_base == "fhibe":
        valid_attributes = FHIBE_ATTRIBUTE_LIST
    else:
        valid_attributes = FHIBE_FACE_ATTRIBUTE_LIST
    if not isinstance(attributes, list):
        raise ValueError("Attributes must be a list")
    for attr in attributes:
        if not isinstance(attr, str):
            raise ValueError("Attributes must be strings")

        if attr not in valid_attributes:
            raise ValueError(
                f"{attr} is not a valid attribute. "
                f"It must be one of {valid_attributes}. "
            )
    # Now reorder according to hardcoded order
    reordered_attrs = []
    for attr in valid_attributes:
        if attr in attributes:
            reordered_attrs.append(attr)
    return reordered_attrs


def format_threshold_results(
    in_results: Dict[str, Any], metric_name: str
) -> Dict[float, float]:
    """Converts the metric-specific thresholded results into a consistent format.

    Args:
        in_results: The results dictionary
        metric_name: The name of the metric used when computing the thresholded results.

    Return:
        A dictionary of thresholded results in consistent format
            to be used in downstream functions.
    """
    if metric_name in ["iou", "oks"]:
        results = {float(key): float(in_results[key]["summary"]) for key in in_results}

    elif metric_name == "pck":
        keys = sorted(list(in_results["results"].keys()))
        thresholds = in_results["thresholds"]
        # loop through once to get all pck scores
        all_pck_scores = np.array(
            [in_results["results"][key]["pcks@thresholds"] for key in keys]
        )
        results = {}
        for ix, thr in enumerate(thresholds):
            results[thr] = np.nanmean(all_pck_scores[:, ix])
    else:
        raise NotImplementedError(
            f"This function is not implemented for metric: {metric_name}"
        )
    return results


### Face parsing


def update_prediction_map(
    model_outputs: dict[str, Any], prediction_change_map: dict[int, int]
) -> dict[str, Any]:
    """Update the labels of the model predictions.

    Used for face parsing.

    Args:
        model_outputs: The model outputs dictionary
        prediction_change_map: Dictionary mapping in labels to out labels.

    Return:
        A dictionary of thresholded results in consistent format
            to be used in downstream functions.
    """
    for filepath in tqdm(
        model_outputs.keys(), desc="Face Parsing - Map predicted masks to GT masks"
    ):
        pred = model_outputs[filepath]["detections"]
        for current_label, new_label in prediction_change_map.items():
            pred[pred == current_label] = new_label
    # Keep model_outputs[key]["detections"] as np.array
    return model_outputs


def _encode_mask(decoded_mask: npt.NDArray[np.int64]) -> dict[str, Any]:
    """Encode a segmentation mask using run-length encoding.

    The idea is that the mask has a lot of repeated consecutive
    integer values, so it can be efficiently compressed using
    run-length encoding.

    Args:
        decoded_mask: A numpy array containing the detection masks.

    Return:
        Dictionary mapping the unique mask keys to their run-length
        encodings.

    """
    mask_unique_keys = np.unique(decoded_mask).tolist()
    result = {}
    for mask_key in mask_unique_keys:
        current_mask_rle = encode(
            np.asfortranarray(np.array(decoded_mask == mask_key, dtype=np.uint8))
        )
        current_mask_rle["counts"] = current_mask_rle["counts"].decode("utf-8")
        result[mask_key] = current_mask_rle
    return result


def _decode_mask(encoded_mask: Dict[str, Any]) -> List[List[int]] | Any:
    """Decode a segmentation mask that has been run-length encoded.

    The idea is that the mask has a lot of repeated consecutive
    integer values, so it can be efficiently compressed using
    run-length encoding.

    Args:
        encoded_mask: A mask encoded with the function _encode_mask

    Return:
        The segmentation mask in nested list format.

    """
    mask_keys = encoded_mask.keys()
    result_mask = None

    for mask_key in mask_keys:
        tmp_mask: npt.NDArray[np.int64] = int(mask_key) * decode(encoded_mask[mask_key])
        if result_mask is None:
            result_mask = tmp_mask.copy()
        else:
            result_mask += tmp_mask
    if result_mask is not None:
        return result_mask.tolist()
    else:
        return None
