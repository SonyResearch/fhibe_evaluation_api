# SPDX-License-Identifier: Apache-2.0
"""Main module for model evaluation.

This module contains functions for performing the model evaluation.
"""

import logging
import os
from typing import Any, Dict, List, Optional, cast

import numpy as np
import pandas as pd
import torch
from pycocotools.mask import encode
from tqdm import tqdm

from fhibe_eval_api.common.loggers import setup_logging
from fhibe_eval_api.common.utils import get_project_root, read_json_file, save_json_file
from fhibe_eval_api.evaluate.constants import MODEL_OUTPUT_FILENAME
from fhibe_eval_api.evaluate.utils import (
    _encode_mask,
    get_eval_api,
    prepare_evaluation,
    update_prediction_map,
    validate_attributes,
    validate_dataset_and_task,
    validate_metrics,
)
from fhibe_eval_api.metrics.constants import METRIC_THRESHOLDS_DEFAULTS
from fhibe_eval_api.metrics.face_parsing.utils import CELEBA_MASK_HQ_LABELS_DICT
from fhibe_eval_api.metrics.fhibe_metrics import METRIC_FUNCTION_MAPPER
from fhibe_eval_api.models.base_model import BaseModelWrapper

setup_logging("info")


def evaluate_task(
    data_rootdir: str,
    dataset_name: str,
    model: BaseModelWrapper | None,
    model_name: str,
    task_name: str,
    metrics: List[str] | Dict[str, Dict[str, Any]] | None = None,
    attributes: List[str] = [
        "pronoun",
        "age",
        "apparent_skin_color",
        "ancestry",
    ],
    use_mini_dataset: bool = False,
    mini_dataset_size: int = 50,
    downsampled: bool = True,
    reuse_model_outputs: bool = False,
    results_rootdir: str = None,
    cuda: bool = True,
    **task_kwargs: Dict[str, Any],
) -> None:
    """Run model evaluation for a given task on FHIBE.

    Args:
        data_rootdir: The absolute path to the directory containing the
            filepaths.csv, images/, and annotations/ subdirectories
        dataset_name: "fhibe", or "fhibe_face_crop_align"
        model: The compiled PyTorch model to evaluate. If None,
            reuse_model_outputs must be True.
        model_name: The name of the model to evaluate -- used in result filepaths
        task_name: The name of the model task to evaluation
        metrics: Either a list of metric names, a dictionary, or None.
            If provided as a list, and any of the metrics require thresholds,
            defaults will be used
            (see .metrics.fhibe_metrics.METRIC_THRESHOLDS_DEFAULTS).

            If provided as a dict, the keys of the dict are the metric names
            and the values are dictionaires with a single key "thresholds", and the
            value is a list of the thresholds at which the metric will be calculated.
            This is only relevant for a few metrics
            (see .metrics.fhibe_metrics.METRIC_THRESHOLDS_DEFAULTS).
            Example: {"AR_IOU":{"thresholds":[0.5,0.6,0.7,0.8,0.9]}}.

            If metrics=None, all available metrics will be computed,
            and default thresholds will be used if necessary.
            See .utils.TASK_METRIC_LOOKUP_DICT for the default metrics.
        attributes: A list of attributes over which to aggregrate metric performance,
            e.g., ["pronoun", "age", "apparent_skin_color", "ancestry"]. Each metric
            will be computed for each element of each attribute. E.g., "She/her/hers",
            "He/him/his" are elements of the "pronoun" attribute.
        use_mini_dataset: Whether to use a smaller dataset,
            randomly sampled from the full fhibe dataset.
        mini_dataset_size: The size of the mini dataset to use.
            Only relevant if use_mini_dataset=True.
        downsampled: Whether to use the downsampled FHIBE images. Only relevant if
            dataset_name = "fhibe"
        reuse_model_outputs: Whether to use previously calculated model outputs,
            if they exist. Defaults to False.
        results_rootdir: Where to save the results. If not provided, saves in a
            "results/" subdirectory of the project root.
        cuda: Whether to use the GPU for running inference.
        **task_kwargs: Additional task-specific parameters.

        Task-specific parameters:
        - For "keypoint_estimation":
            - custom_keypoints (List[str]): List of subset of keypoints to evaluate
                (e.g., ["Nose", "Left eye", "Right eye"]).

            Example usage:
                evaluate_task(
                    data_rootdir="path/to/data",
                    dataset_name="fhibe",
                    model=custom_model,
                    model_name="my_custom_model",
                    task_name="keypoint_estimation",
                    custom_keypoints=["Nose", "Left eye", "Right eye"]
                )

    Return:
        A bias report object containing the results of the bias evaluation.
    """
    if model is None and reuse_model_outputs is not True:
        raise ValueError("If reuse_model_outputs is not True, model cannot be None.")
    if "_crop" in dataset_name or "_align" in dataset_name:
        dataset_base = "fhibe_face"
    else:
        dataset_base = "fhibe"

    # Check that dataset, task, and metrics are valid and compatible
    validate_dataset_and_task(task_name, dataset_name, dataset_base)
    attributes = validate_attributes(dataset_base, attributes)
    metrics = validate_metrics(task_name, metrics)

    # The face datasets do not have downsampled versions
    if downsampled and dataset_base == "fhibe":
        dataset_name += "_downsampled"
    logging.info(
        f"Evaluating model: {model_name} on task: {task_name} "
        f"and dataset: {dataset_name}"
    )
    if results_rootdir is None:
        results_rootdir = os.path.join(get_project_root(), "results")

    if use_mini_dataset:
        results_rootdir = os.path.join(results_rootdir, "mini")

    current_results_dir = os.path.join(
        results_rootdir, task_name, dataset_name, model_name
    )
    os.makedirs(current_results_dir, exist_ok=True)

    model_outputs_filepath = os.path.join(current_results_dir, MODEL_OUTPUT_FILENAME)

    # Set processed data directory
    processed_data_dir = os.path.join(data_rootdir, "data", "processed")

    eval_api = get_eval_api(
        dataset_name=dataset_name,
        dataset_base=dataset_base,
        data_dir=data_rootdir,
        processed_data_dir=processed_data_dir,
        intersectional_column_names=attributes,
        use_age_buckets=True,
        use_mini_dataset=use_mini_dataset,
        mini_dataset_size=mini_dataset_size,
    )

    # Prepare data and metadata
    annotations_dataframe, img_filepaths, kwargs = prepare_evaluation(
        eval_api=eval_api,
        task_name=task_name,
        dataset_name=dataset_name,
        model=model,
        model_name=model_name,
        current_results_dir=current_results_dir,
        precomputed_masks=None,
        cuda=True,
        **task_kwargs,
    )

    # Call the evaluation function
    model_outputs = _evaluate(
        task_name=task_name,
        dataset_name=dataset_name,
        annotations_dataframe=annotations_dataframe.copy(deep=True),
        model=model,
        model_name=model_name,
        img_filepaths=img_filepaths,
        model_outputs_filepath=model_outputs_filepath,
        reuse_model_outputs=reuse_model_outputs,
        **kwargs,
    )

    # Evaluate metrics in intersectional groups
    compute_metric_results(
        task_name=task_name,
        metrics=metrics,
        intersectional_groups=attributes,
        img_filepaths=img_filepaths,
        model_outputs=model_outputs,
        annotations_dataframe=annotations_dataframe,
        current_results_dir=current_results_dir,
        **kwargs,
    )
    logging.info("Evaluation complete.")
    return None


def _evaluate(
    task_name: str,
    dataset_name: str,
    annotations_dataframe: pd.DataFrame,
    model: BaseModelWrapper,
    model_name: str,
    img_filepaths: List[str],
    model_outputs_filepath: str,
    reuse_model_outputs: bool = False,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Evaluate the dataset on the model in batches.

    This function instantiates a data loader, passes the data
    through the model's forward pass one batch at a time,
    and then aggregrates the model outputs. Model outputs
    are saved to disk. For all tasks except face_encoding,
    the model outputs are saved as a single JSON file.
    For the face encoding task, the encodings are saved
    in a separate directory:
    data/processed/{dataset_name}/encoded_{model_name}/

    Args:
        task_name: The name of the model task to evaluation
        dataset_name: "fhibe", "fhibe_face_crop", or "fhibe_face_crop_align"
        annotations_dataframe: Contains the annotation metadata
        model: The CV model wrapped using BaseModelWrapper
        model_name: The name of the model to evaluate
        img_filepaths: A list of the filepaths of the images to evaluate.
        model_outputs_filepath: The full path where the model outputs will be saved
            when appropriate.
        reuse_model_outputs: If True, searches for existing model outputs
            on disk and uses them instead of recomputing them.
        kwargs: Additional keyword arguments, used to pass task-specific parameters.

    Return:
        Model inference results in a dictionary. The face encoding task
        returns an empty dict, and it saves the encodings to disk instead.

    """
    if reuse_model_outputs:
        if task_name == "face_encoding":
            all_encodings_exist = False
            for encoded_filepath in kwargs["encoded_filepaths"]:
                if not os.path.isfile(encoded_filepath):
                    break
            else:
                all_encodings_exist = True
            if all_encodings_exist:
                logging.info("Loading existing encodings files from file.")
                return {}
            else:
                logging.info("Generating encodings.")

        elif os.path.exists(model_outputs_filepath):
            logging.info("Loading existing model outputs from file")
            return cast(Dict[str, Any], read_json_file(model_outputs_filepath))
        else:
            raise FileNotFoundError(
                f"Model outputs file expected at: {model_outputs_filepath}. "
                "but not found. Set reuse_model_outputs=False "
                "to recompute model outputs."
            )

    data_loader = model.data_preprocessor(img_filepaths, **kwargs)

    results: Dict[str, Dict[str, Any]] = {}  # maps image filename to a results dict

    with tqdm(total=len(img_filepaths), desc="Running model inference") as pbar:
        with torch.no_grad():
            if task_name == "person_localization":
                idx = 0
                for batch in data_loader:
                    result_list = model(batch)
                    for ii in range(len(result_list)):
                        result = result_list[ii]
                        _bboxes = result.get("bboxes") or []
                        _scores = result.get("scores") or []
                        _labels = result.get("labels") or []

                        dets = []
                        scores = []

                        n_bboxes = len(_bboxes)
                        for jj in range(n_bboxes):
                            if _labels[jj] == 0:
                                dets.append(_bboxes[jj])
                                scores.append(_scores[jj])

                        img_filepath = img_filepaths[idx]
                        if len(dets) == 0:
                            results[img_filepath] = {"detections": None, "scores": None}
                        else:
                            results[img_filepath] = {
                                "detections": dets,
                                "scores": scores,
                            }
                        idx += 1
                        pbar.update(1)
            elif task_name == "person_parsing":
                idx = 0
                for batch in data_loader:
                    result_list = model(batch)
                    for ii in range(len(result_list)):
                        result = result_list[ii]
                        _masks = result.get("masks") or []
                        _scores = result.get("scores") or []
                        _labels = result.get("labels") or []
                        masks = []
                        scores = []
                        n_masks = len(_labels)
                        for jj in range(n_masks):
                            if _labels[jj] == 0:
                                mask_rle = encode(np.asfortranarray(_masks[jj]))
                                mask_rle["counts"] = mask_rle["counts"].decode("utf-8")
                                masks.append(mask_rle)
                                scores.append(_scores[jj])

                        img_filepath = img_filepaths[idx]
                        if len(masks) == 0:
                            results[img_filepath] = {"detections": None, "scores": None}
                        else:
                            results[img_filepath] = {
                                "detections": masks,
                                "scores": scores,
                            }
                        idx += 1
                        pbar.update(1)
            elif task_name == "keypoint_estimation":
                idx = 0
                for batch in data_loader:
                    result_list = model(batch)
                    for ii in range(len(result_list)):
                        result = result_list[ii]
                        keypoints = result.get("keypoints") or []
                        scores = result.get("scores") or []
                        img_filepath = img_filepaths[idx]
                        results[img_filepath] = {
                            "detections": keypoints,
                            "scores": scores,
                        }
                        idx += 1
                        pbar.update(1)
            elif task_name == "face_localization":
                idx = 0
                for batch in data_loader:
                    result_list = model(batch)
                    for ii in range(len(result_list)):
                        result = result_list[ii]
                        img_filepath = img_filepaths[idx]
                        dets = result.get("detections") or []
                        scores = result.get("scores") or []
                        results[img_filepath] = {"detections": dets, "scores": scores}
                        idx += 1
                        pbar.update(1)
            elif task_name == "body_parts_detection":
                idx = 0
                for batch in data_loader:
                    result_list = model(batch)
                    for ii in range(len(result_list)):
                        # result is a list of dicts, one per gt bbox,
                        # mapping body part to probability of detection
                        result = result_list[ii]
                        img_filepath = img_filepaths[idx]
                        results[img_filepath] = {"detections": result}
                        idx += 1
                        pbar.update(1)
            elif task_name == "face_verification":
                aligned_img_filepaths = kwargs["aligned_img_filepaths"]
                success = kwargs["success"]  # whether alignment was successful

                embeddings = []
                with torch.no_grad():
                    for batch in data_loader:
                        normed_batch_embeddings = model(batch)
                        embeddings.append(normed_batch_embeddings)

                embeddings = torch.cat(embeddings, dim=0).numpy()

                embedding_idx = 0
                for i, (
                    img_filepath,
                    aligned_img_filepath,
                    aligned_exists,
                ) in enumerate(zip(img_filepaths, aligned_img_filepaths, success)):
                    if aligned_exists:
                        results[aligned_img_filepath] = {
                            "detections": embeddings[embedding_idx].tolist(),
                            "img_filepath": img_filepath,
                        }
                        embedding_idx += 1
                    else:
                        results[aligned_img_filepath] = {
                            "detections": None,
                            "img_filepath": img_filepath,
                        }
                    pbar.update(1)
            elif task_name == "face_parsing":
                idx = 0
                for batch in data_loader:
                    result_list = model(batch)
                    for ii in range(len(result_list)):
                        result = result_list[ii]
                        img_filepath = img_filepaths[idx]
                        results[img_filepath] = {
                            "detections": result["detections"],
                        }
                        idx += 1
                        pbar.update(1)

                # Map ear masks to skin
                mapped_predictions: Optional[Dict[int, int]] = None
                if hasattr(model, "map_ears_to_skin"):
                    if model.map_ears_to_skin is True:
                        prediction_change_map = kwargs["prediction_change_map"]
                        mapped_predictions = {
                            CELEBA_MASK_HQ_LABELS_DICT[
                                fromLabel
                            ]: CELEBA_MASK_HQ_LABELS_DICT[toLabel]
                            for fromLabel, toLabel in prediction_change_map.items()
                        }
                        update_prediction_map(results, mapped_predictions)
                else:
                    raise AttributeError(
                        "Model must have a 'map_ears_to_skin' attribute"
                    )
                # Perform run-length encoding
                for key in tqdm(results, desc="Encoding detection masks)"):
                    results[key]["detections_rle"] = _encode_mask(
                        results[key]["detections"]
                    )
                    results[key].pop("detections")
            elif task_name == "face_encoding":
                # Make directory in which to save encodings to disk
                encoded_filepaths = kwargs["encoded_filepaths"]
                dir_to_make = os.path.dirname(encoded_filepaths[0])
                os.makedirs(dir_to_make, exist_ok=True)

                idx = 0
                for batch in data_loader:
                    result_list = model(batch)
                    # Save each encoding to disk
                    for ii in range(len(result_list)):
                        result = result_list[ii]
                        if hasattr(model, "save_encoding"):
                            model.save_encoding(
                                result["encoding"], encoded_filepaths[idx]
                            )
                        else:
                            raise AttributeError(
                                "Model must have a 'save_encoding' method."
                            )
                        idx += 1
                        pbar.update(1)
            elif task_name == "face_super_resolution":
                save_dir, _ = os.path.split(model_outputs_filepath)
                idx = 0
                for batch in data_loader:
                    result_list = model(batch)
                    # Save each super resolution array to disk
                    for ii in range(len(result_list)):

                        img_filepath = img_filepaths[idx]
                        array = result_list[ii]
                        _, file_basename = os.path.split(img_filepath)
                        file_savename = os.path.join(save_dir, f"super_{file_basename}")
                        if hasattr(model, "save_array"):
                            model.save_array(array, file_savename)
                        else:
                            raise AttributeError(
                                "Model must have a 'save_array' method."
                            )
                        idx += 1
                        results[img_filepath] = {"super_res_filename": file_savename}
                        pbar.update(1)
    # Save out model outputs except for face encoding task
    if task_name not in ["face_encoding"]:
        # Make sure directory exists
        os.makedirs(os.path.dirname(model_outputs_filepath), exist_ok=True)
        save_json_file(filepath=model_outputs_filepath, data=results, indent=4)
        logging.info(f"Saved model outputs to: {model_outputs_filepath}")
    logging.info("Inference complete.\n")
    return results


def compute_metric_results(
    task_name: str,
    metrics: List[str] | Dict[str, Dict[str, Any]],
    intersectional_groups: List[str],
    img_filepaths: list[str],
    model_outputs: Dict[str, Any],
    annotations_dataframe: pd.DataFrame,
    current_results_dir: str,
    **kwargs: Any,
) -> None:
    """Calculate metrics over intersectional groups and save results to disk.

    In addition to named keyword arguments, additional task-specific kwargs
    can be passed.

    Args:
        task_name: The name of the task
        metrics: Either a list of metric names or a dictionary.
            If provided as a list, and any of the metrics require thresholds,
            defaults will be used
            (see .metrics.fhibe_metrics.METRIC_THRESHOLDS_DEFAULTS).
            If provided as a dict, the keys of the dict are the metric names
            and the values are dictionaires with a single key "thresholds", and the
            value is either None or a list of the thresholds
            at which the metric will be calculated.
            If the key is None, and the metric requires thresholds,
            then the default thresholds will be used.
        intersectional_groups: A list of the demographic attributes to split
            the data on. The four available attributes are:
            "pronoun", "age", "apparent_skin_color", and "ancestry".
        img_filepaths: A list of the filepaths of the images to evaluate.
        model_outputs: The resulting dictionary returned by _evaluate(),
            containing the model outputs for all images
        annotations_dataframe: Contains the annotation metadata
        current_results_dir: The directory in which to save the results.
        kwargs: Additional keyword arguments. Used to pass task-specific parameters.

    Returns:
        None
    """
    logging.info("Computing metrics.")
    for metric_name in metrics:
        metric_fn = METRIC_FUNCTION_MAPPER[metric_name]
        if isinstance(metrics, Dict):
            thresholds = metrics[metric_name].get("thresholds", None)
        else:
            thresholds = None
        if not thresholds and metric_name in METRIC_THRESHOLDS_DEFAULTS:
            thresholds = METRIC_THRESHOLDS_DEFAULTS.get(metric_name)

        if thresholds is not None:
            thresholds_str = [f"{thr:.2f}" for thr in thresholds]
        else:
            thresholds_str = "None"
        logging.info(f"Evaluating metric: {metric_name}")
        logging.info(
            f"Using thresholds: {thresholds_str}",
        )
        metric_fn(
            task_name=task_name,
            intersectional_groups=intersectional_groups,
            filepaths=img_filepaths,
            model_outputs=model_outputs,
            annotations_dataframe=annotations_dataframe,
            thresholds=thresholds,
            current_results_dir=current_results_dir,
            **kwargs,
        )
    logging.info("Done computing metrics.\n")
    return
