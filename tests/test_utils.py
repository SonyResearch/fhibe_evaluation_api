# SPDX-License-Identifier: Apache-2.0
import os
import pickle

import numpy as np
import pytest

from fhibe_eval_api.common.utils import read_json_file
from fhibe_eval_api.evaluate.constants import TASK_DICT, VALID_DATASET_NAMES
from fhibe_eval_api.evaluate.utils import (
    format_threshold_results,
    get_eval_api,
    prepare_evaluation,
    validate_dataset_and_task,
    validate_metrics,
)

CURRENT_DIR = os.path.dirname(__file__)


def test_validate_dataset_and_task():
    dataset_name = "abc"
    dataset_base = "fhibe"
    task_name = "person_localization"
    with pytest.raises(ValueError) as excinfo:
        validate_dataset_and_task(task_name, dataset_name, dataset_base)
    error_msg = f"dataset: {dataset_name} must be one of {VALID_DATASET_NAMES}"
    assert str(excinfo.value) == error_msg

    dataset_name = "fhibe"
    dataset_base = "fhibe"
    task_name = "person detection"
    with pytest.raises(KeyError) as excinfo:
        validate_dataset_and_task(task_name, dataset_name, dataset_base)
    error_msg = (
        f"Task name: {task_name} not found in list of "
        f"available tasks for dataset: {dataset_name}: {TASK_DICT[dataset_base]}"
    )

    assert error_msg in str(excinfo.value)

    dataset_name = "fhibe"
    dataset_base = "fhibe"
    task_name = "person_localization"
    _ = validate_dataset_and_task(task_name, dataset_name, dataset_base)
    assert _ is None

    dataset_name = "fhibe_face"
    dataset_base = "fhibe_face"
    task_name = "face_parsing"
    with pytest.raises(ValueError) as excinfo:
        _ = validate_dataset_and_task(task_name, dataset_name, dataset_base)
    error_msg = f"dataset: {dataset_name} must be one of {VALID_DATASET_NAMES}"
    assert str(excinfo.value) == error_msg

    dataset_name = "fhibe_face_crop_align"
    dataset_base = "fhibe_face"
    task_name = "person_localization"
    with pytest.raises(KeyError) as excinfo:
        validate_dataset_and_task(task_name, dataset_name, dataset_base)
        error_msg = (
            f"Task name: {task_name} not found in list of "
            f"available tasks for dataset: {dataset_name}: {TASK_DICT[dataset_base]}"
        )

        assert error_msg in str(excinfo.value)

    dataset_name = "fhibe_face_crop_align"
    dataset_base = "fhibe_face"
    task_name = "face_parsing"
    _ = validate_dataset_and_task(task_name, dataset_name, dataset_base)
    assert _ is None


def test_validate_metrics():
    task_name = "keypoint_estimation"
    for metrics in [None, {}, []]:
        _metrics = validate_metrics(task_name)
        assert _metrics == ["PCK", "AR_OKS"]

    metrics = {"PCK": [0.1, 0.5, 0.9]}
    with pytest.raises(ValueError) as excinfo:
        _metrics = validate_metrics(task_name, metrics)
    error_msg = "Metric key: PCK must map to a dict."
    assert str(excinfo.value) == error_msg

    metrics = {"PCK": {"a": [0.1, 0.5, 0.9]}}
    with pytest.raises(KeyError) as excinfo:
        _metrics = validate_metrics(task_name, metrics)
    error_msg = "metrics['PCK'] dict must have a 'thresholds' key."
    assert error_msg in str(excinfo.value)

    metrics = {"PCK": {"thresholds": (0.1, 0.5, 0.9)}}
    with pytest.raises(ValueError) as excinfo:
        _metrics = validate_metrics(task_name, metrics)
    error_msg = "metrics['PCK']['thresholds'] must map to a list."
    assert error_msg == str(excinfo.value)

    metrics = {"PCK": {"thresholds": [0.1, 0.5, 0.9], "other_key": "abc"}}
    with pytest.raises(KeyError) as excinfo:
        _metrics = validate_metrics(task_name, metrics)
    error_msg = "Key: other_key should not be present in metrics['PCK'] dict."
    assert error_msg in str(excinfo.value)

    metrics = {
        "PCK": {
            "thresholds": [0.1, 0.5, 0.9],
        },
    }
    _metrics = validate_metrics(task_name, metrics)
    assert _metrics == metrics

    metrics = {
        "PCK": {
            "thresholds": [0.1, 0.5, 0.9],
        },
        "AR_OKS": {
            "thresholds": [0.5, 0.55, 0.6],
        },
    }
    _metrics = validate_metrics(task_name, metrics)
    assert _metrics == metrics

    metrics = ["PCK", "AR_OKS"]
    _metrics = validate_metrics(task_name, metrics)
    assert _metrics == {
        "PCK": {
            "thresholds": None,
        },
        "AR_OKS": {
            "thresholds": None,
        },
    }

    task_name = "face_parsing"
    metrics = ["F1"]
    _metrics = validate_metrics(task_name, metrics)
    assert _metrics == {
        "F1": {
            "thresholds": None,
        },
    }

    task_name = "face_parsing"
    metrics = ["AR_OKS"]
    with pytest.raises(ValueError) as excinfo:
        _metrics = validate_metrics(task_name, metrics)
    error_msg = "Metric: AR_OKS is not implemented for task: face_parsing."
    assert error_msg == str(excinfo.value)

    task_name = "face_parsing"
    metrics = "AR_OKS"
    with pytest.raises(ValueError) as excinfo:
        _metrics = validate_metrics(task_name, metrics)
    error_msg = "metrics must be a dictionary, list, or None"
    assert error_msg == str(excinfo.value)


def test_format_threshold_results(demo_model_fixture):
    task_name = "person_localization"
    dataset_name = "fhibe_downsampled"
    _, model_name = demo_model_fixture(task_name)
    detailed_results_file = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        dataset_name,
        model_name,
        "ground_truth",
        "detailed_results_iou_threshold.json",
    )
    in_results = read_json_file(detailed_results_file)
    res = format_threshold_results(in_results, metric_name="iou")
    assert res == {
        0.5: 1.0,
        0.55: 1.0,
        0.6: 1.0,
        0.65: 1.0,
        0.7: 1.0,
        0.75: 0.98,
        0.8: 0.96,
        0.85: 0.94,
        0.9: 0.84,
        0.95: 0.46,
    }

    task_name = "keypoint_estimation"
    dataset_name = "fhibe_downsampled"
    _, model_name = demo_model_fixture(task_name)
    detailed_results_file = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        dataset_name,
        model_name,
        "ground_truth",
        "detailed_results_oks_threshold.json",
    )
    in_results = read_json_file(detailed_results_file)
    res = format_threshold_results(in_results, metric_name="oks")
    assert res == {
        0.5: 1.0,
        0.55: 1.0,
        0.6: 1.0,
        0.65: 1.0,
        0.7: 0.98,
        0.75: 0.96,
        0.8: 0.96,
        0.85: 0.92,
        0.9: 0.84,
        0.95: 0.52,
    }

    task_name = "person_parsing"
    dataset_name = "fhibe_downsampled"
    _, model_name = demo_model_fixture(task_name)
    detailed_results_file = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        dataset_name,
        model_name,
        "ground_truth",
        "detailed_results_iou_threshold.json",
    )
    in_results = read_json_file(detailed_results_file)
    res = format_threshold_results(in_results, metric_name="iou")
    assert res == {
        0.5: 0.98,
        0.55: 0.98,
        0.6: 0.98,
        0.65: 0.98,
        0.7: 0.98,
        0.75: 0.96,
        0.8: 0.94,
        0.85: 0.88,
        0.9: 0.66,
        0.95: 0.02,
    }

    task_name = "face_localization"
    dataset_name = "fhibe_downsampled"
    _, model_name = demo_model_fixture(task_name)
    detailed_results_file = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        dataset_name,
        model_name,
        "ground_truth",
        "detailed_results_iou_threshold.json",
    )
    in_results = read_json_file(detailed_results_file)
    res = format_threshold_results(in_results, metric_name="iou")

    assert res == {
        0.5: 1.0,
        0.55: 1.0,
        0.6: 1.0,
        0.65: 1.0,
        0.7: 0.98,
        0.75: 0.9,
        0.8: 0.84,
        0.85: 0.74,
        0.9: 0.48,
        0.95: 0.12,
    }

    with pytest.raises(NotImplementedError) as excinfo:
        res = format_threshold_results(in_results, metric_name="gpg")
    error_msg = "This function is not implemented for metric: gpg"
    assert error_msg == str(excinfo.value)


def test_get_eval_api():
    dataset_name = "fhibe_downsampled"
    dataset_base = "fhibe"
    data_dir = os.path.join(CURRENT_DIR, "static", "data")
    processed_data_dir = os.path.join(data_dir, "processed")
    eval_api = get_eval_api(
        dataset_name,
        dataset_base,
        data_dir,
        processed_data_dir,
        intersectional_column_names=None,
        use_age_buckets=True,
        use_mini_dataset=False,
    )

    assert len(eval_api.dataframe) == 50
    assert len(eval_api.dataframe.columns) == 48
    assert eval_api.intersectional_column_names == (
        "pronoun",
        "age",
        "apparent_skin_color",
        "ancestry",
    )
    assert eval_api.age_map == {
        0: "[18, 30)",
        1: "[30, 40)",
        2: "[40, 50)",
        3: "[50, 60)",
        4: "[60, +]",
    }

    # Test that using mini dataset works
    eval_api_mini = get_eval_api(
        dataset_name,
        dataset_base,
        data_dir,
        processed_data_dir,
        intersectional_column_names=None,
        use_age_buckets=True,
        use_mini_dataset=True,
        mini_dataset_size=10,
    )
    assert len(eval_api_mini.dataframe) == 10
    assert len(eval_api_mini.dataframe.columns) == 49
    assert eval_api_mini.intersectional_column_names == (
        "pronoun",
        "age",
        "apparent_skin_color",
        "ancestry",
    )
    assert eval_api_mini.age_map == {
        0: "[18, 30)",
        1: "[30, 40)",
        2: "[40, 50)",
        3: "[50, 60)",
        4: "[60, +]",
    }

    # Test using face dataset
    dataset_name = "fhibe_face_crop_align"
    dataset_base = "fhibe_face"
    data_dir = os.path.join(CURRENT_DIR, "static", "data")
    processed_data_dir = os.path.join(data_dir, "processed")
    eval_api_face = get_eval_api(
        dataset_name,
        dataset_base,
        data_dir,
        processed_data_dir,
        intersectional_column_names=None,
        use_age_buckets=True,
        use_mini_dataset=False,
    )
    assert len(eval_api_face.dataframe) == 50
    assert len(eval_api_face.dataframe.columns) == 75
    assert eval_api_face.intersectional_column_names == (
        "pronoun",
        "age",
        "apparent_skin_color",
        "ancestry",
    )
    assert eval_api_face.age_map == {
        0: "[18, 30)",
        1: "[30, 40)",
        2: "[40, 50)",
        3: "[50, 60)",
        4: "[60, +]",
    }


def test_prepare_evaluation(eval_api_fixture, demo_model_fixture):
    task_name = "person_localization"
    dataset_name = "fhibe_downsampled"
    eval_api = eval_api_fixture(task_name)
    model, model_name = demo_model_fixture(task_name)
    current_results_dir = os.path.join(
        CURRENT_DIR, "static", "results", "mini", task_name, dataset_name, model_name
    )
    annotations_dataframe, img_filepaths, kwargs = prepare_evaluation(
        eval_api=eval_api,
        task_name=task_name,
        dataset_name=dataset_name,
        model=model,
        model_name=model_name,
        current_results_dir=current_results_dir,
        cuda=False,
    )
    assert len(annotations_dataframe) == 50
    assert len(annotations_dataframe.columns) == 49
    # there are some duplicate filenames in the dataframe
    # img_filepaths is the unique list
    assert len(img_filepaths) == 50
    assert kwargs == {"gt_column_name": "person_bbox"}

    task_name = "keypoint_estimation"
    dataset_name = "fhibe_downsampled"
    eval_api = eval_api_fixture(task_name)
    model, model_name = demo_model_fixture(task_name)
    current_results_dir = os.path.join(
        CURRENT_DIR, "static", "results", "mini", task_name, dataset_name, model_name
    )
    precomputed_mask_file = os.path.join(
        current_results_dir,
        "ground_truth",
        "precomputed_person_segment_areas.pkl",
    )
    with open(precomputed_mask_file, "rb") as file:
        precomputed_areas = pickle.load(file)
    annotations_dataframe, img_filepaths, kwargs = prepare_evaluation(
        eval_api=eval_api,
        task_name=task_name,
        dataset_name=dataset_name,
        model=model,
        model_name=model_name,
        current_results_dir=current_results_dir,
        precomputed_masks=precomputed_areas,
        cuda=False,
    )
    assert len(annotations_dataframe) == 50
    assert len(annotations_dataframe.columns) == 51
    # there are some duplicate filenames in the dataframe
    # img_filepaths is the unique list
    assert len(img_filepaths) == 50
    assert "img_filepath_gt_bboxes" in kwargs
    assert "gt_keypoint_column_name" in kwargs
    assert "gt_face_bbox_column_name" in kwargs
    assert "kpt_oks_sigmas" in kwargs

    assert len(kwargs["img_filepath_gt_bboxes"]) == 50
    assert kwargs["gt_keypoint_column_name"] == "keypoints_coco_fmt"
    assert kwargs["gt_face_bbox_column_name"] == "face_bbox"
    compare_array = np.array(
        [
            0.026,
            0.025,
            0.025,
            0.035,
            0.035,
            0.079,
            0.079,
            0.072,
            0.072,
            0.062,
            0.062,
            0.107,
            0.107,
            0.087,
            0.087,
            0.089,
            0.089,
        ]
    )
    assert kwargs["kpt_oks_sigmas"] == pytest.approx(compare_array)

    task_name = "face_localization"
    dataset_name = "fhibe_downsampled"
    eval_api = eval_api_fixture(task_name)
    current_results_dir = os.path.join(
        CURRENT_DIR, "static", "results", "mini", task_name, dataset_name, model_name
    )
    model, model_name = demo_model_fixture(task_name)
    annotations_dataframe, img_filepaths, kwargs = prepare_evaluation(
        eval_api=eval_api,
        task_name=task_name,
        dataset_name=dataset_name,
        model=model,
        model_name=model_name,
        current_results_dir=current_results_dir,
        cuda=False,
    )
    assert len(annotations_dataframe) == 50
    assert len(annotations_dataframe.columns) == 50
    assert kwargs == {"gt_column_name": "face_bbox"}

    task_name = "body_parts_detection"
    dataset_name = "fhibe_downsampled"
    eval_api = eval_api_fixture(task_name, use_mini_dataset=True, mini_dataset_size=50)
    model, model_name = demo_model_fixture(task_name)
    current_results_dir = os.path.join(
        CURRENT_DIR, "static", "results", "mini", task_name, dataset_name, model_name
    )
    annotations_dataframe, img_filepaths, kwargs = prepare_evaluation(
        eval_api=eval_api,
        task_name=task_name,
        dataset_name=dataset_name,
        model=model,
        current_results_dir=current_results_dir,
        model_name=model_name,
        cuda=False,
    )
    assert len(annotations_dataframe) == 50
    assert len(annotations_dataframe.columns) == 50
    assert list(kwargs.keys()) == [
        "gt_column_name",
        "img_filepath_gt_bboxes",
    ]
    assert kwargs["gt_column_name"] is None
    assert isinstance(kwargs["img_filepath_gt_bboxes"], dict)
    assert len(kwargs["img_filepath_gt_bboxes"]) == 50
    for img_fp, gt_bboxes in kwargs["img_filepath_gt_bboxes"].items():
        assert len(gt_bboxes) in [1, 2]
        for gt_bbox in gt_bboxes:
            assert len(gt_bbox) == 4
    # Verify that derivative body parts are calculated correctly
    for bp_list in annotations_dataframe["visible_body_parts"]:
        # Face should be in all of them
        assert "Face" in bp_list

        # If left or right leg in then "leg" needs to be in as well
        if "Left leg" in bp_list or "Right leg" in bp_list:
            assert "Leg" in bp_list

        # If left or right hand in then "Hand" needs to be in as well
        if any([x in bp_list for x in ["Left hand", "Right hand", "Glove"]]):
            assert "Hand" in bp_list

        assert 1 <= len(bp_list) <= 35

    task_name = "person_parsing"
    dataset_name = "fhibe_downsampled"
    eval_api = eval_api_fixture(task_name, use_mini_dataset=True, mini_dataset_size=50)
    model, model_name = demo_model_fixture(task_name)
    current_results_dir = os.path.join(
        CURRENT_DIR, "static", "results", "mini", task_name, dataset_name, model_name
    )
    precomputed_mask_file = os.path.join(
        current_results_dir,
        "ground_truth",
        "precomputed_person_masks.pkl",
    )
    with open(precomputed_mask_file, "rb") as file:
        precomputed_masks = pickle.load(file)
    annotations_dataframe, img_filepaths, kwargs = prepare_evaluation(
        eval_api=eval_api,
        task_name=task_name,
        dataset_name=dataset_name,
        model=model,
        model_name=model_name,
        current_results_dir=current_results_dir,
        precomputed_masks=precomputed_masks,
        cuda=False,
    )
    assert len(annotations_dataframe) == 50
    assert len(annotations_dataframe.columns) == 50
    assert kwargs == {"to_rle": True, "gt_column_name": "person_mask"}

    task_name = "face_super_resolution"
    dataset_name = "fhibe_face_crop_align"
    eval_api = eval_api_fixture(task_name, use_mini_dataset=True, mini_dataset_size=50)
    model, model_name = demo_model_fixture(task_name)
    current_results_dir = os.path.join(
        CURRENT_DIR, "static", "results", "mini", task_name, dataset_name, model_name
    )
    annotations_dataframe, img_filepaths, kwargs = prepare_evaluation(
        eval_api=eval_api,
        task_name=task_name,
        dataset_name=dataset_name,
        model=model,
        model_name=model_name,
        current_results_dir=current_results_dir,
        cuda=False,
    )
    assert len(annotations_dataframe) == 50
    assert len(annotations_dataframe.columns) == 76
    assert kwargs == {"aligned": True}
