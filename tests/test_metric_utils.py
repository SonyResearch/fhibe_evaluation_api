# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest

from fhibe_eval_api.common.utils import read_json_file
from fhibe_eval_api.metrics.utils import (
    compute_gt_bbox_iou_scores,
    compute_gt_mask_iou_scores,
    gather_body_part_detection_scores,
    group_f1_results,
    group_thresholded_body_part_results,
    group_thresholded_metric_results,
    save_thresholded_results_per_file,
)

CURRENT_DIR = os.path.dirname(__file__)


def test_compute_gt_bbox_iou_scores(prepare_task_fixture):
    task_name = "person_localization"
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name=task_name)
    )
    model_outputs_fp = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
        "ground_truth",
        "fixed_model_outputs.json",
    )
    model_outputs = read_json_file(model_outputs_fp)

    gt_iou_scores = compute_gt_bbox_iou_scores(
        filepaths=list(model_outputs.keys()),
        model_outputs=model_outputs,
        annotations_dataframe=annotations_df,
        gt_column_name="person_bbox",
    )
    assert len(gt_iou_scores) >= len(model_outputs)
    assert all([x >= 0 for x in gt_iou_scores])
    assert sum(gt_iou_scores) > 0


def test_compute_gt_mask_iou_scores(prepare_task_fixture):
    task_name = "person_parsing"
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name=task_name)
    )
    model_outputs_fp = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
        "ground_truth",
        "fixed_model_outputs.json",
    )
    model_outputs = read_json_file(model_outputs_fp)

    gt_iou_scores = compute_gt_mask_iou_scores(
        filepaths=list(model_outputs.keys()),
        model_outputs=model_outputs,
        annotations_dataframe=annotations_df,
        gt_mask_column_name="person_mask",
        to_rle=True,
    )
    assert len(gt_iou_scores) >= len(model_outputs)
    assert all([x >= 0 for x in gt_iou_scores])
    assert sum(gt_iou_scores) > 0


def test_gather_body_part_detection_scores(prepare_task_fixture):
    task_name = "body_parts_detection"
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name, use_mini_dataset=True)
    )
    current_results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
    )
    fixed_results_dir = os.path.join(current_results_dir, "ground_truth")
    model_outputs_fp = os.path.join(
        fixed_results_dir,
        "fixed_model_outputs.json",
    )
    model_outputs = read_json_file(model_outputs_fp)

    bp_det_list = gather_body_part_detection_scores(
        filepaths=list(model_outputs.keys()),
        model_outputs=model_outputs,
        annotations_dataframe=annotations_df,
    )
    assert len(bp_det_list) >= len(model_outputs)
    for det_dict in bp_det_list:
        assert "Face" in det_dict
        assert "Hand" in det_dict
        assert det_dict["Face"] >= 0
        assert det_dict["Hand"] >= 0


def test_save_thresholded_results_per_file(prepare_task_fixture):
    task_name = "person_localization"
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name=task_name)
    )
    current_results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
    )
    fixed_results_dir = os.path.join(current_results_dir, "ground_truth")
    gt_iou_scores_fp = os.path.join(fixed_results_dir, "gt_bbox_iou_scores.json")
    gt_iou_scores = read_json_file(gt_iou_scores_fp)
    results_path = os.path.join(
        current_results_dir, "compare", "results_iou_threshold.json"
    )
    detailed_results_path = os.path.join(
        current_results_dir, "compare", "detailed_results_iou_threshold.json"
    )
    thresholds = list(np.arange(0.5, 1.0, 0.05))
    if os.path.isfile(results_path):
        os.remove(results_path)
    if os.path.isfile(detailed_results_path):
        os.remove(detailed_results_path)

    metric_threshold_dict = save_thresholded_results_per_file(
        annotations_dataframe=annotations_df,
        metric_vals=gt_iou_scores,
        results_path=results_path,
        detailed_results_path=detailed_results_path,
        thresholds=thresholds,
    )
    assert os.path.isfile(results_path) is True
    assert os.path.isfile(detailed_results_path) is True
    assert "0.50" in metric_threshold_dict
    assert "0.75" in metric_threshold_dict
    assert "0.95" in metric_threshold_dict
    assert sum(metric_threshold_dict["0.50"].values()) == 50
    assert sum(metric_threshold_dict["0.70"].values()) == 50
    assert sum(metric_threshold_dict["0.90"].values()) == 42

    # cleanup
    os.remove(results_path)
    os.remove(detailed_results_path)


def test_group_thresholded_metric_results(prepare_task_fixture):
    task_name = "person_localization"
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name=task_name)
    )
    current_results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
    )
    fixed_results_dir = os.path.join(current_results_dir, "ground_truth")
    threshold_dict_fp = os.path.join(fixed_results_dir, "results_iou_threshold.json")
    threshold_dict = read_json_file(threshold_dict_fp)
    save_json_fp = os.path.join(
        current_results_dir, "compare", "intersectional_results_AR_IOU.json"
    )
    if os.path.isfile(save_json_fp):
        os.remove(save_json_fp)
    grouped_results_dict = group_thresholded_metric_results(
        threshold_dict=threshold_dict,
        metric_name="AR_IOU",
        annotations_dataframe=annotations_df,
        intersectional_groups=["pronoun", "age", "ancestry", "apparent_skin_color"],
        save_json_filepath=save_json_fp,
    )
    assert os.path.isfile(save_json_fp) is True

    # compare to pre-generated result
    pregenerated_fp = os.path.join(
        fixed_results_dir, "intersectional_results_AR_IOU.json"
    )
    pregen_grouped_results_dict = read_json_file(pregenerated_fp)
    assert len(grouped_results_dict.keys()) == len(pregen_grouped_results_dict.keys())
    # print("grouped_results_dict: ", grouped_results_dict)
    for attr_combo in grouped_results_dict:
        for attr_val_str, mdict in grouped_results_dict[attr_combo].items():
            for key, val in mdict.items():
                assert val == pytest.approx(
                    pregen_grouped_results_dict[attr_combo][attr_val_str][key]
                )


def test_group_f1_results(prepare_task_fixture):
    task_name = "face_parsing"
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name)
    )
    current_results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_face_crop_align",
        model_name,
    )
    fixed_results_dir = os.path.join(current_results_dir, "ground_truth")
    f1_results_file = os.path.join(fixed_results_dir, "F1_scores.json")
    all_f1_scores = read_json_file(f1_results_file)

    save_json_fp = os.path.join(
        current_results_dir, "compare", "intersectional_results_F1.json"
    )
    if os.path.isfile(save_json_fp):
        os.remove(save_json_fp)

    grouped_results_dict = group_f1_results(
        all_f1_scores=all_f1_scores,
        annotations_dataframe=annotations_df,
        intersectional_groups=["pronoun", "age", "ancestry", "apparent_skin_color"],
        save_json_filepath=save_json_fp,
    )
    assert os.path.isfile(save_json_fp)

    # compare to pre-generated result
    pregenerated_fp = os.path.join(fixed_results_dir, "intersectional_results_F1.json")
    pregen_grouped_results_dict = read_json_file(pregenerated_fp)

    assert len(grouped_results_dict.keys()) == len(pregen_grouped_results_dict.keys())
    for attr_combo in grouped_results_dict:
        for attr_val_str, mdict in grouped_results_dict[attr_combo].items():
            for key, val in mdict.items():
                assert val == pytest.approx(
                    pregen_grouped_results_dict[attr_combo][attr_val_str][key]
                )


def test_group_thresholded_body_part_results(prepare_task_fixture):
    task_name = "body_parts_detection"
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name)
    )
    current_results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
    )
    fixed_results_dir = os.path.join(current_results_dir, "ground_truth")
    threshold_dict_fp = os.path.join(
        fixed_results_dir, "results_body_parts_AR_DET_threshold.json"
    )
    threshold_dict = read_json_file(threshold_dict_fp)
    save_json_fp = os.path.join(
        current_results_dir, "compare", "intersectional_results_AR_DET.json"
    )
    if os.path.isfile(save_json_fp):
        os.remove(save_json_fp)
    grouped_results_dict = group_thresholded_body_part_results(
        threshold_dict=threshold_dict,
        metric_name="AR_DET",
        annotations_dataframe=annotations_df,
        intersectional_groups=["pronoun", "age", "ancestry", "apparent_skin_color"],
        save_json_filepath=save_json_fp,
    )
    assert os.path.isfile(save_json_fp) is True
    # compare to pre-generated result
    pregenerated_fp = os.path.join(
        fixed_results_dir, "intersectional_results_AR_DET.json"
    )
    pregen_grouped_results_dict = read_json_file(pregenerated_fp)
    assert len(grouped_results_dict) == len(pregen_grouped_results_dict)
    for attr_combo in grouped_results_dict:
        for attr_val_str, bp_dict in grouped_results_dict[attr_combo].items():
            for bp, mdict in bp_dict.items():
                for key, val in mdict.items():
                    if isinstance(val, list):
                        assert val == pytest.approx(
                            pregen_grouped_results_dict[attr_combo][attr_val_str][bp][
                                key
                            ]
                        )
                        continue
                    if np.isnan(val):
                        assert np.isnan(
                            pregen_grouped_results_dict[attr_combo][attr_val_str][bp][
                                key
                            ]
                        )
                    else:
                        assert val == pytest.approx(
                            pregen_grouped_results_dict[attr_combo][attr_val_str][bp][
                                key
                            ]
                        )

    # Cleanup
    os.remove(save_json_fp)
