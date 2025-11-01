# SPDX-License-Identifier: Apache-2.0
import os

import numpy as np
import pytest

from fhibe_eval_api.common.metrics import f1_score
from fhibe_eval_api.common.utils import read_json_file
from fhibe_eval_api.metrics.fhibe_metrics import (
    accuracy_body_part_detection,
    average_recall_bbox,
    average_recall_body_part_detection,
    average_recall_mask,
    f1_scores_parsing,
    learned_perceptual_image_patch_similarity,
    object_keypoint_similarity,
    percentage_correct_keypoints,
)

current_dir = os.path.dirname(__file__)


def test_average_recall_bbox(prepare_task_fixture):
    task_name = "person_localization"
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name)
    )
    base_results_dir = os.path.join(
        current_dir,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
    )
    fixed_results_dir = os.path.join(base_results_dir, "ground_truth")
    current_results_dir = os.path.join(base_results_dir, "compare")

    # Read existing model outputs on disk
    model_outputs_fp = os.path.join(fixed_results_dir, "fixed_model_outputs.json")
    model_outputs = read_json_file(model_outputs_fp)

    # Write to subdirectory compare/
    results_path = os.path.join(current_results_dir, "results_iou_threshold.json")
    detailed_results_path = os.path.join(
        current_results_dir, "detailed_results_iou_threshold.json"
    )
    if os.path.isfile(results_path):
        os.remove(results_path)
    if os.path.isfile(detailed_results_path):
        os.remove(detailed_results_path)

    grouped_results_dict = average_recall_bbox(
        task_name,
        intersectional_groups=[
            "pronoun",
            "age",
            "apparent_skin_color",
            "ancestry",
        ],
        filepaths=img_filepaths,
        model_outputs=model_outputs,
        annotations_dataframe=annotations_df,
        thresholds=list(np.arange(0.5, 1.0, 0.05)),
        current_results_dir=current_results_dir,
        **kwargs,
    )
    assert os.path.isfile(results_path)
    os.path.isfile(detailed_results_path)

    # cleanup
    if os.path.isfile(results_path):
        os.remove(results_path)
    if os.path.isfile(detailed_results_path):
        os.remove(detailed_results_path)

    assert len(grouped_results_dict) == 15

    pronoun_subdict = grouped_results_dict["['pronoun']"]
    assert len(pronoun_subdict) == 3
    she_subdict = pronoun_subdict["['0. She/her/hers']"]
    assert len(she_subdict) == 3
    assert list(she_subdict.keys()) == ["scores", "AR_IOU", "Class_Size"]
    assert len(she_subdict["scores"]) == she_subdict["Class_Size"]
    assert she_subdict["AR_IOU"] == pytest.approx(0.8863636363636364)
    assert np.mean(she_subdict["scores"]) == pytest.approx(she_subdict["AR_IOU"])
    longkey = "['pronoun', 'age', 'apparent_skin_color', 'ancestry']"
    assert longkey in grouped_results_dict
    longcol = "['0. She/her/hers', '[18, 30)', '1. [136, 105, 81]', '0. Africa']"
    assert longcol in grouped_results_dict[longkey]


def test_average_recall_mask(prepare_task_fixture):
    task_name = "person_parsing"
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name)
    )
    base_results_dir = os.path.join(
        current_dir,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
    )
    fixed_results_dir = os.path.join(base_results_dir, "ground_truth")
    current_results_dir = os.path.join(base_results_dir, "compare")

    # Read existing model outputs on disk
    model_outputs_fp = os.path.join(fixed_results_dir, "fixed_model_outputs.json")
    model_outputs = read_json_file(model_outputs_fp)

    # Write to subdirectory compare/
    results_path = os.path.join(current_results_dir, "results_iou_threshold.json")
    detailed_results_path = os.path.join(
        current_results_dir, "detailed_results_iou_threshold.json"
    )
    if os.path.isfile(results_path):
        os.remove(results_path)
    if os.path.isfile(detailed_results_path):
        os.remove(detailed_results_path)

    grouped_results_dict = average_recall_mask(
        task_name,
        intersectional_groups=["nationality", "lighting", "head_pose"],
        filepaths=img_filepaths,
        model_outputs=model_outputs,
        annotations_dataframe=annotations_df,
        thresholds=list(np.arange(0.5, 1.0, 0.05)),
        current_results_dir=current_results_dir,
        **kwargs,
    )

    assert os.path.isfile(results_path)
    os.path.isfile(detailed_results_path)

    # cleanup
    if os.path.isfile(results_path):
        os.remove(results_path)
    if os.path.isfile(detailed_results_path):
        os.remove(detailed_results_path)

    assert len(grouped_results_dict) == 7  # 2^3 -1
    assert "['nationality']" in grouped_results_dict
    nat_subdict = grouped_results_dict["['nationality']"]
    assert "['3. American']" in nat_subdict
    american_subdict = nat_subdict["['3. American']"]
    assert len(american_subdict) == 3
    assert list(american_subdict.keys()) == ["scores", "AR_MASK", "Class_Size"]
    assert american_subdict["Class_Size"] == len(american_subdict["scores"])
    assert american_subdict["scores"] == [0.8, 0.8, 0.9, 0.8]
    assert american_subdict["AR_MASK"] == pytest.approx(np.mean([0.8, 0.8, 0.9, 0.8]))

    assert "['nationality', 'lighting', 'head_pose']" in grouped_results_dict
    intersect_subdict = grouped_results_dict["['nationality', 'lighting', 'head_pose']"]
    assert len(intersect_subdict) == 80
    longkey = "['5. Angolan', '0. Lighting from above the head/face', '0. Typical']"
    assert longkey in intersect_subdict


def test_percentage_correct_keypoints(prepare_task_fixture):
    task_name = "keypoint_estimation"
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name)
    )
    base_results_dir = os.path.join(
        current_dir,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
    )
    fixed_results_dir = os.path.join(base_results_dir, "ground_truth")
    current_results_dir = os.path.join(base_results_dir, "compare")

    # Read existing model outputs on disk
    model_outputs_fp = os.path.join(fixed_results_dir, "fixed_model_outputs.json")
    model_outputs = read_json_file(model_outputs_fp)

    # Write to subdirectory compare/
    results_path = os.path.join(current_results_dir, "pck_scores_threshold.json")
    if os.path.isfile(results_path):
        os.remove(results_path)

    grouped_results_dict = percentage_correct_keypoints(
        task_name,
        intersectional_groups=[
            "camera_position",
            "camera_distance",
            "natural_skin_color",
            "hairstyle",
        ],
        filepaths=img_filepaths,
        model_outputs=model_outputs,
        annotations_dataframe=annotations_df,
        thresholds=[0.1, 0.25, 0.5, 0.75, 0.9],
        current_results_dir=current_results_dir,
        **kwargs,
    )

    assert os.path.isfile(results_path)

    # cleanup
    if os.path.isfile(results_path):
        os.remove(results_path)

    assert len(grouped_results_dict) == 15
    assert "['camera_position']" in grouped_results_dict
    assert "['hairstyle']" in grouped_results_dict
    camera_subdict = grouped_results_dict["['camera_position']"]
    assert "['0. Typical']" in camera_subdict
    typical_subdict = camera_subdict["['0. Typical']"]
    assert list(typical_subdict.keys()) == ["scores", "PCK", "Class_Size"]
    assert len(typical_subdict["scores"]) == typical_subdict["Class_Size"]
    assert np.mean(typical_subdict["scores"]) == typical_subdict["PCK"]

    longkey = (
        "['camera_position', 'camera_distance', 'natural_skin_color', 'hairstyle']"
    )
    assert longkey in grouped_results_dict
    longcol = "['0. Typical', '1. CD II', '0. [102, 78, 65]', '1. Buzz cut']"
    assert longcol in grouped_results_dict[longkey]
    longdict = grouped_results_dict[longkey][longcol]
    assert longdict["scores"][0] == pytest.approx(0.0)
    assert longdict["Class_Size"] == 1
    assert longdict["PCK"] == pytest.approx(0.0)


def test_object_keypoint_similarity(prepare_task_fixture):
    task_name = "keypoint_estimation"
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name)
    )
    base_results_dir = os.path.join(
        current_dir,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
    )
    fixed_results_dir = os.path.join(base_results_dir, "ground_truth")
    current_results_dir = os.path.join(base_results_dir, "compare")

    # Read existing model outputs on disk
    model_outputs_fp = os.path.join(fixed_results_dir, "fixed_model_outputs.json")
    model_outputs = read_json_file(model_outputs_fp)

    # Write to subdirectory compare/
    scores_path = os.path.join(current_results_dir, "oks_scores.json")
    results_path = os.path.join(current_results_dir, "results_oks_threshold.json")
    detailed_results_path = os.path.join(
        current_results_dir, "detailed_results_oks_threshold.json"
    )
    if os.path.isfile(scores_path):
        os.remove(scores_path)
    if os.path.isfile(results_path):
        os.remove(results_path)
    if os.path.isfile(detailed_results_path):
        os.remove(detailed_results_path)

    grouped_results_dict = object_keypoint_similarity(
        task_name,
        intersectional_groups=[
            "camera_position",
            "camera_distance",
            "natural_skin_color",
            "hairstyle",
        ],
        filepaths=img_filepaths,
        model_outputs=model_outputs,
        annotations_dataframe=annotations_df,
        thresholds=list(np.arange(0.5, 1.0, 0.05)),
        current_results_dir=current_results_dir,
        **kwargs,
    )

    assert os.path.isfile(scores_path)
    assert os.path.isfile(results_path)
    assert os.path.isfile(detailed_results_path)

    # Compare oks resuls
    pregen_oks_scores_filename = os.path.join(fixed_results_dir, "oks_scores.json")
    pregen_oks_scores = read_json_file(pregen_oks_scores_filename)
    oks_scores = read_json_file(scores_path)
    assert len(oks_scores) == len(pregen_oks_scores)
    assert oks_scores == pytest.approx(pregen_oks_scores)
    # cleanup
    if os.path.isfile(scores_path):
        os.remove(scores_path)
    if os.path.isfile(results_path):
        os.remove(results_path)
    if os.path.isfile(detailed_results_path):
        os.remove(detailed_results_path)

    assert len(grouped_results_dict) == 15
    assert "['camera_position']" in grouped_results_dict
    assert "['hairstyle']" in grouped_results_dict
    camera_subdict = grouped_results_dict["['camera_position']"]
    assert "['0. Typical']" in camera_subdict
    typical_subdict = camera_subdict["['0. Typical']"]
    assert list(typical_subdict.keys()) == ["scores", "AR_OKS", "Class_Size"]
    assert len(typical_subdict["scores"]) == typical_subdict["Class_Size"]
    assert np.mean(typical_subdict["scores"]) == typical_subdict["AR_OKS"]

    longkey = (
        "['camera_position', 'camera_distance', 'natural_skin_color', 'hairstyle']"
    )
    assert longkey in grouped_results_dict
    longcol = "['0. Typical', '1. CD II', '0. [102, 78, 65]', '1. Buzz cut']"
    assert longcol in grouped_results_dict[longkey]
    longdict = grouped_results_dict[longkey][longcol]
    assert longdict["scores"][0] == pytest.approx(0.0)
    assert longdict["Class_Size"] == 1
    assert longdict["AR_OKS"] == pytest.approx(0.0)


def test_average_recall_body_part_detection(prepare_task_fixture):
    task_name = "body_parts_detection"
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name)
    )
    base_results_dir = os.path.join(
        current_dir,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
    )
    fixed_results_dir = os.path.join(base_results_dir, "ground_truth")
    current_results_dir = os.path.join(base_results_dir, "compare")

    # Read existing model outputs on disk
    model_outputs_fp = os.path.join(fixed_results_dir, "fixed_model_outputs.json")
    model_outputs = read_json_file(model_outputs_fp)

    # Write to subdirectory compare/
    results_path = os.path.join(
        current_results_dir, "results_body_parts_AR_DET_threshold.json"
    )
    detailed_results_path = os.path.join(
        current_results_dir, "detailed_results_body_parts_AR_DET_threshold.json"
    )
    if os.path.isfile(results_path):
        os.remove(results_path)
    if os.path.isfile(detailed_results_path):
        os.remove(detailed_results_path)

    grouped_results_dict = average_recall_body_part_detection(
        task_name,
        intersectional_groups=["pronoun", "age"],
        filepaths=img_filepaths,
        model_outputs=model_outputs,
        annotations_dataframe=annotations_df,
        thresholds=list(np.arange(0.5, 1.0, 0.05)),
        current_results_dir=current_results_dir,
        **kwargs,
    )
    assert os.path.isfile(results_path)
    os.path.isfile(detailed_results_path)

    # cleanup
    if os.path.isfile(results_path):
        os.remove(results_path)
    if os.path.isfile(detailed_results_path):
        os.remove(detailed_results_path)

    assert len(grouped_results_dict) == 3
    assert "['pronoun']" in grouped_results_dict
    assert "['age']" in grouped_results_dict
    assert "['pronoun', 'age']" in grouped_results_dict
    age_subdict = grouped_results_dict["['age']"]
    assert "['[18, 30)']" in age_subdict
    young_subdict = age_subdict["['[18, 30)']"]
    assert list(young_subdict.keys()) == ["Face", "Hand"]
    face_subdict = young_subdict["Face"]
    assert len(face_subdict) == 3
    assert len(face_subdict["scores"]) == face_subdict["Class_Size"]
    assert face_subdict["AR_DET"] == pytest.approx(0.6)


def test_accuracy_body_part_detection(prepare_task_fixture):
    task_name = "body_parts_detection"
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name)
    )
    base_results_dir = os.path.join(
        current_dir,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
    )
    fixed_results_dir = os.path.join(base_results_dir, "ground_truth")
    current_results_dir = os.path.join(base_results_dir, "compare")

    # Read existing model outputs on disk
    model_outputs_fp = os.path.join(fixed_results_dir, "fixed_model_outputs.json")
    model_outputs = read_json_file(model_outputs_fp)

    # Write to subdirectory compare/
    results_path = os.path.join(
        current_results_dir, "results_body_parts_ACC_DET_threshold.json"
    )
    detailed_results_path = os.path.join(
        current_results_dir, "detailed_results_body_parts_ACC_DET_threshold.json"
    )
    if os.path.isfile(results_path):
        os.remove(results_path)
    if os.path.isfile(detailed_results_path):
        os.remove(detailed_results_path)

    grouped_results_dict = accuracy_body_part_detection(
        task_name,
        intersectional_groups=["pronoun", "age"],
        filepaths=img_filepaths,
        model_outputs=model_outputs,
        annotations_dataframe=annotations_df,
        thresholds=[0.5],
        current_results_dir=current_results_dir,
        **kwargs,
    )
    assert os.path.isfile(results_path)
    os.path.isfile(detailed_results_path)

    # cleanup
    if os.path.isfile(results_path):
        os.remove(results_path)
    if os.path.isfile(detailed_results_path):
        os.remove(detailed_results_path)

    assert len(grouped_results_dict) == 3
    assert "['pronoun']" in grouped_results_dict
    assert "['age']" in grouped_results_dict
    assert "['pronoun', 'age']" in grouped_results_dict
    age_subdict = grouped_results_dict["['age']"]
    assert "['[18, 30)']" in age_subdict
    young_subdict = age_subdict["['[18, 30)']"]
    assert list(young_subdict.keys()) == ["Face", "Hand"]
    face_subdict = young_subdict["Face"]
    assert len(face_subdict) == 3
    assert len(face_subdict["scores"]) == face_subdict["Class_Size"]
    assert face_subdict["ACC_DET"] == pytest.approx(1.0)


def test_lpips_face_super_resolution(prepare_task_fixture):
    task_name = "face_super_resolution"
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name)
    )
    results_base_dir = os.path.join(
        current_dir,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_face_crop_align",
        model_name,
    )
    fixed_results_dir = os.path.join(results_base_dir, "ground_truth")

    # Read existing model outputs on disk
    model_outputs_fp = os.path.join(fixed_results_dir, "fixed_model_outputs.json")
    model_outputs = read_json_file(model_outputs_fp)

    # Write to subdirectory compare/
    current_results_dir = os.path.join(results_base_dir, "compare")
    lpips_filename = os.path.join(current_results_dir, "lpips_scores.json")
    if os.path.isfile(lpips_filename):
        os.remove(lpips_filename)
    grouped_results_dict = learned_perceptual_image_patch_similarity(
        task_name,
        intersectional_groups=["pronoun", "scene"],
        filepaths=img_filepaths,
        model_outputs=model_outputs,
        annotations_dataframe=annotations_df,
        thresholds=None,
        current_results_dir=current_results_dir,
        **kwargs,
    )
    assert os.path.isfile(lpips_filename)

    assert "['pronoun']" in grouped_results_dict
    assert "['scene']" in grouped_results_dict
    assert "['pronoun', 'scene']" in grouped_results_dict
    assert len(grouped_results_dict) == 3
    scene_subdict = grouped_results_dict["['scene']"]
    assert len(scene_subdict) == 9
    assert "['3. Outdoor: Man-made elements']" in scene_subdict
    man_made_dict = scene_subdict["['3. Outdoor: Man-made elements']"]
    print(man_made_dict["scores"])
    assert man_made_dict["scores"] == pytest.approx(
        [0.4467528164386749, 0.43641766905784607, 0.43755948543548584]
    )
    assert man_made_dict["Class_Size"] == 3
    assert np.mean(man_made_dict["scores"]) == pytest.approx(man_made_dict["LPIPS"])
    assert len(man_made_dict["scores"]) == man_made_dict["Class_Size"]


# def test_peak_signal_to_noise_ratio(prepare_task_fixture):
#     ## This metric requires having access to the images
#     ## but we cannot upload such images to the cloud in the CI tests
#     ## So until we figure out a secure way to do this we must
#     ## Disable this test.
#     task_name = "face_encoding"
#     (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
#         prepare_task_fixture(task_name)
#     )
#     fixed_results_dir = os.path.join(
#         current_dir,
#         "static",
#         "results",
#         "mini",
#         task_name,
#         "fhibe_face_crop_align",
#         model_name
#     )

#     # Write to subdirectory compare/
#     current_results_dir = os.path.join(fixed_results_dir, "compare")
#     psnr_filename = os.path.join(current_results_dir, "psnr_scores.json")
#     if os.path.isfile(psnr_filename):
#         os.remove(psnr_filename)
#     grouped_results_dict = peak_signal_to_noise_ratio(
#         task_name,
#         intersectional_groups=["apparent_right_eye_color","facial_marks"],
#         filepaths=img_filepaths,
#         model_outputs={},
#         annotations_dataframe=annotations_df,
#         thresholds=None,
#         current_results_dir=current_results_dir,
#         **kwargs,
#     )

#     assert os.path.isfile(psnr_filename)
#     # Compare psnr
#     pregen_psnr_filename = os.path.join(fixed_results_dir, "psnr_scores.json")
#     psnr_pregen = read_json_file(pregen_psnr_filename)
#     psnr_gen = read_json_file(psnr_filename)

#     assert np.array_equal(psnr_gen, psnr_pregen)
#     # cleanup
#     if os.path.isfile(psnr_filename):
#         os.remove(psnr_filename)

#     assert len(grouped_results_dict) == 3
#     eye_color_subdict = grouped_results_dict["['apparent_right_eye_color']"]
#     assert len(eye_color_subdict) == 4
#     brown_subdict =  eye_color_subdict["['5. Brown']"]
#     assert len(brown_subdict["scores"]) == brown_subdict["Class_Size"]
#     assert np.mean(brown_subdict["scores"]) == brown_subdict["PSNR"]


def test_lpips_face_encoding(prepare_task_fixture):
    task_name = "face_encoding"
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name)
    )
    results_base_dir = os.path.join(
        current_dir,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_face_crop_align",
        model_name,
    )

    # Write to subdirectory compare/
    current_results_dir = os.path.join(results_base_dir, "compare")
    lpips_filename = os.path.join(current_results_dir, "lpips_scores.json")
    if os.path.isfile(lpips_filename):
        os.remove(lpips_filename)
    grouped_results_dict = learned_perceptual_image_patch_similarity(
        task_name,
        intersectional_groups=["facial_hairstyle", "natural_facial_haircolor"],
        filepaths=img_filepaths,
        model_outputs={},
        annotations_dataframe=annotations_df,
        thresholds=None,
        current_results_dir=current_results_dir,
        **kwargs,
    )

    assert os.path.isfile(lpips_filename)

    # cleanup
    if os.path.isfile(lpips_filename):
        os.remove(lpips_filename)

    assert len(grouped_results_dict) == 3
    hairstyle_subdict = grouped_results_dict["['facial_hairstyle']"]
    assert len(hairstyle_subdict) == 4
    beard_subdict = hairstyle_subdict["['1. Beard']"]
    assert len(beard_subdict) == 3
    assert len(beard_subdict["scores"]) == beard_subdict["Class_Size"]
    assert np.mean(beard_subdict["scores"]) == beard_subdict["LPIPS"]


def test_f1_score():
    np.random.seed(0)
    pred_mask = np.random.randint(0, 255, size=(800, 600))
    gt_mask = np.random.randint(0, 255, size=(800, 600))
    f1 = f1_score(pred_mask=pred_mask, gt_mask=gt_mask)
    assert f1 == pytest.approx(0.9958861855118002)


def test_f1_score_parsing(prepare_task_fixture):
    task_name = "face_parsing"
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name)
    )
    results_base_dir = os.path.join(
        current_dir,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_face_crop_align",
        model_name,
    )
    fixed_results_dir = os.path.join(results_base_dir, "ground_truth")

    # Read existing model outputs on disk
    model_outputs_fp = os.path.join(fixed_results_dir, "fixed_model_outputs.json")
    model_outputs = read_json_file(model_outputs_fp)

    # Write to subdirectory compare/
    current_results_dir = os.path.join(results_base_dir, "compare")
    f1_scores_filename = os.path.join(current_results_dir, "F1_scores.json")

    if os.path.isfile(f1_scores_filename):
        os.remove(f1_scores_filename)

    grouped_results_dict = f1_scores_parsing(
        task_name,
        intersectional_groups=[
            "pronoun",
            "age",
            "apparent_skin_color",
        ],
        filepaths=img_filepaths,
        model_outputs=model_outputs,
        annotations_dataframe=annotations_df,
        thresholds=None,
        current_results_dir=current_results_dir,
        **kwargs,
    )
    assert os.path.isfile(f1_scores_filename)

    # cleanup
    if os.path.isfile(f1_scores_filename):
        os.remove(f1_scores_filename)
    assert len(grouped_results_dict) == 7
    pronoun_dict = grouped_results_dict["['pronoun']"]
    assert len(pronoun_dict) == 3
    assert "['0. She/her/hers']" in pronoun_dict
    assert "['1. He/him/his']" in pronoun_dict
    assert "['2. They/them/their']" in pronoun_dict
    age_subdict = grouped_results_dict["['age']"]
    assert len(age_subdict) == 5
    old_subdict = age_subdict["['[60, +]']"]

    assert len(old_subdict) == 3
    assert len(old_subdict["scores"]) == old_subdict["Class_Size"]
    assert np.mean(old_subdict["scores"]) == old_subdict["F1"]
