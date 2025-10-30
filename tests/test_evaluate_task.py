# SPDX-License-Identifier: Apache-2.0

"""Integration tests."""

import os

import pytest

from fhibe_eval_api.common.utils import read_json_file
from fhibe_eval_api.evaluate.evaluate import evaluate_task

CURRENT_DIR = os.path.dirname(__file__)


def test_evaluate_person_localization(prepare_task_fixture):
    task_name = "person_localization"
    data_rootdir = os.path.join(CURRENT_DIR, "static")
    dataset_name = "fhibe"
    downsampled = True
    attributes = [
        "pronoun",
        "age",
        "apparent_skin_color",
        "apparent_skin_color_hue_lum",
        "ancestry",
    ]
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name)
    )
    results_rootdir = os.path.join(CURRENT_DIR, "static", "results")
    current_results_dir = os.path.join(
        results_rootdir,
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
    )
    model_outputs_fp = os.path.join(current_results_dir, "model_outputs.json")
    threshold_fp = os.path.join(current_results_dir, "results_iou_threshold.json")
    detailed_threshold_fp = os.path.join(
        current_results_dir, "detailed_results_iou_threshold.json"
    )
    iou_fp = os.path.join(current_results_dir, "gt_bbox_iou_scores.json")
    intersect_fp = os.path.join(
        current_results_dir, "intersectional_results_AR_IOU.json"
    )
    filelist = [
        model_outputs_fp,
        threshold_fp,
        detailed_threshold_fp,
        iou_fp,
        intersect_fp,
    ]
    for f in filelist:
        if os.path.isfile(f):
            os.remove(f)

    evaluate_task(
        data_rootdir=data_rootdir,
        dataset_name=dataset_name,
        model=wrapped_model,
        model_name=model_name,
        task_name=task_name,
        metrics=None,
        attributes=attributes,
        use_mini_dataset=True,
        mini_dataset_size=50,
        downsampled=downsampled,
        reuse_model_outputs=False,
        results_rootdir=results_rootdir,
        cuda=False,
    )
    for f in filelist:
        assert os.path.isfile(f)

    # cleanup all except model outputs
    for f in filelist:
        if f == model_outputs_fp:
            continue
        if os.path.isfile(f):
            os.remove(f)

    # Run reusing model outputs
    evaluate_task(
        data_rootdir=data_rootdir,
        dataset_name=dataset_name,
        model=None,
        model_name=model_name,
        task_name=task_name,
        metrics=None,
        attributes=attributes,
        use_mini_dataset=True,
        mini_dataset_size=50,
        downsampled=downsampled,
        reuse_model_outputs=True,
        results_rootdir=results_rootdir,
        cuda=False,
    )
    for f in filelist:
        assert os.path.isfile(f)

    # cleanup
    for f in filelist:
        if os.path.isfile(f):
            os.remove(f)

    # Ensure that if you use model=None and reuse_model_outputs=False,
    # an error will be raised
    # Run reusing model outputs
    with pytest.raises(ValueError) as excinfo:
        evaluate_task(
            data_rootdir=data_rootdir,
            dataset_name=dataset_name,
            model=None,
            model_name=model_name,
            task_name=task_name,
            metrics=None,
            attributes=attributes,
            use_mini_dataset=True,
            mini_dataset_size=50,
            downsampled=downsampled,
            reuse_model_outputs=False,
            results_rootdir=results_rootdir,
            cuda=False,
        )

    error_str = "If reuse_model_outputs is not True, model cannot be None."
    assert error_str == str(excinfo.value)

    # Ensure that if you use reuse_model_outputs=True,
    # but the model_outputs_file is not found,
    # an error will be raised
    with pytest.raises(FileNotFoundError) as excinfo:
        evaluate_task(
            data_rootdir=data_rootdir,
            dataset_name=dataset_name,
            model=None,
            model_name="bad_model",
            task_name=task_name,
            metrics=None,
            attributes=attributes,
            use_mini_dataset=True,
            mini_dataset_size=50,
            downsampled=downsampled,
            reuse_model_outputs=True,
            results_rootdir=results_rootdir,
            cuda=False,
        )

    error_str1 = "Model outputs file expected at: "
    error_str2 = (
        "but not found. Set reuse_model_outputs=False " "to recompute model outputs."
    )
    assert error_str1 in str(excinfo.value)
    assert error_str2 in str(excinfo.value)


def test_evaluate_person_parsing(prepare_task_fixture):
    task_name = "person_parsing"
    data_rootdir = os.path.join(CURRENT_DIR, "static")
    dataset_name = "fhibe"
    downsampled = True
    attributes = [
        "pronoun",
        "age",
        "apparent_skin_color",
        "ancestry",
    ]
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name)
    )
    results_rootdir = os.path.join(CURRENT_DIR, "static", "results")
    current_results_dir = os.path.join(
        results_rootdir,
        task_name,
        "fhibe_downsampled",
        model_name,
    )
    model_outputs_fp = os.path.join(current_results_dir, "model_outputs.json")
    threshold_fp = os.path.join(current_results_dir, "results_iou_threshold.json")
    detailed_threshold_fp = os.path.join(
        current_results_dir, "detailed_results_iou_threshold.json"
    )
    intersect_fp = os.path.join(
        current_results_dir, "intersectional_results_AR_MASK.json"
    )
    filelist = [model_outputs_fp, threshold_fp, detailed_threshold_fp, intersect_fp]
    for f in filelist:
        if os.path.isfile(f):
            os.remove(f)

    evaluate_task(
        data_rootdir=data_rootdir,
        dataset_name=dataset_name,
        model=wrapped_model,
        model_name=model_name,
        task_name=task_name,
        metrics=None,
        attributes=attributes,
        use_mini_dataset=False,
        downsampled=downsampled,
        reuse_model_outputs=False,
        results_rootdir=results_rootdir,
        cuda=False,
    )
    for f in filelist:
        assert os.path.isfile(f)

    # cleanup
    for f in filelist:
        if os.path.isfile(f):
            os.remove(f)


def test_evaluate_face_localization(prepare_task_fixture):
    task_name = "face_localization"
    data_rootdir = os.path.join(CURRENT_DIR, "static")
    dataset_name = "fhibe"
    downsampled = True
    attributes = [
        "pronoun",
        "age",
        "apparent_skin_color",
        "ancestry",
    ]
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name)
    )
    results_rootdir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
    )
    current_results_dir = os.path.join(
        results_rootdir,
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
    )
    model_outputs_fp = os.path.join(current_results_dir, "model_outputs.json")
    threshold_fp = os.path.join(current_results_dir, "results_iou_threshold.json")
    detailed_threshold_fp = os.path.join(
        current_results_dir, "detailed_results_iou_threshold.json"
    )
    iou_fp = os.path.join(current_results_dir, "gt_bbox_iou_scores.json")
    intersect_fp = os.path.join(
        current_results_dir, "intersectional_results_AR_IOU.json"
    )
    filelist = [
        model_outputs_fp,
        threshold_fp,
        detailed_threshold_fp,
        iou_fp,
        intersect_fp,
    ]
    for f in filelist:
        if os.path.isfile(f):
            os.remove(f)

    evaluate_task(
        data_rootdir=data_rootdir,
        dataset_name=dataset_name,
        model=wrapped_model,
        model_name=model_name,
        task_name=task_name,
        metrics=None,
        attributes=attributes,
        use_mini_dataset=True,
        mini_dataset_size=50,
        downsampled=downsampled,
        reuse_model_outputs=False,
        results_rootdir=results_rootdir,
        cuda=False,
    )
    for f in filelist:
        assert os.path.isfile(f)

    # cleanup
    for f in filelist:
        if os.path.isfile(f):
            os.remove(f)


def test_evaluate_body_parts_detection(prepare_task_fixture):
    task_name = "body_parts_detection"
    data_rootdir = os.path.join(CURRENT_DIR, "static")
    dataset_name = "fhibe"
    downsampled = True
    attributes = [
        "pronoun",
        "age",
        "apparent_skin_color",
        "ancestry",
    ]
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name)
    )
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name)
    )
    results_rootdir = os.path.join(CURRENT_DIR, "static", "results")
    current_results_dir = os.path.join(
        results_rootdir,
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
    )
    model_outputs_fp = os.path.join(current_results_dir, "model_outputs.json")
    threshold_acc_fp = os.path.join(
        current_results_dir, "results_body_parts_ACC_DET_threshold.json"
    )
    threshold_ar_fp = os.path.join(
        current_results_dir, "results_body_parts_AR_DET_threshold.json"
    )
    detailed_threshold_acc_fp = os.path.join(
        current_results_dir, "detailed_results_body_parts_ACC_DET_threshold.json"
    )
    detailed_threshold_ar_fp = os.path.join(
        current_results_dir, "detailed_results_body_parts_AR_DET_threshold.json"
    )
    bp_fp = os.path.join(current_results_dir, "body_part_detections.json")
    intersect_acc_fp = os.path.join(
        current_results_dir, "intersectional_results_ACC_DET.json"
    )
    intersect_ar_fp = os.path.join(
        current_results_dir, "intersectional_results_AR_DET.json"
    )
    filelist = [
        model_outputs_fp,
        threshold_acc_fp,
        threshold_ar_fp,
        detailed_threshold_acc_fp,
        detailed_threshold_ar_fp,
        bp_fp,
        intersect_acc_fp,
        intersect_ar_fp,
    ]
    for f in filelist:
        if os.path.isfile(f):
            os.remove(f)

    evaluate_task(
        data_rootdir=data_rootdir,
        dataset_name=dataset_name,
        model=wrapped_model,
        model_name=model_name,
        task_name=task_name,
        metrics=None,
        attributes=attributes,
        use_mini_dataset=True,
        mini_dataset_size=50,
        downsampled=downsampled,
        reuse_model_outputs=False,
        results_rootdir=results_rootdir,
        cuda=False,
    )
    for f in filelist:
        assert os.path.isfile(f)

    # cleanup
    for f in filelist:
        if os.path.isfile(f):
            os.remove(f)


def test_evaluate_keypoint_estimation(prepare_task_fixture):
    task_name = "keypoint_estimation"
    data_rootdir = os.path.join(CURRENT_DIR, "static")
    dataset_name = "fhibe"
    downsampled = True
    attributes = [
        "pronoun",
        "age",
        "apparent_skin_color",
        "ancestry",
    ]
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name)
    )
    results_rootdir = os.path.join(CURRENT_DIR, "static", "results")
    current_results_dir = os.path.join(
        results_rootdir,
        task_name,
        "fhibe_downsampled",
        model_name,
    )
    model_outputs_fp = os.path.join(current_results_dir, "model_outputs.json")
    threshold_fp = os.path.join(current_results_dir, "results_oks_threshold.json")
    detailed_threshold_fp = os.path.join(
        current_results_dir, "detailed_results_oks_threshold.json"
    )
    oks_fp = os.path.join(current_results_dir, "oks_scores.json")
    pck_fp = os.path.join(current_results_dir, "pck_scores_threshold.json")
    intersect_oks_fp = os.path.join(
        current_results_dir, "intersectional_results_AR_OKS.json"
    )
    intersect_pck_fp = os.path.join(
        current_results_dir, "intersectional_results_PCK.json"
    )
    filelist = [
        model_outputs_fp,
        threshold_fp,
        detailed_threshold_fp,
        oks_fp,
        pck_fp,
        intersect_oks_fp,
        intersect_pck_fp,
    ]
    for f in filelist:
        if os.path.isfile(f):
            os.remove(f)

    evaluate_task(
        data_rootdir=data_rootdir,
        dataset_name=dataset_name,
        model=wrapped_model,
        model_name=model_name,
        task_name=task_name,
        metrics=None,
        attributes=attributes,
        use_mini_dataset=False,
        downsampled=downsampled,
        reuse_model_outputs=False,
        results_rootdir=results_rootdir,
        cuda=False,
    )
    for f in filelist:
        assert os.path.isfile(f)

    # cleanup
    for f in filelist:
        if os.path.isfile(f):
            os.remove(f)


def test_custom_keypoints(prepare_task_fixture):
    custom_keypoints = ["Nose", "Left eye", "Right eye"]

    task_name = "keypoint_estimation"
    data_rootdir = os.path.join(CURRENT_DIR, "static")
    dataset_name = "fhibe"
    downsampled = True
    attributes = [
        "pronoun",
        "age",
        "apparent_skin_color",
        "ancestry",
    ]
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name, custom_keypoints=custom_keypoints)
    )
    results_rootdir = os.path.join(CURRENT_DIR, "static", "results")
    current_results_dir = os.path.join(
        results_rootdir,
        task_name,
        "fhibe_downsampled",
        model_name,
    )
    model_outputs_fp = os.path.join(current_results_dir, "model_outputs.json")
    threshold_fp = os.path.join(current_results_dir, "results_oks_threshold.json")
    detailed_threshold_fp = os.path.join(
        current_results_dir, "detailed_results_oks_threshold.json"
    )
    oks_fp = os.path.join(current_results_dir, "oks_scores.json")
    pck_fp = os.path.join(current_results_dir, "pck_scores_threshold.json")
    intersect_oks_fp = os.path.join(
        current_results_dir, "intersectional_results_AR_OKS.json"
    )
    intersect_pck_fp = os.path.join(
        current_results_dir, "intersectional_results_PCK.json"
    )
    filelist = [
        model_outputs_fp,
        threshold_fp,
        detailed_threshold_fp,
        oks_fp,
        pck_fp,
        intersect_oks_fp,
        intersect_pck_fp,
    ]
    for f in filelist:
        if os.path.isfile(f):
            os.remove(f)

    evaluate_task(
        data_rootdir=data_rootdir,
        dataset_name=dataset_name,
        model=wrapped_model,
        model_name=model_name,
        task_name=task_name,
        metrics=None,
        attributes=attributes,
        use_mini_dataset=False,
        downsampled=downsampled,
        reuse_model_outputs=False,
        results_rootdir=results_rootdir,
        cuda=False,
        custom_keypoints=custom_keypoints,
    )
    for f in filelist:
        assert os.path.isfile(f)
    # Ensure that custom keypoints are saved to the intersectional results files
    intersect_oks = read_json_file(intersect_oks_fp)
    assert "custom_keypoints" in intersect_oks
    assert intersect_oks["custom_keypoints"] == custom_keypoints

    intersect_pck = read_json_file(intersect_pck_fp)
    assert "custom_keypoints" in intersect_pck
    assert intersect_pck["custom_keypoints"] == custom_keypoints
    # cleanup
    for f in filelist:
        if os.path.isfile(f):
            os.remove(f)


def test_custom_keypoints_bad(prepare_task_fixture):
    task_name = "keypoint_estimation"

    # wrong datatype
    custom_keypoints_bad_name = [0, "Left eye", "Right eye"]
    with pytest.raises(ValueError) as excinfo:
        (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
            prepare_task_fixture(task_name, custom_keypoints=custom_keypoints_bad_name)
        )
    error_str = "A keypoint you specified: 0 is not a string."
    assert error_str in str(excinfo.value)

    # bad name
    custom_keypoints_bad_name = ["Nos", "Left eye", "Right eye"]
    with pytest.raises(ValueError) as excinfo:
        (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
            prepare_task_fixture(task_name, custom_keypoints=custom_keypoints_bad_name)
        )
    error_str = "A keypoint you specified: 'Nos' is not a valid FHIBE keypoint."
    assert error_str in str(excinfo.value)

    # wrong order
    custom_keypoints_wrong_order = ["Left eye", "Right eye", "Nose"]
    with pytest.raises(ValueError) as excinfo:
        (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
            prepare_task_fixture(
                task_name, custom_keypoints=custom_keypoints_wrong_order
            )
        )
    error_str = (
        "Custom keypoints are not in the correct order. "
        "Please output your keypoints in this order: "
        "['Nose', 'Left eye', 'Right eye']."
    )
    assert error_str == str(excinfo.value)

    # empty list
    custom_keypoints_empty = []
    with pytest.raises(ValueError) as excinfo:
        (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
            prepare_task_fixture(task_name, custom_keypoints=custom_keypoints_empty)
        )
    error_str = (
        "You cannot provide an empty list of custom keypoints. "
        "Please provide a valid list of keypoints or set custom_keypoints=None."
    )
    assert error_str == str(excinfo.value)

    # wrong task -- no error should be raised
    task_name = "person_localization"
    custom_keypoints = ["Nose", "Left eye", "Right eye"]
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name, custom_keypoints=custom_keypoints)
    )


def test_evaluate_face_parsing(prepare_task_fixture):
    task_name = "face_parsing"
    data_rootdir = os.path.join(CURRENT_DIR, "static")
    dataset_name = "fhibe_face_crop_align"
    attributes = [
        "pronoun",
        "age",
        "apparent_skin_color",
        "apparent_skin_color_hue_lum",
        "ancestry",
    ]
    (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
        prepare_task_fixture(task_name)
    )
    results_rootdir = os.path.join(CURRENT_DIR, "static", "results")
    current_results_dir = os.path.join(
        results_rootdir,
        "mini",
        task_name,
        "fhibe_face_crop_align",
        model_name,
    )
    model_outputs_fp = os.path.join(current_results_dir, "model_outputs.json")
    # Don't remove f1 score file because in order to make it
    # the mask file is needed and that won't be available on github runner
    intersect_fp = os.path.join(current_results_dir, "intersectional_results_F1.json")
    filelist = [model_outputs_fp, intersect_fp]
    for f in filelist:
        if os.path.isfile(f):
            os.remove(f)

    evaluate_task(
        data_rootdir=data_rootdir,
        dataset_name=dataset_name,
        model=wrapped_model,
        model_name=model_name,
        task_name=task_name,
        metrics=None,
        attributes=attributes,
        use_mini_dataset=True,
        mini_dataset_size=50,
        reuse_model_outputs=False,
        results_rootdir=results_rootdir,
        cuda=False,
    )
    for f in filelist:
        if not os.path.isfile(f):
            print(f)
        assert os.path.isfile(f)

    # cleanup
    for f in filelist:
        if os.path.isfile(f):
            os.remove(f)


# def test_evaluate_face_encoding(prepare_task_fixture):
#     import numpy as np
#     np.random.seed(0)
#     task_name = "face_encoding"
#     dataset_name = "fhibe_face_crop_align"
#     attributes = [
#         "pronoun",
#         "age",
#         "apparent_skin_color",
#         "ancestry",
#     ]
#     (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
#         prepare_task_fixture(task_name)
#     )
#     results_rootdir = os.path.join(CURRENT_DIR, "static", "results")
#     current_results_dir = os.path.join(
#         results_rootdir,
#         "mini",
#         task_name,
#         "fhibe_face_crop_align",
#         model_name,
#     )
#     encoding_dir = os.path.join(
#         current_results_dir,
#         "compare",
#         "encodings",
#     )
#     if os.path.exists(encoding_dir):
#         # Remove any existing encodings
#         enc_fps = os.listdir(encoding_dir)
#         for fp in enc_fps:
#             full_fp = os.path.join(encoding_dir, fp)
#             if os.path.isfile(full_fp):
#                 os.remove(full_fp)

#     lpips_fp = os.path.join(current_results_dir, "lpips_scores.json")
#     psnr_fp = os.path.join(current_results_dir, "psnr_scores.json")
#     intersec_lpips_fp = os.path.join(
#         current_results_dir, "intersectional_results_LPIPS.json"
#     )
#     intersec_psnr_fp = os.path.join(
#         current_results_dir, "intersectional_results_PSNR.json"
#     )
#     filelist = [lpips_fp, psnr_fp, intersec_lpips_fp, intersec_psnr_fp]
#     for f in filelist:
#         if os.path.isfile(f):
#             os.remove(f)

#     data_rootdir = os.path.join(CURRENT_DIR, "static")
#     evaluate_task(
#         data_rootdir=data_rootdir,
#         dataset_name=dataset_name,
#         model=wrapped_model,
#         model_name=model_name,
#         task_name=task_name,
#         metrics=["PSNR", "LPIPS"],
#         attributes=attributes,
#         use_mini_dataset=True,
#         mini_dataset_size=50,
#         reuse_model_outputs=False,
#         results_rootdir=results_rootdir,
#         cuda=False,
#     )
#     for f in filelist:
#         print(f)
#         assert os.path.isfile(f)
#     # Face encoding doesn't save a model outputs json
#     # Instead, it saves encodings to disk
#     # ground truth encodings

#     # Check that encodings were created
#     enc_fps = os.listdir(encoding_dir)
#     assert len(enc_fps) == 50

#     for enc_fp in enc_fps:
#         fullpath_gen = os.path.join(encoding_dir, enc_fp)
#         assert os.path.isfile(fullpath_gen)
#         enc_gen = np.array(Image.open(fullpath_gen))
#         assert enc_gen.shape == (32, 32, 3)
#         assert enc_gen.dtype == "uint8"

#     # Cleanup all except encodings
#     for f in filelist:
#         if os.path.isfile(f):
#             os.remove(f)

#     # Run reusing model outputs
#     evaluate_task(
#         data_rootdir=data_rootdir,
#         dataset_name=dataset_name,
#         model=wrapped_model,
#         model_name=model_name,
#         task_name=task_name,
#         metrics=["PSNR", "LPIPS"],
#         attributes=attributes,
#         use_mini_dataset=True,
#         mini_dataset_size=50,
#         reuse_model_outputs=True,
#         results_rootdir=results_rootdir,
#         cuda=False,
#     )
#     for f in filelist:
#         assert os.path.isfile(f)

#     # Cleanup all files
#     for f in filelist:
#         if os.path.isfile(f):
#             os.remove(f)

#     for enc_fp in enc_fps:
#         if os.path.isfile(enc_fp):
#             os.remove(enc_fp)


# def test_evaluate_face_super_resolution(prepare_task_fixture):
#     ## This task produces super resolution images,
#     ## but we cannot upload such images to the cloud in the CI tests
#     ## So until we figure out a secure way to do this we must
#     ## Disable this test.
#     np.random.seed(0)
#     task_name = "face_super_resolution"
#     (annotations_df, img_filepaths, wrapped_model, model_name, kwargs) = (
#         prepare_task_fixture(task_name)
#     )
#     results_dir = os.path.join(
#         CURRENT_DIR,
#         "static",
#         "results",
#         "mini",
#         task_name,
#         "fhibe_face_crop_align",
#         model_name,
#         "compare",
#     )
#     model_outputs_filepath = os.path.join(
#         results_dir,
#         "model_outputs.json",
#     )
#     if os.path.isfile(model_outputs_filepath):
#         os.remove(model_outputs_filepath)
#     results = _evaluate(
#         task_name=task_name,
#         dataset_name="fhibe_face_crop_align",
#         annotations_dataframe=annotations_df,
#         model=wrapped_model,
#         model_name=model_name,
#         img_filepaths=img_filepaths,
#         model_outputs_filepath=model_outputs_filepath,
#         reuse_model_outputs=False,
#         **kwargs,
#     )
#     assert os.path.isfile(model_outputs_filepath)
#     assert len(results) == 50
#     # Instead, it saves encodings to disk
#     super_fps = [x for x in os.listdir(results_dir) if x.startswith("super_")]
#     assert len(super_fps) == 50

#     # Compare contents with pregenerated encodings
#     pregen_super_res_dir = os.path.join(
#         CURRENT_DIR,
#         "static",
#         "results",
#         "mini",
#         task_name,
#         "fhibe_face_crop_align",
#         "face_super_resolution_test_model",
#         "fixed_super_resolution_files",
#     )

#     for super_fp in super_fps:
#         fullpath_gen = os.path.join(results_dir, super_fp)
#         assert os.path.isfile(fullpath_gen)
#         fullpath_pregen = os.path.join(pregen_super_res_dir, super_fp)
#         with open(fullpath_gen, "rb") as file:
#             enc_gen = np.array(Image.open(file))
#         with open(fullpath_pregen, "rb") as file:
#             enc_pregen = np.array(Image.open(file))
#         assert np.array_equal(enc_gen, enc_pregen)
#         # Cleanup
#         os.remove(fullpath_gen)
