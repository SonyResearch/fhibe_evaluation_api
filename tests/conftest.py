# SPDX-License-Identifier: Apache-2.0

import os
import pickle

import pytest

from fhibe_eval_api.common.utils import read_json_file, save_json_file
from fhibe_eval_api.evaluate.constants import TASK_DICT
from fhibe_eval_api.evaluate.utils import get_eval_api, prepare_evaluation
from fhibe_eval_api.reporting.reporting import BiasReport

from .models import (
    TestBodyPartsDetector,
    TestFaceEncoder,
    TestFaceLocalizer,
    TestFaceParser,
    TestFaceSuperResolver,
    TestFaceVerifier,
    TestKeypointEstimator,
    TestPersonLocalizer,
    TestPersonParser,
)

TASK_MODEL_NAME_DICT = {
    "person_localization": "person_localizer_test_model",
    "person_parsing": "person_parser_test_model",
    "keypoint_estimation": "keypoint_estimator_test_model",
    "face_local˚ization": "face_localizer_test_model",
    "body_parts_detection": "body_parts_detector_test_model",
    "face_super_resolution": "face_super_resolution_test_model",
    "face_encoding": "face_encoder_test_model",
    "face_parsing": "face_parser_test_model",
    "face_verification": "face_verifier_test_model",
}

CURRENT_DIR = os.path.dirname(__file__)


def get_dataset_name_and_base(task_name):
    if task_name in TASK_DICT["fhibe"]:
        dataset_name = "fhibe_downsampled"
        dataset_base = "fhibe"
    elif task_name in TASK_DICT["fhibe_face"]:
        dataset_name = "fhibe_face_crop_align"
        dataset_base = "fhibe_face"
    else:
        raise ValueError(f"Task: {task_name} not recognized")
    return dataset_name, dataset_base


@pytest.fixture
def demo_model_fixture():
    def build_model(task_name, **task_kwargs):
        model = None
        if task_name == "person_localization":
            model_name = "person_localizer_test_model"
            wrapped_model = TestPersonLocalizer(model)
        elif task_name == "keypoint_estimation":
            model_name = "keypoint_estimator_test_model"
            custom_keypoints = task_kwargs.get("custom_keypoints")
            wrapped_model = TestKeypointEstimator(
                model, custom_keypoints=custom_keypoints
            )
        elif task_name == "face_localization":
            model_name = "face_localizer_test_model"
            wrapped_model = TestFaceLocalizer(model)
        elif task_name == "body_parts_detection":
            model_name = "body_parts_detector_test_model"
            wrapped_model = TestBodyPartsDetector(model)
        elif task_name == "person_parsing":
            model_name = "person_parser_test_model"
            wrapped_model = TestPersonParser(model)
        elif task_name == "face_parsing":
            model_name = "face_parser_test_model"
            wrapped_model = TestFaceParser(model, map_ears_to_skin=True)
        elif task_name == "face_encoding":
            model_name = "face_encoder_test_model"
            wrapped_model = TestFaceEncoder(model)
        elif task_name == "face_super_resolution":
            model_name = "face_super_resolution_test_model"
            wrapped_model = TestFaceSuperResolver(model)
        elif task_name == "face_verification":
            model_name = "face_verifier_test_model"
            wrapped_model = TestFaceVerifier(model)
        return wrapped_model, model_name

    return build_model


@pytest.fixture()
def update_fixed_model_outputs(demo_model_fixture):
    """Update paths in fixed_model_outputs.json files.

    It simulates the process of having made the file on the machine
    where the test is being run. Helps fix a hardcode issue with the
    CI tests.
    """

    def update_paths(task_name):
        if task_name == "face_encoding":
            # Face encoding does not save model results to disk in a JSON file
            return
        _, model_name = demo_model_fixture(task_name)
        dataset_name, dataset_base = get_dataset_name_and_base(task_name)
        data_dir = os.path.join(CURRENT_DIR, "static", "data")
        fixed_model_outputs_fp = os.path.join(
            CURRENT_DIR,
            "static",
            "results",
            "mini",
            task_name,
            dataset_name,
            model_name,
            "ground_truth",
            "fixed_model_outputs.json",
        )
        fixed_model_outputs = read_json_file(fixed_model_outputs_fp)
        new_model_outputs = {}
        for key in fixed_model_outputs:
            newkey = key.replace(
                "/home/austin.hoag/fhibe_evaluation_api/tests/static/data", data_dir
            )
            if task_name == "face_super_resolution":
                old_super_res_fp = fixed_model_outputs[key]
                new_super_res_fp = old_super_res_fp["super_res_filename"].replace(
                    "/home/austin.hoag/fhibe_evaluation_api/tests",
                    CURRENT_DIR,
                )
                new_model_outputs[newkey] = {"super_res_filename": new_super_res_fp}
            else:
                new_model_outputs[newkey] = fixed_model_outputs[key]
        # Overwrite orig file
        save_json_file(fixed_model_outputs_fp, new_model_outputs, indent=4)
        return

    return update_paths


@pytest.fixture
def eval_api_fixture():
    def prepare_eval_api(task_name, use_mini_dataset=True, mini_dataset_size=50):
        dataset_name, dataset_base = get_dataset_name_and_base(task_name)

        data_dir = os.path.join(CURRENT_DIR, "static")
        processed_data_dir = os.path.join(data_dir, "data", "processed")
        print("CURRENT_DIR: ", CURRENT_DIR)
        print("data_dir: ", data_dir)
        eval_api = get_eval_api(
            dataset_name,
            dataset_base,
            data_dir,
            processed_data_dir,
            intersectional_column_names=None,
            use_age_buckets=True,
            use_mini_dataset=use_mini_dataset,
            mini_dataset_size=mini_dataset_size,
        )
        return eval_api

    return prepare_eval_api


@pytest.fixture
def prepare_task_fixture(
    demo_model_fixture, eval_api_fixture, update_fixed_model_outputs
):
    def _prepare_task(
        task_name, use_mini_dataset=False, mini_dataset_size=50, **task_kwargs
    ):
        update_fixed_model_outputs(task_name)
        eval_api = eval_api_fixture(
            task_name,
            use_mini_dataset=use_mini_dataset,
            mini_dataset_size=mini_dataset_size,
        )
        wrapped_model, model_name = demo_model_fixture(task_name, **task_kwargs)
        dataset_name, dataset_base = get_dataset_name_and_base(task_name)
        current_results_dir = os.path.join(
            CURRENT_DIR,
            "static",
            "results",
            "mini",
            task_name,
            dataset_name,
            model_name,
        )
        if task_name == "person_parsing":
            # Don't use mini because we need to maintain the order
            # without shuffling so that the precomputed masks will
            # be in the same order as the fixed model outputs
            current_results_dir = os.path.join(
                CURRENT_DIR,
                "static",
                "results",
                task_name,
                dataset_name,
                model_name,
            )
            precomputed_mask_file = os.path.join(
                current_results_dir,
                "ground_truth",
                "precomputed_person_masks.pkl",
            )
            with open(precomputed_mask_file, "rb") as file:
                precomputed_masks = pickle.load(file)
        elif task_name == "keypoint_estimation":
            precomputed_mask_file = os.path.join(
                current_results_dir,
                "ground_truth",
                "precomputed_person_segment_areas.pkl",
            )
            with open(precomputed_mask_file, "rb") as file:
                precomputed_masks = pickle.load(file)
        else:
            precomputed_masks = None
        annotations_dataframe, img_filepaths, kwargs = prepare_evaluation(
            eval_api=eval_api,
            task_name=task_name,
            dataset_name=dataset_name,
            model=wrapped_model,
            model_name=model_name,
            current_results_dir=current_results_dir,
            precomputed_masks=precomputed_masks,
            cuda=False,
            **task_kwargs,
        )
        return annotations_dataframe, img_filepaths, wrapped_model, model_name, kwargs

    return _prepare_task


@pytest.fixture
def bias_report_fixture():
    def create_bias_report(task_name, results_dir=None):
        dataset_name, dataset_base = get_dataset_name_and_base(task_name)
        downsampled = True
        model_name = TASK_MODEL_NAME_DICT[task_name]
        data_rootdir = os.path.join(CURRENT_DIR, "static", "data")
        results_base_dir = os.path.join(CURRENT_DIR, "static", "results")
        use_mini_dataset = True
        dataset_version = "testing"
        bias_report = BiasReport(
            model_name=model_name,
            task_name=task_name,
            dataset_version=dataset_version,
            data_rootdir=data_rootdir,
            results_base_dir=results_base_dir,
            dataset_name=dataset_name,
            downsampled=downsampled,
            use_mini_dataset=use_mini_dataset,
            results_dir=results_dir,
        )
        return bias_report

    yield create_bias_report
