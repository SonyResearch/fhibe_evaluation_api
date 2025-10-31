# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Dict, List

import numpy as np
from PIL import Image
from pycocotools.mask import decode

from fhibe_eval_api.common.utils import read_json_file
from fhibe_eval_api.evaluate.utils import _decode_mask
from fhibe_eval_api.models.base_model import BaseModelWrapper

CURRENT_DIR = os.path.dirname(__file__)


class TestDataLoader:
    def __init__(self, img_filepaths, batch_size):
        self.img_filepaths = img_filepaths
        total_elements = len(img_filepaths)
        self.total_elements = total_elements
        self.batch_size = batch_size

    def __iter__(self):
        """Iterate once in the data loader."""
        for i in range(0, self.total_elements, self.batch_size):
            batch = {
                "image_paths": [
                    self.img_filepaths[j]
                    for j in range(i, min(i + self.batch_size, self.total_elements))
                ],
                "images": list(range(i, min(i + self.batch_size, self.total_elements))),
            }
            yield batch


class TestPersonLocalizer(BaseModelWrapper):
    def __init__(self, model):
        super().__init__(model)
        self.batch_size = 4
        self.fixed_model_outputs_file = os.path.join(
            CURRENT_DIR,
            "static",
            "results",
            "mini",
            "person_localization",
            "fhibe_downsampled",
            "person_localizer_test_model",
            "ground_truth",
            "fixed_model_outputs.json",
        )
        self.fixed_model_outputs = read_json_file(self.fixed_model_outputs_file)
        self.fixed_filenames = list(self.fixed_model_outputs.keys())
        self.batch_index = 0

    def data_preprocessor(self, img_filepaths, **kwargs) -> TestDataLoader:
        data_loader = TestDataLoader(img_filepaths, batch_size=self.batch_size)
        return data_loader

    def __call__(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        result_list = []

        for i in range(len(batch["image_paths"])):
            output_ix = self.batch_index * self.batch_size + i
            key = self.fixed_filenames[output_ix]
            _result = self.fixed_model_outputs[key]
            bboxes = _result["detections"]
            bbox_scores = _result["scores"]
            if bboxes is None:
                labels = []
            else:
                labels = [0 for _ in bboxes]
            # bboxes = np.random.uniform(10,500,size=(self.batch_size,4)).tolist()
            # bbox_scores = np.random.uniform(0,1,size=(self.batch_size)).tolist()
            # labels = np.random.randint(0,2,size=(self.batch_size)).tolist()

            result_list.append(
                {"bboxes": bboxes, "scores": bbox_scores, "labels": labels}
            )
        self.batch_index += 1
        return result_list


class TestPersonParser(BaseModelWrapper):
    def __init__(self, model):
        super().__init__(model)
        self.batch_size = 4
        self.fixed_model_outputs_file = os.path.join(
            CURRENT_DIR,
            "static",
            "results",
            "mini",
            "person_parsing",
            "fhibe_downsampled",
            "person_parser_test_model",
            "ground_truth",
            "fixed_model_outputs.json",
        )
        self.fixed_model_outputs = read_json_file(self.fixed_model_outputs_file)
        self.fixed_filenames = list(self.fixed_model_outputs.keys())
        self.batch_index = 0

    def data_preprocessor(self, img_filepaths, **kwargs) -> TestDataLoader:
        data_loader = TestDataLoader(img_filepaths, batch_size=self.batch_size)
        return data_loader

    def __call__(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        result_list = []

        for i in range(len(batch["image_paths"])):
            output_ix = self.batch_index * self.batch_size + i
            key = self.fixed_filenames[output_ix]
            _result = self.fixed_model_outputs[key]
            masks = _result["detections"]
            decoded_masks = [decode(pm) for pm in masks]
            bbox_scores = _result["scores"]
            if decoded_masks == []:
                labels = []
            else:
                labels = [0 for _ in decoded_masks]

            result_list.append(
                {"masks": decoded_masks, "scores": bbox_scores, "labels": labels}
            )
        self.batch_index += 1
        return result_list


class TestFaceLocalizer(BaseModelWrapper):
    def __init__(self, model):
        super().__init__(model)
        self.batch_size = 4
        self.fixed_model_outputs_file = os.path.join(
            CURRENT_DIR,
            "static",
            "results",
            "mini",
            "face_localization",
            "fhibe_downsampled",
            "face_localizer_test_model",
            "ground_truth",
            "fixed_model_outputs.json",
        )
        self.fixed_model_outputs = read_json_file(self.fixed_model_outputs_file)
        self.fixed_filenames = list(self.fixed_model_outputs.keys())
        self.batch_index = 0

    def data_preprocessor(self, img_filepaths, **kwargs) -> TestDataLoader:
        data_loader = TestDataLoader(img_filepaths, batch_size=self.batch_size)
        return data_loader

    def __call__(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        result_list = []

        for i in range(len(batch["image_paths"])):
            output_ix = self.batch_index * self.batch_size + i
            key = self.fixed_filenames[output_ix]
            _result = self.fixed_model_outputs[key]
            bboxes = _result["detections"]
            bbox_scores = _result["scores"]

            result_list.append(
                {
                    "detections": bboxes,
                    "scores": bbox_scores,
                }
            )
        self.batch_index += 1
        return result_list


class TestBodyPartsDetector(BaseModelWrapper):
    def __init__(self, model):
        super().__init__(model)
        self.batch_size = 4
        self.fixed_model_outputs_file = os.path.join(
            CURRENT_DIR,
            "static",
            "results",
            "mini",
            "body_parts_detection",
            "fhibe_downsampled",
            "body_parts_detector_test_model",
            "ground_truth",
            "fixed_model_outputs.json",
        )
        self.fixed_model_outputs = read_json_file(self.fixed_model_outputs_file)
        self.fixed_filenames = list(self.fixed_model_outputs.keys())
        self.batch_index = 0

    def data_preprocessor(self, img_filepaths, **kwargs) -> TestDataLoader:
        data_loader = TestDataLoader(img_filepaths, batch_size=self.batch_size)
        return data_loader

    def __call__(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        result_list = []

        for i in range(len(batch["image_paths"])):
            output_ix = self.batch_index * self.batch_size + i
            key = self.fixed_filenames[output_ix]
            _result = self.fixed_model_outputs[key]
            dets = _result["detections"]

            result_list.append(dets)
        self.batch_index += 1
        return result_list


class TestKeypointEstimator(BaseModelWrapper):
    def __init__(self, model, custom_keypoints=None):
        super().__init__(model)
        self.batch_size = 4
        self.fixed_model_outputs_file = os.path.join(
            CURRENT_DIR,
            "static",
            "results",
            "keypoint_estimation",
            "fhibe_downsampled",
            "keypoint_estimator_test_model",
            "fixed_model_outputs.json",
        )
        self.fixed_model_outputs = read_json_file(self.fixed_model_outputs_file)
        self.fixed_filenames = list(self.fixed_model_outputs.keys())
        self.batch_index = 0
        self.custom_keypoints = custom_keypoints
        if self.custom_keypoints is not None:
            from fhibe_eval_api.datasets.fhibe import FHIBE_COMMON_KEYPOINTS

            self.keypoints_ixs = []
            for keypoint in custom_keypoints:
                if keypoint in FHIBE_COMMON_KEYPOINTS:
                    self.keypoints_ixs.append(FHIBE_COMMON_KEYPOINTS.index(keypoint))

    def data_preprocessor(self, img_filepaths, **kwargs) -> TestDataLoader:
        data_loader = TestDataLoader(img_filepaths, batch_size=self.batch_size)
        return data_loader

    def __call__(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        result_list = []

        for i in range(len(batch["image_paths"])):
            output_ix = self.batch_index * self.batch_size + i
            key = self.fixed_filenames[output_ix]
            _result = self.fixed_model_outputs[key]
            keypoints = _result["detections"][0]
            scores = _result["scores"][0]
            if self.custom_keypoints is not None:
                keypoints = [keypoints[ix] for ix in self.keypoints_ixs]
                scores = [scores[ix] for ix in self.keypoints_ixs]
            result_list.append(
                {
                    "keypoints": [keypoints],
                    "scores": [scores],
                }
            )
        self.batch_index += 1
        return result_list


class TestFaceParser(BaseModelWrapper):
    def __init__(self, model, map_ears_to_skin=True):
        super().__init__(model)
        self.batch_size = 4
        self.map_ears_to_skin = map_ears_to_skin
        self.fixed_model_outputs_file = os.path.join(
            CURRENT_DIR,
            "static",
            "results",
            "mini",
            "face_parsing",
            "fhibe_face_crop_align",
            "face_parser_test_model",
            "ground_truth",
            "fixed_model_outputs.json",
        )
        self.fixed_model_outputs = read_json_file(self.fixed_model_outputs_file)
        self.fixed_filenames = list(self.fixed_model_outputs.keys())
        self.batch_index = 0

    def data_preprocessor(self, img_filepaths, **kwargs) -> TestDataLoader:
        data_loader = TestDataLoader(img_filepaths, batch_size=self.batch_size)
        return data_loader

    def __call__(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        result_list = []

        for i in range(len(batch["image_paths"])):
            output_ix = self.batch_index * self.batch_size + i
            key = self.fixed_filenames[output_ix]
            _result = self.fixed_model_outputs[key]
            detections = np.array(_decode_mask(_result["detections_rle"]))

            result_list.append(
                {
                    "detections": detections,
                }
            )
        self.batch_index += 1
        return result_list


class TestFaceEncoder(BaseModelWrapper):
    def __init__(self, model):
        super().__init__(model)
        self.batch_size = 4
        self.batch_index = 0

    def data_preprocessor(self, img_filepaths, **kwargs) -> TestDataLoader:
        data_loader = TestDataLoader(img_filepaths, batch_size=self.batch_size)
        return data_loader

    def __call__(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        result_list = []

        for i in range(len(batch["image_paths"])):
            rand_array = np.random.uniform(0, 255, (32, 32, 3)).astype("uint8")
            result_list.append(
                {
                    "encoding": rand_array,
                }
            )
        self.batch_index += 1
        return result_list

    def save_encoding(self, encoding: Any, filepath: str):
        """Save the encoded image to disk as a png file.

        Args:
            encoding: The encoded face image
            filepath: The filepath where to save the image
        """
        img_name = filepath.split("/")[-1]
        encodings_dir = os.path.join(
            CURRENT_DIR,
            "static",
            "results",
            "mini",
            "face_encoding",
            "fhibe_face_crop_align",
            "face_encoder_test_model",
            "compare",
            "encodings",
        )
        os.makedirs(encodings_dir, exist_ok=True)
        savename = os.path.join(
            encodings_dir,
            img_name,
        )
        im = Image.fromarray(encoding)
        im.save(savename)
        return


class TestFaceSuperResolver(BaseModelWrapper):
    def __init__(self, model):
        super().__init__(model)
        self.batch_size = 4
        self.batch_index = 0
        self.fixed_results_dir = os.path.join(
            CURRENT_DIR,
            "static",
            "results",
            "mini",
            "face_super_resolution",
            "fhibe_face_crop_align",
            "face_super_resolution_test_model",
            "ground_truth",
        )
        self.fixed_model_outputs_file = os.path.join(
            self.fixed_results_dir,
            "fixed_model_outputs.json",
        )
        self.fixed_model_outputs = read_json_file(self.fixed_model_outputs_file)
        self.fixed_filenames = list(self.fixed_model_outputs.keys())

    def data_preprocessor(self, img_filepaths, **kwargs) -> TestDataLoader:
        data_loader = TestDataLoader(img_filepaths, batch_size=self.batch_size)
        return data_loader

    def __call__(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        result_list = []

        for i in range(len(batch["image_paths"])):
            output_ix = self.batch_index * self.batch_size + i
            key = self.fixed_filenames[output_ix]
            fp = self.fixed_model_outputs[key]["super_res_filename"]
            basename = fp.split("/")[-1]
            super_res_filename = os.path.join(
                self.fixed_results_dir, "fixed_super_resolution_files", basename
            )

            array = np.array(Image.open(super_res_filename))
            result_list.append(array)
        self.batch_index += 1
        return result_list

    def save_array(self, array: Any, filepath: str) -> None:
        """Save the super resolution image to disk as a png file.

        Args:
            array: The super resolution face image
            filepath: The filepath where to save the image

        Return:
            None
        """
        img_name = filepath.split("/")[-1]
        savename = os.path.join(
            CURRENT_DIR,
            "static",
            "results",
            "mini",
            "face_super_resolution",
            "fhibe_face_crop_align",
            "face_super_resolution_test_model",
            "compare",
            img_name,
        )
        im = Image.fromarray(array)
        im.save(savename)
        return


class TestFaceVerifier(BaseModelWrapper):
    def __init__(self, model, map_ears_to_skin=True):
        super().__init__(model)
        self.batch_size = 4
        self.map_ears_to_skin = map_ears_to_skin
        self.fixed_model_outputs_file = os.path.join(
            CURRENT_DIR,
            "static",
            "results",
            "mini",
            "face_parsing",
            "fhibe_face_crop_align",
            "face_verifier_test_model",
            "ground_truth",
            "fixed_model_outputs.json",
        )
        # self.fixed_model_outputs = read_json_file(self.fixed_model_outputs_file)
        # self.fixed_filenames = list(self.fixed_model_outputs.keys())
        self.batch_index = 0

    def data_preprocessor(self, img_filepaths, **kwargs) -> TestDataLoader:
        data_loader = TestDataLoader(img_filepaths, batch_size=self.batch_size)
        return data_loader

    def __call__(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        # batch_embeddings = self.model(batch["images"].cuda()).cpu()
        # normed_batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
        normed_batch_embeddings = np.array(
            [np.random.uniform(0, 1, 512) for _ in range(len(batch["image_paths"]))]
        )
        # for i in range(len(batch["image_paths"])):
        #     output_ix = self.batch_index * self.batch_size + i
        #     key = self.fixed_filenames[output_ix]
        #     _result = self.fixed_model_outputs[key]
        return normed_batch_embeddings
