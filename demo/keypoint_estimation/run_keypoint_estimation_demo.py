# Copyright (c) Sony AI Inc.
# All rights reserved.
import os
from typing import Any, Dict, List

import yaml
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms

from fhibe_eval_api.common.data import ImageDataset
from fhibe_eval_api.common.utils import (
    get_project_root,
    open_image_with_pil,
    read_json_file,
)
from fhibe_eval_api.evaluate import evaluate_task
from fhibe_eval_api.models.base_model import BaseModelWrapper
from fhibe_eval_api.reporting import BiasReport

project_root = get_project_root()
batch_size = 4


class CustomModel(nn.Module):
    """A dummy model illustrating an example custom model for this task."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.batch_size = batch_size
        self.fixed_model_outputs_file = os.path.join(
            project_root,
            "tests",
            "static",
            "results",
            "mini",
            "keypoint_estimation",
            "fhibe_downsampled",
            "keypoint_estimator_test_model",
            "ground_truth",
            "fixed_model_outputs.json",
        )
        self.fixed_model_outputs = read_json_file(self.fixed_model_outputs_file)
        self.fixed_filenames = list(self.fixed_model_outputs.keys())
        self.batch_index = 0

    def forward(self, batch: List[str]) -> List[Dict[str, Any]]:
        """Perform a forward pass (inference) of a batch of data.

        Args:
            batch: A batch from a data loader

        Return:
            List of dicts, where each dict contains the keypoints
            and confidence scores for each set of keypoints
            (one per subject per image).
        """
        results = []
        for i in range(len(batch)):
            output_ix = self.batch_index * self.batch_size + i
            key = self.fixed_filenames[output_ix]
            _result = self.fixed_model_outputs[key]
            keypoints = _result["detections"]
            scores = _result["scores"]
            results.append(
                {
                    "keypoints": keypoints,
                    "scores": scores,
                }
            )
        self.batch_index += 1
        return results


class KeypointDataset(ImageDataset):
    """A reusable dataset object for the keypoint estimation task."""

    def __init__(  # noqa: D107
        self, image_paths: List[str], gt_bboxes: List[List[float]]
    ) -> None:
        super().__init__(
            image_paths,
            exif_transpose=False,
            transform=None,
            grayscale=False,
        )
        self.gt_bboxes = gt_bboxes

    def __getitem__(self, image_index: int) -> Dict[str, Any]:
        """Obtain ground truth bbox in addition to image and image path.

        Args:
            image_index: The 0-indexed index

        Return:
            Dict containing the image, image paths, and
            ground truth bounding boxes.
        """
        image_path = self.image_paths[image_index]
        pil_image = open_image_with_pil(
            image_path=image_path,
            exif_transpose=self.exif_transpose,
            grayscale=self.grayscale,
        )
        gt_bboxes = self.gt_bboxes[image_path]  # could be more than one per image
        return {
            "images": self.transform(pil_image) if pil_image is not None else None,
            "image_paths": image_path,
            "gt_bboxes": gt_bboxes,
        }


def keypoint_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Custom collate function for handling a batch for the keypoint task.

    Args:
        batch: List of PIL images, image paths, and ground truth bboxes.

    Return:
        Dict[str, List[Any]]: The collated list of PIL images, image paths,
        and ground truth bounding boxes.
    """
    images = [item["images"] for item in batch]
    image_paths = [item["image_paths"] for item in batch]
    gt_bboxes = [item["gt_bboxes"] for item in batch]

    return {"images": images, "image_paths": image_paths, "gt_bboxes": gt_bboxes}


class DemoKeypointEstimator(BaseModelWrapper):
    """Model wrapper to comply with API standards."""

    def __init__(self, model: Any) -> None:
        """Initialize the object by referencing the base class.

        Args:
            model: An instance of your custom model class.

        Return:
            None
        """
        super().__init__(model)
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                ),
            ]
        )
        self.input_size = [288, 384]
        self.cuda = True

    def data_preprocessor(
        self, img_filepaths: List[str], **kwargs: Dict[str, Any]
    ) -> DataLoader:
        """Perform batch preprocessing and return a data loader.

        Args:
            img_filepaths: List of unique image filepaths
            **kwargs: additional keyword arguments.

        Return:
            Torch dataloader
        """
        image_dataset = KeypointDataset(
            image_paths=img_filepaths, gt_bboxes=kwargs["img_filepath_gt_bboxes"]
        )
        data_loader = DataLoader(
            image_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=keypoint_collate_fn,
        )
        return data_loader

    def __call__(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run forward pass over the demo model.

        Args:
            batch: a batch containing images and ground truth bounding boxes

        Return:
            List of dicts, where each dict contains the keypoints
            and confidence scores for each set of keypoints
            (one per subject per image).
        """
        result_list = []
        _results = self.model(batch["image_paths"])
        for i in range(len(_results)):
            result = {
                "keypoints": _results[i]["keypoints"],
                "scores": _results[i]["scores"],
            }
            result_list.append(result)
        return result_list


def main() -> None:
    """Run the demo."""
    with open("./keypoint_estimation.yaml", "r") as f:
        config = yaml.safe_load(f)

    task_name = config["task_name"]
    dataset_name = config["dataset_name"]
    data_rootdir = config["data_rootdir"]
    model_name = config["model_name"]
    metrics = config["metrics"]
    attributes = config["attributes"]
    downsampled = config["downsampled"]
    use_mini_dataset = config["use_mini_dataset"]
    mini_dataset_size = config["mini_dataset_size"]
    dataset_version = config["dataset_version"]
    results_basedir = config["results_basedir"]

    keypoint_estimator = CustomModel()
    wrapped_model = DemoKeypointEstimator(keypoint_estimator)

    evaluate_task(
        task_name=task_name,
        dataset_name=dataset_name,
        data_rootdir=data_rootdir,
        model=wrapped_model,
        model_name=model_name,
        metrics=metrics,
        attributes=attributes,
        results_rootdir=results_basedir,
        use_mini_dataset=use_mini_dataset,
        mini_dataset_size=mini_dataset_size,
        downsampled=downsampled,
    )
    bias_report = BiasReport(
        model_name=model_name,
        task_name=task_name,
        data_rootdir=data_rootdir,
        dataset_version=dataset_version,
        results_base_dir=results_basedir,
        dataset_name=dataset_name,
        downsampled=downsampled,
        use_mini_dataset=use_mini_dataset,
    )
    bias_report.generate_pdf_report(
        attributes=attributes,
    )


if __name__ == "__main__":
    main()
