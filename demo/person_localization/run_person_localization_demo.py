# Copyright (c) Sony AI Inc.
# All rights reserved.
from typing import Any, Dict, List

import numpy as np
import yaml
from torch import nn
from torch.utils.data import DataLoader

from fhibe_eval_api.common.data import pil_image_collate_function
from fhibe_eval_api.common.loaders import image_data_loader_from_paths
from fhibe_eval_api.common.utils import get_project_root
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
        # self.batch_index = 0

    def forward(self, batch: List[str]) -> List[Dict[str, Any]]:
        """Perform a forward pass (inference) of a batch of data.

        In reality, predicts randomized bounding boxes
        to illustrate what a forward pass would look like.

        Args:
            batch: A batch from a data loader

        Return:
            List of dicts, where each dict contains the predicted
            bounding boxes, confidence scores, and class labels
            for each bbox.
        """
        results = []
        for i in range(len(batch)):
            # Randomly predict some bounding boxes for demo purposes
            scale_factor = 2048
            n_boxes = np.random.randint(1, 5)
            bboxes = []  # list of bboxes
            scores = []  # list of confidence scores
            for _ in range(n_boxes):
                x1, y1 = np.random.uniform(0.3, 0.7, size=(2,))
                w, h = np.random.uniform(0.1, 0.3, size=(2,))
                x2 = min(x1 + w, 1.0)
                y2 = min(y1 + h, 1.0)
                bbox = [x1, y1, x2, y2]

                # scale to image size
                bbox = [coord * scale_factor for coord in bbox]
                score = np.random.uniform(0, 1)
                bboxes.append(bbox)
                scores.append(score)

            results.append(
                {
                    "bboxes": bboxes,
                    "bbox_scores": scores,
                    "labels": [0 for _ in range(len(bboxes))],
                }
            )
        return results


class DemoPersonLocalizer(BaseModelWrapper):
    """Model wrapper to comply with API standards."""

    def __init__(self, model: Any) -> None:
        """Initialize the object by referencing the base class.

        Args:
            model: An instance of your custom model class.

        Return:
            None
        """
        super().__init__(model)

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
        data_loader = image_data_loader_from_paths(
            image_paths_1=img_filepaths,
            image_paths_2=None,
            transform=None,
            num_workers=8,
            batch_size=batch_size,
            collate_fn=pil_image_collate_function,
        )
        return data_loader

    def __call__(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Run forward pass over the demo model.

        Args:
            batch: a batch containing images and ground truth bounding boxes

        Return:
            List of dicts, where each dcit contains the predicted
            bounding boxes, confidence scores, and class labels
            for each bbox.
        """
        result_list = []
        _results = self.model(batch["image_paths"])
        for i in range(len(_results)):
            result_list.append(
                {
                    "bboxes": _results[i]["bboxes"],
                    "scores": _results[i]["bbox_scores"],
                    "labels": _results[i]["labels"],
                }
            )
        return result_list


def main() -> None:
    """Run the demo."""
    with open("./person_localization.yaml", "r") as f:
        config = yaml.safe_load(f)

    task_name = config["task_name"]
    dataset_name = config["dataset_name"]
    data_rootdir = config["data_rootdir"]
    model_name = config["model_name"]
    metrics = config["metrics"]
    attributes = config["attributes"]
    downsampled = config["downsampled"]
    use_mini_dataset = config["use_mini_dataset"]
    dataset_version = config["dataset_version"]
    results_basedir = config["results_basedir"]

    face_detection_model = CustomModel()
    wrapped_model = DemoPersonLocalizer(face_detection_model)

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
