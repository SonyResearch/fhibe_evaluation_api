# SPDX-License-Identifier: Apache-2.0
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

    def forward(self, batch: List[str]) -> List[Dict[str, Any]]:
        """Perform a forward pass (inference) of a batch of data.

        Args:
            batch: A batch from a data loader

        Return:
            List of dicts, where each dict contains the predicted
            person binary masks, confidence scores, and class labels for each
            person mask. Use 0 for the class label for 'person'.
        """
        results = []
        for img in batch["images"]:
            # Predict one or two people per image
            n_people = np.random.randint(1, 3)
            masks = []
            scores = []
            labels = []
            for _ in range(n_people):
                # Create a random binary mask
                mask = np.zeros((img.height, img.width), dtype=np.uint8)
                x1, y1 = np.random.randint(0, img.width // 2), np.random.randint(
                    0, img.height // 2
                )
                x2, y2 = np.random.randint(
                    img.width // 2, img.width
                ), np.random.randint(img.height // 2, img.height)
                mask[y1:y2, x1:x2] = 1
                masks.append(mask)
                scores.append(np.random.rand())
                labels.append(0)  # '0' is the class label for 'person'

            results.append(
                {
                    "masks": masks,
                    "scores": scores,
                    "labels": labels,
                }
            )
        return results


class DemoPersonParser(BaseModelWrapper):
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
            List of dicts, where each dict contains the predicted
            person masks, confidence scores, and class labels for each
            person mask.
        """
        return self.model(batch)


def main() -> None:
    """Run the demo."""
    with open("./person_parsing.yaml", "r") as f:
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

    person_parsing_model = CustomModel()
    wrapped_model = DemoPersonParser(person_parsing_model)

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
