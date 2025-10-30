# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Dict, List

import numpy as np
import torchvision.transforms as transforms
import yaml
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader

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

    def forward(self, batch: Any) -> List[Dict[str, Any]]:
        """Perform a forward pass (inference) of a batch of data.

        Args:
            batch: A batch from a data loader

        Return:
            List of dictionaries, where each dictionary contains the encoding
            for each image in the batch.
        """
        results = []

        for i in range(len(batch["image_paths"])):
            rand_array = np.random.uniform(0, 255, (32, 32, 3)).astype("uint8")
            results.append(
                {
                    "encoding": rand_array,
                }
            )
        return results


class DemoFaceEncoder(BaseModelWrapper):
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
        aligned_filepaths = kwargs["aligned_filepaths"]
        success_aligned_img_filepaths = [
            fp for fp in aligned_filepaths if os.path.isfile(fp)
        ]

        trans = transforms.Compose(
            [
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )

        dataset_loader = image_data_loader_from_paths(
            image_paths_1=success_aligned_img_filepaths,
            transform=trans,
            batch_size=batch_size,
            num_workers=8,
        )

        return dataset_loader

    def save_encoding(self, encoding: Any, filepath: str) -> None:
        """Save the encoded image to disk as a png file.

        Args:
            encoding: The encoded face image
            filepath: The filepath where to save the image

        Return:
            None
        """
        im = Image.fromarray(encoding)
        im.save(filepath)
        return

    def __call__(self, batch: Any) -> List[Dict[str, Any]]:
        """Run forward pass over the demo model.

        Args:
            batch: a batch containing images

        Return:
            List of dictionaries, where each dictionary contains the encoding
            for each image in the batch.
        """
        return self.model(batch)


def main() -> None:
    """Run the demo."""
    np.random.seed(0)
    with open("./face_encoding.yaml", "r") as f:
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

    face_encoding_model = CustomModel()
    wrapped_model = DemoFaceEncoder(face_encoding_model)
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
