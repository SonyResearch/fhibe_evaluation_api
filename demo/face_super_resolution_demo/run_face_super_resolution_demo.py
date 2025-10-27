# Copyright (c) Sony AI Inc.
# All rights reserved.

from typing import Any, Dict, List

import cv2
import numpy as np
import torchvision.transforms as transforms
import yaml
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

    def forward(self, batch: Any) -> List[np.ndarray]:
        """Perform a forward pass (inference) of a batch of data.

        Args:
            batch: A batch from a data loader

        Return:
            List of predicted super resolution arrays
            corresponding to each image in the batch.
        """
        results = []
        for i, image in enumerate((batch["images"])):
            # Add some noise to the original image
            numpy_image = image.cpu().detach()
            numpy_image += np.random.uniform(-0.25, 0.25, size=numpy_image.shape)
            numpy_image = numpy_image.clamp(0, 1)
            numpy_image = (numpy_image.permute(1, 2, 0) * 255).byte().numpy()

            numpy_image = cv2.cvtColor(numpy_image, cv2.COLOR_RGB2BGR)

            numpy_image = numpy_image.astype("uint8")
            results.append(numpy_image)
        return results


class DemoSuperResolver(BaseModelWrapper):
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
        transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
            ]
        )

        dataset_loader = image_data_loader_from_paths(
            image_paths_1=img_filepaths,
            transform=transform,
            batch_size=batch_size,
            num_workers=8,
        )

        return dataset_loader

    def save_array(self, array: Any, filepath: str) -> None:
        """Save the super resolution image to disk as a png file.

        Args:
            array: The super resolution face image
            filepath: The filepath where to save the image

        Return:
            None
        """
        cv2.imwrite(filepath, array)
        return None

    def __call__(self, batch: Any) -> List[np.ndarray]:
        """Run forward pass over the demo model.

        Args:
            batch: a batch containing images and ground truth bounding boxes

        Return:
            List of predicted super resolution arrays
            corresponding to each image in the batch.
        """
        result_list = []
        _results = self.model(batch)
        for i in range(len(_results)):
            numpy_image = _results[i]
            result_list.append(numpy_image)

        return result_list


def main() -> None:
    """Run the demo."""
    np.random.seed(0)
    with open("./face_super_resolution.yaml", "r") as f:
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

    face_sr_model = CustomModel()
    wrapped_model = DemoSuperResolver(face_sr_model)
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
