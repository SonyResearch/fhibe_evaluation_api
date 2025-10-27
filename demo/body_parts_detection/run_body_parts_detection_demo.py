# Copyright (c) Sony AI Inc.
# All rights reserved.
from typing import Any, Dict, List

import numpy as np
import yaml
from torch import nn
from torch.utils.data import DataLoader

from fhibe_eval_api.common.utils import get_project_root, open_image_with_pil
from fhibe_eval_api.evaluate import evaluate_task
from fhibe_eval_api.models.base_model import BaseModelWrapper
from fhibe_eval_api.reporting import BiasReport

project_root = get_project_root()
batch_size = 4


class CustomModel(nn.Module):
    """A dummy model illustrating an example custom model for this task."""

    def __init__(self):  # noqa: D107
        super().__init__()
        self.batch_size = batch_size

    def forward(self, batch: Any) -> List[List[Dict[str, float]]]:
        """Perform a forward pass (inference) of a batch of data.

        Args:
            batch: A batch from a data loader

        Return:
            List of lists, where each outer list is at the image level
            and each inner list is at the subject level. Each sublist
            contains a dict mapping body part string to its detection
            probability.
        """
        results = []
        for i in range(len(batch["image_paths"])):
            gt_bboxes = batch["gt_bboxes"][i]
            dets = []
            for gt_bbox in gt_bboxes:
                # One could crop to each gt_bbox here
                face_prob = np.random.uniform(0.5, 1)
                hand_prob = np.random.uniform(0.5, 1)
                det = {"Face": face_prob, "Hand": hand_prob}
                dets.append(det)

            results.append(dets)
        return results


class BodyPartsDataset:
    """A reusable dataset object for the body parts detection task."""

    def __init__(self, image_paths, gt_bboxes):  # noqa: D107
        self.image_paths = image_paths
        self.exif_transpose = False
        self.transform = None
        self.grayscale = False
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
            "images": pil_image if pil_image is not None else None,
            "image_paths": image_path,
            "gt_bboxes": gt_bboxes,
        }

    def __len__(self):  # noqa: D105
        return len(self.image_paths)


def body_parts_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Custom collate function for handling a batch from body parts detectiont ask.

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


class DemoBodyPartsDetector(BaseModelWrapper):
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
        image_dataset = BodyPartsDataset(
            image_paths=img_filepaths, gt_bboxes=kwargs["img_filepath_gt_bboxes"]
        )
        data_loader = DataLoader(
            image_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=8,
            collate_fn=body_parts_collate_fn,
        )
        return data_loader

    def __call__(self, batch: Dict[str, Any]) -> List[List[Dict[str, Any]]]:
        """Run forward pass over the demo model.

        Args:
            batch: a batch containing images and ground truth bounding boxes

        Return:
            A list of lists. Each inner list represents a single image and contains
            1 dictionary per ground truth person bounding box. The dictionaries
            map the body part to the predicted probability that body parts exists
            in the image for that person.
        """
        return self.model(batch)


def main() -> None:
    """Run the demo."""
    np.random.seed(42)
    with open("./body_parts_detection.yaml", "r") as f:
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
    body_parts_detection_model = CustomModel()
    wrapped_model = DemoBodyPartsDetector(body_parts_detection_model)

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
