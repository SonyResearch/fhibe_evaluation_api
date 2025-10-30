# SPDX-License-Identifier: Apache-2.0

from typing import List

import numpy as np
import torch
import torch.nn.functional as F  # noqa: N812
import torchvision.transforms as transforms
import yaml
from facenet_pytorch import MTCNN
from torch.utils.data import DataLoader
from tqdm import tqdm
from verifier import facenet_model

from fhibe_eval_api.common.data import pil_image_collate_function
from fhibe_eval_api.common.loaders import image_data_loader_from_paths
from fhibe_eval_api.common.utils import create_folders, get_project_root
from fhibe_eval_api.evaluate.evaluate import evaluate_task
from fhibe_eval_api.models.base_model import BaseModelWrapper
from fhibe_eval_api.reporting.reporting import BiasReport

project_root = get_project_root()


class WrappedFaceNetModel(BaseModelWrapper):

    def __init__(self, model):
        """Wraps a FaceNet model for use in the evaluation API."""
        super().__init__(model)

    def data_preprocessor(self, img_filepaths: List[str], **kwargs) -> DataLoader:
        """Preprocesses the input images by aligning faces and creating a data loader.

        Args:
            img_filepaths (List[str]): List of file paths to the input images.
            **kwargs: Additional keyword arguments.

        Returns:
            DataLoader: A data loader for the preprocessed images.
        """
        success_aligned_img_filepaths = kwargs["success_aligned_img_filepaths"]
        transform = transforms.Compose(
            [np.float32, transforms.ToTensor(), self.fixed_image_standardization]
        )
        data_loader = image_data_loader_from_paths(
            image_paths_1=success_aligned_img_filepaths,
            transform=transform,
            batch_size=4,
            num_workers=8,
        )
        return data_loader

    def fixed_image_standardization(self, image_tensor):
        """Applies fixed image standardization to the input image tensor.

        Args:
            image_tensor (torch.Tensor): Input image tensor.

        Returns:
            torch.Tensor: Standardized image tensor.
        """
        processed_tensor = (image_tensor - 127.5) / 128.0
        return processed_tensor

    def align_faces(
        self,
        img_filepaths: List[str],
        aligned_img_filepaths: List[str],
        batch_size: int,
        num_workers: int,
        cuda: bool,
    ) -> List[bool]:
        """Aligns faces in the input images and saves the aligned images.

        Args:
            img_filepaths (List[str]): List of file paths to the input images.
            aligned_img_filepaths (List[str]): List of file paths to save the
                aligned images.
            batch_size (int): Batch size for processing images.
            num_workers (int): Number of workers for data loading.
            cuda (bool): Whether to use CUDA for processing.

        Returns:
            List[bool]: List indicating success of alignment for each image.
        """
        if len(img_filepaths) != len(aligned_img_filepaths):
            raise ValueError("len(img_filepaths) != len(aligned_img_filepaths)")

        detector = MTCNN(
            image_size=160,
            margin=14,
            device="cuda" if cuda else "cpu",
            selection_method="center_weighted_size",
        )

        detector_transform = None
        dataset_loader = image_data_loader_from_paths(
            image_paths_1=img_filepaths,
            transform=detector_transform,
            batch_size=batch_size,
            num_workers=num_workers,
            collate_fn=pil_image_collate_function,
        )

        for aligned_img_filepath in aligned_img_filepaths:
            create_folders(filepath=aligned_img_filepath)

        success = []
        idx = 0
        with tqdm(total=len(img_filepaths), desc="Processing Images") as pbar:
            with torch.no_grad():
                for _, batch in enumerate(dataset_loader):
                    batch_images = batch["images"]
                    batch_filepaths = batch["image_paths"]

                    detected_faces = detector(
                        batch_images,
                        save_path=aligned_img_filepaths[
                            idx : idx + len(batch_filepaths)
                        ],
                        return_prob=False,
                    )

                    for face_i, face in enumerate(detected_faces):
                        if face is None:
                            success.append(False)
                            continue
                        else:
                            success.append(True)

                    idx += len(batch_filepaths)
                    pbar.update(len(batch_filepaths))
        return success

    def __call__(self, batch) -> torch.Tensor:
        """Call the model.

        Generates normalized embeddings for a batch of images.

        Args:
            batch: A batch of data containing images.

        Returns:
            Normalized embeddings for the input images.
        """
        batch_embeddings = self.model(batch["images"].cuda()).cpu()
        normed_batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
        return normed_batch_embeddings


def main():
    """Run the demo."""
    np.random.seed(0)
    with open("./face_verification.yaml", "r") as f:
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

    face_verification_model = facenet_model()
    wrapped_model = WrappedFaceNetModel(face_verification_model)
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
