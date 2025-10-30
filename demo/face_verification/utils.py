# SPDX-License-Identifier: Apache-2.0

from typing import List

import torch
from facenet_pytorch import MTCNN
from tqdm import tqdm

from fhibe_eval_api.common.data import pil_image_collate_function
from fhibe_eval_api.common.loaders import image_data_loader_from_paths
from fhibe_eval_api.common.utils import create_folders


def align_faces(
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
                    save_path=aligned_img_filepaths[idx : idx + len(batch_filepaths)],
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
