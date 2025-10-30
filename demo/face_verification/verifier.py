# SPDX-License-Identifier: Apache-2.0

import os
from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchvision.transforms as transforms
from facenet_pytorch import InceptionResnetV1
from tqdm import tqdm
from utils import align_faces

from fhibe_eval_api.common.loaders import image_data_loader_from_paths
from fhibe_eval_api.common.utils import save_json_file


def facenet_model(cuda: bool = True) -> nn.Module:
    """Load the Inception ResNet V1 model.

    Args:
        cuda (bool): If True model is moved to cuda device.

    Returns:
        nn.Module: A PyTorch model instance of the FaceNet model trained on VGGFace2
        with an Inception ResNet V1 backbone.
    """
    model = InceptionResnetV1(pretrained="vggface2")
    if cuda:
        return model.cuda().eval()
    else:
        return model.eval()


def run_facenet(
    save_json_filepath: str,
    img_filepaths: List[str],
    aligned_img_filepaths: List[str],
    aligned: bool,
    batch_size: int = 128,
    num_workers: int = 8,
    cuda: bool = True,
) -> Dict[str, Any]:
    """This function runs facenet over a list of images.

    Args:
        save_json_filepath (str): Save path for storing the results of the dlib
            detector.
        img_filepaths (List[str]): List of image filepaths.
        aligned_img_filepaths (List[str]): List of aligned image filepaths.
        aligned (bool): Flag to indicate if the images are already aligned
        batch_size (int): Batch size.
        num_workers (int): Number of workers for data loading.
        cuda (bool): If True model is moved to cuda device.

    Returns:
        Dict[Any, Any]: Returns a dictionary of bbox detections and bbox confidence
            scores.
    """

    def fixed_image_standardization(image_tensor):
        processed_tensor = (image_tensor - 127.5) / 128.0
        return processed_tensor

    save_path, json_basename = os.path.split(save_json_filepath)
    if not json_basename.endswith(".json"):
        raise ValueError("`save_json_filepath` must have a .json extension.")
    os.makedirs(save_path, exist_ok=True)

    model = facenet_model(cuda=cuda)

    if aligned:
        success = [True for _ in aligned_img_filepaths]
    else:
        success = align_faces(
            img_filepaths=img_filepaths,
            aligned_img_filepaths=aligned_img_filepaths,
            batch_size=1,
            num_workers=num_workers,
            cuda=cuda,
        )

    success_aligned_img_filepaths = [
        filepath for exists, filepath in zip(success, aligned_img_filepaths) if exists
    ]

    transform = transforms.Compose(
        [np.float32, transforms.ToTensor(), fixed_image_standardization]
    )

    dataset_loader = image_data_loader_from_paths(
        image_paths_1=success_aligned_img_filepaths,
        transform=transform,
        batch_size=batch_size,
        num_workers=num_workers,
    )

    results_dict: Dict[str, Any] = {}
    embeddings = []
    with tqdm(
        total=len(success_aligned_img_filepaths), desc="Processing Images"
    ) as pbar:
        with torch.no_grad():
            for _, batch in enumerate(dataset_loader):
                batch_embeddings = model(batch["images"].cuda()).cpu()
                normed_batch_embeddings = F.normalize(batch_embeddings, p=2, dim=1)
                embeddings.append(normed_batch_embeddings)

                pbar.update(batch_embeddings.shape[0])

    embeddings = torch.cat(embeddings, dim=0).numpy()

    embedding_idx = 0
    for i, (img_filepath, aligned_img_filepath, aligned_exists) in enumerate(
        zip(img_filepaths, aligned_img_filepaths, success)
    ):
        if aligned_exists:
            results_dict[aligned_img_filepath] = {
                "detections": embeddings[embedding_idx].tolist(),
                "img_filepath": img_filepath,
            }
            embedding_idx += 1
        else:
            results_dict[aligned_img_filepath] = {
                "detections": None,
                "img_filepath": img_filepath,
            }

    save_json_file(filepath=save_json_filepath, data=results_dict)
    return results_dict
