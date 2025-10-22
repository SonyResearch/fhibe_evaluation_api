# Copyright (c) Sony AI Inc.
# All rights reserved.
"""Module implementing the curricular face network."""

import os
from typing import Any, Dict, List

import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
import torchvision.transforms as transforms
from tqdm import tqdm

from fhibe_eval_api.common.loaders import image_data_loader_from_paths
from fhibe_eval_api.common.utils import save_json_file
from fhibe_eval_api.metrics.face_verification.curricular_face.backbone import IR101
from fhibe_eval_api.metrics.face_verification.utils import align_faces

CURRENT_DIR = os.path.dirname(__file__)


def curricular_face_model(cmd_auth: bool = True, cuda: bool = True) -> nn.Module:
    """Load the CurricularFace backbone model.

    Args:
        cmd_auth (bool, optional): True if command-line authentication is desired,
            False otherwise.
        cuda (bool): If True model is moved to cuda device.

    Returns:
        nn.Module: A PyTorch model instance of the CurricularFace model with an
        IR-101 backbone.

    Raises:
        Exception: If there is an error loading the model weights.
    """
    model = IR101()

    model_weights_file_path = os.path.join(
        CURRENT_DIR,
        "CurricularFace_Backbone.pth",
    )

    if os.path.exists(model_weights_file_path):
        try:
            model.load_state_dict(
                torch.load(model_weights_file_path, map_location=torch.device("cuda")),
                strict=True,
            )
        except Exception as e:
            raise type(e)(f"Error loading model: {e}") from e
    else:
        raise FileNotFoundError(f"File not found: {model_weights_file_path}.")
    if cuda:
        return model.cuda().eval()
    else:
        return model.eval()


def run_curricular_face(
    save_json_filepath: str,
    img_filepaths: List[str],
    aligned_img_filepaths: List[str],
    aligned: bool,
    batch_size: int = 128,
    num_workers: int = 8,
    cuda: bool = True,
) -> Dict[str, Any]:
    """This function runs curricular_face over a list of images.

    Args:
        save_json_filepath (str): Save path for storing the results of the dlib
            detector.
        img_filepaths (List[str]): List of image filepaths.
        aligned_img_filepaths (List[str]): List of aligned image filepaths.
        aligned (bool): Flag to indicate if the images are already aligned
        batch_size (int): Batch size.
        num_workers: The number of parallel cpus to use for data processing.
        cuda (bool): If True model is moved to cuda device.

    Returns:
        Dict[Any, Any]: Returns a dictionary of bbox detections and bbox confidence
            scores.
    """
    save_path, json_basename = os.path.split(save_json_filepath)
    if not json_basename.endswith(".json"):
        raise ValueError("`save_json_filepath` must have a .json extension.")
    os.makedirs(save_path, exist_ok=True)

    model = curricular_face_model(cuda=cuda)

    if aligned:
        success = [True for _ in aligned_img_filepaths]
    else:
        success = align_faces(
            img_filepaths=img_filepaths,
            aligned_img_filepaths=aligned_img_filepaths,
            crop_size=112,
            cuda=cuda,
        )

    success_aligned_img_filepaths = [
        filepath for exists, filepath in zip(success, aligned_img_filepaths) if exists
    ]

    transform = transforms.Compose(
        [
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
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
                batch_embeddings, _ = model(batch["images"].cuda())
                batch_embeddings = batch_embeddings.cpu()
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
