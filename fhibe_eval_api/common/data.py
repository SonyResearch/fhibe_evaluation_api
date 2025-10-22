# Copyright (c) Sony AI Inc.
# All rights reserved.
"""Module containing data utilities.

This module contains functions for performing data transformations.
"""

from typing import Any, Dict, List, Optional

import torchvision.transforms as transforms  # type: ignore
from torch.utils.data import Dataset  # type: ignore

from fhibe_eval_api.common.utils import open_image_with_pil


def identity_function(x: Any) -> Any:  # noqa: D103
    return x


def pil_image_collate_function(batch: List[Dict[str, Any]]) -> Dict[str, List[Any]]:
    """Collate function for handling a batch of PIL images.

    Args:
        batch (List[Dict[str, Any]]): List of PIL images and image paths.

    Returns:
        Dict[str, List[Any]]: The collated list of PIL images and image paths.
    """
    images = [item["images"] for item in batch]
    image_paths = [item["image_paths"] for item in batch]
    return {"images": images, "image_paths": image_paths}


class ImageDataset(Dataset):  # type: ignore # noqa: D101
    def __init__(
        self,
        image_paths: List[str],
        exif_transpose: bool,
        transform: Optional[transforms.Compose] = None,
        grayscale: bool = False,
    ) -> None:
        """A PyTorch Dataset class for loading images from a list of file paths.

        Args:
            image_paths: A list of strings representing the paths to image
                files.
            exif_transpose: A boolean indicating whether to apply EXIF
                orientation correction.
            transform: A composition of PyTorch image
                transformations to apply to each image. Default is None.
            grayscale: If True, images will be converted to grayscale
                before being transformed. Default is False.
        """
        self.image_paths = image_paths
        self.exif_transpose = exif_transpose
        if transform is None:
            self.transform = identity_function
        else:
            self.transform = transform
        self.grayscale = grayscale

    def __getitem__(self, image_index):  # noqa: D105
        image_path = self.image_paths[image_index]
        pil_image = open_image_with_pil(
            image_path=image_path,
            exif_transpose=self.exif_transpose,
            grayscale=self.grayscale,
        )

        return {
            "images": self.transform(pil_image) if pil_image is not None else None,
            "image_paths": image_path,
        }

    def __len__(self):  # noqa: D105
        return len(self.image_paths)
