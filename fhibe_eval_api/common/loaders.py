# Copyright (c) Sony AI Inc.
# All rights reserved.
"""Module containing data loader utilities.

This module contains helper functions for data loading.
"""

from typing import Any, List, Optional, Tuple, Union

from torch.utils.data import DataLoader
from torchvision import transforms

from fhibe_eval_api.common.data import ImageDataset


def image_data_loader_from_paths(
    transform: Optional[transforms.Compose],
    image_paths_1: List[str],
    image_paths_2: Optional[List[str]] = None,
    exif_transpose: bool = True,
    batch_size: int = 32,
    shuffle: bool = False,
    num_workers: int = 8,
    grayscale: bool = False,
    collate_fn: Optional[Any] = None,
) -> Union[DataLoader, Tuple[DataLoader, DataLoader]]:
    """Create a PyTorch DataLoader that loads images from paths.

    Args:
        transform (Optional[transforms.Compose]): A composition of PyTorch transforms
            to be applied to the images.
        image_paths_1 (List[str]): A list of strings representing the paths to the
            first set of image files.
        image_paths_2 (List[str], optional): A list of strings representing the paths
            to the second set of image files.
        exif_transpose (bool): A boolean indicating whether to apply EXIF orientation
            correction. Default is True.
        batch_size (int): The batch size to use for the DataLoader. Default is 32.
        shuffle (bool): A boolean indicating whether to shuffle the data. Default
            is False.
        num_workers (int): Number of workers to load data in parallel.
        grayscale (bool): A boolean indicating whether to convert images to
            grayscale. Default is False, which converts images to RGB.
        collate_fn (Optional[Any]): Collate function. Default is None.

    Returns:
        Union[DataLoader, Tuple[DataLoader, DataLoader]]: A single DataLoader is
            returned if only image_paths_1 is specified, otherwise two DataLoaders
            are returned.
    """
    image_dataset_1 = ImageDataset(
        image_paths=image_paths_1,
        exif_transpose=exif_transpose,
        transform=transform,
        grayscale=grayscale,
    )

    data_loader_1 = DataLoader(
        image_dataset_1,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_fn,
    )

    if image_paths_2 is not None:
        image_dataset_2 = ImageDataset(
            image_paths=image_paths_2,
            exif_transpose=exif_transpose,
            transform=transform,
            grayscale=grayscale,
        )

        data_loader_2 = DataLoader(
            image_dataset_2,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            collate_fn=collate_fn,
        )

        return data_loader_1, data_loader_2
    else:
        return data_loader_1
