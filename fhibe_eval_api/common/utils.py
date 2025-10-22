# Copyright (c) Sony AI Inc.
# All rights reserved.
"""Module containing generatl utility functions.

This module contains helper functions used throughout the API.
"""

import json
import os
from typing import Any, Dict, List, Optional, cast

import pandas as pd
from PIL import Image, ImageOps


def read_json_file(filepath: str) -> Dict[Any, Any]:
    """Read data from a JSON file.

    Args:
        filepath (str): The path to the JSON file.

    Return:
        Dict[Any, Any]: The data loaded from the JSON file.
    """
    with open(filepath, "r") as json_file:
        data: Dict[Any, Any] = json.load(json_file)
    return data


def save_json_file(
    filepath: str,
    data: Dict[Any, Any] | List[Any],
    indent: Optional[int | str | None] = None,
) -> None:
    """Save data to a JSON file.

    Args:
        filepath (str): The path to the JSON file.
        data (Dict[Any, Any]): The data to be saved.
        indent: Number of indentation spaces (optional).

    Return:
        None
    """
    with open(filepath, "w") as json_file:
        json.dump(data, json_file, indent=indent)


def save_df_to_latex(df: pd.DataFrame, path: str, index: bool) -> None:
    """Save dataframe as a latex table.

    Args:
        df: Pandas dataframe
        path: The filename for the latex file.
        index: The index column to use.

    Return:
        None
    """
    latex_code = df.to_latex(index=index)
    # Write the LaTeX code to a .tex file
    with open(path, "w") as file:
        file.write(latex_code)


def open_image_with_pil(
    image_path: str, exif_transpose: bool = True, grayscale: bool = False
) -> Optional[Image.Image]:
    """Opens an image file using the PIL library.

    Optionally applies EXIF orientation correction.

    Args:
        image_path: A string representing the path to the image file.
        exif_transpose: A boolean indicating whether to apply EXIF orientation
            correction. Default is True.
        grayscale: A boolean indicating whether to convert the image to
            grayscale. Default is False, which converts the image to RGB.

    Return:
        An Image object with three channels (RGB) if grayscale
            is set to False or one channel (L) if grayscale is set to True.
            Returns None if the image file cannot be opened.
    """
    try:
        pil_image = Image.open(image_path)
        converted_pil_image = pil_image.convert("RGB" if not grayscale else "L")
    except IOError:
        print(ValueError(f"Failed to open image file {image_path}"))
        return None
    return (
        ImageOps.exif_transpose(converted_pil_image)
        if exif_transpose
        else converted_pil_image
    )


def check_list_lengths(list_1: list[Any], list_2: list[Any]) -> None:
    """Raises a ValueError if the length of list_1 and list_2 are different.

    Args:
        list_1 (list): First list to compare.
        list_2 (list): Second list to compare.

    Return:
        None
    """
    if len(list_1) != len(list_2):
        raise ValueError(
            f"Length of list_1 ({len(list_1)}) and list_2 ({len(list_2)}) are "
            f"different."
        )


def create_folders(filepath: str) -> None:
    """Create folders for the given file path if they don't exist.

    Args:
        filepath (str): The path of the file.

    Returns:
        None
    """
    file_dir = os.path.dirname(filepath)
    if not os.path.exists(file_dir):
        os.makedirs(file_dir, exist_ok=True)


def get_project_root() -> str:
    """Get the project root.

    Returns:
        None
    """
    utility_path = __file__

    while "/fhibe_eval_api" in utility_path:
        utility_path = os.path.dirname(utility_path)

    return utility_path


def eval_custom(x: str | List[Any]) -> List[Any]:
    """Customized eval() function to handle multiple types.

    Args:
        x: input string or list.

    Returns:
        A list from evalating a string, or simply the input list.
    """
    if isinstance(x, str):
        return cast(List[Any], eval(x))
    elif isinstance(x, list):
        return x
