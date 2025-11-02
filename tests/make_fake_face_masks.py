# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pandas as pd
from PIL import Image

CURRENT_DIR = os.path.dirname(__file__)


def make_fake_face_masks(dataframe):
    # Save fake combined.png images to tests/static/processed/masks/ with the
    # correct prefixes and of the correct shape

    fake_base_path = os.path.join(
        CURRENT_DIR,
        "static",
        "data",
        "processed",
        "fhibe_face_crop_align",
        "masks",
        "CelebAMask-HQ_format",
    )
    os.makedirs(fake_base_path, exist_ok=True)
    image_shape = (24, 24)
    for image_id in dataframe.image_id:
        random_array = np.random.randint(0, 19, size=image_shape, dtype=np.uint8)
        fake_full_path = os.path.join(fake_base_path, image_id, "combined.png")
        parent_dir = os.path.dirname(fake_full_path)
        os.makedirs(parent_dir, exist_ok=True)
        img = Image.fromarray(random_array)
        img.save(fake_full_path)
        print(f"Saved {fake_full_path}")


if __name__ == "__main__":
    f = "./static/data/processed/fhibe_face_crop_align/fhibe_face_crop_align.csv"
    df = pd.read_csv(f)
    make_fake_face_masks(df)
