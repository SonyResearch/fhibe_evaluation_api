# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pandas as pd
from PIL import Image

CURRENT_DIR = os.path.dirname(__file__)


def make_fake_face_images(dataframe):
    # Save fake png images to tests/static/data/images_faces/ with the correct prefixes
    # and of the correct shape
    fake_base_path = os.path.join(CURRENT_DIR, "static")
    image_shape = (24, 24, 3)  # for testing purposes
    for ix, row in dataframe.iterrows():
        fp = row["filepath"]
        random_array = np.random.randint(0, 255, size=image_shape, dtype=np.uint8)
        fake_full_path = os.path.join(fake_base_path, fp)
        parent_dir = os.path.dirname(fake_full_path)
        os.makedirs(parent_dir, exist_ok=True)
        img = Image.fromarray(random_array)
        img.save(fake_full_path)
        print(f"Saved {fake_full_path}")


if __name__ == "__main__":
    f = "./static/data/processed/fhibe_downsampled/fhibe_downsampled.csv"
    df = pd.read_csv(f)
    make_fake_face_images(df)
