# SPDX-License-Identifier: Apache-2.0

import os

import numpy as np
import pandas as pd
from PIL import Image

CURRENT_DIR = os.path.dirname(__file__)


def make_fake_face_encodings(dataframe):
    # Save fake png images to tests/static/data/images_faces/ with the correct prefixes
    # and of the correct shape
    fake_encoding_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        "face_encoding",
        "fhibe_face_crop_align",
        "face_encoder_test_model",
        "face_encodings",
    )
    os.makedirs(fake_encoding_dir, exist_ok=True)
    image_shape = (24, 24, 3)
    for fp in dataframe.filepath:
        basename = os.path.basename(fp)
        fake_full_path = os.path.join(fake_encoding_dir, basename)
        random_array = np.random.randint(0, 255, size=image_shape, dtype=np.uint8)
        img = Image.fromarray(random_array)
        img.save(fake_full_path)
        print(f"Saved {fake_full_path}")


if __name__ == "__main__":
    f = "./static/data/processed/fhibe_face_crop_align/fhibe_face_crop_align.csv"
    df = pd.read_csv(f)
    make_fake_face_encodings(df)
