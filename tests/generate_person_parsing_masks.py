# SPDX-License-Identifier: Apache-2.0

import pickle

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from fhibe_eval_api.datasets.fhibe import _get_img_masks
from fhibe_eval_api.evaluate.constants import DEFAULT_RANDOM_STATE


def prepare_person_parsing(dataframe, to_rle=True):
    dataframe = dataframe.copy(deep=True)
    use_mini_dataset = False
    mini_dataset_size = 50
    if use_mini_dataset:
        dataframe = dataframe.sample(
            n=mini_dataset_size if mini_dataset_size else 50,
            random_state=DEFAULT_RANDOM_STATE,
        ).reset_index()

    mask_info = zip(
        dataframe["segments"], dataframe["image_height"], dataframe["image_width"]
    )

    person_only = True
    iterable = [
        (seg, im_h, im_w, person_only, to_rle) for seg, im_h, im_w in tqdm(mask_info)
    ]

    print("[Start] Extracting FHIBE person binary masks")
    fhibe_masks = process_map(_get_img_masks, iterable, max_workers=12, chunksize=16)
    with open("object.pickle", "wb") as file:
        pickle.dump(fhibe_masks, file)


if __name__ == "__main__":
    f = "./static/data/processed/fhibe_downsampled/fhibe_downsampled.csv"
    df = pd.read_csv(f)
    prepare_person_parsing(df)
