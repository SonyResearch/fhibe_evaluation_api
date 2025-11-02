# SPDX-License-Identifier: Apache-2.0

import pickle

import pandas as pd
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from fhibe_eval_api.datasets.fhibe import get_person_segments_area


def prepare_keypoint_estimation(dataframe, keypoints_fmt="keypoints_coco_fmt"):
    dataframe = dataframe.copy(deep=True)

    # use_mini_dataset = True
    # mini_dataset_size = 50
    # if use_mini_dataset:
    #     dataframe = dataframe.sample(
    #         n=mini_dataset_size if mini_dataset_size else 50,
    #         random_state=DEFAULT_RANDOM_STATE,
    #     ).reset_index()

    # # Convert the bbox columns from stringified list to list
    # dataframe["person_bbox"] = [eval(x) for x in dataframe["person_bbox"]]
    # dataframe["face_bbox"] = [eval(x) for x in dataframe["face_bbox"]]

    # # Convert coords [xmin, ymin, width, height] -> [xmin, ymin, xmax, ymax]
    # dataframe["person_bbox"] = [
    #     [x[0], x[1], x[0] + x[2], x[1] + x[3]] for x in dataframe["person_bbox"]
    # ]
    # dataframe["face_bbox"] = [
    #     [x[0], x[1], x[0] + x[2], x[1] + x[3]] for x in dataframe["face_bbox"]
    # ]

    # # Convert keypoints to COCO format
    # dataframe["keypoints_coco_fmt"] = [
    #     np.array(covert_keypoints_to_COCO_format(eval(x))).reshape((-1, 3))
    #     for x in dataframe["keypoints"]
    # ]

    # img_filepaths = dataframe["filepath"].unique().tolist()
    # img_filepath_gt_bboxes = {
    #     filepath: dataframe.loc[
    #         dataframe["filepath"] == filepath, "person_bbox"
    #     ].tolist()
    #     for filepath in img_filepaths
    # }

    # Compute person segments area size
    mask_info = zip(
        dataframe["segments"], dataframe["image_height"], dataframe["image_width"]
    )
    iterable = [(seg, im_h, im_w) for seg, im_h, im_w in tqdm(mask_info)]
    fhibe_masks_areas = process_map(
        get_person_segments_area,
        iterable,
        max_workers=12,
        chunksize=16,
        desc="Computing area size of person segments",
    )
    with open("precomputed_person_segment_areas.pkl", "wb") as file:
        pickle.dump(fhibe_masks_areas, file)


if __name__ == "__main__":
    f = "./static/data/processed/fhibe_downsampled/fhibe_downsampled.csv"
    df = pd.read_csv(f)
    prepare_keypoint_estimation(df)
