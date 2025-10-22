# Copyright (c) Sony AI Inc.
# All rights reserved.
"""Module for preparing the full body FHIBE dataset.

This module contains a class and functions for preparing the FHIBE
dataset for use in the evaluation of each task.
"""
import os
from typing import Any, Dict, List, Tuple, cast

import cv2
import numpy as np
import pandas as pd
from numpy.typing import NDArray
from PIL import Image
from pycocotools.mask import area as get_rle_mask_area
from pycocotools.mask import encode
from tqdm import tqdm
from tqdm.contrib.concurrent import process_map

from fhibe_eval_api.common.loggers import setup_logging
from fhibe_eval_api.datasets.utils import fix_location_country

setup_logging("info")

# suppress warnings like:
# DecompressionBombWarning: Image size (*** pixels) exceeds limit of *** pixels,
# could be decompression bomb DOS attack.
Image.MAX_IMAGE_PIXELS = None

# Set number of parallel threads
NUM_CPUS = min(32, os.cpu_count())  # type: ignore

# Keypoints that exist both in FHIBE and COCO
FHIBE_COMMON_KEYPOINTS = [
    "Nose",
    "Left eye",
    "Right eye",
    "Left ear",
    "Right ear",
    "Left shoulder",
    "Right shoulder",
    "Left elbow",
    "Right elbow",
    "Left wrist",
    "Right wrist",
    "Left hip",
    "Right hip",
    "Left knee",
    "Right knee",
    "Left ankle",
    "Right ankle",
]


def validate_custom_keypoints(custom_keypoints: List[str]) -> None:
    """Ensure that the custom keypoints are valid.

    Checks names and order.

    Args:
        custom_keypoints: List of keypoints to validate.

    Return:
        None

    Raises:
        ValueError: If any of the custom keypoints are not valid
            or if no keypoints are provided.
    """
    if not custom_keypoints:
        raise ValueError(
            "You cannot provide an empty list of custom keypoints. "
            "Please provide a valid list of keypoints or set custom_keypoints=None."
        )
    for keypoint in custom_keypoints:
        if not isinstance(keypoint, str):
            raise ValueError(
                f"A keypoint you specified: {keypoint} is not a string. "
                "Please provide a list of strings."
            )
        if keypoint not in FHIBE_COMMON_KEYPOINTS:
            raise ValueError(
                f"A keypoint you specified: '{keypoint}' "
                "is not a valid FHIBE keypoint. "
                "Please provide a list of valid keypoints. "
                f"Valid keypoints are: {FHIBE_COMMON_KEYPOINTS}"
            )
    custom_keypoints_proper_order = [
        k for k in FHIBE_COMMON_KEYPOINTS if k in custom_keypoints
    ]
    if custom_keypoints != custom_keypoints_proper_order:
        raise ValueError(
            "Custom keypoints are not in the correct order. "
            "Please output your keypoints in this order: "
            f"{custom_keypoints_proper_order}."
        )
    return


def convert_keypoints_to_coco_format(
    fhibe_kps: Dict[str, List[float | int]] | str,
    custom_keypoints: List[str] | None = None,
) -> List[float | int]:
    """Convert FHIBE keypoints to COCO format.

    Args:
        fhibe_kps: Dictionary containing the FHIBE keypoints
        custom_keypoints: Optional list of keypoints to use for the task.

    Return:
        A list containing triplets of [x,y,visibility]
    """
    # COCO_kps = [
    #     "nose","left_eye","right_eye","left_ear","right_ear",
    #     "left_shoulder","right_shoulder","left_elbow","right_elbow",
    #     "left_wrist","right_wrist","left_hip","right_hip",
    #     "left_knee","right_knee","left_ankle","right_ankle"
    # ]

    def _update_visibility(kp: List[float | int]) -> List[float | int]:
        kp[-1] = int(kp[-1])
        if kp[-1] == 1:
            kp[-1] = 2
        return kp

    # If custom keypoints are provided, filter to only include those
    if custom_keypoints is not None:
        # Initialize everything to 0 visibility
        # Ensures order of keypoints follows FHIBE_COMMON_KEYPOINTS
        fhibe_common_kps_dict: Dict[str, List[float | int]] = {
            k: [0.0, 0.0, 0] for k in FHIBE_COMMON_KEYPOINTS if k in custom_keypoints
        }
        # restrict fhibe_kps to only those in custom_keypoints
        fhibe_kps = {
            k: v for k, v in fhibe_kps.items() if k.split(". ")[-1] in custom_keypoints
        }
    else:
        # Initialize everything to 0 visibility
        fhibe_common_kps_dict: Dict[str, List[float | int]] = {
            k: [0.0, 0.0, 0] for k in FHIBE_COMMON_KEYPOINTS
        }

    if isinstance(fhibe_kps, str):
        fhibe_kps = cast(Dict[str, List[float | int]], eval(fhibe_kps))

    # Update the keypoints in COCO format, based on the values of the input dict
    for k, v in fhibe_kps.items():
        if k.split(". ")[-1] in FHIBE_COMMON_KEYPOINTS:
            fhibe_common_kps_dict.update({k.split(". ")[-1]: _update_visibility(v)})
    # Convert everything to a list of triplets [x,y,visibility]
    # final_kps = np.array(list(fhibe_common_kps_dict.values()), dtype="object")
    final_kps = [x for y in fhibe_common_kps_dict.values() for x in y]
    return final_kps


def get_person_segments_area(
    args: Tuple[str | List[Dict[str, List[Dict[str, float]]]], int, int]
) -> int:
    """Get the integer area of a person segment.

    Args:
        args: Tuple containing
        - segments: Union[str, list], list of segments containing x, y coordinates
            (or string containing the list; str param will be deprecated)
        - image_height: int
        - image_width: int

    Return:
        Integer area
    """
    segments, image_height, image_width = args
    person_only = True
    to_rle = True
    rle_person_mask = _get_img_masks(
        (segments, image_height, image_width, person_only, to_rle)
    )
    area = cast(int, get_rle_mask_area(rle_person_mask))
    return area


def _get_img_masks(
    args: Tuple[str | List[Dict[str, Any]], int, int, bool, bool]
) -> Dict[str, Any] | NDArray[np.uint8] | str:
    """Given a list of segments it converts them to binary masks.

    Args:
        args: Tuple consisting of:
        - segments: Union[str, list], list of segments containing x, y coordinates
            (or string containing the list; str param will be deprecated)
        - image_height: int
        - image_width: int
        - person_only: bool, Whether to return only the whole person mask
        - toRLE: bool, Whether to encode the mask using RLE, defaults to False

    Return:
        Either a dictionary with all mask categories and their associated masks,
        or a single binary mask (for person_only=True)
        or a string encoded using RLE (for person_only=True)
    """
    # this is needed because process_map calls _get_img_masks on iterable of tuples.
    segments, image_height, image_width, person_only, to_rle = args

    # TODO: get rid of eval, keep only list.
    if isinstance(segments, str):
        segments = eval(segments)

    masks = {"person_mask": np.zeros((image_height, image_width), dtype=np.uint8)}

    canvas = np.zeros((image_height, image_width), dtype=np.uint8)
    for seg in segments:
        points = np.array([[int(p["x"]), int(p["y"])] for p in seg["polygon"]])
        cv2.fillPoly(canvas, pts=[points], color=1)

        masks["person_mask"] = np.logical_or(canvas, masks["person_mask"]).astype(
            np.uint8
        )

        if not person_only:
            segm_name = seg["class_name"]
            if segm_name not in masks:
                masks[segm_name] = np.zeros((image_height, image_width), dtype=np.uint8)
            masks[segm_name] = np.logical_or(canvas, masks[segm_name]).astype(np.uint8)

        canvas = 0 * canvas

    if person_only:
        if to_rle:
            masks["person_mask"] = encode(
                np.asfortranarray(np.array(masks["person_mask"]))
            )
        return masks["person_mask"]

    if to_rle:
        masks = {
            key: encode(np.asfortranarray(np.array(m))) for key, m in masks.items()
        }
    return masks


class FHIBEPublicEval:
    """Evaluation API wrapper class for full body FHIBE tasks."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        intersectional_column_names: List[str] | None = None,
        age_buckets: bool = True,
    ) -> None:
        """Constructor.

        Args:
            dataframe: The dataframe containing FHIBE annotations
            intersectional_column_names: List of demographic groups
                for aggregation.
            age_buckets: Whether to use age buckets.

        Return:
            None
        """
        self.dataframe = dataframe
        self._intersectional_column_names = intersectional_column_names
        self.age_buckets = age_buckets

        if intersectional_column_names is None:
            self._intersectional_column_names = (
                "pronoun",
                "age",
                "apparent_skin_color",
                "ancestry",
            )

        self.age_map = {
            0: "[18, 30)",
            1: "[30, 40)",
            2: "[40, 50)",
            3: "[50, 60)",
            4: "[60, +]",
        }

    def _get_age_buckets(self, dataframe: pd.DataFrame) -> pd.DataFrame:
        """Reassign the 'age' column in the dataframe to string buckets.

        Args:
            dataframe: The dataframe containing FHIBE annotations

        Return:
            None
        """
        age_buckets = []
        for age in dataframe["age"]:
            digit = age // 10
            if digit == 1:
                age_buckets.append(self.age_map[0])
            elif digit >= 6:
                age_buckets.append(self.age_map[4])
            else:
                age_buckets.append(self.age_map[digit - 2])

        dataframe["age"] = age_buckets
        return dataframe

    @property
    def intersectional_column_names(self):  # noqa: D102
        return self._intersectional_column_names

    @staticmethod
    def _face_area(list_of_face_bboxes):
        face_areas = []
        for bb in list_of_face_bboxes:
            # assuming [xmin, ymin, xmax, ymax]
            area = (bb[2] - bb[0]) * (bb[3] - bb[1])
            if area <= 49:
                face_areas.append("0")
            elif 50 <= area <= 299:
                face_areas.append("1")
            elif 300 <= area <= 899:
                face_areas.append("2")
            elif 900 <= area <= 1499:
                face_areas.append("3")
            else:
                face_areas.append("4")
        return face_areas

    def prepare_person_localization(
        self,
    ) -> Tuple[pd.DataFrame, List[str], str]:
        """Run preparation steps for the person localization task.

        This task maps to the person detection task in the utility evaluation
        code, so this method is just a wrapper for those preparation steps.

        Return:
            Annotation dataframe, unique image list, dict mapping
            filepath to ground truth person bounding boxes, and
            the name of the column name for the person bounding box.
        """
        dataframe = self.dataframe.copy(deep=True)

        # Convert the bbox columns from stringified list to list
        dataframe["person_bbox"] = [eval(x) for x in dataframe["person_bbox"]]

        # Convert coords [xmin, ymin, width, height] -> [xmin, ymin, xmax, ymax]
        dataframe["person_bbox"] = [
            [x[0], x[1], x[0] + x[2], x[1] + x[3]] for x in dataframe["person_bbox"]
        ]

        img_filepaths = dataframe["filepath"].unique().tolist()

        # Specify the ground-truth
        gt_column_name = "person_bbox"

        if self.age_buckets:
            dataframe = self._get_age_buckets(dataframe)

        # Fix location_country
        dataframe["location_country"] = dataframe["location_country"].apply(
            fix_location_country
        )

        return dataframe, img_filepaths, gt_column_name

    def prepare_keypoint_estimation(
        self,
        keypoints_fmt: str = "keypoints_coco_fmt",
        precomputed_areas: List[int] = None,
        custom_keypoints: List[str] | None = None,
    ):
        """Run preparation steps for the keypoint estimation task.

        Args:
            keypoints_fmt: The format to use for the keypoints
            precomputed_areas: Optional list of precomputed areas for each
                row in the dataframe. Can speed up the preprocessing if
                saved to disk. Used in testing.
            custom_keypoints: Optional list of keypoints to use for the task.

        Return:
            Tuple containing:
                dataframe: annotation dataframe
                img_filepaths: List of unique image filepaths
                img_filepath_gt_bboxes: Dict mapping filepath to list of ground
                    truth person bounding boxes
                gt_keypoint_column_name: Column name in dataframe for ground
                    truth keypoints
                gt_face_bbox_column_name: Column name in datafrmae for ground
                    truth face bounding boxes.
                kpt_oks_sigmas: Array/List of uncertainties for each keypoint
                    to be used in OKS metric calculation.
        """
        dataframe = self.dataframe.copy(deep=True)

        # Convert the bbox columns from stringified list to list
        dataframe["person_bbox"] = [eval(x) for x in dataframe["person_bbox"]]
        dataframe["face_bbox"] = [eval(x) for x in dataframe["face_bbox"]]

        # Convert coords [xmin, ymin, width, height] -> [xmin, ymin, xmax, ymax]
        dataframe["person_bbox"] = [
            [x[0], x[1], x[0] + x[2], x[1] + x[3]] for x in dataframe["person_bbox"]
        ]
        dataframe["face_bbox"] = [
            [x[0], x[1], x[0] + x[2], x[1] + x[3]] for x in dataframe["face_bbox"]
        ]

        # Validate custom keypoint list
        if custom_keypoints is not None:
            validate_custom_keypoints(custom_keypoints)

        # Convert keypoints to COCO format
        dataframe["keypoints_coco_fmt"] = [
            np.array(
                convert_keypoints_to_coco_format(eval(x), custom_keypoints)
            ).reshape((-1, 3))
            for x in dataframe["keypoints"]
        ]

        img_filepaths = dataframe["filepath"].unique().tolist()
        img_filepath_gt_bboxes = {
            filepath: dataframe.loc[
                dataframe["filepath"] == filepath, "person_bbox"
            ].tolist()
            for filepath in img_filepaths
        }

        if precomputed_areas is None:
            # Compute person segments area size
            mask_info = zip(
                dataframe["segments"],
                dataframe["image_height"],
                dataframe["image_width"],
            )
            iterable = [(seg, im_h, im_w) for seg, im_h, im_w in tqdm(mask_info)]
            fhibe_masks_areas = process_map(
                get_person_segments_area,
                iterable,
                max_workers=NUM_CPUS,
                chunksize=16,
                desc="Computing area size of person segments",
            )
            dataframe["area"] = fhibe_masks_areas
        else:
            dataframe["area"] = precomputed_areas

        # kpt_oks_sigmas
        if keypoints_fmt == "keypoints_coco_fmt":
            kpt_oks_sigmas = (
                np.array(
                    [
                        0.26,
                        0.25,
                        0.25,
                        0.35,
                        0.35,
                        0.79,
                        0.79,
                        0.72,
                        0.72,
                        0.62,
                        0.62,
                        1.07,
                        1.07,
                        0.87,
                        0.87,
                        0.89,
                        0.89,
                    ]
                )
                / 10.0
            )
        else:
            # initialize the sigmas sampled from a uniform distribution
            # with a probability score of 1/ #kpts.
            num_kpts = len(eval(dataframe.iloc[0]["keypoints"]))
            kpt_oks_sigmas = np.ones(num_kpts) / num_kpts

        if custom_keypoints is not None:
            # If custom keypoints are provided, filter the sigmas,
            # keeping the order of FHIBE_COMMON_KEYPOINTS.
            kpt_oks_sigmas = kpt_oks_sigmas[
                [k in custom_keypoints for k in FHIBE_COMMON_KEYPOINTS]
            ]
        gt_keypoint_column_name = keypoints_fmt
        gt_face_bbox_column_name = "face_bbox"

        # Create age buckets if applicable
        if self.age_buckets:
            dataframe = self._get_age_buckets(dataframe)

        # Fix location_country
        dataframe["location_country"] = dataframe["location_country"].apply(
            fix_location_country
        )

        return (
            dataframe,
            img_filepaths,
            img_filepath_gt_bboxes,
            gt_keypoint_column_name,
            gt_face_bbox_column_name,
            kpt_oks_sigmas,
        )

    def prepare_face_localization(
        self,
    ) -> Tuple[pd.DataFrame, List[str], str]:
        """Run preparation steps for the face localization task.

        This task maps to the face detection task in the utility evaluation code.

        Return:
            Annotation dataframe, unique image list, dict mapping
            filepath to ground truth face bounding boxes, and
            the name of the column name for the face bounding box.
        """
        dataframe = self.dataframe.copy(deep=True)

        # Convert the face_bbox column from stringified list to list
        dataframe["face_bbox"] = [eval(x) for x in dataframe["face_bbox"]]

        # Convert coords [xmin, ymin, width, height] -> [xmin, ymin, xmax, ymax]
        dataframe["face_bbox"] = [
            [x[0], x[1], x[0] + x[2], x[1] + x[3]] for x in dataframe["face_bbox"]
        ]

        # Compute the face area based on the face bounding box
        dataframe["face_area"] = self._face_area(dataframe["face_bbox"].tolist())

        # Get image filepaths. Call unique() due to multiple subjects per image
        img_filepaths = dataframe["filepath"].unique().tolist()

        # Specify the ground-truth
        gt_column_name = "face_bbox"

        # Create age buckets if applicable
        if self.age_buckets:
            dataframe = self._get_age_buckets(dataframe)

        # Fix location_country
        dataframe["location_country"] = dataframe["location_country"].apply(
            fix_location_country
        )

        return dataframe, img_filepaths, gt_column_name

    def prepare_person_parsing(
        self, to_rle: bool, precomputed_masks=None
    ) -> Tuple[pd.DataFrame, List[str], str]:
        """Run preparation steps for the person parsing task.

        Args:
            to_rle: Whether to use run-length encoding for the masks.
            precomputed_masks: Optional list of precomputed masks for each
                row in the dataframe. Can speed up the preprocessing if
                saved to disk. Used in testing.

        Return:
            Tuple containing:
                - dataframe: Annotation dataframe
                - img_filepaths: unique image list
                - gt_column_name: The column name in the dataframe
                    containing the person masks.
        """
        dataframe = self.dataframe.copy(deep=True)
        if precomputed_masks is None:
            mask_info = zip(
                dataframe["segments"],
                dataframe["image_height"],
                dataframe["image_width"],
            )

            person_only = True
            iterable = [
                (seg, im_h, im_w, person_only, to_rle)
                for seg, im_h, im_w in tqdm(mask_info)
            ]

            print("[Start] Extracting FHIBE person binary masks")
            fhibe_masks = process_map(
                _get_img_masks, iterable, max_workers=NUM_CPUS, chunksize=16
            )
            dataframe["person_mask"] = fhibe_masks
            print("[Done] Extracting FHIBE person binary masks\n")
        else:
            dataframe["person_mask"] = precomputed_masks

        img_filepaths = dataframe["filepath"].unique().tolist()

        # Specify the ground-truth
        gt_column_name = "person_mask"

        if self.age_buckets:
            dataframe = self._get_age_buckets(dataframe)

        # Fix location_country
        dataframe["location_country"] = dataframe["location_country"].apply(
            fix_location_country
        )

        return dataframe, img_filepaths, gt_column_name

    def prepare_body_parts_detection(
        self,
    ) -> Tuple[pd.DataFrame, List[str], Dict[str, List[List[float]]], None]:
        """Run preparation steps for the body_parts_detection task.

        Return:
            Annotation dataframe, unique image list, dict mapping
            filepath to ground truth person bounding boxes, and
            the name of the column name for the person bounding box.
        """
        dataframe = self.dataframe.copy(deep=True)
        # Convert bbox from string to list
        dataframe["person_bbox"] = [eval(x) for x in dataframe["person_bbox"]]
        # Get list of visible body parts for each image

        def get_visible_body_parts(row: Any) -> List[str]:
            """Extract visible body parts from each dataframe row.

            Args:
                row: A row in the annotation dataframe

            Return:
                A list of the visible ground truth body parts in the given
                row in the dataframe.
            """
            kp_json, seg_json = row["keypoints"], row["segments"]
            seg_list = eval(seg_json)
            kp_dict = eval(kp_json)
            # Face is always visible
            vis_list = ["Face"]
            # First, add all segment mask categories
            for seg_dict in seg_list:
                bp_hr = seg_dict["class_name"].split(". ")[-1]
                # Avoid duplicate - some overlap between masks and keypoint body parts
                if bp_hr not in vis_list:
                    vis_list.append(bp_hr)
                # If a segement is present it is visible and therefore detectable

            # Make list of visible keypoints for making derivative body parts
            vis_kp_list = []

            for bp, kl in kp_dict.items():
                if kl[-1] > 0:  # if visibility > 0 then visible
                    bp_hr = bp.split(". ")[-1]
                    vis_kp_list.append(bp_hr)
            # Hand is present if any of the knuckle keypoints present or glove mask
            left_hand_keypoints = [
                "Left pinky knuckle",
                "Left index knuckle",
                "Left thumb knuckle",
            ]
            right_hand_keypoints = [
                "Right pinky knuckle",
                "Right index knuckle",
                "Right thumb knuckle",
            ]
            # Add left hand if any lh kps visible
            if any([kp in vis_kp_list for kp in left_hand_keypoints]):
                vis_list.append("Left hand")

            # Add right hand if any rh kps visible
            if any([kp in vis_kp_list for kp in right_hand_keypoints]):
                vis_list.append("Right hand")

            # Either hand (inclusive or)
            if any([bp in vis_list for bp in ["Left hand", "Right hand", "Glove"]]):
                vis_list.append("Hand")

            # Legs
            left_leg_keypoints = [
                "Left knee," "Left ankle",
                "Left heel",
                "Left foot index",
            ]
            right_leg_keypoints = [
                "Right knee," "Right ankle",
                "Right heel",
                "Right foot index",
            ]
            # Add left leg if any lh kps visible or left shoe mask present
            if (
                any([kp in vis_kp_list for kp in left_leg_keypoints])
                or "Left shoe" in vis_list
            ):
                vis_list.append("Left leg")

            # Add right leg if any rh kps visible or right shoe mask present
            if (
                any([kp in vis_kp_list for kp in right_leg_keypoints])
                or "Right shoe" in vis_list
            ):
                vis_list.append("Right leg")

            # Either leg (inclusive or)
            if any([bp in vis_list for bp in ["Left leg", "Right leg"]]):
                vis_list.append("Leg")

            return vis_list

        tqdm.pandas(desc="Extracting ground truth visible body parts")
        dataframe["visible_body_parts"] = dataframe.progress_apply(
            get_visible_body_parts, axis=1
        )

        img_filepaths = dataframe["filepath"].unique().tolist()
        img_filepath_gt_bboxes = {
            filepath: dataframe.loc[
                dataframe["filepath"] == filepath, "person_bbox"
            ].tolist()
            for filepath in img_filepaths
        }  # maps filepath to a list of ground truth person bounding boxes

        gt_column_name = None  # not used for this task.

        if self.age_buckets:
            dataframe = self._get_age_buckets(dataframe)  # type: ignore

        # Fix location_country
        dataframe["location_country"] = dataframe["location_country"].apply(
            fix_location_country
        )

        return dataframe, img_filepaths, img_filepath_gt_bboxes, gt_column_name
