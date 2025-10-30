# SPDX-License-Identifier: Apache-2.0
"""Module for preparing the FHIBE-face dataset.

This module contains a class for preparing the FHIBE face
dataset for use in the evaluation of each task.
"""
import os
from typing import List, Tuple

import pandas as pd

from fhibe_eval_api.common.utils import eval_custom
from fhibe_eval_api.datasets.utils import fix_location_country


class FHIBEFacePublicEval:
    """Evaluation API wrapper class for FHIBE-face tasks."""

    def __init__(
        self,
        dataframe: pd.DataFrame,
        aligned: bool,
        data_dir: str,
        processed_data_dir: str,
        intersectional_column_names: List[str] | None = None,
        age_buckets: bool = True,
    ) -> None:
        """Constructor.

        Args:
            dataframe: The dataframe containing FHIBE annotations
            aligned: Whether the images are already self-aligned
            data_dir: Raw data directory
            processed_data_dir: Processed data directory
            intersectional_column_names: List of demographic groups
                for aggregation.
            age_buckets: Whether to use age buckets.

        Return:
            None
        """
        self.dataframe = dataframe
        self._aligned = aligned
        self.data_dir = data_dir
        self.processed_data_dir = processed_data_dir
        self.intersectional_column_names = intersectional_column_names
        self.age_buckets = age_buckets

        if intersectional_column_names is None:
            self.intersectional_column_names = (
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

    @property
    def is_aligned(self):  # noqa: D102
        return self._aligned

    def _get_age_buckets(self, dataframe):
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

    def prepare_face_parsing(self, mask_fmt: str = "CelebA"):
        """Run preparation steps for the face parsing task.

        Args:
            mask_fmt: The format to use for the masks

        Return:
            Tuple containing:
                dataframe: annotation dataframe
                img_filepaths: List of unique image filepaths
                mask_filepaths: List of filepaths to face parsing masks
                prediction_change_map: An optional dict mapping the
                    face labels to new labels for compliance with different
                    mask formats.
        """
        # mask_fmt = CelebA or FHIBE
        dataframe = self.dataframe.copy(deep=True)

        # This is used to modify predictions that do not have GT labels.
        # For instance, we do not have "ear" labels in FHIBE
        # so everything must be changed into "skin"
        if mask_fmt == "CelebA":
            prediction_change_map = {"l_ear": "skin", "r_ear": "skin"}
        else:
            prediction_change_map = None

        # FHIBE Face (crop & align)
        img_filepaths = dataframe["filepath"].tolist()
        mask_filepaths = dataframe[mask_fmt].tolist()
        mask_filepaths = [
            next(
                (element for element in eval_custom(mask_fp) if "combined" in element),
                None,
            )
            for mask_fp in mask_filepaths
        ]
        mask_filepaths = [os.path.join(self.data_dir, fp) for fp in mask_filepaths]

        # Create age buckets if applicable
        if self.age_buckets:
            dataframe = self._get_age_buckets(dataframe)

        # Fix location_country
        dataframe["location_country"] = dataframe["location_country"].apply(
            fix_location_country
        )

        return dataframe, img_filepaths, mask_filepaths, prediction_change_map

    def prepare_face_super_resolution(self) -> Tuple[pd.DataFrame, List[str]]:
        """Run preparation steps for the face super resolution task.

        Return:
            Tuple containing:
                dataframe: annotation dataframe
                img_filepaths: List of unique image filepaths
        """
        dataframe = self.dataframe.copy(deep=True)

        img_filepaths = dataframe["filepath"].tolist()

        # Create age buckets if applicable
        if self.age_buckets:
            dataframe = self._get_age_buckets(dataframe)  # type:ignore

        # Fix location_country
        dataframe["location_country"] = dataframe["location_country"].apply(
            fix_location_country
        )

        if not self._aligned:
            aligned_filepaths = [
                original_path.replace(self.data_dir, self.processed_data_dir)
                for original_path in dataframe["filepath"]
            ]
            dataframe["aligned_filepath"] = aligned_filepaths
        return dataframe, img_filepaths

    def prepare_face_verification(
        self, drop_two_subjects: bool = True
    ) -> Tuple[pd.DataFrame, List[str]]:
        """Run preparation steps for the face verficiation task.

        Args:
            drop_two_subjects: Whether to not include images containing
                multiple subjects.

        Return:
            Tuple containing:
                dataframe: annotation dataframe
                img_filepaths: List of unique image filepaths
        """
        dataframe = self.dataframe.copy(deep=True)

        if drop_two_subjects:
            dataframe = dataframe.loc[~dataframe["multiple_subjects"].astype(bool)]

        dataframe["person"] = dataframe["subject_id"]
        dataframe.reset_index(inplace=True, drop=True)

        # keep subjects with at least 2 appearances
        dataframe.sort_values(by="person", inplace=True)
        dataframe = dataframe[dataframe.groupby("person").person.transform("count") > 1]

        img_filepaths = dataframe["filepath"].tolist()

        # Create age buckets if applicable
        if self.age_buckets:
            dataframe = self._get_age_buckets(dataframe)

        # Fix location_country
        dataframe["location_country"] = dataframe["location_country"].apply(
            fix_location_country
        )

        if not self._aligned:
            aligned_filepaths = [
                original_path.replace(self.data_dir, self.processed_data_dir)
                for original_path in dataframe["filepath"]
            ]
            dataframe["aligned_filepath"] = aligned_filepaths

        return dataframe, img_filepaths

    def prepare_face_encoding(self):
        """Run preparation steps for the face encoding task.

        Return:
            Tuple containing:
                dataframe: annotation dataframe
                img_filepaths: List of unique image filepaths
        """
        dataframe = self.dataframe.copy(deep=True)
        img_filepaths = dataframe["filepath"].tolist()

        # Create age buckets if applicable
        if self.age_buckets:
            dataframe = self._get_age_buckets(dataframe)

        # Fix location_country
        dataframe["location_country"] = dataframe["location_country"].apply(
            fix_location_country
        )

        if not self._aligned:
            aligned_filepaths = [
                original_path.replace(self.data_dir, self.processed_data_dir)
                for original_path in dataframe["filepath"]
            ]
            dataframe["aligned_filepath"] = aligned_filepaths

        return dataframe, img_filepaths
