# SPDX-License-Identifier: Apache-2.0
"""Reporting utility functions.

This module contains functions for calculating the disparity between 
attribute groups as well as formatting attribute group names and values.
"""

import itertools
import os
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind
from tqdm.contrib.concurrent import process_map

from fhibe_eval_api.evaluate.constants import FITZPATRICK_TYPE_DICT


def process_pair(
    args: Tuple[Tuple[str, str], Dict[str, List[float]], str]
) -> List[Any]:
    """Calculate the disparity, p-value, and other stats for an attribute pair.

    Args:
        args: Tuple containing the attribute pair, metric scores,
            and the statistic to use for calculating
            significance/effect size.

    Return:
        List containing:
            worst_group: str,
            worst_class_size: int,
            best_group: str,
            best_class_size: int,
            disparity: float,
            p_value: float,
            effect_size: float
    """
    pair, data, statistic = args
    group1, group2 = pair
    group1_scores = data[group1]
    group2_scores = data[group2]
    if statistic == "t":
        t_stat, p_value = ttest_ind(
            group1_scores, group2_scores, alternative="two-sided"
        )
    elif statistic == "U":
        u, p_value = mannwhitneyu(group1_scores, group2_scores, alternative="two-sided")
    # Calculate the median scores for both groups
    median1 = np.median(group1_scores)
    median2 = np.median(group2_scores)
    assert median1 >= 0
    assert median2 >= 0
    n1 = len(group1_scores)
    n2 = len(group2_scores)

    # Determine the worst and best group based on the medians
    if median1 < median2:
        if median1 == 0.0 and median2 == 0.0:
            disparity = 1.0
        elif median1 == 0.0:
            disparity = 0.0
        else:
            disparity = float(median1 / median2)
        worst_group, best_group = group1, group2
        worst_class_size = len(group1_scores)
        best_class_size = len(group2_scores)
    else:
        if median1 == 0 and median2 == 0:
            disparity = 1
        elif median2 == 0:
            disparity = 0
        else:
            disparity = float(median2 / median1)
        worst_group, best_group = group2, group1
        worst_class_size = len(group2_scores)
        best_class_size = len(group1_scores)

    if np.isnan(disparity):
        disparity = 0.0
    else:
        disparity = 1.0 - disparity

    # Calculate effect size
    if statistic == "t":
        # Cohen's D (effect size)
        mean1 = np.mean(group1_scores)
        mean2 = np.mean(group2_scores)
        std1 = np.std(group1_scores, ddof=1)
        std2 = np.std(group2_scores, ddof=1)
        pooled_std = np.sqrt(((n1 - 1) * std1**2 + (n2 - 1) * std2**2) / (n1 + n2 - 2))
        effect_size = (mean1 - mean2) / pooled_std
    elif statistic == "U":
        mean_u = n1 * n2 / 2
        std_u = np.sqrt(n1 * n2 * (n1 + n2 + 1) / 12)
        effect_size = (u - mean_u) / std_u

    return [
        worst_group,
        worst_class_size,
        best_group,
        best_class_size,
        disparity,
        p_value,
        effect_size,
    ]


def find_significant_pairs(
    data: Dict[str, List[float]], alpha: float = 0.05, statistic: str = "t"
) -> List[List[Any]] | None:
    """Find all pairs with significant disparity.

    Return None if no significant pairs are found.

    Args:
        data: A dictionary mapping attribute string to list of
            metric scores
        alpha: Significance threshold for a single test
        statistic: The significance test to use, i.e.,
            t: t-test
            U: Mann-Whitney U test
    Return:
        List of statistically significant pairs
    """
    attribute_values = list(data.keys())
    combs = itertools.combinations(attribute_values, 2)
    tasks = [(pair, data, statistic) for pair in combs]
    cpu_count = os.cpu_count()
    if cpu_count is None:
        cpu_count = 1

    chunksize = max(1, len(tasks) // min(40, cpu_count))

    pairs = process_map(
        process_pair,
        tasks,
        max_workers=min(40, cpu_count),
        chunksize=chunksize,
        disable=True,
    )
    # Apply Bonferroni correction to adjust alpha
    m = len(pairs)
    alpha_bf = alpha / m
    # Find all pairs with stat-sig disparities
    stat_sig_pairs = [p for p in pairs if p[-2] < alpha_bf]
    if len(stat_sig_pairs) > 0:
        return stat_sig_pairs
    else:
        return None


def format_attribute_name(attribute_name: str) -> str:
    """Reformat the name of an attribute.

    E.g., "location_country" -> "Location Country"

    Args:
        attribute_name: An attribute name, e.g., "pronoun"

    Return:
        The reformatted attribute name
    """
    return attribute_name.replace("_", " ").title()


def format_single_attribute_value(attribute_name: str, attribute_value: str) -> str:
    """Reformat an attribute value.

    E.g., "['0. Africa']" -> "Africa"

    Args:
        attribute_name: A lower-case attribute name, e.g., "pronoun"
        attribute_value: The string value of the attribute,
            from the intersectional results JSON file.

    Return:
        The reformatted attribute value
    """
    if attribute_name == "age":
        return attribute_value.split("'")[1]
    elif attribute_name in [
        "user_hour_captured",
        "location_country",
    ]:
        return attribute_value[2:-2]
    elif attribute_name == "apparent_skin_color_hue_lum":
        return attribute_value[2:-2].replace("_", " ")
    elif attribute_name in ["apparent_skin_color", "natural_skin_color"]:
        sc_tup = tuple(eval(attribute_value.split(". ")[-1].split("']")[0]))
        return FITZPATRICK_TYPE_DICT[sc_tup]
    else:
        return attribute_value.split(". ")[-1].split("']")[0]


def format_attribute_list(
    attribute_names: List[str], attribute_values: List[str]
) -> List[str]:
    """Reformat a list of attribute values.

    E.g., "['0. Africa']" -> "Africa"

    Args:
        attribute_names: List of attribute names, e.g.,
            ["pronoun", "age", "ancestry"]
        attribute_values: List of string values of the attributes,
            e.g., ["['1. He/him/his']","['0. Africa']"]

    Return:
        List of reformatted attribute values
    """
    reformatted_attributes = []
    for ix, attr_name in enumerate(attribute_names):
        if attr_name in ["age", "user_hour_captured", "location_country"]:
            reformatted_attributes.append(attribute_values[ix])
        else:
            reformatted_attributes.append(
                format_single_attribute_value(attr_name, attribute_values[ix])
            )
    return reformatted_attributes
