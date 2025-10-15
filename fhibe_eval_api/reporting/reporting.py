# Copyright (c) Sony AI Inc.
# All rights reserved.
"""Reporting module.

This module contains a BiasReport class for creating a bias report,
including figures and tables summarizes model bias on the FHIBE datasets. 
"""

import itertools
import logging
import math
import os
import textwrap
from datetime import datetime
from typing import Dict, List, Union

import matplotlib  # type: ignore
import matplotlib.pyplot as plt  # type: ignore
import numpy as np
import pandas as pd
import seaborn as sns  # type: ignore
from matplotlib.ticker import FixedFormatter, FixedLocator
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter  # type: ignore
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet  # type: ignore
from reportlab.platypus import (  # type: ignore
    Image,
    ListFlowable,
    ListItem,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from tqdm import tqdm

from fhibe_eval_api.common.loggers import setup_logging
from fhibe_eval_api.common.utils import read_json_file
from fhibe_eval_api.evaluate.constants import (
    ATTRIBUTE_CONSOLIDATION_DICT,
    ATTRIBUTES_DESCRIPTION_DICT,
    DATASET_DESCRIPTION_DICT,
    DATASET_VERSION,
    FITZPATRICK_RGB_DICT,
    METADATA_UNIVERSAL_DESCRIPTION,
    TASK_DESCRIPTION_DICT,
    TASK_METADATA_DESCRIPTION_DICT,
)
from fhibe_eval_api.evaluate.utils import format_threshold_results, validate_attributes
from fhibe_eval_api.metrics.constants import (
    METRIC_DESCRIPTION_DICT,
    TASK_METRIC_LOOKUP_DICT,
)
from fhibe_eval_api.reporting.utils import (
    find_significant_pairs,
    format_attribute_list,
    format_attribute_name,
    format_single_attribute_value,
)

setup_logging("info")

HUMAN_READABLE_TASK_DICT = {
    "person_localization": "Person Localization",
    "keypoint_estimation": "Keypoint Estimation",
    "person_parsing": "Person Parsing",
    "face_localization": "Face Detection",
    "body_parts_detection": "Body Parts Detection",
    "face_parsing": "Face Parsing",
    "face_verification": "Face Verification",
    "face_encoding": "Face Encoding",
    "face_super_resolution": "Face Super Resolution",
}


class BiasReport:
    """Class providing methods for bias report generation."""

    def __init__(
        self,
        model_name: str,
        task_name: str,
        data_rootdir: str,
        results_base_dir: str,
        dataset_name: str,
        results_dir: str | None = None,
        downsampled: bool = False,
        use_mini_dataset: bool = False,
    ) -> None:
        """Constructor.

        Args:
            model_name: The name of the model to evaluate.
                It must be same as one used to generate eval results.
            task_name: The name of the task for which you want to
                generate a bias report.
            data_rootdir: The absolute filepath containing the raw/ and processed/
                subdirectories
            results_base_dir: The absolute filepath to the top-level results/ directory
                written out during bias evaluation.
            dataset_name: "fhibe", "fhibe_face_crop", or "fhibe_face_crop_align"
            results_dir: (Optional) Directly provide the directory containing the
                results. If not supplied, results_base_dir is used to find the
                directory containing the results.
            downsampled: Whether to use the downsampled FHIBE images
            use_mini_dataset: Whether to use the results evaluated on the mini dataset.

        Return:
            None
        """
        self.model_name = model_name
        self.task_name = task_name
        self.data_rootdir = data_rootdir
        self.data_dir = os.path.join(self.data_rootdir, "raw")
        self.processed_data_dir = os.path.join(self.data_rootdir, "processed")
        self.downsampled = downsampled
        self.use_mini_dataset = use_mini_dataset

        if "_crop" in dataset_name or "_align" in dataset_name:
            self.dataset_base = "fhibe_face"
        else:
            self.dataset_base = "fhibe"

        self.dataset_name = dataset_name
        # The face datasets do not have downsampled versions
        if (
            downsampled
            and self.dataset_base == "fhibe"
            and "downsampled" not in self.dataset_name
        ):
            self.dataset_name += "_downsampled"

        self.results_base_dir = results_base_dir
        if self.use_mini_dataset:
            self.results_base_dir = os.path.join(self.results_base_dir, "mini")

        if results_dir is not None:
            self.results_dir = results_dir
        else:
            self.results_dir = os.path.join(
                self.results_base_dir,
                self.task_name,
                self.dataset_name,
                self.model_name,
            )

        # Ensure that there are results generated for this model + task
        self.valid_tasks = [
            x
            for x in os.listdir(self.results_base_dir)
            if x in TASK_METRIC_LOOKUP_DICT.keys()
        ]
        self.task_name = self._validate_task(task_name)
        if self.task_name == "body_parts_detection":
            intersectional_results_file = os.path.join(
                self.results_dir, "intersectional_results_AR_DET.json"
            )
            intersectional_results = read_json_file(intersectional_results_file)
            first_key = list(intersectional_results.keys())[0]
            first_subkey = list(intersectional_results[first_key].keys())[0]
            _body_parts = list(intersectional_results[first_key][first_subkey].keys())
            self.evaluated_body_parts = [x for x in _body_parts if x != "All"]

        self.report_subtitle_style = ParagraphStyle(
            name="CustomTitle",
            parent=getSampleStyleSheet()["Title"],
            fontName="Helvetica",
            fontSize=14,
            textColor="black",
            spaceAfter=12,
            bold=False,
        )
        self.report_date_style = ParagraphStyle(
            name="CustomDate",
            parent=getSampleStyleSheet()["Title"],
            fontName="Helvetica",
            fontSize=10,
            textColor="black",
            spaceAfter=12,
            bold=False,
        )
        self.table_attr_val_style = ParagraphStyle(
            name="CustomStyle",
            alignment=0,  # left alignment
            parent=getSampleStyleSheet()["Normal"],
            fontSize=8,  # Set the desired font size here
        )
        self.table_attr_header_style = ParagraphStyle(
            name="TableHeaderStyle",
            alignment=1,  # horizontal centering
            textColor=colors.white,
            parent=getSampleStyleSheet()["Normal"],
            fontSize=9,  # Set the desired font size here
            fontName="Helvetica-Bold",
        )
        self.table_attr_title_style = ParagraphStyle(
            name="TableTitleStyle",
            alignment=1,  # horizontal centering
            textColor=colors.white,
            parent=getSampleStyleSheet()["Normal"],
            fontSize=12,  # Set the desired font size here
            fontName="Helvetica-Bold",
        )
        self.report_table_style = TableStyle(
            [
                ("SPAN", (0, 0), (-1, 0)),  # Span the title row across all columns
                (
                    "BACKGROUND",
                    (0, 0),
                    (-1, 1),
                    (0.4, 0.4, 0.4),
                ),  # Gray background for header
                (
                    "TEXTCOLOR",
                    (0, 0),
                    (-1, 1),
                    (1, 1, 1),
                ),  # White text color for header
                ("ALIGN", (0, 0), (-1, -1), "CENTER"),
                ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),  # Vertically center all cells
                ("BOTTOMPADDING", (0, 0), (-1, 0), 12),  # Title padding
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("FONTSIZE", (0, 1), (-1, 0), 8),  # Rest of table font size
                (
                    "BACKGROUND",
                    (0, 2),
                    (-1, -1),
                    (0.9, 0.9, 0.9),
                ),  # Light gray background for data cells
                ("BOX", (0, 0), (-1, -1), 1, (0, 0, 0)),  # Black border for all cells
                (
                    "GRID",
                    (0, 0),
                    (-1, -1),
                    1,
                    (0, 0, 0),
                ),  # Black gridlines for all cells
            ]
        )

    def _validate_task(self, task_name: str) -> str:
        """Verify that a provided task_name has been evaluated.

        Args:
            task_name: The name of the task in question

        Return:
            The task name, if it has been validated.
        """
        if task_name not in self.valid_tasks:
            raise ValueError(
                f"Task: {task_name} has not yet been evaluated for this model: "
                f"{self.model_name}"
            )
        return task_name

    def validate_metric(self, metric_name: str) -> str:
        """Verify that a provided metric name is valid for the task.

        Args:
            metric_name: The name of the metric in question

        Return:
            The metric name if no ValueError is raised.
        """
        if metric_name.upper() in TASK_METRIC_LOOKUP_DICT[self.task_name]:
            return metric_name.upper()
        else:
            raise ValueError(
                f"{metric_name} is not a valid metric for task: {self.task_name}"
            )

    def get_available_metrics(self) -> List[str]:
        """Retrieve a list of metric names that have been evaluated.

        Return:
            A list of metric names that have been evaluated.
        """
        ir_files = [
            x
            for x in os.listdir(self.results_dir)
            if x.startswith("intersectional_results") and x.endswith(".json")
        ]
        metrics = [
            x.lstrip("intersectional_results_").rstrip(".json") for x in ir_files
        ]
        return metrics

    def get_metric_thresholds(self, metric_name: str) -> str | None:
        """Retrieve the thresholds used for a given metric in the evaluation.

        Args:
            metric_name: The name of the metric whose thresholds to lookup.

        Return:
            The list of thresholds or None if thresholds aren't used for this task.
        """
        if metric_name == "AR_OKS":
            threshold_file = os.path.join(
                self.results_dir, "detailed_results_oks_threshold.json"
            )
            threshold_results = read_json_file(threshold_file)
            return ", ".join(list(threshold_results.keys()))
        elif metric_name in ["AR_IOU", "AR_MASK"]:
            threshold_file = os.path.join(
                self.results_dir, "detailed_results_iou_threshold.json"
            )
            threshold_results = read_json_file(threshold_file)
            return ", ".join(list(threshold_results.keys()))
        elif metric_name == "PCK":
            threshold_file = os.path.join(self.results_dir, "pck_scores_threshold.json")
            threshold_results = read_json_file(threshold_file)
            return ", ".join([f"{x:.2f}" for x in threshold_results["thresholds"]])
        elif self.task_name == "body_parts_detection":
            threshold_file = os.path.join(
                self.results_dir,
                f"detailed_results_body_parts_{metric_name}_threshold.json",
            )
            threshold_results = read_json_file(threshold_file)
            return ", ".join(list(threshold_results.keys()))
        else:
            return None

    def get_attr_scores(
        self,
        intersectional_results: Union[
            Dict[str, Dict[str, int | float]],
            Dict[str, Dict[str, Dict[str, int | float]]],
        ],
        metric_name: str,
        attr_name: str,
        group_largest_regions: bool,
    ) -> Dict[
        str,
        List[str]
        | List[int]
        | List[float]
        | List[Dict[str, float]]
        | List[Dict[str, int | float]],
    ]:
        """Obtain all scores for individuals in a single attribute group.

        Also format the group names so they are more human readable.

        Args:
            intersectional_results: A dictionary containing the metric
                results in each intersectional group from the evaluation.
            metric_name: The name of the metric to group.
            attr_name: The attribute to aggregate over, e.g., "age"
            group_largest_regions: Whether to group nationality and ancestry
                at the continent level.

        Return:
            A dictionary containing the metric results for each class in the group
        """
        attr_needs_grouping = False
        if group_largest_regions and attr_name in ATTRIBUTE_CONSOLIDATION_DICT:
            attr_needs_grouping = True
            lg_regions = list(ATTRIBUTE_CONSOLIDATION_DICT[attr_name].keys())

        key = f"['{attr_name}']"
        grouped_data = intersectional_results[key]

        data = []
        if self.task_name == "body_parts_detection":
            # data = {attr_name: [], "Body Part": [], metric_name: [], "Class_Size": []}
            for elem, bp_dict in grouped_data.items():
                if attr_needs_grouping and eval(elem)[0] not in lg_regions:
                    continue
                formatted_elem = format_single_attribute_value(attr_name, elem)
                for bp, mdict in bp_dict.items():
                    scores = mdict["scores"]
                    for score in scores:
                        data.append(
                            {
                                attr_name: formatted_elem,
                                "Body Part": bp,
                                metric_name: score,
                            }
                        )
        else:
            for elem, mdict in grouped_data.items():
                if attr_needs_grouping and eval(elem)[0] not in lg_regions:
                    continue
                formatted_elem = format_single_attribute_value(attr_name, elem)
                if self.task_name == "face_verification":
                    data.append(
                        {
                            attr_name: formatted_elem,
                            metric_name: mdict[metric_name],
                            metric_name + "_Error": mdict[f"{metric_name}_Error"],
                            "class_size": mdict["Class_Size"],
                        }
                    )
                else:
                    scores = mdict["scores"]
                    for score in scores:
                        data.append({attr_name: formatted_elem, metric_name: score})

        return data

    def calculate_disparity(
        self,
        intersectional_results: Union[
            Dict[str, Dict[str, int | float]],
            Dict[str, Dict[str, Dict[str, int | float]]],
        ],
        attributes_to_consider: List[str],
        metric_name: str,
        group_largest_regions: bool = True,
        k: int = 0,
        min_group_size: int | None = 20,
    ) -> List[List[str | int | float]]:
        """Calculate disparity for each attribute combo.

        This function can be used to calculate disparity for a fixed combination number
        (k) or for all combination numbers. For example, if k = 1, then
        only calculate disparity for single attributes (no combinations).
        If k = 2, calculate for all bisections
        (e.g., age x pronoun, age x ancestry, ...).
        If attr_number_per_combo = 0, calculate for all combination numbers 1-N,
        where N is the number of attributes.

        Args:
            intersectional_results: A dictionary containing the metric
                results in each intersectional group from the evaluation.
            attributes_to_consider: The attributes (and their combinations)
                for which to calculate disparity.
            metric_name: The name of the metric for this disparity calculation
            group_largest_regions: Whether to group nationality and ancestry
                at the continent level.
            k: The number of attributes in each combination. If k=0,
                calculate disparity using all possible k's: k=1,2,..,N.
            min_group_size: Filter out group sizes smaller than this value.
                If None, allow all group sizes.

        Return:
            List of lists, where each sublist contains:
            - attr_key (str)
            - body_part if relevant to the task (str)
            - worst_group (str)
            - worst_class_size (int),
            - best_group (str)
            - best_class_size (int)
            - disparity (float)
            - p_value (float)
            - effect_size (float)
        """
        # Make sure attributes to consider is validated and ordered correctly
        attributes_to_consider = validate_attributes(
            self.dataset_base, attributes_to_consider
        )
        results = []
        if k == 0:
            total = int(2 ** len(attributes_to_consider)) - 1
            r_lower = 1
            r_upper = len(attributes_to_consider) + 1
        else:
            n = len(attributes_to_consider)
            total = math.factorial(n) // (math.factorial(k) * math.factorial(n - k))
            r_lower = k
            r_upper = k + 1

        with tqdm(
            total=total,
            desc="Calculating highest disparity attributes.",
        ) as pbar:
            for r in range(r_lower, r_upper):
                combinations = itertools.combinations(attributes_to_consider, r)
                for combo in combinations:
                    attr_key = str(list(combo))
                    attr_dict = intersectional_results[attr_key]
                    if group_largest_regions:
                        # Find indices of attrs in combo that need to be grouped
                        ixs_to_group = [
                            ii
                            for ii in range(len(combo))
                            if combo[ii] in ATTRIBUTE_CONSOLIDATION_DICT.keys()
                        ]
                    if self.task_name == "body_parts_detection":
                        # Make data dict mapping class name to metric scores
                        # for each member of the class and body part
                        # e.g., "['0. She/her/hers', '5. Africa']":
                        # {"Face": [0.88,0.94,0.51], "Hand": [0.58,0.89]}
                        for bp in self.evaluated_body_parts:
                            data = {}
                            for attr_val, bp_dict in attr_dict.items():
                                if group_largest_regions and ixs_to_group != []:
                                    if any(
                                        [
                                            eval(attr_val)[ix]
                                            not in ATTRIBUTE_CONSOLIDATION_DICT[
                                                combo[ix]
                                            ]
                                            for ix in ixs_to_group
                                        ]
                                    ):
                                        continue
                                class_size = bp_dict[bp]["Class_Size"]
                                # Filter out small (and zero sized) groups
                                if (
                                    min_group_size is not None
                                    and class_size < min_group_size
                                ):
                                    continue
                                if class_size > 0:
                                    data[attr_val] = bp_dict[bp]["scores"]

                            if len(data) < 2:
                                pairs = None
                            else:
                                pairs = find_significant_pairs(
                                    data, alpha=0.05, statistic="U"
                                )

                            if pairs is None:
                                continue
                            for pair in pairs:
                                result = [attr_key, bp]
                                result.extend(pair)
                                results.append(result)
                    else:
                        # Make data dict mapping class name to metric scores
                        # for each member of the class
                        # e.g., "['0. She/her/hers', '5. Africa']": [0.88,0.94,0.51]
                        data = {}
                        for attr_val, val_dict in attr_dict.items():
                            if group_largest_regions and ixs_to_group != []:
                                if any(
                                    [
                                        eval(attr_val)[ix]
                                        not in ATTRIBUTE_CONSOLIDATION_DICT[combo[ix]]
                                        for ix in ixs_to_group
                                    ]
                                ):
                                    continue
                            # Filter out small groups
                            if (
                                min_group_size is not None
                                and len(val_dict["scores"]) < min_group_size
                            ):
                                continue
                            data[attr_val] = val_dict["scores"]
                        # Find the best,worst pairs using U-test
                        if len(data) < 2:
                            pairs = None
                        else:
                            pairs = find_significant_pairs(
                                data, alpha=0.05, statistic="U"
                            )
                        if pairs is None:
                            pbar.update(1)
                            continue
                        for pair in pairs:
                            result = [attr_key]
                            result.extend(pair)
                            results.append(result)
                    pbar.update(1)
        # return result_dict
        return results

    def generate_pdf_report(
        self,
        attributes: List[str],
        group_largest_regions: bool = True,
        show_significance_on_plots: bool = True,
        report_savename: str = "bias_report.pdf",
    ) -> SimpleDocTemplate:
        """Create a complete PDF bias report and save it to disk.

        The PDF report includes:
          - Description of the task, dataset, metrics, and attributes used
          - Plots of the metric value aggregated over
            each individual attribute.
          - Table of the attributes with the largest disparity
            for each attribute and combination of attributes

        Args:
            attributes: A list of string attributes to include in the report.
                Must be a subset of the attributes that were evaluated
                for this task. Defaults to empty list, which uses all
                available attributes.
            group_largest_regions: Whether to group ancestry and nationality
                at the continent level.
            show_significance_on_plots: Whether to show stat sig pairs
                on metric vs. attribute plots (brackets + asterisks).
            report_savename: A custom filename. This is a relative filename.
            The report will always be saved in self.results_dir.

        Return:
            The pdf document.
        """
        logging.info("Generating bias report...")
        attributes = validate_attributes(self.dataset_base, attributes)
        if len(attributes) > 8 and self.task_name != "face_verification":
            logging.warning(
                "Generating the bias report with 8 or more attributes "
                "can be computationally expensive and may take a few hours, "
                "depending on the number of available cpu cores."
            )
        if self.task_name == "face_verification" and show_significance_on_plots:
            logging.warning(
                "Showing significance on plots is not supported "
                "for the face verification task."
            )
            show_significance_on_plots = False
        report_dir = os.path.join(self.results_dir, "bias_report")
        os.makedirs(report_dir, exist_ok=True)
        report_savename = os.path.join(report_dir, report_savename)
        doc = SimpleDocTemplate(report_savename, pagesize=letter)

        # container for all content
        content = []

        # Title
        title = "<center>FHIBE Bias Evaluation Report</center><br/>"
        content.append(Paragraph(title, getSampleStyleSheet()["Title"]))

        # Subtitle
        hr_task_name = HUMAN_READABLE_TASK_DICT[self.task_name]
        today = datetime.today()
        formatted_date = today.strftime("%B %d, %Y")
        subtitle = (
            f"<center>Task: {hr_task_name}</center><br/>"
            f"<center>Model: {self.model_name}</center><br/>"
            f"<center>Dataset version: {DATASET_VERSION}</center><br/>"
        )
        date = f"<center>Reported generated on {formatted_date}</center>"
        content.append(Paragraph(subtitle, self.report_subtitle_style))
        content.append(Paragraph(date, self.report_date_style))

        # Task description
        task_descr = TASK_DESCRIPTION_DICT[self.task_name]
        task_heading = "Task Description"
        content.append(Paragraph(task_heading, getSampleStyleSheet()["Heading1"]))
        content.append(Paragraph(task_descr, getSampleStyleSheet()["BodyText"]))
        content.append(Spacer(1, 12))

        # Dataset description
        dataset_heading = "Evaluation Dataset"
        dataset_desc = DATASET_DESCRIPTION_DICT[self.dataset_name]
        univ_metadata_desc = METADATA_UNIVERSAL_DESCRIPTION
        task_metadata_desc = TASK_METADATA_DESCRIPTION_DICT[self.task_name]

        content.append(Paragraph(dataset_heading, getSampleStyleSheet()["Heading1"]))
        content.append(Paragraph(dataset_desc, getSampleStyleSheet()["BodyText"]))
        content.append(Paragraph(univ_metadata_desc, getSampleStyleSheet()["BodyText"]))
        if task_metadata_desc is not None:
            content.append(
                Paragraph(task_metadata_desc, getSampleStyleSheet()["BodyText"])
            )

        if self.task_name == "body_parts_detection":
            body_parts_desc = (
                "The following body part annotations were used "
                "in the evaluation of this model: "
            )
            body_parts_desc += ", ".join(self.evaluated_body_parts)
            content.append(
                Paragraph(body_parts_desc, getSampleStyleSheet()["BodyText"])
            )
        # Attributes description
        demography_ts = (
            "The following attributes were used to aggregate results in this report. "
            "For descriptions and definitions of these "
            "attributes, see the supplement. "
        )
        content.append(Paragraph(demography_ts, getSampleStyleSheet()["BodyText"]))
        attributes_list = "Attributes: "
        attributes_list += ", ".join(attributes)
        content.append(Paragraph(attributes_list, getSampleStyleSheet()["BodyText"]))

        # Results
        content.append(PageBreak())
        results_heading = "Evaluation Results"
        content.append(Paragraph(results_heading, getSampleStyleSheet()["Heading1"]))
        metrics = self.get_available_metrics()
        metric_overview_text = (
            "Here we report the results for each metric that was evaluated. "
        )
        content.append(
            Paragraph(metric_overview_text, getSampleStyleSheet()["BodyText"])
        )

        metric_index = 1
        for metric_name in metrics:
            intersectional_results_json_file = os.path.join(
                self.results_dir, f"intersectional_results_{metric_name}.json"
            )
            intersectional_results = read_json_file(intersectional_results_json_file)

            metric_title = METRIC_DESCRIPTION_DICT[metric_name]["title"]
            metric_desc = METRIC_DESCRIPTION_DICT[metric_name]["description"]
            metric_heading = f"Metric {metric_index}. {metric_name}: {metric_title}"

            content.append(Paragraph(metric_heading, getSampleStyleSheet()["Heading2"]))
            content.append(Paragraph("Description:", getSampleStyleSheet()["Heading3"]))
            content.append(Paragraph(metric_desc, getSampleStyleSheet()["BodyText"]))

            # If custom keypoints were used, add them to the report
            if self.task_name == "keypoint_estimation":
                if intersectional_results.get("custom_keypoints") is not None:
                    keypoints_used = intersectional_results["custom_keypoints"]
                else:
                    from fhibe_eval_api.datasets.fhibe import FHIBE_COMMON_KEYPOINTS

                    keypoints_used = FHIBE_COMMON_KEYPOINTS
                kp_desc = (
                    "The following keypoints were used in the evaluation: "
                    f"{', '.join(keypoints_used)}"
                )
                content.append(Paragraph(kp_desc, getSampleStyleSheet()["BodyText"]))
            # Get thresholds to display
            thresholds = self.get_metric_thresholds(metric_name)
            if thresholds is not None:
                content.append(
                    Paragraph(
                        f"Thresholds: {thresholds}", getSampleStyleSheet()["BodyText"]
                    )
                )
            content.append(Paragraph("Results:", getSampleStyleSheet()["Heading3"]))

            # threshold plots
            if metric_name in ["AR_IOU", "AR_MASK"]:
                plot_desc = "IoU vs. threshold curve calculated over all images"
                content.append(Paragraph(plot_desc, getSampleStyleSheet()["Heading4"]))
                savename = os.path.join(report_dir, "iou_thresh.png")
                _ = self.plot_iou_vs_threshold(savefig=True, fig_savename=savename)
                image = Image(savename, width=400, height=300)
                content.append(image)
            elif metric_name == "AR_OKS":
                plot_desc = "OKS vs. threshold curve calculated over all images"
                content.append(Paragraph(plot_desc, getSampleStyleSheet()["Heading4"]))
                savename = os.path.join(report_dir, "oks_thresh.png")
                _ = self.plot_oks_vs_threshold(savefig=True, fig_savename=savename)
                image = Image(savename, width=400, height=300)
                content.append(image)
            elif metric_name == "PCK":
                plot_desc = "PCK vs. threshold curve calculated over all images"
                content.append(Paragraph(plot_desc, getSampleStyleSheet()["Heading4"]))
                savename = os.path.join(report_dir, "pck_thresh.png")
                _ = self.plot_pck_vs_threshold(savefig=True, fig_savename=savename)
                image = Image(savename, width=400, height=300)
                content.append(image)

            # metric in individual demographic groups plots
            plot_desc = "Metric performance aggregated in demographic groups"
            content.append(Paragraph(plot_desc, getSampleStyleSheet()["Heading4"]))
            plot_note1 = "Bar plots show mean and 68% confidence intervals. "
            if self.task_name != "body_parts_detection":
                plot_note1 += (
                    "Group sizes are included in parentheses in the x-axis labels.\n"
                )
            content.append(
                Paragraph(
                    plot_note1,
                    getSampleStyleSheet()["BodyText"],
                )
            )
            if self.task_name not in ("face_verification", "body_parts_detection"):
                plot_note2 = (
                    "Significant differences are indicated with brackets and "
                    "asterisks, where * indicates p<0.05/m, ** indicates p<0.01/m, "
                    "and *** indicates p<0.001/m, where p is the p-value and "
                    "and m is the number of attribute pairs tested for significance "
                    "(Bonferroni correction). Groups with size < 20 "
                    "are not included in significance calculations.\n"
                    "Note that a significant difference does not necessarily "
                    "mean a large difference. See the disparity table below for the "
                    "largest disparities in metric performance across attribute. \n"
                    "Note also that if a difference is not labeled as significant, "
                    "it can still be a large difference worth investigating, "
                    "especially if the group sizes are small. "
                )
                content.append(
                    Paragraph(
                        plot_note2,
                        getSampleStyleSheet()["BodyText"],
                    )
                )
            for attr_name in attributes:
                savename = os.path.join(
                    report_dir, f"intersectional_results_{metric_name}_{attr_name}.png"
                )
                self.plot_metric_by_intersectional_group(
                    metric_name=metric_name,
                    attr_name=attr_name,
                    savefig=True,
                    fig_savename=savename,
                    intersectional_results=intersectional_results,
                    group_largest_regions=group_largest_regions,
                    show_medians=False,
                    show_significance=show_significance_on_plots,
                )
                image = Image(savename, width=400, height=300)
                content.append(image)

            if self.task_name != "face_verification":
                # Tables of best/worst performance for each attribute
                # and combination of attributes
                table_overview_desc = (
                    "Table of highest disparity in metric performance "
                    "across all attribute intersections"
                )
                content.append(PageBreak())
                content.append(
                    Paragraph(table_overview_desc, getSampleStyleSheet()["Heading4"])
                )
                disparity_text = (
                    "Disparity is defined as 1 - ( AVG(worst group) / AVG(best group) )"
                )
                content.append(
                    Paragraph(
                        disparity_text,
                        getSampleStyleSheet()["Normal"],
                    )
                )
                content.append(Spacer(2, 6))
                significance_text = (
                    "Only statistically significant results are shown. "
                    "Significance is determined via p < alpha/m, where p "
                    "is the p value, alpha=0.05, and m is the number of attribute "
                    "pair for a single attribute (or attribute combination). "
                )

                table = self.make_disparity_table(
                    intersectional_results,
                    attributes,
                    metric_name,
                    group_largest_regions,
                )
                if table is not None:
                    content.append(
                        Paragraph(
                            significance_text,
                            getSampleStyleSheet()["Normal"],
                        )
                    )
                    content.append(Spacer(2, 12))
                    content.append(table)
                else:
                    content.append(
                        Paragraph(
                            "There were no statistically significant disparities "
                            "in the model performance for this metric.",
                            getSampleStyleSheet()["Normal"],
                        )
                    )
                content.append(Spacer(2, 12))
                metric_index += 1

        # Supplement
        content.append(PageBreak())
        supp_heading = "Supplement"
        supp_desc = (
            "In this section, we describe the attributes "
            "that were used to aggregate results in this report. "
        )
        if group_largest_regions is True:
            if "ancestry" in attributes or "nationality" in attributes:
                supp_desc += (
                    "The following attributes were grouped at the continent level: "
                )
            attrs_to_group = [x for x in ["ancestry", "nationality"] if x in attributes]
            supp_desc += ", ".join(attrs_to_group)
            supp_desc += "."
        item_list = [
            ListItem(
                Paragraph(
                    ATTRIBUTES_DESCRIPTION_DICT[attr], getSampleStyleSheet()["BodyText"]
                ),
                bulletColor="black",
            )
            for attr in attributes
        ]
        attr_text = ListFlowable(item_list, bulletType="bullet")

        content.append(Paragraph(supp_heading, getSampleStyleSheet()["Heading1"]))
        content.append(Paragraph(supp_desc, getSampleStyleSheet()["BodyText"]))
        content.append(attr_text)
        # Build the PDF document
        doc.build(content)
        logging.info(f"Bias report complete. Saved PDF to: {report_savename}")
        return doc

    def make_disparity_table(
        self,
        intersectional_results: Union[
            Dict[str, Dict[str, int | float]],
            Dict[str, Dict[str, Dict[str, int | float]]],
        ],
        attributes_to_consider: List[str],
        metric_name: str,
        group_largest_regions: bool,
    ) -> Table:
        """Create a table containing of highest disparity attributes.

        The table includes the top 50 single attributes, i.e.,
        no intersections. Restrict minimum group size to be 20.

        Args:
            intersectional_results: A dictionary containing the metric
                results in each intersectional group from the evaluation.
            attributes_to_consider: List of attributes for which to
                calculate disparities. Enables subsetting the results.
            metric_name: The name of the metric to use for disparity calculation.
            group_largest_regions: Whether to group nationality and ancestry
                at the continent level.

        Return:
            A reportlab table to be included in the bias report pdf.
        """
        # First get the k=1 (single attribute) disparities
        disparity_list = self.calculate_disparity(
            intersectional_results,
            attributes_to_consider=attributes_to_consider,
            metric_name=metric_name,
            group_largest_regions=group_largest_regions,
            k=1,
            min_group_size=20,
        )

        # Sort by disparity from highest to lowest and take n highest disparities
        n = 50
        highest_disparity_list = sorted(
            disparity_list, key=lambda pair: pair[-3], reverse=True
        )[0:n]
        data_list = []
        if self.task_name == "body_parts_detection":
            header_row = [
                Paragraph("Attribute(s)", self.table_attr_header_style),
                Paragraph("Body Part", self.table_attr_header_style),
                Paragraph("Worst group", self.table_attr_header_style),
                Paragraph(
                    "Worst mean/median (class size)", self.table_attr_header_style
                ),
                Paragraph("Best group", self.table_attr_header_style),
                Paragraph(
                    "Best mean/median (class size)", self.table_attr_header_style
                ),
                Paragraph("Disparity", self.table_attr_header_style),
                Paragraph("P value", self.table_attr_header_style),
                Paragraph("Effect size", self.table_attr_header_style),
            ]

            for entry in highest_disparity_list:
                (
                    attr_key,
                    bp,
                    worst_group,
                    worst_class_size,
                    best_group,
                    best_class_size,
                    disparity,
                    p_value,
                    effect_size,
                ) = entry
                attr_list = eval(attr_key)

                worst_performance = intersectional_results[attr_key][worst_group][bp][
                    metric_name
                ]
                worst_median = np.nanmedian(
                    intersectional_results[attr_key][worst_group][bp]["scores"]
                )
                best_performance = intersectional_results[attr_key][best_group][bp][
                    metric_name
                ]
                best_median = np.nanmedian(
                    intersectional_results[attr_key][best_group][bp]["scores"]
                )

                # Reformat the attribute values to be more readable
                if len(attr_list) == 1:
                    attr_str = attr_list[0]
                    worst_group_str = format_single_attribute_value(
                        attr_str, worst_group
                    )
                    best_group_str = format_single_attribute_value(attr_str, best_group)
                    attr_str = f"&#8226; {attr_str}"
                    worst_group_str = f"&#8226; {worst_group_str}"
                    best_group_str = f"&#8226; {best_group_str}"
                else:
                    attr_str = attr_key
                    attr_list = eval(attr_str)
                    attr_str = "<br/>".join([f"&#8226; {item}" for item in attr_list])
                    worst_group_list = eval(worst_group)
                    best_group_list = eval(best_group)
                    worst_group_list = format_attribute_list(
                        attr_list, worst_group_list
                    )
                    best_group_list = format_attribute_list(attr_list, best_group_list)
                    worst_group_str = "<br/>".join(
                        [f"&#8226; {item}" for item in worst_group_list]
                    )
                    best_group_str = "<br/>".join(
                        [f"&#8226; {item}" for item in best_group_list]
                    )
                p_value_str = "< 0.001" if p_value < 0.001 else f"{p_value:.3f}"
                worst_perf_str = (
                    f"{worst_performance:.2f}/{worst_median:.2f} "
                    f"({worst_class_size})"
                )
                row = [
                    Paragraph(attr_str, self.table_attr_val_style),
                    Paragraph(bp, self.table_attr_val_style),
                    Paragraph(worst_group_str, self.table_attr_val_style),
                    Paragraph(worst_perf_str, self.table_attr_val_style),
                    Paragraph(best_group_str, self.table_attr_val_style),
                    Paragraph(
                        f"{best_performance:.2f}/{best_median:.2f} ({best_class_size})",
                        self.table_attr_val_style,
                    ),
                    Paragraph(f"{disparity:.2f}", self.table_attr_val_style),
                    Paragraph(p_value_str, self.table_attr_val_style),
                    Paragraph(f"{abs(effect_size):.2f}", self.table_attr_val_style),
                ]

                data_list.append(row)
        else:
            header_row = [
                Paragraph("Attribute(s)", self.table_attr_header_style),
                Paragraph("Worst group", self.table_attr_header_style),
                Paragraph(
                    "Worst mean/median (class size)", self.table_attr_header_style
                ),
                Paragraph("Best group", self.table_attr_header_style),
                Paragraph(
                    "Best mean/median (class size)", self.table_attr_header_style
                ),
                Paragraph("Mean/median disparity", self.table_attr_header_style),
                Paragraph("P value", self.table_attr_header_style),
                Paragraph("Effect size", self.table_attr_header_style),
            ]
            for entry in highest_disparity_list:
                (
                    attr_key,
                    worst_group,
                    worst_class_size,
                    best_group,
                    best_class_size,
                    disparity,
                    p_value,
                    effect_size,
                ) = entry
                attr_list = eval(attr_key)
                worst_median = np.nanmedian(
                    intersectional_results[attr_key][worst_group]["scores"]
                )
                worst_performance = intersectional_results[attr_key][worst_group][
                    metric_name
                ]
                best_median = np.nanmedian(
                    intersectional_results[attr_key][best_group]["scores"]
                )
                best_performance = intersectional_results[attr_key][best_group][
                    metric_name
                ]
                mean_disparity = 1 - worst_performance / best_performance

                # Reformat the attribute values to be more readable
                if len(attr_list) == 1:
                    attr_str = format_attribute_name(attr_list[0])
                    worst_group_str = format_single_attribute_value(
                        attr_list[0].lower(), worst_group
                    )
                    best_group_str = format_single_attribute_value(
                        attr_list[0].lower(), best_group
                    )
                    attr_str = f"&#8226; {attr_str}"
                    worst_group_str = f"&#8226; {worst_group_str}"
                    best_group_str = f"&#8226; {best_group_str}"
                else:
                    attr_key
                    attr_list = eval(attr_key)
                    attr_str = "<br/>".join(
                        [f"&#8226; {format_attribute_name(item)}" for item in attr_list]
                    )
                    worst_group_list = eval(worst_group)
                    best_group_list = eval(best_group)
                    worst_group_list = format_attribute_list(
                        attr_list, worst_group_list
                    )
                    best_group_list = format_attribute_list(attr_list, best_group_list)
                    worst_group_str = "<br/>".join(
                        [f"&#8226; {item}" for item in worst_group_list]
                    )
                    best_group_str = "<br/>".join(
                        [f"&#8226; {item}" for item in best_group_list]
                    )

                p_value_str = "< 0.001" if p_value < 0.001 else f"{p_value:.3f}"
                worst_perf_str = (
                    f"{worst_performance:.2f}/{worst_median:.2f} ({worst_class_size})"
                )
                row = [
                    Paragraph(attr_str, self.table_attr_val_style),
                    Paragraph(worst_group_str, self.table_attr_val_style),
                    Paragraph(worst_perf_str, self.table_attr_val_style),
                    Paragraph(best_group_str, self.table_attr_val_style),
                    Paragraph(
                        f"{best_performance:.2f}/{best_median:.2f} ({best_class_size})",
                        self.table_attr_val_style,
                    ),
                    Paragraph(
                        f"{mean_disparity:.2f} / {disparity:.2f}",
                        self.table_attr_val_style,
                    ),
                    Paragraph(p_value_str, self.table_attr_val_style),
                    Paragraph(f"{abs(effect_size):.2f}", self.table_attr_val_style),
                ]

                data_list.append(row)
        if len(data_list) == 0:  # only the header
            return None
        table_data = []
        title_row_text = (
            f"Attributes with highest disparities in {metric_name} performance"
        )
        title_row = [Paragraph(title_row_text, self.table_attr_title_style)]
        table_data.append(title_row)
        table_data.append(header_row)
        table_data.extend(data_list)
        # Last three columns only need 150 pixels width total
        if self.task_name == "body_parts_detection":
            col_widths = [80, 50, 75, 75, 75, 75, 55, 50, 50]
        else:
            col_widths = [80, 75, 75, 75, 75, 75, 50, 50]
        table = Table(table_data, colWidths=col_widths)
        table.setStyle(self.report_table_style)
        return table

    def plot_iou_vs_threshold(
        self,
        savefig: bool = False,
        fig_savename: str | None = None,
    ) -> matplotlib.figure.Figure:
        """Plot mean iou vs. threshold over all images in evaluation.

        Args:
            savefig: Whether to save the figure to disk
            fig_savename: The filename to save to disk

        Return:
            The matplotlib figure object.
        """
        if self.task_name not in [
            "person_localization",
            "person_parsing",
            "face_localization",
        ]:
            raise RuntimeError(f"IoU cannot be plotted for task: {self.task_name}")

        # Look for detailed iou file with these thresholds
        detailed_results_file = os.path.join(
            self.results_dir, "detailed_results_iou_threshold.json"
        )
        results_from_file = read_json_file(detailed_results_file)
        y_label = "Fraction of images with IoU >= threshold"
        x_label = "Threshold"
        formatted_results = format_threshold_results(
            results_from_file, metric_name="iou"
        )
        # Assumes results is a dict where keys are threshold values (float)
        # and values are the fraction fo images where metric >= threshold
        x = []
        y = []
        for thr in sorted(formatted_results.keys()):
            x.append(thr)
            y.append(formatted_results[thr])
        # Set the seaborn style with gridlines
        fig, ax = plt.subplots()
        sns.set_style("darkgrid")

        # Plot the line and points
        sns.lineplot(x=x, y=y, marker="o", ax=ax)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.ylim(-0.05, 1.05)
        if savefig and fig_savename:
            plt.savefig(fig_savename, format=fig_savename.split(".")[-1])
            logging.info(f"Saved {fig_savename}")
        return fig

    def plot_oks_vs_threshold(
        self,
        savefig: bool = False,
        fig_savename: str | None = None,
    ) -> matplotlib.figure.Figure:
        """Plot mean object keypoint similarity (OKS) vs. threshold in evaluation.

        Args:
            savefig: Whether to save the figure to disk
            fig_savename: The filename to save to disk

        Return:
            The matplotlib figure object.
        """
        if self.task_name not in ["keypoint_estimation"]:
            raise RuntimeError("OKS is only available for task: keypoint_estimation")

        # Look for detailed oks file with these thresholds
        detailed_results_file = os.path.join(
            self.results_dir, "detailed_results_oks_threshold.json"
        )
        results_from_file = read_json_file(detailed_results_file)
        y_label = "Fraction of images with OKS >= threshold"
        x_label = "Threshold"
        formatted_results = format_threshold_results(
            results_from_file, metric_name="oks"
        )
        # Assumes results is a dict where keys are threshold values (float)
        # and values are the fraction fo images where metric >= threshold
        x = []
        y = []
        for thr in sorted(formatted_results.keys()):
            x.append(thr)
            y.append(formatted_results[thr])

        fig = plt.figure()

        # Set the seaborn style with gridlines
        sns.set_style("darkgrid")

        # Plot the line and points
        sns.lineplot(x=x, y=y, marker="o")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.ylim(-0.05, 1.05)
        if savefig and fig_savename:
            plt.savefig(fig_savename, format=fig_savename.split(".")[-1])
            logging.info(f"Saved {fig_savename}")
        return fig

    def plot_pck_vs_threshold(
        self,
        savefig: bool = False,
        fig_savename: str | None = None,
    ) -> matplotlib.figure.Figure:
        """Plot mean percentage correct keypoints (PCK) vs. threshold in evaluation.

        Args:
            savefig: Whether to save the figure to disk
            fig_savename: The filename to save to disk

        Return:
            The matplotlib figure object.
        """
        if self.task_name != "keypoint_estimation":
            raise RuntimeError("PCK is only available for task: keypoint_estimation.")

        # Look for PCK file with these thresholds
        detailed_results_file = os.path.join(
            self.results_dir, "pck_scores_threshold.json"
        )
        results_from_file = read_json_file(detailed_results_file)

        y_label = "Percentage correct keypoints (pck)"
        x_label = "Threshold"
        formatted_results = format_threshold_results(
            results_from_file, metric_name="pck"
        )
        # Assumes results is a dict where keys are threshold values (float)
        # and values are the fraction fo images where metric >= threshold
        x = []
        y = []
        for thr in sorted(formatted_results.keys()):
            x.append(thr)
            y.append(formatted_results[thr])

        fig = plt.figure()
        # Set the seaborn style with gridlines
        sns.set_style("darkgrid")

        # Plot the line and points
        sns.lineplot(x=x, y=y, marker="o")
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.ylim(-0.05, 1.05)
        if savefig and fig_savename:
            plt.savefig(fig_savename, format=fig_savename.split(".")[-1])
            logging.info(f"Saved {fig_savename}")
        return fig

    def plot_metric_by_intersectional_group(
        self,
        metric_name: str,
        attr_name: str,
        intersectional_results: Dict | None = None,
        group_largest_regions: bool = True,
        show_medians: bool = False,
        show_significance: bool = False,
        savefig: bool = False,
        fig_savename: str | None = None,
    ) -> matplotlib.figure.Figure:
        """Make bar plot of metric values in different group classes.

        Shows mean with 68% confidence intervals, and optionally median.
        Uses seaborn's bootstrapping method for estimating confidence intervals.

        Args:
            metric_name: The name of the metric to plot
            attr_name: The name of the attribute to plot,
                e.g., "pronoun"
            intersectional_results: Pre-loaded dictionary containing
                the intersectional results. If None, these are loaded
                from the file.
            group_largest_regions: Whether to group nationality and ancestry
                at the continent level.
            show_medians: Whether to show the medians on the bar plot,
                defaults to False.
            show_significance: Whether to show statistically significant
                disparities as brackets with asterisks.
            savefig: Whether to save the figure
            fig_savename: The filename to save
        Return:
            The matplotlib figure of the plot
        """
        metric_name = self.validate_metric(metric_name)
        _ = validate_attributes(self.dataset_base, [attr_name])

        if intersectional_results is None:
            intersectional_results = read_json_file(
                os.path.join(
                    self.results_dir, f"intersectional_results_{metric_name}.json"
                )
            )
        data = self.get_attr_scores(
            intersectional_results, metric_name, attr_name, group_largest_regions
        )
        df = pd.DataFrame(data)
        sns.set_style("darkgrid", rc={"xtick.bottom": True, "ytick.left": False})
        fig, ax = plt.subplots(figsize=(8, 6))
        if self.task_name == "body_parts_detection":
            hue = "Body Part"
        else:
            hue = attr_name
        palette = "Blues"

        if (
            attr_name in ["apparent_skin_color", "natural_skin_color"]
            and self.task_name != "body_parts_detection"
        ):
            # Remap skin color to Fitzpatrick type
            rgb_tuples = [
                FITZPATRICK_RGB_DICT[sc_str] for sc_str in df[attr_name].unique()
            ]
            hex_colors = ["#%02x%02x%02x" % rgb for rgb in rgb_tuples]
            palette = hex_colors

        if self.task_name == "face_verification":
            g = sns.barplot(
                x=attr_name,
                y=metric_name,
                data=df,
                palette=palette,
                hue=hue,
                ax=ax,
                errorbar=None,
            )
            # Add the error bars manually
            for bar, err in zip(g.patches, df["VAL_Error"]):
                # Get the center of the bar and its height
                bar_x = bar.get_x() + bar.get_width() / 2
                bar_height = bar.get_height()

                # Add the error bar
                ax.errorbar(
                    bar_x, bar_height, yerr=err, fmt="none", c="black", capsize=3
                )

        else:
            g = sns.barplot(
                x=attr_name,
                y=metric_name,
                data=df,
                estimator="mean",
                palette=palette,
                hue=hue,
                ax=ax,
                errorbar=("ci", 68),
                capsize=0.1,  # Adds caps to the error bars
                err_kws={"linewidth": 1},
            )

            if show_medians:
                # Overlay medians
                medians = df.groupby([attr_name])[metric_name].median().reset_index()
                for i, group in enumerate(df[attr_name].unique()):
                    data = medians[medians[attr_name] == group]
                    ax.scatter(
                        x=data[attr_name],
                        y=data[metric_name],
                        color="black",
                        marker="o",
                        edgecolor="black",
                        facecolors="none",
                    )

        if self.task_name == "body_parts_detection":
            sns.move_legend(g, loc="center left", bbox_to_anchor=(1.02, 0.5))

        # Optionally show significances as asterisks and brackets
        if show_significance is True and self.task_name not in [
            "body_parts_detection",
            "face_verification",
        ]:
            disparity_list = self.calculate_disparity(
                intersectional_results,
                attributes_to_consider=[attr_name],
                metric_name=metric_name,
                group_largest_regions=group_largest_regions,
                k=1,
                min_group_size=20,
            )
            y_max = ax.get_ylim()[1]
            x_coords = {}
            for i, container in enumerate(ax.containers):
                for j, bar in enumerate(container):
                    ax_text = ax.get_xticklabels()[i + j].get_text()
                    x_coords[ax_text] = bar.get_x() + bar.get_width() / 2

            # Set the bracket parameters
            bracket_height = 0.05 * y_max
            text_height = 0.05 * y_max

            # Add stat sig annotations
            for i, entry in enumerate(disparity_list):
                (
                    attr_key,
                    worst_group,
                    worst_class_size,
                    best_group,
                    best_class_size,
                    disparity,
                    p_value,
                    effect_size,
                ) = entry
                # Add significance asterisks
                # Use bonferroni correction, i.e. p < alpha/m, where m is
                # number of tests run
                m = len(disparity_list)
                if p_value < 0.05 / m:
                    worst_group_str = format_single_attribute_value(
                        attr_name, worst_group
                    )
                    best_group_str = format_single_attribute_value(
                        attr_name, best_group
                    )
                    x1, x2 = x_coords[worst_group_str], x_coords[best_group_str]
                    y = y_max + (i * bracket_height * 1.5)

                    # Draw the bracket
                    ax.plot(
                        [x1, x1, x2, x2],
                        [y, y + bracket_height, y + bracket_height, y],
                        color="black",
                        linewidth=1.5,
                    )
                    if p_value < 0.001 / m:
                        sig_text = "***"
                    elif p_value < 0.01 / m:
                        sig_text = "**"
                    else:
                        sig_text = "*"

                    ax.text(
                        (x1 + x2) / 2,
                        y + text_height,
                        sig_text,
                        ha="center",
                        va="bottom",
                    )
        # Add class size to x label
        class_size_dict = {}
        if self.task_name != "body_parts_detection":
            if self.task_name == "face_verification":
                class_size_dict = df.set_index(attr_name)["class_size"].to_dict()
            else:
                class_size_dict = df[attr_name].value_counts().to_dict()

        # Get current x-tick labels
        labels = []
        xtick_labels = ax.get_xticklabels()
        for ii in range(len(xtick_labels)):
            label = xtick_labels[ii].get_text()
            # Add class size
            if self.task_name != "body_parts_detection":
                label += f" ({class_size_dict[label]})"
            labels.append(label)

        # Wrap the labels
        wrapped_labels = [textwrap.fill(label, 30) for label in labels]

        # Set the new labels
        ax.xaxis.set_major_locator(FixedLocator(range(len(labels))))
        ax.xaxis.set_major_formatter(FixedFormatter(wrapped_labels))
        for label in ax.get_xticklabels():
            if attr_name not in [
                "apparent_skin_color",
                "natural_skin_color",
                "apparent_skin_color_hue_lum",
                "age",
                "camera_position",
                "user_hour_captured",
            ]:
                label.set_rotation(90)
            label.set_ha("center")  # Adjust horizontal alignment

        ax.tick_params(axis="x", labelsize=12)  # X-axis tick labels
        ax.tick_params(axis="y", labelsize=12)  # Y-axis tick labels
        ax.set_xlabel(format_attribute_name(attr_name), fontsize=12)  # X-axis label
        ax.set_ylabel(metric_name, fontsize=12)  # Y-axis label
        ax.set_ylim(0, ax.get_ylim()[1])
        plt.tight_layout()
        if savefig and fig_savename:
            plt.savefig(fig_savename, format=fig_savename.split(".")[-1])
            logging.info(f"Saved {fig_savename}")
        return fig
