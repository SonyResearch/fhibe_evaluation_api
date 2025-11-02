# SPDX-License-Identifier: Apache-2.0
import glob
import os

import pytest

from fhibe_eval_api.common.utils import read_json_file
from fhibe_eval_api.reporting.reporting import BiasReport

CURRENT_DIR = os.path.dirname(__file__)


def test_bias_report(bias_report_fixture, demo_model_fixture):
    # Person detection
    task_name = "person_localization"
    _, model_name = demo_model_fixture(task_name)
    results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
        "ground_truth",
    )
    report = bias_report_fixture(task_name, results_dir)
    assert report.dataset_name == "fhibe_downsampled"
    assert report.dataset_base == "fhibe"
    assert task_name in report.valid_tasks

    # Body parts detection
    task_name = "body_parts_detection"
    _, model_name = demo_model_fixture(task_name)
    results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
        "ground_truth",
    )
    report = bias_report_fixture(task_name, results_dir)
    assert report.dataset_name == "fhibe_downsampled"
    assert report.dataset_base == "fhibe"
    assert task_name in report.valid_tasks

    # Face encoding
    task_name = "face_encoding"
    _, model_name = demo_model_fixture(task_name)
    results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_face_crop_align",
        model_name,
        "ground_truth",
    )
    report = bias_report_fixture(task_name, results_dir)
    assert report.dataset_name == "fhibe_face_crop_align"
    assert report.dataset_base == "fhibe_face"
    assert task_name in report.valid_tasks

    # Face verification
    task_name = "face_verification"
    _, model_name = demo_model_fixture(task_name)
    results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_face_crop_align",
        model_name,
        "ground_truth",
    )
    report = bias_report_fixture(task_name, results_dir)
    assert report.dataset_name == "fhibe_face_crop_align"
    assert report.dataset_base == "fhibe_face"
    assert task_name in report.valid_tasks

    # Invalid task name
    with pytest.raises(ValueError) as excinfo:
        report = BiasReport(
            model_name="some_model",
            task_name="bad_task_name",
            dataset_version="testing",
            data_rootdir=report.data_rootdir,
            results_base_dir=os.path.join(CURRENT_DIR, "static", "results"),
            dataset_name=report.dataset_name,
            downsampled=True,
            use_mini_dataset=True,
        )
    error_str = (
        "Task: bad_task_name has not yet been evaluated for this model: some_model"
    )
    assert error_str == str(excinfo.value)


def test_plot_iou_vs_threshold(bias_report_fixture, demo_model_fixture):
    # Person detection
    task_name = "person_localization"
    _, model_name = demo_model_fixture(task_name)
    results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
        "ground_truth",
    )
    report = bias_report_fixture(task_name, results_dir)
    fig_savename = os.path.join(
        report.results_base_dir,
        task_name,
        "fhibe_downsampled",
        model_name,
        "compare",
        "iou_vs_threshold.png",
    )
    if os.path.isfile(fig_savename):
        os.remove(fig_savename)
    fig = report.plot_iou_vs_threshold(savefig=True, fig_savename=fig_savename)
    assert os.path.isfile(fig_savename)
    ax = fig.axes[0]
    lines = ax.lines
    line = lines[0]
    x_data = line.get_xdata()
    y_data = line.get_ydata()
    assert x_data == pytest.approx(
        [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    )
    assert y_data == pytest.approx(
        [1.0, 1.0, 1.0, 1.0, 1.0, 0.98, 0.96, 0.94, 0.84, 0.46]
    )

    # cleanup
    os.remove(fig_savename)

    # Check that error is raised if we try to make this plot
    # for a task that does not support it.
    task_name = "keypoint_estimation"
    report = bias_report_fixture(task_name)

    with pytest.raises(RuntimeError) as excinfo:
        fig = report.plot_iou_vs_threshold()
    error_str = f"IoU cannot be plotted for task: {task_name}"
    assert str(excinfo.value) == error_str


def test_plot_oks_vs_threshold(bias_report_fixture, demo_model_fixture):
    task_name = "keypoint_estimation"
    _, model_name = demo_model_fixture(task_name)
    results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
        "ground_truth",
    )
    report = bias_report_fixture(task_name, results_dir)
    fig_savename = os.path.join(
        report.results_base_dir,
        task_name,
        "fhibe_downsampled",
        model_name,
        "compare",
        "oks_vs_threshold.png",
    )
    if os.path.isfile(fig_savename):
        os.remove(fig_savename)
    fig = report.plot_oks_vs_threshold(savefig=True, fig_savename=fig_savename)
    assert os.path.isfile(fig_savename)
    ax = fig.axes[0]
    lines = ax.lines
    line = lines[0]
    x_data = line.get_xdata()
    y_data = line.get_ydata()
    assert x_data == pytest.approx(
        [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    )
    assert y_data == pytest.approx(
        [1.0, 1.0, 1.0, 1.0, 0.98, 0.96, 0.96, 0.92, 0.84, 0.52]
    )

    # cleanup
    os.remove(fig_savename)

    # Check that error is raised if we try to make this plot
    # for a task that does not support it.
    task_name = "person_localization"
    report = bias_report_fixture(task_name)

    with pytest.raises(RuntimeError) as excinfo:
        fig = report.plot_oks_vs_threshold()
    error_str = "OKS is only available for task: keypoint_estimation"
    assert str(excinfo.value) == error_str


def test_plot_pck_vs_threshold(bias_report_fixture, demo_model_fixture):
    task_name = "keypoint_estimation"
    _, model_name = demo_model_fixture(task_name)
    results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
        "ground_truth",
    )
    report = bias_report_fixture(task_name, results_dir)
    fig_savename = os.path.join(
        report.results_base_dir,
        task_name,
        "fhibe_downsampled",
        model_name,
        "compare",
        "pck_vs_threshold.png",
    )

    if os.path.isfile(fig_savename):
        os.remove(fig_savename)
    fig = report.plot_pck_vs_threshold(savefig=True, fig_savename=fig_savename)
    assert os.path.isfile(fig_savename)
    ax = fig.axes[0]
    lines = ax.lines
    line = lines[0]
    x_data = line.get_xdata()
    y_data = line.get_ydata()
    assert x_data == pytest.approx([0.1, 0.25, 0.5, 0.75, 0.9])
    assert y_data == pytest.approx(
        [0.60342146, 0.93793776, 0.98220232, 0.98719697, 0.98886364]
    )

    # cleanup
    os.remove(fig_savename)

    # Check that error is raised if we try to make this plot
    # for a task that does not support it.
    task_name = "person_localization"
    report = bias_report_fixture(task_name)

    with pytest.raises(RuntimeError) as excinfo:
        fig = report.plot_pck_vs_threshold()
    error_str = "PCK is only available for task: keypoint_estimation."
    assert str(excinfo.value) == error_str


def test_plot_metric_by_intersectional_group(bias_report_fixture, demo_model_fixture):
    # Person localization
    task_name = "person_localization"
    _, model_name = demo_model_fixture(task_name)
    results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
        "ground_truth",
    )
    report = bias_report_fixture(task_name, results_dir)

    bar_height_dict = {
        "pronoun": [0.8863636363636364, 0.9428571428571427, 0.8],
        "age": [0.909375, 0.94, 0.9142857142857144, 0.95, 0.95],
        "ancestry": [0.9291666666666666, 0.908, 0.925],
    }
    for attr_name in ["pronoun", "age", "ancestry"]:
        fig_savedir = os.path.join(
            report.results_base_dir,
            task_name,
            "fhibe_downsampled",
            model_name,
            "compare",
        )
        os.makedirs(fig_savedir, exist_ok=True)
        fig_savename = os.path.join(
            fig_savedir,
            f"AR_IOU_vs_{attr_name}.png",
        )
        if os.path.isfile(fig_savename):
            os.remove(fig_savename)
        fig = report.plot_metric_by_intersectional_group(
            metric_name="AR_IOU",
            attr_name=attr_name,
            savefig=True,
            fig_savename=fig_savename,
        )
        assert os.path.isfile(fig_savename)
        ax = fig.axes[0]

        bars = ax.patches

        # Extract the bar heights
        bar_heights = [bar.get_height() for bar in bars]
        assert bar_heights == pytest.approx(bar_height_dict[attr_name])
        # cleanup
        os.remove(fig_savename)

    # Keypoint estimation
    task_name = "keypoint_estimation"
    _, model_name = demo_model_fixture(task_name)
    results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
        "ground_truth",
    )
    report = bias_report_fixture(task_name, results_dir)

    bar_height_dict = {
        "pronoun": [0.8714285714285714, 0.9517241379310343],
        "age": [0.9342857142857142, 0.9222222222, 0.7, 0.9333333333333332],
        "ancestry": [0.9233333333333332, 0.9421052631578947, 0.7333333333333334, 0.9],
    }
    for attr_name in ["pronoun", "age", "ancestry"]:
        fig_savedir = os.path.join(
            report.results_base_dir,
            task_name,
            "fhibe_downsampled",
            model_name,
            "compare",
        )
        os.makedirs(fig_savedir, exist_ok=True)

        fig_savename = os.path.join(
            fig_savedir,
            f"AR_OKS_vs_{attr_name}.png",
        )
        if os.path.isfile(fig_savename):
            os.remove(fig_savename)
        fig = report.plot_metric_by_intersectional_group(
            metric_name="AR_OKS",
            attr_name=attr_name,
            savefig=True,
            fig_savename=fig_savename,
        )
        assert os.path.isfile(fig_savename)
        ax = fig.axes[0]

        bars = ax.patches

        # Extract the bar heights
        bar_heights = [bar.get_height() for bar in bars]
        assert bar_heights == pytest.approx(bar_height_dict[attr_name])

        # cleanup
        os.remove(fig_savename)

    # Body parts detection
    task_name = "body_parts_detection"
    _, model_name = demo_model_fixture(task_name)
    results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
        "ground_truth",
    )
    report = bias_report_fixture(task_name, results_dir)
    bar_height_dict = {
        "pronoun": [
            0.5772727272727273,
            0.5535714285714286,
            1.0,
            0.16,
            0.33076923076923076,
            1.0,
        ],
        "ancestry": [
            0.6208333333333333,
            0.512,
            0.6,
            0.2833333333333334,
            0.22727272727272727,
            0.0,
        ],
    }
    for attr_name in ["pronoun", "ancestry"]:
        fig_savedir = os.path.join(
            report.results_base_dir,
            task_name,
            "fhibe_downsampled",
            model_name,
            "compare",
        )
        os.makedirs(fig_savedir, exist_ok=True)
        fig_savename = os.path.join(
            fig_savedir,
            f"AR_IOU_vs_{attr_name}.png",
        )
        if os.path.isfile(fig_savename):
            os.remove(fig_savename)
        fig = report.plot_metric_by_intersectional_group(
            metric_name="AR_DET",
            attr_name=attr_name,
            savefig=True,
            fig_savename=fig_savename,
        )
        assert os.path.isfile(fig_savename)

        ax = fig.axes[0]
        bars = ax.patches

        # Extract the bar heights
        bar_heights = [bar.get_height() for bar in bars][0:-2]  # don't include legend
        assert bar_heights == pytest.approx(bar_height_dict[attr_name])

        # cleanup
        os.remove(fig_savename)

    # Face verification
    task_name = "face_verification"
    _, model_name = demo_model_fixture(task_name)
    results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_face_crop_align",
        model_name,
        "ground_truth",
    )
    report = bias_report_fixture(task_name, results_dir)

    bar_height_dict = {
        "pronoun": [0.66, 0.95],
        "age": [0.89, 0.74, 0.0, 0.0, 0.0],
        "apparent_skin_color": [0.82, 0.0, 0.0, 0.06, 0.0, 0.0],
    }
    errorbar_height_dict = {
        "pronoun": [[0.42, 0.9], [0.87, 1.03]],
        "age": [[0.79, 0.99], [0.59, 0.89], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        "apparent_skin_color": [
            [0.63, 1.01],
            [0.0, 0.0],
            [0.0, 0.0],
            [-0.06, 0.18],
            [0.0, 0.0],
            [0.0, 0.0],
        ],
    }
    for attr_name in ["pronoun", "age", "apparent_skin_color"]:
        fig_savedir = os.path.join(
            report.results_base_dir,
            task_name,
            "fhibe_face_crop_align",
            model_name,
            "compare",
        )
        os.makedirs(fig_savedir, exist_ok=True)
        fig_savename = os.path.join(
            fig_savedir,
            f"VAL_vs_{attr_name}.png",
        )
        if os.path.isfile(fig_savename):
            os.remove(fig_savename)
        fig = report.plot_metric_by_intersectional_group(
            metric_name="VAL",
            attr_name=attr_name,
            savefig=True,
            fig_savename=fig_savename,
        )
        assert os.path.isfile(fig_savename)
        ax = fig.axes[0]

        bars = ax.patches
        # Extract the bar heights
        bar_heights = [bar.get_height() for bar in bars]
        assert bar_heights == pytest.approx(bar_height_dict[attr_name])
        # Ensure the error bars are shown and the correct height
        lines = ax.get_lines()
        constructed_error_bars = [[] for _ in range(len(bars))]
        for i, line in enumerate(lines):
            line_ix = int(line.get_xdata()[0])
            constructed_error_bars[line_ix].append(line.get_ydata()[0])
        assert len(constructed_error_bars) == len(errorbar_height_dict[attr_name])
        for ii, sublist in enumerate(constructed_error_bars):
            assert len(sublist) == 2
            assert pytest.approx(sublist) == errorbar_height_dict[attr_name][ii]
        # cleanup
        os.remove(fig_savename)


def test_plot_with_significance(bias_report_fixture, demo_model_fixture):
    # Person localization
    task_name = "person_parsing"
    _, model_name = demo_model_fixture(task_name)
    results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
        "ground_truth",
    )
    report = bias_report_fixture(task_name, results_dir)

    bar_height_dict = {
        "pronoun": [0.7761904761904762, 0.8793103448275861],
        "age": [
            0.8342857142857143,
            0.8444444444444446,
            0.8000000000000002,
            0.8666666666666667,
        ],
        "ancestry": [0.8533333333333332, 0.8157894736842106, 0.7999999999999999, 0.9],
    }
    line_number_dict = {
        "pronoun": 3,  # Two error bars (one for each bar) and the significance bracket
        "age": 4,
        "ancestry": 4,
    }
    text_dict = {
        "pronoun": ["*"],
        "age": [],
        "ancestry": [],
    }

    for attr_name in ["pronoun", "age", "ancestry"]:
        fig = report.plot_metric_by_intersectional_group(
            metric_name="AR_MASK",
            attr_name=attr_name,
            savefig=False,
            show_significance=True,
        )
        ax = fig.axes[0]

        bars = ax.patches
        bar_heights = [bar.get_height() for bar in bars]
        assert bar_heights == pytest.approx(bar_height_dict[attr_name])
        lines = ax.lines
        assert len(lines) == line_number_dict[attr_name]
        texts = [t.get_text() for t in ax.texts]
        assert texts == text_dict[attr_name]


def test_calculate_disparity(bias_report_fixture, demo_model_fixture):
    task_name = "person_localization"
    _, model_name = demo_model_fixture(task_name)
    results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
        "ground_truth",
    )
    report = bias_report_fixture(task_name, results_dir)
    intersectional_results_fp = os.path.join(
        report.results_dir, "intersectional_results_AR_IOU.json"
    )
    intersectional_results = read_json_file(intersectional_results_fp)
    disparity_results = report.calculate_disparity(
        intersectional_results=intersectional_results,
        attributes_to_consider=["pronoun", "age"],
        metric_name="AR_IOU",
        group_largest_regions=True,
        min_group_size=5,
    )
    assert disparity_results == []

    task_name = "keypoint_estimation"
    _, model_name = demo_model_fixture(task_name)
    results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
        "ground_truth",
    )
    report = bias_report_fixture(task_name, results_dir)
    intersectional_results_fp = os.path.join(
        report.results_dir, "intersectional_results_PCK.json"
    )
    intersectional_results = read_json_file(intersectional_results_fp)
    disparity_results = report.calculate_disparity(
        intersectional_results=intersectional_results,
        attributes_to_consider=["pronoun", "age"],
        metric_name="PCK",
        group_largest_regions=True,
        min_group_size=5,
    )
    assert disparity_results == [
        [
            "['age']",
            "['[30, 40)']",
            9,
            "['[18, 30)']",
            35,
            0.01449275362318836,
            0.024944098507239155,
            2.2549181991052545,
        ]
    ]

    task_name = "body_parts_detection"
    _, model_name = demo_model_fixture(task_name)
    results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
        "ground_truth",
    )
    report = bias_report_fixture(task_name, results_dir)
    intersectional_results_fp = os.path.join(
        report.results_dir, "intersectional_results_AR_DET.json"
    )
    intersectional_results = read_json_file(intersectional_results_fp)
    disparity_results = report.calculate_disparity(
        intersectional_results=intersectional_results,
        attributes_to_consider=["pronoun", "apparent_skin_color"],
        metric_name="AR_DET",
        group_largest_regions=True,
        min_group_size=5,
    )
    assert disparity_results == []


def test_generate_pdf_report(bias_report_fixture, demo_model_fixture):
    # Person detection
    task_name = "person_localization"
    _, model_name = demo_model_fixture(task_name)
    results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
        "ground_truth",
    )
    report = bias_report_fixture(task_name, results_dir)
    pdf_savename = os.path.join(report.results_dir, "bias_report", "bias_report.pdf")
    if os.path.isfile(pdf_savename):
        os.remove(pdf_savename)
    _ = report.generate_pdf_report(attributes=["pronoun", "age"])
    assert os.path.isfile(pdf_savename)
    with open(pdf_savename, "rb") as file:
        # Read the contents of the file
        doc_content = file.read()
    assert doc_content is not None

    # Keypoint estimation
    task_name = "keypoint_estimation"
    # First with default keypoints
    _, model_name = demo_model_fixture(task_name)
    results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
        "ground_truth",
    )
    report = bias_report_fixture(task_name, results_dir)
    pdf_savename = os.path.join(report.results_dir, "bias_report", "bias_report.pdf")
    if os.path.isfile(pdf_savename):
        os.remove(pdf_savename)
    _ = report.generate_pdf_report(attributes=["pronoun", "age"])
    assert os.path.isfile(pdf_savename)
    with open(pdf_savename, "rb") as file:
        # Read the contents of the file
        doc_content = file.read()
    assert doc_content is not None

    # Now with custom keypoints
    _, model_name = demo_model_fixture(
        task_name, custom_keypoints=["Nose", "Left eye", "Right eye"]
    )
    model_name = "keypoint_estimator_test_model_custom_keypoints"

    results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
        "ground_truth",
    )
    report = bias_report_fixture(task_name, results_dir)
    pdf_savename = os.path.join(report.results_dir, "bias_report", "bias_report.pdf")
    if os.path.isfile(pdf_savename):
        os.remove(pdf_savename)
    _ = report.generate_pdf_report(attributes=["pronoun", "age"])
    assert os.path.isfile(pdf_savename)
    with open(pdf_savename, "rb") as file:
        # Read the contents of the file
        doc_content = file.read()
    assert doc_content is not None

    # cleanup
    report_files = glob.glob(report.results_dir + "/bias_report/*")
    for fp in report_files:
        if os.path.isfile(fp):
            os.remove(fp)

    # Body parts detection
    task_name = "body_parts_detection"
    _, model_name = demo_model_fixture(task_name)
    results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
        "ground_truth",
    )
    report = bias_report_fixture(task_name, results_dir)
    pdf_savename = os.path.join(report.results_dir, "bias_report", "bias_report.pdf")
    if os.path.isfile(pdf_savename):
        os.remove(pdf_savename)
    _ = report.generate_pdf_report(
        attributes=["pronoun", "ancestry", "apparent_skin_color"],
    )
    assert os.path.isfile(pdf_savename)
    with open(pdf_savename, "rb") as file:
        # Read the contents of the file
        doc_content = file.read()
    assert doc_content is not None

    # cleanup
    report_files = glob.glob(report.results_dir + "/bias_report/*")
    for fp in report_files:
        if os.path.isfile(fp):
            os.remove(fp)

    # Face encoding
    task_name = "face_encoding"
    _, model_name = demo_model_fixture(task_name)
    results_dir = os.path.join(
        CURRENT_DIR,
        "static",
        "results",
        "mini",
        task_name,
        "fhibe_downsampled",
        model_name,
        "ground_truth",
    )
    report = bias_report_fixture(task_name, results_dir)
    pdf_savename = os.path.join(report.results_dir, "bias_report", "bias_report.pdf")
    if os.path.isfile(pdf_savename):
        os.remove(pdf_savename)
    _ = report.generate_pdf_report(attributes=["pronoun", "ancestry"])
    assert os.path.isfile(pdf_savename)
    with open(pdf_savename, "rb") as file:
        # Read the contents of the file
        doc_content = file.read()
    assert doc_content is not None

    # cleanup
    report_files = glob.glob(report.results_dir + "/bias_report/*")
    for fp in report_files:
        if os.path.isfile(fp):
            os.remove(fp)
