# SPDX-License-Identifier: Apache-2.0

import json
import os

import pandas as pd

CURRENT_DIR = os.path.dirname(__file__)


def make_fake_model_outputs(task_name: str, model_name: str, use_mini_dataset: bool):
    # Update the keys in the model_outputs.json files to the correct paths in the
    # tests/static directory.
    dataset_name = "fhibe_downsampled"
    if task_name in [
        "face_parsing",
        "face_encoding",
        "face_verification",
        "face_super_resolution",
    ]:
        dataset_name = "fhibe_face_crop_align"
    fake_base_path = os.path.join(CURRENT_DIR, "static")
    results_base_path = os.path.join(fake_base_path, "results")
    if use_mini_dataset:
        results_base_path = os.path.join(results_base_path, "mini")
    model_outputs_fp = os.path.join(
        results_base_path,
        task_name,
        dataset_name,
        model_name,
        "ground_truth",
        "fixed_model_outputs.json",
    )
    print(model_outputs_fp)
    # Load json
    with open(model_outputs_fp, "r") as f:
        model_outputs = json.load(f)

    dataframe_fp = os.path.join(
        fake_base_path, "data", "processed", dataset_name, f"{dataset_name}.csv"
    )
    dataframe = pd.read_csv(dataframe_fp)

    model_values = list(model_outputs.values())

    # Assign them to the new keys
    new_model_outputs = {}
    counter = 0
    for ix, row in dataframe.iterrows():
        fp = row["filepath"]
        fake_full_path = os.path.join(fake_base_path, fp)
        new_model_outputs[fake_full_path] = model_values[counter]
        counter += 1

    # Overwrite the original fixed model outputs file
    with open(model_outputs_fp, "w") as f:
        json.dump(new_model_outputs, f, indent=4)

    print(f"Made fake model outputs for task {task_name} at {model_outputs_fp}")


if __name__ == "__main__":
    task_name = "face_super_resolution"
    model_name = "face_super_resolution_test_model"
    use_mini_dataset = True
    make_fake_model_outputs(task_name, model_name, use_mini_dataset)
