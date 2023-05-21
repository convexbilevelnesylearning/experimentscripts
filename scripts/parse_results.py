#!/usr/bin/env python3

import os
import pandas as pd
import re

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
RESULTS_BASE_DIR = os.path.join(THIS_DIR, "../results")

DATASET_NAME_OUTPUT_KEYS = {
    "epinions": {
        "test_eval": {"name": "AUROC", "key": "Results -- AUROC:"}
    },
    "citeseer": {
        "test_eval": {"name": "Accuracy", "key": "Results -- Categorical Accuracy:"}
    },
    "cora": {
        "test_eval": {"name": "Accuracy", "key": "Results -- Categorical Accuracy:"}
    },
    "yelp": {
        "test_eval": {"name": "MAE", "key": "Results -- MAE:"}
    }
}


def parse_experiment_output(dataset_name: str, output_file_path: str) -> pd.Series:
    experiment_output = pd.Series({"learning_objective": [], "inference_time": [], }, dtype=object)

    with open(output_file_path, "r") as output_file:
        for line in output_file.readlines():
            if "Reasoner  - Final Objective:" in line:
                experiment_output["inference_time"].append(float(re.search(r"Total Optimization Time: (-?\d+)", line).group(1)))

            if "Weight Learning Objective:" in line:
                experiment_output["learning_objective"].append(float(re.search(r"Weight Learning Objective: (-?\d+.\d+|Infinity|NaN)", line).group(1)))

            if DATASET_NAME_OUTPUT_KEYS[dataset_name]["test_eval"]["key"] in line:
                experiment_output[DATASET_NAME_OUTPUT_KEYS[dataset_name]["test_eval"]["name"]] = float(
                    re.search(r"{} (-?\d+.\d+)".format(DATASET_NAME_OUTPUT_KEYS[dataset_name]["test_eval"]["key"]), line).group(1)
                )

    if DATASET_NAME_OUTPUT_KEYS[dataset_name]["test_eval"]["name"] not in experiment_output:
        experiment_output["wl_num_steps"] = None
        experiment_output["inference_time"] = None
        experiment_output["total_inference_time"] = None
        experiment_output[DATASET_NAME_OUTPUT_KEYS[dataset_name]["test_eval"]["name"]] = None
        return experiment_output

    experiment_output["wl_num_steps"] = len(experiment_output["learning_objective"])

    experiment_output["inference_time"] = experiment_output["inference_time"][:-1]
    experiment_output["total_inference_time"] = sum(experiment_output["inference_time"])

    return experiment_output


def parse_results_recursive(dataset_name: str, current_directory: str, model_results: list, parameters: dict):
    # Base Cases
    if os.path.exists(os.path.join(current_directory, "out.txt")):
        results = pd.concat([pd.Series(parameters), parse_experiment_output(dataset_name, os.path.join(current_directory, "out.txt"))])
        model_results.append(results)

    # Recursive Cases
    for parameter_setting_dir in os.listdir(current_directory):
        if "::" not in parameter_setting_dir:
            continue

        parameter, value = parameter_setting_dir.split("::")
        next_parameters = parameters.copy()
        next_parameters[parameter] = value
        parse_results_recursive(dataset_name, os.path.join(current_directory, parameter_setting_dir), model_results, next_parameters)


def parse_results(dataset_name: str, model_dir: str) -> pd.DataFrame:
    model_results = []
    parse_results_recursive(dataset_name, model_dir, model_results, {})

    if len(model_results) == 0:
        print("No results found for model in directory: {}".format(model_dir))
        return pd.DataFrame()

    return pd.DataFrame(model_results)


def save_model_results(model_results: pd.DataFrame, model_dir: str):
    summarized_model_results = model_results
    if not model_results.empty:
        summarized_model_results = model_results.drop(columns=["learning_objective", "inference_time"])

    summarized_model_results.to_csv(os.path.join(model_dir, "results.csv"), index=False)


def main():
    for experiment in os.listdir(RESULTS_BASE_DIR):
        experiment_dir = os.path.join(RESULTS_BASE_DIR, experiment)
        for dataset_name in os.listdir(experiment_dir):
            dataset_dir = os.path.join(experiment_dir, dataset_name)
            for model_type_name in os.listdir(os.path.join(dataset_dir)):
                model_type_dir = os.path.join(dataset_dir, model_type_name)
                for model_name in os.listdir(model_type_dir):
                    model_dir = os.path.join(model_type_dir, model_name)
                    model_results = parse_results(dataset_name, model_dir)
                    save_model_results(model_results, model_dir)


if __name__ == '__main__':
    main()