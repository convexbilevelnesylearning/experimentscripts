#!/usr/bin/env python3

import json
import os
import signal
import sys

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))
RESULTS_BASE_DIR = os.path.join(THIS_DIR, "../results")
PSL_EXAMPLES_DIR = os.path.join(THIS_DIR, "../psl-examples")
EXPERIMENT_RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, "dual_bcd_regularization")

STANDARD_EXPERIMENT_OPTIONS = {
    "runtime.log.level": "TRACE",
    "runtime.learn": "false",
    "runtime.inference.method": "DistributedDualBCDInference",
    "duallcqp.computeperiod": "10",
    "duallcqp.maxiterations": "10000"
}

STANDARD_DATASET_OPTIONS = {
    "epinions": {
        "duallcqp.primaldualthreshold": "0.1"
    },
    "citeseer": {
        "eval.closetruth": "true",
        "duallcqp.primaldualthreshold": "0.1"
    },
    "cora": {
        "eval.closetruth": "true",
        "duallcqp.primaldualthreshold": "0.1"
    },
    "yelp": {
        "duallcqp.primaldualthreshold": "0.1"
    },
}

INFERENCE_OPTION_RANGES = {
    "duallcqp.regularizationparameter": ["100", "10", "1", "0.1", "0.01"]
}


def enumerate_hyperparameters(hyperparameters_dict: dict, current_hyperparameters={}):
    for key in sorted(hyperparameters_dict):
        hyperparameters = []
        for value in hyperparameters_dict[key]:
            next_hyperparameters = current_hyperparameters.copy()
            next_hyperparameters[key] = value

            remaining_hyperparameters = hyperparameters_dict.copy()
            remaining_hyperparameters.pop(key)

            if remaining_hyperparameters:
                hyperparameters = hyperparameters + enumerate_hyperparameters(remaining_hyperparameters, next_hyperparameters)
            else:
                hyperparameters.append(next_hyperparameters)
        return hyperparameters


def run_inference_params(dataset: str):
    base_out_dir = os.path.join(EXPERIMENT_RESULTS_DIR, "DistributedDualBCDInference/DistributedDualBCDInference", dataset)
    os.makedirs(base_out_dir, exist_ok=True)

    # Load the json file with the dataset options.
    dataset_cli_path = os.path.join(PSL_EXAMPLES_DIR, "{}/cli".format(dataset))
    dataset_json_path = os.path.join(dataset_cli_path, "{}.json".format(dataset))

    dataset_json = None
    with open(dataset_json_path, "r") as file:
        dataset_json = json.load(file)
    original_options = dataset_json["options"]

    standard_experiment_option_ranges = {**INFERENCE_OPTION_RANGES}

    for split in range(0, 5):
        method_extension_out_dir = os.path.join(base_out_dir, "split::{}".format(split))
        os.makedirs(method_extension_out_dir, exist_ok=True)

        # Iterate over every combination options values.
        method_options_dict = {**standard_experiment_option_ranges}

        for options in enumerate_hyperparameters(method_options_dict):
            experiment_out_dir = method_extension_out_dir
            for key, value in sorted(options.items()):
                experiment_out_dir = os.path.join(experiment_out_dir, "{}::{}".format(key, value))
            os.makedirs(experiment_out_dir, exist_ok=True)

            if os.path.exists(os.path.join(experiment_out_dir, "out.txt")):
                print("Skipping experiment: {}.".format(experiment_out_dir))
                continue

            dataset_json.update({"options":{**original_options,
                                            **STANDARD_EXPERIMENT_OPTIONS,
                                            **STANDARD_DATASET_OPTIONS[dataset],
                                            **options}})

            # Write the options the json file.
            with open(dataset_json_path, "w") as file:
                json.dump(dataset_json, file, indent=4)

            # SED the split data.
            os.system("sed -i 's#/[0-9]/#/{}/#g' {}".format(split, dataset_json_path))

            # Run the experiment.
            print("Running experiment: {}.".format(experiment_out_dir))

            def timeoutHandler(signum, frame):
                print("Experiment Timeout: {}.".format(experiment_out_dir))

                raise Exception("Timeout")

            signal.signal(signal.SIGALRM, timeoutHandler)
            signal.alarm(14400)

            try:
                exit_code = os.system(
                    "cd {} && ./run.sh {} > out.txt 2> out.err".format(dataset_cli_path, experiment_out_dir))
            except Exception as e:
                if e.args[0] != "Timeout":
                    raise e

                exit_code = 0

            signal.alarm(0)

            if exit_code != 0:
                print("Experiment failed: {}.".format(experiment_out_dir))
                exit()

            # Save the output and json file.
            os.system("mv {} {}".format(os.path.join(dataset_cli_path, "out.txt"), experiment_out_dir))
            os.system("mv {} {}".format(os.path.join(dataset_cli_path, "out.err"), experiment_out_dir))
            os.system("cp {} {}".format(os.path.join(dataset_cli_path, "{}.json".format(dataset)), experiment_out_dir))

            # Reset the json file.
            dataset_json.update({"options": original_options})
            with open(dataset_json_path, "w") as file:
                json.dump(dataset_json, file)

            print("Finished experiment: Dataset:{}".format(dataset))


def parse_args() -> str:
    if len(sys.argv) != 2:
        print("Usage: python3 run_weight_learning_performance_experiments.py <dataset>")
        sys.exit(1)

    dataset = sys.argv[1]
    return dataset

def main():
    dataset = parse_args()

    run_inference_params(dataset)


if __name__ == '__main__':
    main()
