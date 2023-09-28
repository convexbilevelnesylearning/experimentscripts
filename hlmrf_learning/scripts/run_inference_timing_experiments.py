#!/usr/bin/env python3

import json
import os
import signal
import sys

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..'))
import util

RESULTS_BASE_DIR = os.path.join(THIS_DIR, "../results")
PSL_EXAMPLES_DIR = os.path.join(THIS_DIR, "../psl-examples")
PSL_EXTENDED_EXAMPLES_DIR = os.path.join(THIS_DIR, "../psl-extended-examples")
EXPERIMENT_RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, "inference")

STANDARD_EXPERIMENT_OPTIONS = {
    "runtime.log.level": "TRACE",
    "runtime.learn": "false",
    "admmreasoner.primaldualbreak": "false",
    "duallcqp.primaldualbreak": "false",
    "duallcqp.computeperiod": "1",
    "duallcqp.maxiterations": "25000",
    "reasoner.gradientdescent.maxiterations": "25000",
    "reasoner.gradientdescent.firstorderbreak": "false",
    "reasoner.variablemovementbreak": "true",
    "reasoner.variablemovementnorm": "Infinity",
    "reasoner.variablemovementtolerance": "1.0e-3"
}

STANDARD_DATASET_OPTIONS = {
    "epinions": {
    },
    "citeseer": {
        "eval.closetruth": "true",
    },
    "cora": {
        "eval.closetruth": "true",
    },
    "yelp": {
    },
    "drug-drug-interaction": {
    },
    "stance-4forums": {
    },
    "stance-createdebate": {
    }
}

INFERENCE_METHODS = ["DualBCDInference", "DistributedDualBCDInference", "ADMMInference", "GradientDescentInference"]

INFERENCE_METHOD_OPTION_RANGES = {
    "GradientDescentInference": {
        "runtime.inference.method": ["GradientDescentInference"],
        "gradientdescent.stepsize": ["10.0", "1.0", "0.1", "0.01", "0.001"],
    },
    "DualBCDInference": {
        "runtime.inference.method": ["DualBCDInference"],
        "duallcqp.regularizationparameter": ["100.0", "10.0", "1.0", "0.1", "0.01"],
    },
    "DistributedDualBCDInference": {
        "runtime.inference.method": ["DistributedDualBCDInference"],
        "duallcqp.regularizationparameter": ["100.0", "10.0", "1.0", "0.1", "0.01"],
    },
    "ADMMInference": {
        "runtime.inference.method": ["ADMMInference"],
        "admmreasoner.stepsize": ["10", "1", "0.1", "0.01", "0.001"]
    }
}


def run_inference_methods(dataset: str):
    base_out_dir = os.path.join(EXPERIMENT_RESULTS_DIR, dataset)
    os.makedirs(base_out_dir, exist_ok=True)

    # Load the json file with the dataset options.
    dataset_cli_path = os.path.join(PSL_EXAMPLES_DIR, "{}/cli".format(dataset))
    dataset_json_path = os.path.join(dataset_cli_path, "{}.json".format(dataset))

    dataset_original_json = None
    with open(dataset_json_path, "r") as file:
        dataset_original_json = json.load(file)

    original_options = {}
    if "options" in dataset_original_json:
        original_options = dataset_original_json["options"]

    if dataset in ["stance-4forums", "stance-createdebate"]:
        dataset_extended_json_path = os.path.join(PSL_EXTENDED_EXAMPLES_DIR, "{}/cli/{}.json".format(dataset, dataset))

        with open(dataset_extended_json_path, "r") as file:
            dataset_json = json.load(file)
    else:
        dataset_json = dataset_original_json

    for inference_method in INFERENCE_METHODS:
        for split in range(0, 5):
            method_extension_out_dir = os.path.join(base_out_dir, "{}/split::{}".format(inference_method, split))
            os.makedirs(method_extension_out_dir, exist_ok=True)

            # Iterate over every combination options values.
            method_options_dict = {**INFERENCE_METHOD_OPTION_RANGES[inference_method]}

            for options in util.enumerate_hyperparameters(method_options_dict):
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
                    exit_code = os.system("cd {} && ./run.sh {} > out.txt 2> out.err".format(dataset_cli_path, experiment_out_dir))
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
                dataset_original_json.update({"options": original_options})
                with open(dataset_json_path, "w") as file:
                    json.dump(dataset_original_json, file, indent=4)

                print("Finished experiment: Dataset:{}.".format(dataset))


def parse_args() -> str:
    if len(sys.argv) != 2:
        print("Usage: python3 run_inference_timing_experiments.py <dataset>")
        sys.exit(1)

    dataset = sys.argv[1]
    return dataset


def main():
    dataset = parse_args()

    run_inference_methods(dataset)


if __name__ == '__main__':
    main()
