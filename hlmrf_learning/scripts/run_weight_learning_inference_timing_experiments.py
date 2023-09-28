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
EXPERIMENT_RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, "wl_inference_timing")

STANDARD_EXPERIMENT_OPTIONS = {
    "runtime.log.level": "TRACE",
    "inference.normalize": "false",
    "admmreasoner.primaldualbreak": "false",
    "duallcqp.primaldualbreak": "false",
    "duallcqp.computeperiod": "1",
    "duallcqp.maxiterations": "25000",
    "reasoner.variablemovementbreak": "true",
    "reasoner.variablemovementnorm": "Infinity",
    "reasoner.variablemovementtolerance": "1.0e-3",
    "gradientdescent.scalestepsize": "false",
    "gradientdescent.numsteps": "500",
    "gradientdescent.runfulliterations": "true",
    "gradientdescent.batchgenerator": "FullBatchGenerator"
}

STANDARD_DATASET_OPTIONS = {
    "epinions": {
        "admmreasoner.stepsize": "1.0e-1",
        "duallcqp.regularizationparameter": "0.1",
    },
    "citeseer": {
        "eval.closetruth": "true",
        "admmreasoner.stepsize": "10.0",
        "duallcqp.regularizationparameter": "10.0",
    },
    "cora": {
        "eval.closetruth": "true",
        "admmreasoner.stepsize": "10.0",
        "duallcqp.regularizationparameter": "10.0",
    },
    "yelp": {
        "admmreasoner.stepsize": "1.0",
        "duallcqp.regularizationparameter": "0.1",
    },
    "drug-drug-interaction": {
        "admmreasoner.stepsize": "1.0",
        "duallcqp.regularizationparameter": "1.0"
    },
    "stance-4forums": {
        "admmreasoner.stepsize": "1.0",
        "duallcqp.regularizationparameter": "0.1"
    },
    "stance-createdebate": {
        "admmreasoner.stepsize": "1.0",
        "duallcqp.regularizationparameter": "0.1"
    }
}

FIRST_ORDER_WL_METHODS = ["SquaredError", "StructuredPerceptron"]
FIRST_ORDER_WL_INFERENCE_METHODS = ["DistributedDualBCDInference", "ADMMInference"]

FIRST_ORDER_WL_METHODS_STANDARD_OPTION_RANGES = {
    "gradientdescent.negativelogregularization": ["1.0e-3"],
    "gradientdescent.stepsize": ["1.0e-3"],
    "gradientdescent.negativeentropyregularization": ["0.0"]
}

FIRST_ORDER_WL_INFERENCE_METHOD_OPTION_RANGES = {
    "DualBCDInference": {
        "weightlearning.inference": ["DualBCDInference"],
        "runtime.inference.method": ["DualBCDInference"],
    },
    "DistributedDualBCDInference": {
        "weightlearning.inference": ["DistributedDualBCDInference"],
        "runtime.inference.method": ["DistributedDualBCDInference"],
    },
    "ADMMInference": {
        "weightlearning.inference": ["ADMMInference"],
        "runtime.inference.method": ["ADMMInference"],
    }
}

FIRST_ORDER_WL_METHODS_OPTION_RANGES = {
    "StructuredPerceptron": {
        "runtime.learn.method": ["StructuredPerceptron"],
    },
    "Energy": {
        "runtime.learn.method": ["Energy"],
    },
    "BinaryCrossEntropy": {
        "runtime.learn.method": ["BinaryCrossEntropy"],
        "minimizer.initialsquaredpenalty": ["2.0"],
        "minimizer.energylosscoefficient": ["0.1"],
        "minimizer.proxvaluestepsize": ["1.0e-2"],
        "minimizer.squaredpenaltyincreaserate": ["2.0"],
        "minimizer.objectivedifferencetolerance": ["0.1"],
        "minimizer.proxruleweight": ["1.0e-2"]
    },
    "SquaredError": {
        "runtime.learn.method": ["SquaredError"],
        "minimizer.initialsquaredpenalty": ["2.0"],
        "minimizer.energylosscoefficient": ["0.1"],
        "minimizer.proxvaluestepsize": ["1.0e-2"],
        "minimizer.squaredpenaltyincreaserate": ["2.0"],
        "minimizer.objectivedifferencetolerance": ["0.1"],
        "minimizer.proxruleweight": ["1.0e-2"]
    }
}


def run_first_order_wl_methods(dataset: str):
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

    if dataset in ["stance-4forums", "stance-createdebate", "citeseer", "cora"]:
        dataset_extended_json_path = os.path.join(PSL_EXTENDED_EXAMPLES_DIR, "{}/cli/{}.json".format(dataset, dataset))

        with open(dataset_extended_json_path, "r") as file:
            dataset_json = json.load(file)
    else:
        dataset_json = dataset_original_json

    standard_experiment_option_ranges = {**FIRST_ORDER_WL_METHODS_STANDARD_OPTION_RANGES}

    for method in FIRST_ORDER_WL_METHODS:
        for inference_method in FIRST_ORDER_WL_INFERENCE_METHODS:
            for split in range(0, 5):
                method_extension_out_dir = os.path.join(base_out_dir, "{}/{}/split::{}".format(inference_method, method, split))
                os.makedirs(method_extension_out_dir, exist_ok=True)

                # Iterate over every combination options values.
                method_options_dict = {**standard_experiment_option_ranges,
                                       **FIRST_ORDER_WL_METHODS_OPTION_RANGES[method],
                                       **FIRST_ORDER_WL_INFERENCE_METHOD_OPTION_RANGES[inference_method]}

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
                                                    **options,
                                                    "runtime.learn.output.model.path": "./{}_learned.psl".format(dataset)}})

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
                    os.system("cp {} {}".format(os.path.join(dataset_cli_path, "{}_learned.psl".format(dataset)), experiment_out_dir))

                    # Reset the json file.
                    dataset_original_json.update({"options": original_options})
                    with open(dataset_json_path, "w") as file:
                        json.dump(dataset_original_json, file, indent=4)

                    print("Finished experiment: Dataset:{}, Weight Learning Method: {}.".format(dataset, method))


def parse_args() -> str:
    if len(sys.argv) != 2:
        print("Usage: python3 run_weight_learning_inference_timing_experiments.py <dataset>")
        sys.exit(1)

    dataset = sys.argv[1]
    return dataset


def main():
    dataset = parse_args()

    run_first_order_wl_methods(dataset)


if __name__ == '__main__':
    main()
