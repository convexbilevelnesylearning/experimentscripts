#!/usr/bin/env python3

import json
import os
import re
import sys

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..'))
import util

CLI_DIR = os.path.join(THIS_DIR, "../cli")
RESULTS_BASE_DIR = os.path.join(THIS_DIR, "../results")
PERFORMANCE_RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, "performance")

DATASETS = ["citeseer", "cora"]
SPLITS = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

STANDARD_EXPERIMENT_OPTIONS = {
    "inference.normalize": "false",
    "runtime.log.level": "TRACE",
    "runtime.validation": "true",
    "gradientdescent.batchgenerator": "FullBatchGenerator",
    "gradientdescent.trainingcomputeperiod": "5",
    "gradientdescent.savevalidationweights": "true",
    "gradientdescent.validationcomputeperiod": "5",
    "gradientdescent.validationbreak": "true",
    "gradientdescent.validationpatience": "50",
    "gradientdescent.stopcomputeperiod": "5",
    "weightlearning.inference": "DistributedDualBCDInference",
    "runtime.inference.method": "DistributedDualBCDInference",
    "gradientdescent.numsteps": "1000",
    "gradientdescent.runfulliterations": "false",
    "duallcqp.computeperiod": "10",
    "duallcqp.maxiterations": "25000",
}

STANDARD_DATASET_OPTIONS = {
    "citeseer": {
        "duallcqp.primaldualthreshold": "1.0e-3"
    },
    "cora": {
        "duallcqp.primaldualthreshold": "1.0e-3"
    }
}

DATAPATH_NAME = {
    "citeseer": "citeseer",
    "cora": "cora"
}

INFERENCE_OPTION_RANGES = {
    "duallcqp.regularizationparameter": ["1.0e-3"]
}

FIRST_ORDER_WL_METHODS = ["SquaredError", "BinaryCrossEntropy", "Energy", "StructuredPerceptron"]

FIRST_ORDER_WL_METHODS_STANDARD_OPTION_RANGES = {
    "gradientdescent.stepsize": ["1.0e-2", "1.0e-3"],
    "gradientdescent.negativelogregularization": ["1.0e-3"],
    "gradientdescent.negativeentropyregularization": ["0.0"],
}

FIRST_ORDER_WL_METHODS_OPTION_RANGES = {
    "Energy": {
        "runtime.learn.method": ["Energy"]
    },
    "StructuredPerceptron": {
        "runtime.learn.method": ["StructuredPerceptron"]
    },
    "SquaredError": {
        "runtime.learn.method": ["SquaredError"],
        "minimizer.initialsquaredpenalty": ["2.0"],
        "minimizer.energylosscoefficient": ["0.1", "1.0", "10.0"],
        "minimizer.proxvaluestepsize": ["1.0e-2", "1.0e-3"],
        "minimizer.squaredpenaltyincreaserate": ["2.0"],
        "minimizer.objectivedifferencetolerance": ["1.0e-3"],
        "minimizer.finalparametermovementconvergencetolerance": ["1.0e-1"],
        "minimizer.proxruleweight": ["1.0e-1", "1.0e-2", "1.0e-3"]
    },
    "BinaryCrossEntropy": {
        "runtime.learn.method": ["BinaryCrossEntropy"],
        "minimizer.initialsquaredpenalty": ["2.0"],
        "minimizer.energylosscoefficient": ["0.1", "1.0", "10.0"],
        "minimizer.proxvaluestepsize": ["1.0e-2", "1.0e-3"],
        "minimizer.squaredpenaltyincreaserate": ["2.0"],
        "minimizer.objectivedifferencetolerance": ["1.0e-3"],
        "minimizer.finalparametermovementconvergencetolerance": ["1.0e-1"],
        "minimizer.proxruleweight": ["1.0e-1", "1.0e-2", "1.0e-3"]
    }
}


def set_data_path(dataset_json, split):
    # sed dataset paths
    for predicate in dataset_json["predicates"]:
        if "options" in dataset_json["predicates"][predicate]:
            if "entity-data-map-path" in dataset_json["predicates"][predicate]["options"]:
                dataset_json["predicates"][predicate]["options"]["entity-data-map-path"] = \
                    re.sub(r"split::[0-9]+", "split::{}".format(split), dataset_json["predicates"][predicate]["options"]["entity-data-map-path"])

            if "save-path" in dataset_json["predicates"][predicate]["options"]:
                dataset_json["predicates"][predicate]["options"]["save-path"] = \
                    re.sub(r"split::[0-9]+", "split::{}".format(split), dataset_json["predicates"][predicate]["options"]["save-path"])

            if "load-path" in dataset_json["predicates"][predicate]["options"]:
                dataset_json["predicates"][predicate]["options"]["load-path"] = \
                    re.sub(r"split::[0-9]+", "split::{}".format(split), dataset_json["predicates"][predicate]["options"]["load-path"])

        if "targets" in dataset_json["predicates"][predicate]:
            if "learn" in dataset_json["predicates"][predicate]["targets"]:
                dataset_json["predicates"][predicate]["targets"]["learn"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), target) for target in dataset_json["predicates"][predicate]["targets"]["learn"]]

            if "validation" in dataset_json["predicates"][predicate]["targets"]:
                dataset_json["predicates"][predicate]["targets"]["validation"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), target) for target in dataset_json["predicates"][predicate]["targets"]["validation"]]

            if "infer" in dataset_json["predicates"][predicate]["targets"]:
                dataset_json["predicates"][predicate]["targets"]["infer"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), target) for target in dataset_json["predicates"][predicate]["targets"]["infer"]]

        if "truth" in dataset_json["predicates"][predicate]:
            if "learn" in dataset_json["predicates"][predicate]["truth"]:
                dataset_json["predicates"][predicate]["truth"]["learn"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["learn"]]

            if "validation" in dataset_json["predicates"][predicate]["truth"]:
                dataset_json["predicates"][predicate]["truth"]["validation"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["validation"]]

            if "infer" in dataset_json["predicates"][predicate]["truth"]:
                dataset_json["predicates"][predicate]["truth"]["infer"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["infer"]]

        if "observations" in dataset_json["predicates"][predicate]:
            if "learn" in dataset_json["predicates"][predicate]["observations"]:
                dataset_json["predicates"][predicate]["observations"]["learn"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in dataset_json["predicates"][predicate]["observations"]["learn"]]

            if "validation" in dataset_json["predicates"][predicate]["observations"]:
                dataset_json["predicates"][predicate]["observations"]["validation"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in dataset_json["predicates"][predicate]["observations"]["validation"]]

            if "infer" in dataset_json["predicates"][predicate]["observations"]:
                dataset_json["predicates"][predicate]["observations"]["infer"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in dataset_json["predicates"][predicate]["observations"]["infer"]]


def run_first_order_wl_methods(dataset_name):
    base_out_dir = os.path.join(PERFORMANCE_RESULTS_DIR, dataset_name, "first_order_wl_methods")
    os.makedirs(base_out_dir, exist_ok=True)

    # Load the json file with the dataset options.
    dataset_json_path = os.path.join(CLI_DIR, "neupsl-models/experiment::{}.json".format(dataset_name))

    dataset_json = None
    with open(dataset_json_path, "r") as file:
        dataset_json = json.load(file)
    original_options = dataset_json["options"]

    standard_experiment_option_ranges = {**INFERENCE_OPTION_RANGES,
                                         **FIRST_ORDER_WL_METHODS_STANDARD_OPTION_RANGES}

    for method in FIRST_ORDER_WL_METHODS:
        for split in SPLITS:
            split_out_dir = os.path.join(base_out_dir, "{}/split::{}".format(method, split))
            os.makedirs(split_out_dir, exist_ok=True)

            # Iterate over every combination options values.
            method_options_dict = {**standard_experiment_option_ranges,
                                   **FIRST_ORDER_WL_METHODS_OPTION_RANGES[method]}
            for options in util.enumerate_hyperparameters(method_options_dict):
                experiment_out_dir = split_out_dir
                for key, value in sorted(options.items()):
                    experiment_out_dir = os.path.join(experiment_out_dir, "{}::{}".format(key, value))

                os.makedirs(experiment_out_dir, exist_ok=True)

                if os.path.exists(os.path.join(experiment_out_dir, "out.txt")):
                    print("Skipping experiment: {}.".format(experiment_out_dir))
                    continue

                dataset_json.update({"options":{**original_options,
                                                **STANDARD_DATASET_OPTIONS[dataset_name],
                                                **STANDARD_EXPERIMENT_OPTIONS,
                                                **options,
                                                "runtime.learn.output.model.path": "./citation_learned.psl"}})

                dataset_json["predicates"]["Neural/2"]["options"]["learning-rate"] = options["gradientdescent.stepsize"]

                # Set the data path.
                set_data_path(dataset_json, split)

                # Write the options the json file.
                with open(os.path.join(CLI_DIR, "citation.json"), "w") as file:
                    json.dump(dataset_json, file, indent=4)

                # Run the experiment.
                print("Running experiment: {}.".format(experiment_out_dir))
                exit_code = os.system("cd {} && ./run.sh {} > out.txt 2> out.err".format(CLI_DIR, experiment_out_dir))

                if exit_code != 0:
                    print("Experiment failed: {}.".format(experiment_out_dir))
                    exit()

                # Save the output and json file.
                os.system("mv {} {}".format(os.path.join(CLI_DIR, "out.txt"), experiment_out_dir))
                os.system("mv {} {}".format(os.path.join(CLI_DIR, "out.err"), experiment_out_dir))
                os.system("cp {} {}".format(os.path.join(CLI_DIR, "citation.json"), experiment_out_dir))
                os.system("cp {} {}".format(os.path.join(CLI_DIR, "citation_learned.psl"), experiment_out_dir))
                os.system("cp -r {} {}".format(os.path.join(CLI_DIR, "inferred-predicates"), experiment_out_dir))


def main():
    for dataset in DATASETS:
        run_first_order_wl_methods(dataset)


if __name__ == '__main__':
    main()
