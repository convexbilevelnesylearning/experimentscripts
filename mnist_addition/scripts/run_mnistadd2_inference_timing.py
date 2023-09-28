#!/usr/bin/env python3

import json
import os
import re
import sys

THIS_DIR = os.path.join(os.path.dirname(os.path.realpath(__file__)))

sys.path.append(os.path.join(THIS_DIR, '..', '..'))
import util

MNIST_CLI_DIR = os.path.join(THIS_DIR, "../cli")
RESULTS_BASE_DIR = os.path.join(THIS_DIR, "../results")
PERFORMANCE_RESULTS_DIR = os.path.join(RESULTS_BASE_DIR, "inference_timing")

SPLITS = ["0", "1", "2", "3", "4"]
TRAIN_SIZES = ["06000"]
OVERLAPS = ["0.00"]

STANDARD_EXPERIMENT_OPTIONS = {
    "inference.normalize": "false",
    "runtime.log.level": "TRACE",
    "gradientdescent.scalestepsize": "false",
    "gradientdescent.trainingcomputeperiod": "5",
    "gradientdescent.validationcomputeperiod": "5",
    "gradientdescent.validationbreak": "true",
    "gradientdescent.validationpatience": "50",
    "gradientdescent.stopcomputeperiod": "5",
    "gradientdescent.numsteps": "500",
    "gradientdescent.runfulliterations": "false",
    "duallcqp.computeperiod": "10",
    "duallcqp.maxiterations": "50000",
    "runtime.validation": "true",
    "gradientdescent.savevalidationweights": "true",
    "gradientdescent.batchgenerator": "ConnectedComponentBatchGenerator"
}

FIRST_ORDER_WL_METHODS = ["BinaryCrossEntropy"]
INFERENCE_METHODS = ["DualBCDInference", "DistributedDualBCDInference", "ADMMInference"]

INFERENCE_METHOD_OPTION_RANGES = {
    "DualBCDInference": {
        "weightlearning.inference": ["DualBCDInference"],
        "runtime.inference.method": ["DualBCDInference"],
        "duallcqp.regularizationparameter": ["1.0e-3"],
    },
    "DistributedDualBCDInference": {
        "weightlearning.inference": ["DualBCDInference"],
        "runtime.inference.method": ["DistributedDualBCDInference"],
        "duallcqp.regularizationparameter": ["1.0e-3"],
    },
    "ADMMInference": {
        "weightlearning.inference": ["DualBCDInference"],
        "runtime.inference.method": ["ADMMInference"],
        "admmreasoner.stepsize": ["1.0"],
    }
}

BEST_HYPERPARAMETERS = {
    "BinaryCrossEntropy": {
        "06000": {
            "0.00": {
                "runtime.learn.method": "BinaryCrossEntropy",
                "minimizer.initialsquaredpenalty": "2.0",
                "minimizer.objectivedifferencetolerance": "1.0e-3",
                "minimizer.finalparametermovementconvergencetolerance": "1.0e-1",
                "minimizer.proxruleweight": "1.0e-3",
                "minimizer.proxvaluestepsize": "1.0e-3",
                "minimizer.squaredpenaltyincreaserate": "2.0",
                "minimizer.energylosscoefficient": "10.0",
                "gradientdescent.stepsize": "1.0e-14",
                "gradientdescent.negativelogregularization": "1.0e-3",
                "gradientdescent.negativeentropyregularization": "0.0",
                "connectedcomponents.batchsize": "32",
            }
        },
    }
}

BEST_NEURAL_NETWORK_HYPERPARAMETERS = {
    "BinaryCrossEntropy": {
        "06000": {
            "0.00": {
                "dropout": "0.0",
                "weight_decay": "1.0e-4",
                "neural_learning_rate": "1.0e-4",
                "learning_rate_decay_step": "30",
                "learning_rate_decay": "1.0",
                "temperature_decay_rate": "1.0e-5",
                "transforms": "false",
                "freeze_resnet": "false"
            }
        }
    }
}


def set_data_path(dataset_json, split, train_size, overlap):
    # sed dataset paths
    for predicate in dataset_json["predicates"]:
        if "options" in dataset_json["predicates"][predicate]:
            if "entity-data-map-path" in dataset_json["predicates"][predicate]["options"]:
                dataset_json["predicates"][predicate]["options"]["entity-data-map-path"] = \
                    re.sub(r"split::[0-9]+", "split::{}".format(split), dataset_json["predicates"][predicate]["options"]["entity-data-map-path"])
                dataset_json["predicates"][predicate]["options"]["entity-data-map-path"] = \
                    re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), dataset_json["predicates"][predicate]["options"]["entity-data-map-path"])
                dataset_json["predicates"][predicate]["options"]["entity-data-map-path"] = \
                    re.sub(r"overlap::[0-9]+\.[0-9]+", "overlap::{}".format(overlap), dataset_json["predicates"][predicate]["options"]["entity-data-map-path"])

            if "save-path" in dataset_json["predicates"][predicate]["options"]:
                dataset_json["predicates"][predicate]["options"]["save-path"] = \
                    re.sub(r"split::[0-9]+", "split::{}".format(split), dataset_json["predicates"][predicate]["options"]["save-path"])
                dataset_json["predicates"][predicate]["options"]["save-path"] = \
                    re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), dataset_json["predicates"][predicate]["options"]["save-path"])
                dataset_json["predicates"][predicate]["options"]["save-path"] = \
                    re.sub(r"overlap::[0-9]+\.[0-9]+", "overlap::{}".format(overlap), dataset_json["predicates"][predicate]["options"]["save-path"])

        if "targets" in dataset_json["predicates"][predicate]:
            if "learn" in dataset_json["predicates"][predicate]["targets"]:
                dataset_json["predicates"][predicate]["targets"]["learn"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), target) for target in dataset_json["predicates"][predicate]["targets"]["learn"]]
                dataset_json["predicates"][predicate]["targets"]["learn"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), target) for target in dataset_json["predicates"][predicate]["targets"]["learn"]]
                dataset_json["predicates"][predicate]["targets"]["learn"] = \
                    [re.sub(r"overlap::[0-9]+\.[0-9]+", "overlap::{}".format(overlap), target) for target in dataset_json["predicates"][predicate]["targets"]["learn"]]

            if "validation" in dataset_json["predicates"][predicate]["targets"]:
                dataset_json["predicates"][predicate]["targets"]["validation"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), target) for target in dataset_json["predicates"][predicate]["targets"]["validation"]]
                dataset_json["predicates"][predicate]["targets"]["validation"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), target) for target in dataset_json["predicates"][predicate]["targets"]["validation"]]
                dataset_json["predicates"][predicate]["targets"]["validation"] = \
                    [re.sub(r"overlap::[0-9]+\.[0-9]+", "overlap::{}".format(overlap), target) for target in dataset_json["predicates"][predicate]["targets"]["validation"]]

            if "infer" in dataset_json["predicates"][predicate]["targets"]:
                dataset_json["predicates"][predicate]["targets"]["infer"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), target) for target in dataset_json["predicates"][predicate]["targets"]["infer"]]
                dataset_json["predicates"][predicate]["targets"]["infer"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), target) for target in dataset_json["predicates"][predicate]["targets"]["infer"]]
                dataset_json["predicates"][predicate]["targets"]["infer"] = \
                    [re.sub(r"overlap::[0-9]+\.[0-9]+", "overlap::{}".format(overlap), target) for target in dataset_json["predicates"][predicate]["targets"]["infer"]]

        if "truth" in dataset_json["predicates"][predicate]:
            if "learn" in dataset_json["predicates"][predicate]["truth"]:
                dataset_json["predicates"][predicate]["truth"]["learn"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["learn"]]
                dataset_json["predicates"][predicate]["truth"]["learn"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["learn"]]
                dataset_json["predicates"][predicate]["truth"]["learn"] = \
                    [re.sub(r"overlap::[0-9]+\.[0-9]+", "overlap::{}".format(overlap), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["learn"]]

            if "validation" in dataset_json["predicates"][predicate]["truth"]:
                dataset_json["predicates"][predicate]["truth"]["validation"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["validation"]]
                dataset_json["predicates"][predicate]["truth"]["validation"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["validation"]]
                dataset_json["predicates"][predicate]["truth"]["validation"] = \
                    [re.sub(r"overlap::[0-9]+\.[0-9]+", "overlap::{}".format(overlap), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["validation"]]

            if "infer" in dataset_json["predicates"][predicate]["truth"]:
                dataset_json["predicates"][predicate]["truth"]["infer"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["infer"]]
                dataset_json["predicates"][predicate]["truth"]["infer"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["infer"]]
                dataset_json["predicates"][predicate]["truth"]["infer"] = \
                    [re.sub(r"overlap::[0-9]+\.[0-9]+", "overlap::{}".format(overlap), observation) for observation in
                     dataset_json["predicates"][predicate]["truth"]["infer"]]

        if "observations" in dataset_json["predicates"][predicate]:
            if "learn" in dataset_json["predicates"][predicate]["observations"]:
                dataset_json["predicates"][predicate]["observations"]["learn"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in dataset_json["predicates"][predicate]["observations"]["learn"]]
                dataset_json["predicates"][predicate]["observations"]["learn"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), observation) for observation in dataset_json["predicates"][predicate]["observations"]["learn"]]
                dataset_json["predicates"][predicate]["observations"]["learn"] = \
                    [re.sub(r"overlap::[0-9]+\.[0-9]+", "overlap::{}".format(overlap), observation) for observation in dataset_json["predicates"][predicate]["observations"]["learn"]]

            if "validation" in dataset_json["predicates"][predicate]["observations"]:
                dataset_json["predicates"][predicate]["observations"]["validation"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in dataset_json["predicates"][predicate]["observations"]["validation"]]
                dataset_json["predicates"][predicate]["observations"]["validation"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), observation) for observation in dataset_json["predicates"][predicate]["observations"]["validation"]]
                dataset_json["predicates"][predicate]["observations"]["validation"] = \
                    [re.sub(r"overlap::[0-9]+\.[0-9]+", "overlap::{}".format(overlap), observation) for observation in dataset_json["predicates"][predicate]["observations"]["validation"]]

            if "infer" in dataset_json["predicates"][predicate]["observations"]:
                dataset_json["predicates"][predicate]["observations"]["infer"] = \
                    [re.sub(r"split::[0-9]+", "split::{}".format(split), observation) for observation in dataset_json["predicates"][predicate]["observations"]["infer"]]
                dataset_json["predicates"][predicate]["observations"]["infer"] = \
                    [re.sub(r"train-size::[0-9]+", "train-size::{}".format(train_size), observation) for observation in dataset_json["predicates"][predicate]["observations"]["infer"]]
                dataset_json["predicates"][predicate]["observations"]["infer"] = \
                    [re.sub(r"overlap::[0-9]+\.[0-9]+", "overlap::{}".format(overlap), observation) for observation in dataset_json["predicates"][predicate]["observations"]["infer"]]


def run_learning_timing():
    base_out_dir = os.path.join(PERFORMANCE_RESULTS_DIR, "mnist-add2", "first_order_wl_methods")
    os.makedirs(base_out_dir, exist_ok=True)

    # Load the json file with the dataset options.
    dataset_json_path = os.path.join(MNIST_CLI_DIR, "neupsl_models/experiment::mnist-add2.json")

    dataset_json = None
    with open(dataset_json_path, "r") as file:
        dataset_json = json.load(file)
    original_options = dataset_json["options"]

    for method in FIRST_ORDER_WL_METHODS:
        method_out_dir = os.path.join(base_out_dir, method)
        os.makedirs(method_out_dir, exist_ok=True)

        for inference_method in INFERENCE_METHODS:
            inference_method_out_dir = os.path.join(method_out_dir, inference_method)
            os.makedirs(inference_method_out_dir, exist_ok=True)

            # Iterate over every combination options values.
            inference_method_options_dict = {**INFERENCE_METHOD_OPTION_RANGES[inference_method]}
            for inference_method_options in util.enumerate_hyperparameters(inference_method_options_dict):
                for train_size in TRAIN_SIZES:
                    for overlap in OVERLAPS:
                        for split in SPLITS:
                            split_out_dir = os.path.join(inference_method_out_dir, "split::{}/train-size::{}/overlap::{}".format(split, train_size, overlap))
                            os.makedirs(split_out_dir, exist_ok=True)

                            learning_options = BEST_HYPERPARAMETERS[method][train_size][overlap]
                            dropout = BEST_NEURAL_NETWORK_HYPERPARAMETERS[method][train_size][overlap]["dropout"]
                            weight_decay = BEST_NEURAL_NETWORK_HYPERPARAMETERS[method][train_size][overlap]["weight_decay"]
                            freeze_resnet = BEST_NEURAL_NETWORK_HYPERPARAMETERS[method][train_size][overlap]["freeze_resnet"]
                            transforms = BEST_NEURAL_NETWORK_HYPERPARAMETERS[method][train_size][overlap]["transforms"]
                            neural_learning_rate = BEST_NEURAL_NETWORK_HYPERPARAMETERS[method][train_size][overlap]["neural_learning_rate"]
                            learning_rate_decay_step = BEST_NEURAL_NETWORK_HYPERPARAMETERS[method][train_size][overlap]["learning_rate_decay_step"]
                            learning_rate_decay = BEST_NEURAL_NETWORK_HYPERPARAMETERS[method][train_size][overlap]["learning_rate_decay"]
                            temperature_decay_rate = BEST_NEURAL_NETWORK_HYPERPARAMETERS[method][train_size][overlap]["temperature_decay_rate"]

                            experiment_out_dir = split_out_dir
                            for key, value in sorted(inference_method_options.items()):
                                experiment_out_dir = os.path.join(experiment_out_dir, "{}::{}".format(key, value))

                            for key, value in sorted(learning_options.items()):
                                experiment_out_dir = os.path.join(experiment_out_dir, "{}::{}".format(key, value))

                            experiment_out_dir = os.path.join(experiment_out_dir, "dropout::{}".format(dropout))
                            experiment_out_dir = os.path.join(experiment_out_dir, "weight_decay::{}".format(weight_decay))
                            experiment_out_dir = os.path.join(experiment_out_dir, "freeze_resnet::{}".format(freeze_resnet))
                            experiment_out_dir = os.path.join(experiment_out_dir, "transforms::{}".format(transforms))
                            experiment_out_dir = os.path.join(experiment_out_dir, "neural_learning_rate::{}".format(neural_learning_rate))
                            experiment_out_dir = os.path.join(experiment_out_dir, "learning_rate_decay_step::{}".format(learning_rate_decay_step))
                            experiment_out_dir = os.path.join(experiment_out_dir, "learning_rate_decay::{}".format(learning_rate_decay))
                            experiment_out_dir = os.path.join(experiment_out_dir, "temperature_decay_rate::{}".format(temperature_decay_rate))

                            os.makedirs(experiment_out_dir, exist_ok=True)

                            if os.path.exists(os.path.join(experiment_out_dir, "out.txt")):
                                print("Skipping experiment: {}.".format(experiment_out_dir))
                                continue

                            dataset_json.update({"options": {**original_options,
                                                             **STANDARD_EXPERIMENT_OPTIONS,
                                                             **inference_method_options,
                                                             **learning_options,
                                                             "runtime.learn.output.model.path": "./mnist-addition_learned.psl"}})

                            dataset_json["predicates"]["NeuralClassifier/2"]["options"]["experiment"] = "mnist-2"
                            dataset_json["predicates"]["NeuralClassifier/2"]["options"]["split"] = split
                            dataset_json["predicates"]["NeuralClassifier/2"]["options"]["train_size"] = train_size
                            dataset_json["predicates"]["NeuralClassifier/2"]["options"]["dropout"] = dropout
                            dataset_json["predicates"]["NeuralClassifier/2"]["options"]["weight_decay"] = weight_decay
                            dataset_json["predicates"]["NeuralClassifier/2"]["options"]["freeze_resnet"] = freeze_resnet
                            dataset_json["predicates"]["NeuralClassifier/2"]["options"]["transforms"] = transforms
                            dataset_json["predicates"]["NeuralClassifier/2"]["options"]["neural_learning_rate"] = neural_learning_rate
                            dataset_json["predicates"]["NeuralClassifier/2"]["options"]["learning_rate_decay_step"] = learning_rate_decay_step
                            dataset_json["predicates"]["NeuralClassifier/2"]["options"]["learning_rate_decay"] = learning_rate_decay
                            dataset_json["predicates"]["NeuralClassifier/2"]["options"]["temperature_decay_rate"] = temperature_decay_rate

                            # Set the data path.
                            set_data_path(dataset_json, split, train_size, overlap)

                            # Write the options the json file.
                            with open(os.path.join(MNIST_CLI_DIR, "mnist-addition.json"), "w") as file:
                                json.dump(dataset_json, file, indent=4)

                            # Run the experiment.
                            print("Running experiment: {}.".format(experiment_out_dir))
                            exit_code = os.system("cd {} && ./run.sh {} > out.txt 2> out.err".format(MNIST_CLI_DIR, experiment_out_dir))

                            if exit_code != 0:
                                print("Experiment failed: {}.".format(experiment_out_dir))
                                exit()

                            # Save the output and json file.
                            os.system("mv {} {}".format(os.path.join(MNIST_CLI_DIR, "out.txt"), experiment_out_dir))
                            os.system("mv {} {}".format(os.path.join(MNIST_CLI_DIR, "out.err"), experiment_out_dir))
                            os.system("cp {} {}".format(os.path.join(MNIST_CLI_DIR, "mnist-addition.json"), experiment_out_dir))
                            os.system("cp {} {}".format(os.path.join(MNIST_CLI_DIR, "mnist-addition_learned.psl"), experiment_out_dir))
                            os.system("cp -r {} {}".format(os.path.join(MNIST_CLI_DIR, "inferred-predicates"), experiment_out_dir))


def main():
    run_learning_timing()


if __name__ == '__main__':
    main()
