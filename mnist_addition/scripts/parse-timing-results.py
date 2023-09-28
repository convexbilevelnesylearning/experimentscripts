#!/usr/bin/env python3

import os
import re
import sys

THIS_DIR = os.path.abspath(os.path.dirname(os.path.realpath(__file__)))
RESULTS_DIR = os.path.join(THIS_DIR, '..', 'results')

LOG_FILENAME = 'out.txt'


def print_results(results):
    for experiment, result in sorted(results.items()):
        for dataset, dataset_result in sorted(result.items()):
            for experiment_group, experiment_group_result in sorted(dataset_result.items()):
                for wl_method, wl_method_result in sorted(experiment_group_result.items()):
                    for inference_method, inference_method_result in sorted(wl_method_result.items()):
                        print("Experiment: {}, Dataset: {}, Experiment Group: {}, WL Method: {}, Inference Method:{}".format(
                            experiment, dataset,experiment_group, wl_method, inference_method))
                        print(' '.join(inference_method_result['header']))
                        for row in inference_method_result['rows']:
                            print(' '.join([str(value) for value in row]))


def get_log_paths(path):
    log_paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if LOG_FILENAME == file.split("/")[-1]:
                log_paths.append(os.path.join(root, file))

    return sorted(log_paths)


def parse_log(log_path):
    results = []
    inference_times = []
    with open(log_path, 'r') as file:
        for line in file:
            if "Reasoner  - Final Objective:" in line:
                inference_times.append(float(re.search(r"Total Optimization Time: (-?\d+)", line).group(1)))

            if 'Final MAP State Validation Evaluation Metric:' in line:
                match = re.search(r': ([\d\.]+)', line)
                results.append(float(match.group(1)))

            if 'Evaluation results: Evaluator: CategoricalEvaluator, Predicate: IMAGESUM' in line:
                match = re.search(r': ([\d\.]+)', line)
                results.append(float(match.group(1)))

    results.append(sum(inference_times[:-1]))
    results.append(inference_times[-1])
    return results


def main():
    results = {}
    for experiment in sorted(os.listdir(RESULTS_DIR)):
        if experiment not in ["inference_timing", "learning_timing"]:
            continue

        results[experiment] = {dataset: dict() for dataset in sorted(os.listdir(os.path.join(RESULTS_DIR, experiment)))}
        for dataset in sorted(os.listdir(os.path.join(RESULTS_DIR, experiment))):
            results[experiment][dataset] = {experiment_group: dict() for experiment_group in
                                            sorted(os.listdir(os.path.join(RESULTS_DIR, experiment, dataset)))}
            for experiment_group in sorted(os.listdir(os.path.join(RESULTS_DIR, experiment, dataset))):
                results[experiment][dataset][experiment_group] = {method: dict() for method in sorted(
                    os.listdir(os.path.join(RESULTS_DIR, experiment, dataset, experiment_group)))}
                for wl_method in sorted(os.listdir(os.path.join(RESULTS_DIR, experiment, dataset, experiment_group))):
                    results[experiment][dataset][experiment_group][wl_method] = {method: dict() for method in sorted(
                        os.listdir(os.path.join(RESULTS_DIR, experiment, dataset, experiment_group, wl_method)))}
                    for inference_method in sorted(os.listdir(os.path.join(RESULTS_DIR, experiment, dataset, experiment_group, wl_method))):
                        results[experiment][dataset][experiment_group][wl_method][inference_method] = {'header': [], 'rows': []}
                        log_paths = get_log_paths(os.path.join(RESULTS_DIR, experiment, dataset, experiment_group, wl_method, inference_method))
                        for log_path in log_paths:
                            parts = os.path.dirname(
                                log_path.split("{}/{}/{}/{}/{}/".format(experiment, dataset, experiment_group, wl_method, inference_method))[
                                    1]).split("/")
                            if len(results[experiment][dataset][experiment_group][wl_method][inference_method]['rows']) == 0:
                                results[experiment][dataset][experiment_group][wl_method][inference_method]['header'] = [row.split("::")[0] for row in parts]

                                results[experiment][dataset][experiment_group][wl_method][inference_method]['header'].append(
                                    'Validation_Categorical_Accuracy')

                                results[experiment][dataset][experiment_group][wl_method][inference_method]['header'].append(
                                    'Test_Categorical_Accuracy')

                                results[experiment][dataset][experiment_group][wl_method][inference_method]['header'].append(
                                    'Learning_Inference_Time')

                                results[experiment][dataset][experiment_group][wl_method][inference_method]['header'].append(
                                    'Test_Inference_Time')

                            results[experiment][dataset][experiment_group][wl_method][inference_method]['rows'].append(
                                [row.split("::")[1] for row in parts])

                            for log_result in parse_log(log_path):
                                results[experiment][dataset][experiment_group][wl_method][inference_method]['rows'][-1].append(log_result)
    print_results(results)


def _load_args(args):
    executable = args.pop(0)
    if len(args) != 0 or ({'h', 'help'} & {arg.lower().strip().replace('-', '') for arg in args}):
        print("USAGE: python3 %s" % (executable,), file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    _load_args(sys.argv)
    main()
