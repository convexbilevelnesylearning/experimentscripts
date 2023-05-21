#!/usr/bin/env bash

# Run all the experiments.

PSL_DATASETS='epinions citeseer cora yelp'

function main() {
    trap exit SIGINT

    # dataset paths to pass to scripts
    psl_dataset_paths=''
    for dataset in $PSL_DATASETS; do
        psl_dataset_paths="${psl_dataset_paths}psl-examples/${dataset} "
    done

    ./scripts/setup_psl_examples.sh
    for dataset in $PSL_DATASETS; do
        echo "Running psl dual bcd regularization inference experiments on dataset: ${dataset}."
        python3 ./scripts/run_dual_bcd_inference_regularization_experiments.py ${dataset}
    done

    ./scripts/setup_psl_examples.sh
    for dataset in $PSL_DATASETS; do
        echo "Running psl weight learning performance experiments on dataset: ${dataset}."
        python3 ./scripts/run_weight_learning_performance_experiments.py ${dataset}
    done

    ./scripts/setup_psl_examples.sh
    for dataset in $PSL_DATASETS; do
        echo "Running psl weight learning inference timing experiments on dataset: ${dataset}."
        python3 ./scripts/run_weight_learning_inference_timing_experiments.py ${dataset}
    done


}

main "$@"
