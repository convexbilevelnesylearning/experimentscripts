#!/usr/bin/env bash

# Run all the experiments.

export PYTORCH_ENABLE_MPS_FALLBACK=1

PSL_DATASETS='epinions stance-4forums stance-createdebate citeseer cora drug-drug-interaction yelp'

function main() {
    trap exit SIGINT

    ./hlmrf_learning/scripts/setup_psl_examples.sh
    for dataset in $PSL_DATASETS; do
        echo "Running psl Dual BCD inference regularization experiments on dataset: ${dataset}."
        python3 ./hlmrf_learning/scripts/run_dual_bcd_inference_regularization_experiments.py ${dataset}
    done

    ./hlmrf_learning/scripts/setup_psl_examples.sh
    for dataset in $PSL_DATASETS; do
        echo "Running psl inference timing experiments on dataset: ${dataset}."
        python3 ./hlmrf_learning/scripts/run_inference_timing_experiments.py ${dataset}
    done

    ./hlmrf_learning/scripts/setup_psl_examples.sh
    for dataset in $PSL_DATASETS; do
        echo "Running psl learning timing experiments on dataset: ${dataset}."
        python3 ./hlmrf_learning/scripts/run_weight_learning_inference_timing_experiments.py ${dataset}
    done

    ./hlmrf_learning/scripts/setup_psl_examples.sh
    for dataset in $PSL_DATASETS; do
        echo "Running psl learning performance experiments on dataset: ${dataset}."
        python3 ./hlmrf_learning/scripts/run_weight_learning_performance_experiments.py ${dataset}
    done

    pushd . >/dev/null
      cd mnist_addition
      python3 ./scripts/run_mnistadd1.py
    popd >/dev/null

    pushd . >/dev/null
      cd mnist_addition
      python3 ./scripts/run_mnistadd2.py
    popd >/dev/null

    pushd . >/dev/null
      cd mnist_addition
      python3 ./scripts/run_mnistadd1_inference_timing.py
    popd >/dev/null

    pushd . >/dev/null
      cd mnist_addition
      python3 ./scripts/run_mnistadd2_inference_timing.py
    popd >/dev/null

    pushd . >/dev/null
      cd mnist_addition
      python3 ./scripts/run_mnistadd1_learning_timing.py
    popd >/dev/null

    pushd . >/dev/null
      cd mnist_addition
      python3 ./scripts/run_mnistadd2_learning_timing.py
    popd >/dev/null

    pushd . >/dev/null
      cd citation
      python3 ./scripts/run_weight_learning_performance_experiments.py
    popd >/dev/null
}

main "$@"
