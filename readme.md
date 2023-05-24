Experiments for "Convex and Bilevel Optimization for Neuro-Symbolic Inference and Learning".

### Requirements
These experiments expect that you are running on a POSIX (Linux/Mac) system.
The specific application dependencies are as follows:
 - Bash >= 4.0
 - Java >= 7
 - Python >= 3.7

### Setup
These scripts assume you have already built and installed NeuPSL from our repository.
If you have not, please follow the instructions in our [NeuPSL repository](https://github.com/convexbilevelnesylearning/psl).

## Data
Data for the HL-MRF experiments in the top-level `scripts` directory will be pulled from the [psl-examples repository](https://github.com/linqs/psl-examples)
Data for the deep HL-MRF experiments must be created by running the `create_data.py` scripts in the `citation` and `mnist-addition` directories.

## Models
Models for the HL-MRF experiments in the top-level `scripts` directory will be pulled from the [psl-examples repository](https://github.com/linqs/psl-examples)
Models for the deep HL-MRF experiments are in the `citation` and `mnist-addition` directories.
An additional step is required to create the deep HL-MRF models for the `citation` experiments as the neural component is pretrained.
After creating the data, run `citation/scripts/setup-networks.py`. 
This will pre-train the neural component for the NeuPSL models.

### Running Experiments
The experiments are organized into a series of scripts.
Each script is responsible for running a single experiment.
To run all experiments, simply run the `run.sh` script in the top level directory.
To run a single experiment, run its corresponding python script.

The HL-MRF timing experiments are found in the top level `scripts` directory.
`scripts/run_dual_bcd_inference_regularization_experiments.py` runs the dual BCD regularization experiments.
`scripts/run_weight_learning_inference_timing_experiments.py` runs the weight learning runtime experiments.
`scripts/run_weight_learning_performance_experiments.py` runs the HL-MRF weight learning runtime experiments.

The deep HL-MRF experiments are in the `citation/scripts` and `mnist-addition/scripts` directories.

### Results
For the HL-MRF experiments, results will be written to the top-level `results` directory.
For the Deep HL-MRF experiments, results will be written to the `results` directory in the `citation` and `mnist-addition` directories.
To parse the results for the HL-MRF experiments, run the `parse_results.py` script in the top-level `scripts` directory.
To parse the results for the Deep HL-MRF experiments, run the `parse_results.py` script in the `citation/scripts` and `mnist-addition/scripts` directories.
