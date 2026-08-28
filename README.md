# CSE 507 Ark+ Container

This repository contains modified Ark+ code for CSE 507. A container for this code is also provided, and can be downloaded by running the following command from the root of the repository:

```bash
wget "https://www.dropbox.com/scl/fi/u2rk1siz34lcokuifejqw/nvidia.sif?rlkey=dk0jprx8xvnxnjddjwopvqmbz&st=hmn0kvkc&dl=1" -O containers/nvidia/nvidia.sif
```

After downloading the above container, you can run experiments using the following instructions

## Experiment Scripts

The experiment_scripts directory provides scripts for running experiments. Scripts are provided for easily running the following four experiments:

- vindr_cxr_only: Train a student/teacher Ark model on VinDr-CXR only.
- chestxray14_only: Train a student/teacher Ark model on ChestX-ray14 only.
- vindr_cxr_chestxray14_cyclic: Train a student/teacher Ark model on VinDr-CXR and ChestX-ray14 cyclically.
- vindr_cxr_chestxray14_concurrent_random_sampling: Train a student/teacher Ark model on VinDr-CXR and ChestX-ray14 concurrently with random sampling.
- vindr_cxr_chestxray14_concurrent_equal_sampling: Train a student/teacher Ark model on VinDr-CXR and ChestX-ray14 concurrently with equal sampling.

(For practice 1/demo 1 you should run the experiments in the above order)

### Running Experiments

A start and resume script are provided for each of the experiments, and they both require a seed as an argument. They should be run by running the following command from the root of the repository:

```bash
# Start experiment (replace vindr_cxr_only with the experiment name, and enter any seed)
./experiment_scripts/vindr_cxr_only/start_vindr_cxr_only.sh 100

# Resume experiment (use the same seed as when you started the experiment)
./experiment_scripts/vindr_cxr_only/resume_vindr_cxr_only.sh 100
```

## Analyzing Results of Experiments

Plotting scripts are also provided for easily analyzing the results of experiments you run using this repository. Matplotlib is required for creating the plots. On sol, you can use matplotlib by running the following commands:

```bash
ml mamba
conda create -n plot python matplotlib
source activate plot
```

See the plot/ directory for more information on how to run the plotting scripts.

## References

Original Ark+ Repo

```text
https://github.com/jlianglab/Ark
```
