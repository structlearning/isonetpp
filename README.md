# ISONET++

[Iteratively Refined Early Interaction Alignment for Subgraph Matching based Graph Retrieval (NeruIPS24)](https://openreview.net/forum?id=udTwwF7tks)

This directory contains code necessary for running all the experiments.


# Steps for training/running the models

### Setting up the environment
- Modify the `prefix` in `environment.yaml` to point to the location where the environment will be stored.
- Create a `conda` environment using

    `conda env create -f environment.yaml`
- Activate the environment

    `conda activate einsmatch`

### Training a model
- Navigate to `scripts/`. You are supposed to run the `run_custom.sh` script.
- Set the `gpus` variable to indicate a tuple of all the GPU indices available for the experiment. If just GPU 2 is available, set it to `(2)`. Any non-zero length for the list works.
- Run `bash run_custom.sh` on the command line. This will start training the model. The model evaluates on the test dataset at the end of training by default.
- Note that in our original codebase, we have used `wandb` to manage and monitor runs. However, we have set `WANDB_MODE=disabled` in the bash script since we don't expect every user to be familiar with `wandb`. In case the user has experience using it, the `WANDB_MODE=disabled` part of the command can be deleted, so that it starts as such - `CUDA_VISIBLE_DEVICES=...`
- Results will be stored in the `<experiment_dir>/<experiment_id>` directory, which in this case is `experiments/rqX_custom_models`. This includes trained models, partial configs and logs. The train/validation scores are printed at every epoch in the corresponding log file, and the test score is evaluated at the end of training.
- Note that the `experiments` directory in this folder is **intentionally** kept empty. It will be the home for logs and configs of any new training runs that are started by the user.

### Additional files
- We provide additional files here - https://rebrand.ly/ohmmjfi.

### Model names
- Some models have a different naming convention in the codebase than in the paper.

- Key models

    1. Multi-round EinsMatch (Node) - `configs\rq4_iterative\iterative___scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=pre___unify=true.yaml`

    2. Multi-layer EinsMatch (Node) - `configs\rq4_baselines\scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=pre___unify=true.yaml`

    3. Multi-round EinsMatch (Edge) - `configs\edge_early_variants\edge_early_interaction.yaml`

    4. Multi-layer EinsMatch (Edge) - `configs\edge_early_variants\edge_early_interaction_baseline.yaml`

    5. GMN-Match - `configs\rq4_baselines\scoring=agg___tp=attention_pp=identity_when=post.yaml`

- Ablations

    1. Multi-round Node pair partner (msg-only) - `configs\rq8\iterative___scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=msg_passing_only___unify=true.yaml`

    2. Multi-layer Node pair partner (msg-only) - `configs\rq8\scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=msg_passing_only___unify=true.yaml`

    3. Multi-round Node partner (with additional MLP) - `configs\rq8\iterative___scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=update_only___unify=true.yaml`

    4. Multi-layer Node partner (with additional MLP) - `configs\rq8\scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=update_only___unify=true.yaml`

    5. Multi-round Node partner - `configs\rq4_iterative\iterative___scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=post___unify=true.yaml`

    6. Multi-layer Node partner - `configs\rq4_baselines\scoring=sinkhorn_pp=lrl___tp=sinkhorn_pp=lrl_when=post___unify=true.yaml`

- Configs to perform ablations on the (T,K) parameters for multi-round and (K) parameter for multi-layer can be found in `configs\rq7_efficiency`.

Reference
---------

If you find the code useful, please cite our paper:

    @inproceedings{ramachandraniteratively,
      title={Iteratively Refined Early Interaction Alignment for Subgraph Matching based Graph Retrieval},
      author={Ramachandran, Ashwin and Raj, Vaibhav and Roy, Indradyumna and Chakrabarti, Soumen and De, Abir},
      booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
    }

Indradyumna Roy, Indian Institute of Technology - Bombay  
indraroy15@cse.iitb.ac.in
