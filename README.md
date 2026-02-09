# Tahoe
Tahoe is a foray into Large Language Model (LLM) autoregressive pretraining. The provided code focuses on clarity rather than efficiency, and is mostly for self-learning purposes.

This repo largely follows Stanford's [cs336](https://stanford-cs336.github.io/spring2025/) course, though with some minor deviations.

## Data
Tahoe models are trained on two datasets: TinyStories v2 ([TSv2](https://huggingface.co/datasets/roneneldan/TinyStories)) and a sampled version of OpenWebText ([OWT](https://huggingface.co/datasets/stanford-cs336/owt-sample)). Both of these datasets can be downloaded by running `./download_data.sh`. No additional preprocessing is applied for these datasets, outside of whitespace stripping.

## Tokenization
Both TSv2 and OWT datasets are BPE tokenized. Pre-tokenization is handled by splitting along categories using GPT-2's regex pattern, and tokenization is split along document boundaries.

TSv2 and OWT tokenizers are trained up to 10k and 32k tokens, respectively. Tokenizer training leverages a single worker process, as neither dataset is large enough to warrant multi-process or multi-node training; concretely, training an OWT tokenizer for 32k tokens takes around ~6 hours on a M1 MacBook Pro. Single-process tracking also enables clever optimizations, such as **incremental tracking** where the byte-pair occurence table is built [once](https://github.com/Rostifar/tahoe/blob/5ef2c24a948dbf9c3cfb64bb48c007ac4fdf991a/tokenizer.py#L182) and incrementally updated with each merge.

TSv2 and OWT train / val datasets were then tokenized, yielding the following `.npy` files for model training:
| Dataset | Split | Tokens |
  |---|---|---|
  | TinyStoriesV2-GPT4 | train | 541,229,223 |
  | TinyStoriesV2-GPT4 | val | 5,430,373 |
  | OpenWebText | train | 2,727,120,452 |
  | OpenWebText | val | 66,401,074 |

Tokenizer training can be replicated by running `uv run tokenizer.py --{train-bpe-tsv2 | train-bpe-owt}`, and dataset tokenization can be replicated with the by running the following:
```
// tokenize OWT
uv run build_datasets.py --build-train-owt --build-val-owt
// tokenize TSv2
uv run build_datasets.py --build-train-tsv2 --build-val-tsv2
```

## Training
Model training relied on identifying the right infra (eg. compute stack, config system, experiment tracking) and several rounds of iteration per dataset. Given the trial-and-error nature of this work, I maintained a **[Work Log](https://github.com/Rostifar/tahoe/blob/4d2a5cc6ff947a0095debd1b61765a201c09cdce/Work%20Log.pdf)** which covers design decisions, experiment results, and other details. 

The sections below provide a high-level overview of infra decisions and experiment results.

### Compute
The budget for this project was roughly $100, which has been sufficient for other [practitioners](https://github.com/karpathy/nanochat). Concretely, the goal was to find a provider which supported:
1. Cheap data transfer costs for both ingress and egress. This requirement was important because I wanted to avoid paying for GBs of cold storage over the course of the project. 
2. A wide range of GPUs at different price points. Instances under $0.60 were particularly attractive because I could easily afford hyperparameter sweeps, recovering from botched training runs, etc.
There are a number of neo-clouds that satisfy this requirement: RunPod, Lambda Labs, etc. I chose RunPod because it satisfied both requirements; in retrospect, JAX + TPUs might have also been a nice choice, particularly for pre-training larger models.

A combination of A40, RTX 4090, and RTX 6000 PRO GPUs were used for model training:
* A40 (~$0.40/hr) pods were used for cheap hyperparameter tuning for TSv2.
* RTX 4090 (~$0.60/hr) pods were used for pre-training with TSv2 and smaller OWT models (<50M parameters).
* RTX 6000 PRO (~$1.60/hr) pods used for large OWT models (~100M parameters). 

Datasets, checkpoints, and code were loaded onto pods via a deploy script, which leverages `rsync`:
```
./deploy.sh "ssh root@<...>"
```

### Experiment Configuration and Tracking
Experiments are defined by YAML files under `./configs`, which encode model attributes, optimizer hyperparameters, datasets, and other metadata per experiment. 

Experiment tracking is handled by Weights & Biases.

### Results
After training, both models 