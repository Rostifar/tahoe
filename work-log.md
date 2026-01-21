## Jan. 20th, 2026
Today was spent finishing up some remaining TinyStories experiments.
#### Experiment: LayerNorm, Pre-norm, NoPE, and SwiGLU Ablations
These experiments tested the impact of model architecture choices, including LayerNorm, pre-norm, and SwiGLU. In the first experiment, LayerNorm was removed entirely; the second experiment replaces pre-norm LayerNorm with post-norm. Following experiments removed RoPE position encodings, and SwiGLU with Swish.

Removing LayerNorm led to significant instability during training (top), which was later mitigated by selecting a lower initial learning rate (bottom):
![[Pasted image 20260120231534.png]]
![[Pasted image 20260120231620.png]]
Even with a lower learning rate, validation loss still remained elevated. This is likely due to elevated gradients early in training:
![[Pasted image 20260120231933.png]]
Removing RoPE has a marginal impact, elevating validation loss at 20k steps by around ~0.20:
![[Pasted image 20260120232044.png]]
The story is similar with SwiGLU:
![[Pasted image 20260120232110.png]]
## Jan. 19th, 2026
Primary goal for today is to finish out my TSv2 experiments. Before going over results, I'll briefly outline the experimentation setup I've landed on.
#### Experiment and Compute Setup
I'm using RunPod as my primary GPU provider, for several reasons: it's almost always the cheaper option, solid availability of previous gen GPUs (e.g A-series), and no ingress/egress data pricing.

My compute budget is around  $\approx \$ 0.50$ per hour. I've experimented with the RTX 6000 Ada and A40; the latter is around ~2x faster than the former while also being ~2x as expensive. I've stuck with the A40 to reduce experiment durations.

All experiments are managed by YAML configs. It's a little cumbersome but is much simpler than remembering how experiment *X* was configured or relying on logs. Each YAML file is a one-to-many association between a top-level experiment (eg. trying out different batch sizes) and sub-experiments (eg. trying out `batch_size: 32`).

Code is pushed to remote via rsync, with code changes always happening locally. With this setup, writes to remote are idempotent, and there's no need to worry about git credentials or containers. scp is used for remote to local data transfer.

All logging happens over WandB. I was skeptical at first, but it alleviates the need to handle a Tensorboard setup, which I like.
#### Experiment: LR Sweep
This experiment involved tuning learning rate with all other hyperparams fixed. AdamW is used for all experiments. An cosine annealing LR scheduler is used, outside of [tiny_stories_no_scheduler_1e_4](https://wandb.ai/torusai/tiny_stories_no_scheduler_1e_4). 

| Name                                                                                          | LR_MIN | LR_MAX | n_warmup | Iterations | Train Loss | Val Loss |
| --------------------------------------------------------------------------------------------- | ------ | ------ | -------- | ---------- | ---------- | -------- |
| [tiny_stories_no_scheduler_1e_4](https://wandb.ai/torusai/tiny_stories_no_scheduler_1e_4)     | 3e-4   | 3e-4   | 5%       | ~5000      | 1.68       | 1.89     |
| [tiny_stories_lrgrid_5p_3ne1_1ne5](https://wandb.ai/torusai/tiny_stories_lrgrid_5p_3ne1_1ne5) | 3e-1   | 1e-5   | 5%       | ~20k       | 3.82       | 3.87     |
| [tiny_stories_lrgrid_5p_3ne2_1ne5](https://wandb.ai/torusai/tiny_stories_lrgrid_5p_3ne2_1ne5) | 3e-2   | 1e-5   | 5%       | 750        | 4.36       | 3.40     |
| [tiny_stories_lrgrid_5p_3ne4_1ne5](https://wandb.ai/torusai/tiny_stories_lrgrid_5p_3ne4_1ne5) | 3e-4   | 1e-5   | 5%       | 11000      | **1.58**   | **1.69** |
| [tiny_stories_lrgrid_5p_3ne3_1ne5](https://wandb.ai/torusai/tiny_stories_lrgrid_5p_3ne3_1ne5) | 3e-5   | 1e-5   | 5%       | 11000      | 2.19       | 2.31     |
| [tiny_stories_lrgrid_5p_3ne5_1ne5](https://wandb.ai/torusai/tiny_stories_lrgrid_5p_3ne5_1ne5) | 3e-5   | 1e-5   | 5%       | 8700       | 2.31       | 2.47     |
| [tiny_stories_lrgrid_5p_3ne2_1ne2](https://wandb.ai/torusai/tiny_stories_lrgrid_5p_3ne2_1ne2) | 3e-2   | 1e-2   | 5%       | 15000      | 3.83       | 3.85     |
| [tiny_stories_lrgrid_5p_3ne3_1ne3](https://wandb.ai/torusai/tiny_stories_lrgrid_5p_3ne3_1ne3) | 3e-3   | 1e-3   | 5%       | 9500       | 2.39       | 2.41     |
![[Pasted image 20260119114527.png]] 
These experiments suffer from some step-size bias since some runs terminated early due to node restarts; however, a trend is clear: 
* Cosine annealing appears to help, though marginally for smaller models. Notice that the no scheduler and best cosine annealing runs are close in validation loss.
* Large starting max LRs may lead to increasing or flat loss due to too much movement on the loss surface. 
* Setting min LR to a smaller LR value helps with learning later in the learning process.
From this, some practical advice:
* Start with a largish LR and slowly taper down until training is stable. A higher LR happens with early exploration and escaping local minima.
* Decaying to a smaller LR after, say 5% of steps, helps with stable learning. 
* It may help to keep a higher LR if the model doesn't appear to be learning after a fixed point.
#### Experiment: Batch Sweep
This is a test in how batch size impacts model performance during training. I attempted several different batch sizes: 1, 64, 128, and 256; I attempted higher batch sizes but subsequently ran out of VRAM. Some of these runs were *not* fully a one-to-one comparison because I did not adjust step size to account for increased token usage per batch; effectively, large runs saw a larger batch size for a higher proportion of the run.

Using a high batch size has proved unstable during training due to OOMs. Turns out I was accumulating gradients during validation and had some other anti-patterns (eg. logging tensor v. item losses)!

| Name                                                                                       | BATCH_SIZE | Iterations | Train Loss | Val Loss |
| ------------------------------------------------------------------------------------------ | ---------- | ---------- | ---------- | -------- |
| [tiny_stories_batch_1](https://wandb.ai/torusai/tiny_stories_batch_1?nw=nwuserrbriden)     | 1          | 18k        | 2.05       | 2.34     |
| [tiny_stories_batch_64](https://wandb.ai/torusai/tiny_stories_batch_64?nw=nwuserrbriden)   | 64         | 11k        | 1.51       | 1.62     |
| [tiny_stories_batch_128](https://wandb.ai/torusai/tiny_stories_batch_128?nw=nwuserrbriden) | 128        | 7.5k       | 1.52       | 1.62     |
Large batches fair better than small batches; small batches tend to produce spiky movements over the loss surface due to updating descending along a single example. 
#### Items
- [x] Finish LR sweep findings
- [x] Finish Batch Sweep
	- [x] 256
	- [x] 2048
- [ ] Ablations
	- [ ] RMSNorm
## Jan. 18th, 2026
I'm continuing my training work from yesterday, with the primary goal being to finish out all experiments with TSv2. 
#### GPU Utilization
While running concurrent experiments, I realized *one* experiment was enough to saturate the GPU:
![[Pasted image 20260118180752.png|500]]
Notice that the GPU is running at 97% utilization. In this state, launching another experiment is fairly counterproductive; namely, each process experiences a reduction in FLOPS due to context switching between work. I'll keep this in mind in the future.
#### LR Sweep Results
More curves to come tomorrow!
#### Items
- [x] Refactor codebase to support concurrent experiments
- [x] No scheduler run
- [x] Perform hyperparameter sweep
	- [x] tiny_stories_lrgrid_5p_3ne1_1ne5
	- [x] tiny_stories_lrgrid_5p_3ne2_1ne5
	- [x] tiny_stories_lrgrid_5p_3ne3_1ne5
	- [x] tiny_stories_lrgrid_5p_3ne4_1ne5
	- [x] tiny_stories_lrgrid_5p_3ne5_1ne5
	- [x] tiny_stories_lrgrid_5p_3ne2_1ne2
	- [x] tiny_stories_lrgrid_5p_3ne3_1ne3
- [ ] Perform batch size sweep
	- [ ] 64, 128, 512, 1024, 4096

## Jan. 17th, 2026
I have a complete model architecture, tokenized data, and a training loop. The next step is to train a model and understand the processes around training.

Two datasets to cover: TinyStoriesV2 (TSv2) and sampled OpenWebText (OWT). I'll focus most of my time on TSv2, at least for today, because it's less resource intensive. There are a number of hyperparameters to tune, particularly model architecture (eg. $d_{\text{model}}$), LR, and batch size. Concretely:

|      | Parameter      | Category               | GPT-2      | cs336 (TSv2) |
| ---- | -------------- | ---------------------- | ---------- | ------------ |
| *1*  | vocab_size     | data                   | 50,257     | 10,000       |
| *2*  | batch_size     | data                   | 512        | Tentative    |
| *3*  | context_length | data / model           | 1024       | 256+         |
| *4*  | num_layers     | model                  | 12-48      | 4            |
| *5*  | d_model        | model                  | 768-1600   | 512          |
| *6*  | num_heads      | model                  | 12-25^a    | 16           |
| *7*  | d_ff           | model                  | 3072-6400  | 1344         |
| *8*  | theta          | model                  | N/A        | 10000        |
| *9*  | lr_max         | LR / hyperparameter    | 2.5e-4     | Tentative    |
| *10* | lr_min         | LR / hyperparameter    | N/A        | Tentative    |
| *11* | t_warmup       | LR / hyperparameter    | N/A        | Tentative    |
| *12* | t_cos          | LR / hyperparameter    | N/A        | Tentative    |
| *13* | betas          | AdamW / hyperparameter | 0.9, 0.999 | Tentative    |
| *14* | weight_decay   | AdamW / hyperparameter | 0.01       | Tentative    |
| *15* | max_grad       | hypterparameter        | 1.0        | Tentative    |
*a - chosen to be a multiple of 64.*
#### Data
The simplest place to start is with **data** and some essential questions:
* How large of a vocab should we use? The is the long tail of tokens? What should be the truncation value?
	* **A**. For TSv2, 10k is a solid truncation point, covering ~99.9% of training tokens.
* How many tokens should a given training run use?
	* **A**. $\geq 20 \cdot \text{model size}$, according to Chinchilla scaling laws. 
Vocab size impacts embedding layer and LM layer weights, and for a small model, these values will contribute significantly to model size. For TSv2, the cumulative distribution of tokens looks like:

![[Pasted image 20260117102049.png|500]]
Around 10k tokens, the CDF effective converges to 1.0, with a density loss of <0.1%:

```
# yielded by `tokens.py`
> 0: 0.0; density_loss=1.0; token_loss=536813210
> 1000: 0.8227953201822288; density_loss=0.17720467981777122; token_loss=95125813
> 2000: 0.9216566745814619; density_loss=0.07834332541853806; token_loss=42055732
> 3000: 0.9545258079621401; density_loss=0.04547419203785985; token_loss=24411147
> 4000: 0.9729588621710706; density_loss=0.027041137828929362; token_loss=14516040
> 5000: 0.9848745879409339; density_loss=0.015125412059066146; token_loss=8119521
> 6000: 0.9909489298894116; density_loss=0.009051070110588366; token_loss=4858734
> 7000: 0.9940057194196097; density_loss=0.00599428058039031; token_loss=3217809
> 8000: 0.9957168714234883; density_loss=0.004283128576511697; token_loss=2299240
> 9000: 0.9967888197088145; density_loss=0.0032111802911855003; token_loss=1723804
> 10000: 0.9975380821943632; density_loss=0.0024619178056367597; token_loss=1321590
```

Truncation is probably safe around that point. I can either compact the vocab or random sample. I'll probably just compact the vocab, training, and validation data. Both approaches prevent biasing towards a particular token. After further investigation, I realized I made a mistake and 10k it the *expected vocab size*.

The next point is on training set size provided model capacity, which is turn is driven by GPU limits. Chinchilla's guidance is 20 tokens per parameter. LLama overtrains by a factor of 10x? It seems like the general guidance is as much tokens as possible. The regime you want to avoid is large model + minimal tokens.

Working backwards, we have $538,511,524$ training tokens with TSv2; assuming Chinchilla scaling laws, this yields a model size of $\leq 538,511,524 / 20 \approx \boxed{25M}$ parameters. This is a rough limit, agnostic of GPU resources.

Small update. My tokenization code was wrong. `<|endoftext|>` tokens were being filtered out :( My decoding code was also wrong. I should've sanity checked the outputs for these files.
#### Model
Based on the analysis above, the model should require $\leq 25M$ parameters. I'll rely on the parameters provided with cs336. 

I was able to train the model to a reasonable state on RunPod. Next, I'll handle grid searching for hyperparams.
#### Compute
I don't need a ton of compute for these tests. I'm considering the following test setup:
* RTX 4000 Ada (20GB vRAM) on RunPod.
* Nvidia A10 on Lambda Labs
#### Training Dynamics
This section is about optimizers, particularly hyperparameters. The first area of focus is *learning rate*, which controls the step size in gradient descent. The model is using a cosine scheduler that's parameterized by: 
* $t$: the current iteration.
* $\alpha_{\text{max}}$: the max learning rate.
* $\alpha_{\text{min}}$: the min learning rate.
* $T_{w}$: the warmup learning rate.
* $T_{c}$: the cosine learning rate.
The cosine scheduler starts by warming up (ie. increasing) the learning rate $\frac{1}{T_{w}} \alpha_{\text{max}}, \dots, \alpha_{\text{max}}$; linear growth. After warm-up, the learning rate is decayed via a cosine-interpolation until it hits $\alpha_{\text{min}}$ at $T_{c}$:
![[Pasted image 20260117195601.png|400]]

#### Experiment 1: Basic Fitting
This run leveraged an RTX 4000 Ada on RunPod.

The parameters for this run are under `./configs/tiny-stories-1.yaml`. Training loop was run for 5000 iterations, yielding $32 * 256 * 5000 \approx 40M$ seen tokens.

Training loss reduced to $\leq 2.0$:
![[Pasted image 20260117170417.png]]
Validation loss was not monitored, unfortunately. After 5000 iterations, the model was able to generate realistic sounding English sentences. The training set, however, did not include any special tokens, including `<|endoftext|>` tokens, leading to meandering responses. I need t 
#### Items
- [x] Truncate TSv2 to 10k tokens.
	- [x] Investigation estimated token coverage.
	- [x] Retrain vocab.
	- [x] Retrain train set on 10k reduction.
	- [x] Retrain val set on 10k reduction.
- [x] Retrain TSv2 and OWT datasets.
	- [x] Debug missing special tokens.
	- [x] Retrain OWT.
	- [x] Retrain TSv2.
- [x] o11y
	- [x] Enable wandb logging for validation loss. 
- [ ] Training Validation
	- [ ] Validation loss drop over 1000 iterations.
- [ ] Parallel grid search / sweep for hyperparams