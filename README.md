# Tahoe

## Training Loop Requirements
### Args

* vocab: str (path to vocabulary)
* train_set: str (path to training set)
* val_set: str (path to validation set)

* batch_size: int (batch size of training iterations)
* context_length: int (max context length)
* num_layers: int (number of attention layers)
* d_model: int (dimension of embedding space)
* num_heads: int (number of heads for multi-head attention)
* d_ff: int (dimension of feed-forward transformation)
* theta: float (RoPE angle)

* device: 'cpu' | 'mps' | 'cuda' (device for training)
* dtype: 'fp16' | 'fp32' (data type for training)

* lr_max: float (max learning rate for the model)
* lr_min: float (min learning rate for the model)
* t_warmup: int (iterations for LR scheduler warmup)
* t_cos: int (iterations for LR scheduler cosine decay)

* betas: beta values for AdamW (default=0.99,0.999)
* weight_decay: weight decay value for AdamW (default=1e-2)
* max_grad: float (maximum grad magnitude for gradient clipping)

* ckpt_iter: int (number of iterations between checkpoints)
* ckpt_path: str (path for model checkpointing)
* from_ckpt: str | None (checkpoint to start from)

### Logic
1. Load model and optimizer from checkpoints. Otherwise, initialize model.
2. mmap `train_set` and `val_set` before entering training loop.
3. Enter training loop:
    1. Sample token batch of size `(batch_size, context_length)` from training set.
    2. Compute forward pass and loss.
    3. Backprop, step, and update LR sechduler.
    4. Every `ckpt_iter` iterations, save checkpoint.
    5. Every `val_iter` iterations, evaluate loss on training set.
