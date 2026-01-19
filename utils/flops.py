def param_estimate(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    d_ff: int
) -> int:
    return 2 * vocab_size * d_model + (2 * num_layers + 1) * d_model\
        + (4 * num_layers) * d_model**2 + 3 * num_layers * d_model * d_ff

def estimate_flops(
    vocab_size: int,
    d_model: int,
    num_layers: int,
    d_ff: int,
    seq_len: int
) -> None:
    print("===Estimating Flops===")
    print(f"Parameters: " + ", ".join([f"{k}={v}" for k, v in locals().items() if k != 'self']))
    info = dict(
        attn_kqv_proj=num_layers * (6 * seq_len * d_model**2),
        attn_self_attn=num_layers * (4 * seq_len**2 * d_model),
        attn_head_proj=num_layers * 2 * seq_len * d_model**2,
        attn_fcn=num_layers * (6 * seq_len * d_model * d_ff),
        lm_proj=2 * seq_len * vocab_size * d_model
    )
    total_flops = sum(info.values())
    for k, v in info.items():
        proportion = v / total_flops if total_flops > 0 else 0
        print(f"{k} := {v / 10**12} TFLOPS ({proportion:.2%} of total)")
    print(f"Total TFLOPS: {total_flops / 10**12}\n")

if __name__ == "__main__":
    GPT2_S =  dict(vocab_size = 50257, seq_len=1024, num_layers=12, d_model=768, d_ff=6400)
    estimate_flops(**GPT2_S)

    GPT2_M =  dict(vocab_size = 50257, seq_len=1024, num_layers=24, d_model=1024, d_ff=6400)
    estimate_flops(**GPT2_M)

    GPT2_XL = dict(vocab_size = 50257, seq_len=1024, num_layers=48, d_model=1600, d_ff=6400)
    estimate_flops(**GPT2_XL)
