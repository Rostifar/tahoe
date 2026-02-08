import torch
import tokenizer as tok
from data import load_checkpoint
from train import Config
from transformer import Transformer, decode


if __name__ == "__main__":
    data_config, exp_config = Config.from_args()
    tokenizer = tok.Tokenizer(
        *tok.load(data_config.vocab), 
        special_tokens=["<|endoftext|>"]
    )
    device = torch.device("cpu")
    
    # create transformer
    model = Transformer(
        vocab_size=tokenizer.vocab_size, 
        context_length=exp_config.context_length, 
        num_layers=exp_config.num_layers,
        d_model=exp_config.d_model,
        num_heads=exp_config.num_heads,
        theta=exp_config.theta,
        d_ff=exp_config.d_ff,
        device=device,
        dtype=exp_config.get_dtype(),
        tie_weights=exp_config.tie_weights,
    )
    ckpt = load_checkpoint(exp_config.from_ckpt, model, None, device)
    model.eval()

    response = decode(
        model=model,
        prompt=torch.tensor(tokenizer.encode("Sangeetha was in the park.")),
        stop_token=tokenizer.encode("<|endoftext|>")[0],
        max_tokens=250,
        temperature=1.0,
        top_p=0.9,
    )
    print(tokenizer.decode(response))
