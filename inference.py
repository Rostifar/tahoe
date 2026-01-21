import torch
import tokenizer as tok
from data import load_checkpoint
from train import Config, get_dtype
from transformer import Transformer, decode


if __name__ == "__main__":
    config = Config.from_args()

    print("--Loading Tokenizer--")
    tokenizer = tok.Tokenizer(
        *tok.load(config.vocab), 
        special_tokens=["<|endoftext|>"]
    )
    print(f"> Vocab Size: {len(tokenizer.vocab)}")
    
    device = torch.device("cpu")
    dtype = get_dtype(config.dtype)
    
    # create transformer
    model = Transformer(
        vocab_size=tokenizer.vocab_size, 
        context_length=config.context_length, 
        num_layers=config.num_layers,
        d_model=config.d_model,
        num_heads=config.num_heads,
        theta=config.theta,
        d_ff=config.d_ff,
        device=device,
        dtype=dtype
    )
    ckpt = load_checkpoint(config.from_ckpt, model, None, device)
    model.eval()

    response = decode(
        model=model,
        prompt=tokenizer.encode("SELECT * FROM "),
        stop_token=tokenizer.encode("<|endoftext|>")[0],
        max_tokens=512,
        temperature=1.0,
        top_p=0.9,
    )
    print(tokenizer.decode(response))
