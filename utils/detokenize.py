#!/usr/bin/env python3
"""Read and detokenize .npy files containing token IDs."""

import argparse
import numpy as np
from tokenizer import Tokenizer


def main():
    parser = argparse.ArgumentParser(description="Detokenize .npy files")
    parser.add_argument("npy_file", help="Path to .npy file containing token IDs")
    parser.add_argument(
        "--tokenizer", "-t",
        default="data/tokenizers/tsv2-bpe",
        help="Path to tokenizer directory (default: data/tokenizers/tsv2-bpe)"
    )
    parser.add_argument(
        "--special-tokens", "-s",
        nargs="+",
        default=["<|endoftext|>"],
        help="Special tokens (default: <|endoftext|>)"
    )
    parser.add_argument(
        "--start", type=int, default=0,
        help="Start index in token array"
    )
    parser.add_argument(
        "--length", "-n", type=int, default=8192,
        help="Number of tokens to decode (default: 1000)"
    )
    args = parser.parse_args()

    # Load tokens
    tokens = np.load(args.npy_file, mmap_mode="r").astype(np.uint16)
    print(f"Loaded {len(tokens):,} tokens from {args.npy_file}")
    print(f"dtype: {tokens.dtype}, shape: {tokens.shape}")

    # Load tokenizer
    tokenizer = Tokenizer.from_files(args.tokenizer, special_tokens=args.special_tokens)
    print(f"Loaded tokenizer with vocab size {tokenizer.vocab_size}")

    # Decode slice
    end = min(args.start + args.length, len(tokens))
    token_slice = tokens[args.start:end].tolist()
    text = tokenizer.decode(token_slice)

    print(f"\n--- Tokens [{args.start}:{end}] ---\n")
    print(text)


if __name__ == "__main__":
    main()
