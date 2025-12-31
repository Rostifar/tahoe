import os
import regex as re

from typing import BinaryIO
from collections import Counter
from multiprocessing import Pool
"""
Steps:
1. Read in data;
2. Remove document boundaries (ie. <|endoftext|>);
3. Pretokenize using regex pattern.
4. Count byte pairs and merge.
5. Return tokens and merge rules.
"""

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SPLIT_TOKEN = b"<|endoftext|>" 

def pretokenize_chunk(filename: str, chunk_start: int, chunk_end: int) -> dict[tuple[bytes], int]:
    with open(filename, "rb") as f:
        # split into docs
        f.seek(chunk_start)
        data = f.read(chunk_end - chunk_start)
        docs = re.split(SPLIT_TOKEN, data)
        
        # split into pretokens per doc
        pretokens = Counter()
        for doc in docs:
            text = doc.decode("utf-8")
            for pretoken in re.finditer(GPT2_PAT, text):
                key = tuple(pretoken.group().encode("utf-8"))
                pretokens[key] += 1
        return pretokens


def get_chunk_boundaries(f: BinaryIO, chunk_hint: int | None = None) -> list[int]:
    f.seek(0, os.SEEK_END)
    file_size = f.tell()
    f.seek(0)
    
    # process 50MB chunks
    chunk_hint = chunk_hint if chunk_hint else file_size // 50_000_000
    chunk_size = file_size // chunk_hint
    chunk_boundaries = [i * chunk_size for i in range(chunk_hint + 1)]

    # split chunks along document boundaries
    mini_chunk_size = 4096
    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        f.seek(initial_position)

        mini_chunk = f.read(mini_chunk_size)

        if mini_chunk == b"":
            chunk_boundaries[bi] = file_size
            break

        found_at = mini_chunk.find(SPLIT_TOKEN)
        if found_at != -1:
            chunk_boundaries[bi] = initial_position + found_at
            break
        initial_position += mini_chunk_size
    return sorted(set(chunk_boundaries))


def pretokenize(
    input_path: str,
    chunk_hint: int | None = None,
) -> dict[tuple[bytes], int]:
    with open(input_path, "rb") as f:
        boundaries = get_chunk_boundaries(f, chunk_hint)
        args = [
            (input_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])
        ]
        with Pool(8) as p:
            pretokens = p.map(pretokenize_chunk, args)

    pretoken_table = {}
    for map in pretokens:
        for k, v in map.items():
            pretoken_table[k] = pretoken_table.get(k, 0) + v
    return pretoken_table


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {i: i for i in range(256)}
    vocab.update({
        len(vocab) + i: t.encode("utf-8") for i, t in enumerate(special_tokens)
    })

    pretokens = pretokenize(input_path)
    merges = []
    while len(vocab) < vocab_size:
        pairs = Counter()
        for pretoken, weight in enumerate(pretokens):
            for pair in zip(pretoken[:-1], pretoken[1:]):
                pairs[pair] += weight
        max_freq = max(pairs, key=pairs.get)
        max_pairs = {k for k, v in pairs.items() if v == max_freq}

        # take largest pair
        merge_rule = sorted(max_pairs)[-1]
        merges.append(merge_rule)

        for i in range(len(pretokens)):
            pretoken = list(pretokens[i])
            new_pretoken = []
            for j in range(1, len(pretoken)):
                if (pretoken[j - 1], pretoken[j]) == merge_rule:
                    new_pretoken.append(vocab[merge_rule])

if __name__ == "__main__":
    pretokens = pretokenize(
        "data/TinyStoriesV2-GPT4-train.txt",
    )

    print("max: {}, min: {}, count: {}" % (max(pretokens.values()), min(pretokens.values()), len(pretokens)))
