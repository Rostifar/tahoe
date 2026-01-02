import cProfile
import os
import regex as re
from hashlib import md5
from dataclasses import dataclass
from collections import Counter
from multiprocessing import Pool

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SPLIT_TOKEN = b"<|endoftext|>" 
MAX_MAPPED_SIZE_BYTES = 2_000_000_000

@dataclass
class MergePair:
    weight: int
    children: Counter[tuple[int, int]]


def get_chunk_boundaries(input_path: str, parallelism: int) -> list[int]:
    with open(input_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)
            
        proc_memory_gb = MAX_MAPPED_SIZE_BYTES // parallelism
        num_chunks = max(1, file_size // proc_memory_gb)

        chunk_size = file_size // num_chunks
        chunk_boundaries = [i * chunk_size for i in range(num_chunks + 1)]

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


def pretokenize_chunk(
    filename: str, 
    chunk_start: int, 
    chunk_end: int, 
    special_pat: str
) -> dict[tuple[bytes], int]:
    with open(filename, "rb") as f:
        f.seek(chunk_start)
        data = f.read(chunk_end - chunk_start)
        docs = re.split(special_pat, data)
        
        pretokens = Counter()
        for doc in docs:
            text = doc.decode("utf-8")
            for pretoken in re.finditer(GPT2_PAT, text):
                key = tuple[bytes, ...](bytes([b]) for b in pretoken.group().encode("utf-8"))
                pretokens[key] += 1
        return pretokens


def pretokenize(
    input_path: str, 
    special_tokens: list[str], 
    parallelism: int = 8
) -> dict[tuple[bytes], int]:
    special_pat = rb"|".join(re.escape(s).encode("utf-8") for s in special_tokens)
    boundaries = get_chunk_boundaries(input_path, parallelism)
    args = [
        (input_path, start, end, special_pat) 
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    with Pool(parallelism) as p:
        pretokens = p.starmap(pretokenize_chunk, args)
    
    pretoken_table = {}
    for subtable in pretokens:
        for k, v in subtable.items():
            pretoken_table[k] = pretoken_table.get(k, 0) + v
    return pretoken_table


def apply_merge(
    merge_rule: tuple[int, tuple[int, int]],
    tokens: dict[tuple[int, int], int],
    pairs: dict[tuple[int, int], MergePair]
) -> dict[tuple[int, int], int]:

    def update_pair_table(ptr: int, token: tuple[int]):
        offsets = [-1, 2]
        for offset in offsets:
            pos = offset + ptr
            if pos < 0 or pos >= len(token):
                continue

            if offset < 0:
                old_token = (token[pos], token[ptr])
                new_token = token[pos], token_id
            else:
                old_token = token[ptr + 1], token[pos]
                new_token = token_id, token[pos]

            # remove one reference and add a new reference
            assert pairs[old_token].children[token] > 0
            pairs[old_token].weight -= tokens[token]
            pairs[old_token].children[token] -= 1

            if not pairs[old_token].children[token]:
                del pairs[old_token].children[token]

            hit_pair = pairs.get(new_token, MergePair(weight=0, children=Counter()))
            hit_pair.weight += tokens[token]
            hit_pair.children[token] += 1
            pairs[new_token] = hit_pair


    # optimization:
    # - compute pair counts once; take largest and merge. 
    # - given merge rule (a, b) -> c, remove a pair x a b y; 
    #   1. decrement (a, b) by weight in pairs.
    #   2. decrement (x, a) by weight in pairs.
    #   3. decrement (b, y) by weight in pairs.
    #   4. increment (x, c) by weight in pairs.
    #   5. increment (c, y) by weight in pairs.
    merged_token = []
    token_id, rule = merge_rule
    for token in list(pairs[rule].children.keys()):
        merged_token = []
        j = 0
        while j < len(token):
            if j < len(token) - 1 and (token[j], token[j + 1]) == rule:
                merged_token.append(token_id)
                update_pair_table(j, token)
                j += 2
            else:
                merged_token.append(token[j])
                j += 1

        # add new candidate token and remove unused keys 
        tokens[tuple(merged_token)] = tokens[token]
        del tokens[token]


def find_merges(
    pretokens: dict[tuple[bytes], int],
    vocab_size: int,
    special_tokens: list[str],
    debug: bool = True
):
    tokens = {tuple(ord(b) for b in k): v for k, v in pretokens.items()}
    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({
        len(vocab) + i: t.encode("utf-8") for i, t in enumerate(special_tokens)
    })

    merges = []
    pairs = {}
    for token, weight in tokens.items():
        for pair in zip(token[:-1], token[1:]):
            hit_pair = pairs.get(pair, MergePair(weight=0, children=Counter()))
            hit_pair.weight += weight
            hit_pair.children[token] += 1
            pairs[pair] = hit_pair

    while len(vocab) < vocab_size:
        if debug and len(vocab) % 1000 == 0:
            print(f"Iteration: {len(vocab)}")
        
        token_id = len(vocab)
        max_pair = max(pairs, key=lambda x: (pairs[x].weight, x))
        tokens = apply_merge((token_id, max_pair), tokens, pairs)
        
        merge_rule = tuple((vocab[max_pair[0]], vocab[max_pair[1]]))
        merges.append(merge_rule)
        vocab[len(vocab)] = b"".join(merge_rule)
    return vocab, merges


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    debug: bool = False
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    print("Constructing pretokens...")
    pretokens = pretokenize(input_path, special_tokens)
    if debug:
        dir = f"tmp/{md5(input_path.encode('utf-8')).hexdigest()}"
        os.makedirs(dir, exist_ok=True)
        with open(f"{dir}/1.pretokens.txt", "w") as f:
            for pretoken, count in pretokens.items():
                f.write(b"".join(pretoken).decode("utf-8") + f"-{count}\n")

    print("Finding Merges...")
    vocab, merges = find_merges(pretokens, vocab_size, special_tokens)

    if debug:
        with open(f"{dir}/1.vocab.txt", "w") as f:
            for id, token in vocab.items():
                f.write(str(id) + ": " + token.decode("utf-8", errors="replace") + "\n")

        with open(f"{dir}/1.merges.txt", "w") as f:
            for merge in merges:
                f.write(",".join([b.decode("utf-8", errors="replace") for b in merge]) + "\n"

)

if __name__ == "__main__":
    train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
        debug=True
    )
    """
    train_bpe(
        input_path="data/owt_train.txt",
        vocab_size=32_000,
        special_tokens=["<|endoftext|>"],
        debug=True
    )
    """
