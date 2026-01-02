import cProfile
import os
import regex as re
from hashlib import md5
from collections import Counter
from multiprocessing import Pool

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SPLIT_TOKEN = b"<|endoftext|>" 
MAX_MAPPED_SIZE_GB = 2_000_000_000


def get_chunk_boundaries(
    input_path: str, 
    parallelism: int
) -> list[int]:
    with open(input_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)
            
        # all processes map 
        proc_memory_gb = MAX_MAPPED_SIZE_GB // parallelism
        num_chunks = max(1, file_size // proc_memory_gb)

        chunk_size = file_size // num_chunks
        chunk_boundaries = [i * chunk_size for i in range(num_chunks + 1)]

        # split chunks along document boundaries
        mini_chunk_size = 4096
        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            f.seek(initial_position)

            mini_chunk = f.read(mini_chunk_size)

            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break
            
            # move boundary to closest EOT token
            found_at = mini_chunk.find(SPLIT_TOKEN)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size
        return sorted(set(chunk_boundaries))


def pretokenize_chunk(filename: str, chunk_start: int, chunk_end: int, special_pat: str) -> dict[tuple[bytes], int]:
    with open(filename, "rb") as f:
        # split into docs
        f.seek(chunk_start)
        data = f.read(chunk_end - chunk_start)
        docs = re.split(special_pat, data)
        
        # split into pretokens per doc
        pretokens = Counter()
        for doc in docs:
            text = doc.decode("utf-8")
            for pretoken in re.finditer(GPT2_PAT, text):
                key = tuple(bytes([b]) for b in pretoken.group().encode("utf-8"))
                pretokens[key] += 1
        return pretokens


def pretokenize(input_path: str, special_tokens: list[str], parallelism: int = 8) -> dict[tuple[bytes], int]:
    special_pat = rb"|".join(re.escape(s).encode("utf-8") for s in special_tokens)
    boundaries = get_chunk_boundaries(input_path, parallelism)
    args = [
        (input_path, start, end, special_pat) 
        for start, end in zip(boundaries[:-1], boundaries[1:])
    ]
    with Pool(parallelism) as p:
        pretokens = p.starmap(pretokenize_chunk, args)
    
    # merge tables
    pretoken_table = {}
    for subtable in pretokens:
        for k, v in subtable.items():
            pretoken_table[k] = pretoken_table.get(k, 0) + v
    return pretoken_table


def find_merges(
    pretokens: dict[tuple[bytes], int],
    vocab_size: int,
    special_tokens: list[str]
):
    def apply_merge(rule: tuple[int, int]):
        merged_token = []
        pruned_tokens = {}
        for token in tokens:
            merged_token = []
            j = 0
            while j < len(token):
                if j < len(token) - 1 and (token[j], token[j + 1]) == rule:
                    merged_token.append(len(vocab))
                    j += 2
                else:
                    merged_token.append(token[j])
                    j += 1
            pruned_tokens[tuple(merged_token)] = tokens[token]
        return pruned_tokens

    tokens = {tuple(ord(b) for b in k): v for k, v in pretokens.items()}
    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({
        len(vocab) + i: t.encode("utf-8") for i, t in enumerate(special_tokens)
    })

    merges = []
    while len(vocab) < vocab_size:
        if len(vocab) % 1000 == 0:
            print(f"Iteration: {len(vocab)}")

        pairs = Counter()
        for token, weight in tokens.items():
            for pair in zip(token[:-1], token[1:]):
                pairs[pair] += weight
        
        max_pair = max(pairs, key=lambda x: (pairs[x], x))

        tokens = apply_merge(max_pair)
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
    # construct pretokens
    print("Constructing pretokens...")
    pretokens = pretokenize(input_path, special_tokens)
    if debug:
        dir = f"tmp/{md5(input_path.encode("utf-8")).hexdigest()}"
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
                f.write(",".join([b.decode("utf-8", errors="replace") for b in merge]) + "\n")
    
if __name__ == "__main__":
    """train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=10_000,
        special_tokens=["<|endoftext|>"],
        debug=True
    )"""

    train_bpe(
        input_path="data/owt_train.txt",
        vocab_size=32_000,
        special_tokens=["<|endoftext|>"],
        debug=True
    )
