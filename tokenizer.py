import os
import time
import base64
import regex as re
import statistics
from typing import Iterable, Iterator
from dataclasses import dataclass
from collections import Counter
from multiprocessing import Pool

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
SPLIT_TOKEN = b"<|endoftext|>" 
MAX_MAPPED_SIZE_BYTES = 2_000_000_000

@dataclass
class MergePair:
    weight: int
    children: Counter[int]


def get_chunk_boundaries(input_path: str, parallelism: int) -> list[int]:
    with open(input_path, "rb") as f:
        f.seek(0, os.SEEK_END)
        file_size = f.tell()
        f.seek(0)
            
        proc_memory_gb = MAX_MAPPED_SIZE_BYTES // parallelism
        num_chunks = max(1, file_size // proc_memory_gb)

        chunk_size = file_size // num_chunks
        chunk_boundaries = [i * chunk_size for i in range(num_chunks + 1)]
        chunk_boundaries[-1] = file_size

        mini_chunk_size = 4096
        for bi in range(1, len(chunk_boundaries) - 1):
            initial_position = chunk_boundaries[bi]
            f.seek(initial_position)

            while True:
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
    special_pat: str,
    split_string: str = GPT2_PAT
) -> dict[tuple[bytes], int]:
    with open(filename, "rb") as f:
        f.seek(chunk_start)
        data = f.read(chunk_end - chunk_start)
        docs = re.split(special_pat, data)
        
        pretokens = Counter()
        for doc in docs:
            # TODO: replace this in the future to handle incorrectly encoded text
            text = doc.decode("utf-8", )
            for pretoken in re.finditer(split_string, text):
                key = tuple[bytes, ...](bytes([b]) for b in pretoken.group().encode("utf-8"))
                pretokens[key] += 1
        return pretokens


def pretokenize(
    input_path: str, 
    special_tokens: list[str], 
    parallelism: int = 8,
    verbose: bool = True
) -> dict[tuple[bytes], int]:
    if verbose:
        print(f"Pretokenizing with parameters: (parallelism={parallelism}, max_memory_mapping_bytes={MAX_MAPPED_SIZE_BYTES})")

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
    tokens: dict[tuple[int, ...], tuple[int, int]],
    pairs: dict[tuple[int, int], MergePair],
    token_aliases: list[tuple[int, ...]]
) -> None:
    token_id, rule = merge_rule
    children = list(pairs[rule].children)

    for alias in children:
        token = token_aliases[alias]
        weight = tokens[token][0]

        # construct new token
        merged_token = []
        j = 0
        while j < len(token):
            if j < len(token) - 1 and (token[j], token[j + 1]) == rule:
                merged_token.append(token_id)
                j += 2
            else:
                merged_token.append(token[j])
                j += 1

        if len(merged_token) == len(token):
            continue
        
        # bookkeeping for decrementing / incrementing pairs table
        old_pairs: Counter[tuple[int, int]] = Counter()
        for j in range(len(token) - 1):
            old_pairs[(token[j], token[j + 1])] += 1

        new_pairs: Counter[tuple[int, int]] = Counter()
        for j in range(len(merged_token) - 1):
            new_pairs[(merged_token[j], merged_token[j + 1])] += 1

        for pair, count in old_pairs.items():
            pairs[pair].weight -= weight * count
            pairs[pair].children[alias] -= count

            # prune old references
            if pairs[pair].children[alias] == 0:
                del pairs[pair].children[alias]
            if not pairs[pair].children:
                del pairs[pair]

        # update for new token
        for pair, count in new_pairs.items():
            if pair not in pairs:
                pairs[pair] = MergePair(weight=0, children=Counter())
            pairs[pair].weight += weight * count
            pairs[pair].children[alias] += count

        new_key = tuple(merged_token)
        token_aliases[alias] = new_key
        tokens[new_key] = tokens[token]
        del tokens[token]


def find_merges(
    pretokens: dict[tuple[bytes], int],
    vocab_size: int,
    special_tokens: list[str],
    verbose: bool = True
):
    tokens = {tuple(ord(b) for b in k): (pretokens[k], i) for i, k in enumerate(pretokens)}
    token_aliases = [(-1,)] * len(tokens)
    for id, (_, alias) in tokens.items():
        token_aliases[alias] = id
    
    assert not any(y < 0 for t in token_aliases for y in t), "invalid token alias!"
    vocab = {i: bytes([i]) for i in range(256)}
    vocab.update({
        len(vocab) + i: t.encode("utf-8") for i, t in enumerate(special_tokens)
    })

    # Optimization: precompute frequency table once and update during merges; only merge impacted pretokens.
    # - Each corpus pair holds a list of pretokens it's contained in `children`, storing table identifiers.
    # - When a pair is merged, it's neighboring pairs are updated, including `children` and `weight` metadata.
    # - Since a token may have multiple pairs with the same id, a counter must be used for reference counting.
    merges = []
    pairs = {}
    for token, (weight, alias) in tokens.items():
        for pair in zip(token[:-1], token[1:]):
            hit_pair = pairs.get(pair, MergePair(weight=0, children=Counter()))
            hit_pair.weight += weight
            hit_pair.children[alias] += 1
            pairs[pair] = hit_pair

    print("Merge Table Stats")
    print(f"- total pairs: {len(pairs)}")
    print(f"- average pair weight: {statistics.mean(p.weight for p in pairs.values())}")
    print(f"- weight variance: {statistics.variance(p.weight for p in pairs.values())}")
    print(f"- p50 weight: {statistics.median(p.weight for p in pairs.values())}")

    start = time.perf_counter()
    while len(vocab) < vocab_size:
        if verbose and len(vocab) % 200 == 0:
            end = time.perf_counter()
            print(f"Updated vocab size to {len(vocab)}...; elapsed time: {(end - start):0.4f} s.")
            start = end
        
        token_id = len(vocab)
        max_pair = max(pairs, key=lambda x: (pairs[x].weight, x))
        apply_merge((token_id, max_pair), tokens, pairs, token_aliases)
        
        merge_rule = tuple((vocab[max_pair[0]], vocab[max_pair[1]]))
        merges.append(merge_rule)
        vocab[len(vocab)] = b"".join(merge_rule)
    return vocab, merges


def save(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    output_path: str
) -> None:
    os.makedirs(output_path, exist_ok=True)
    with open(os.path.join(output_path, "vocab"), "w") as f:
        # format: <int>:<base64>\n...
        successor = False
        for id, token in vocab.items():
            if successor:
                f.write("\n")
            else:
                successor = True
            f.write(f"{id}:{base64.b64encode(token).decode()}")


    with open(os.path.join(output_path, "merges"), "w") as f:
        # format: <base64>:<base64>,...
        successor = False
        for left, right in merges:
            if successor:
                f.write("\n")
            else:
                successor = True
            f.write(f"{base64.b64encode(left).decode()}:{base64.b64encode(right).decode()}")


def load(
    path: str
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = {}
    merges = []
    with open(os.path.join(path, "vocab"), "r") as f:
        for line in f.readlines():
            id, b64_token = line.split(":", maxsplit=1)
            vocab[id] = base64.b64decode(b64_token)

    with open(os.path.join(path, "merges"), "r") as f:
        for line in f.readlines():
            left, right = line.split(":", maxsplit=1)
            merges.append((base64.b64decode(left), base64.b64decode(right)))
    return vocab, merges


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    output_path: str,
    verbose: bool = True
) -> None:
    if verbose:
        start = time.perf_counter()
        print(f"Constructing pretokens for dataset {input_path}...")
    pretokens = pretokenize(input_path, special_tokens, verbose=verbose)
    if verbose:
        print(f"Finding merges for {len(pretokens)} pretokens...")
    vocab, merges = find_merges(pretokens, vocab_size, special_tokens, verbose=verbose)

    if verbose:
        print(f"Saving vocab at {output_path}...")
    
    save(vocab, merges, output_path)
    if verbose:
        end = time.perf_counter()
        print(f"Training finished! Elapsed time: {(end - start):0.4f} s.")


class Tokenizer:
    def __init__(
        self, 
        vocab: dict[int, bytes], 
        merges: list[tuple[bytes, bytes]], 
        special_tokens: list[str] | None = None
    ) -> None:
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.inverse_vocab = {v: i for i, v in vocab.items()} 
        
        self.merges = merges
        self.separator = "(" + "|".join(re.escape(t) for t in special_tokens) + ")" if special_tokens else ""
        self.merge_table = {}


    @classmethod
    def from_files(cls, path: str, special_tokens: list[str] | None = None) -> "Tokenizer":
        return cls(*load(path), special_tokens=special_tokens)


    def _tokenize_pretoken(self, pretoken: str) -> list[int]:
        chunks = [bytes([b]) for b in pretoken.encode("utf-8")]
        for merge_rule in self.merges:
            if len(chunks) < 2:
                break
            
            merged = merge_rule[0] + merge_rule[1]
            i = 0
            new_chunks = []
            
            while i < len(chunks):
                if i < len(chunks) - 1 and (chunks[i], chunks[i + 1]) == merged:
                    new_chunks.append(merged)
                    i += 2
                else:
                    new_chunks.append(chunks[i])
                    i += 1
            chunks = new_chunks
        return [self.inverse_vocab[tok] for tok in chunks]


    def encode(self, text: str) -> list[int]:
        # 1. split by special tokens, yielding segments: [s1]<special>[s2]<special>
        # 2. map <special> to vocab; for each segment, split into pretokens, and encode.
        # 3. per pretoken, take pairs and apply 
        out = []
        segments = re.split(self.separator, text) if self.separator else [text]
        for i, segment in enumerate(segments):
            # ignore if special token resides on left/right boundaries, as this produces an empty string
            if not segment:
                continue 

            if i % 2 != 0:
                special_token = segment.encode("utf-8")
                out.append(self.inverse_vocab[special_token])
                continue

            for pretoken in re.finditer(GPT2_PAT, segment):
                out.extend(self._tokenize_pretoken(pretoken.group()))
        return out


    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        # N.B. ensure that iterables are defined along special token boundaries!
        for text in iterable:
            yield self.encode(text)


    def decode(self, tokens: list[int]) -> str:
        return b"".join([self.vocab[token] for token in tokens]).decode("utf-8", errors="replace")     


if __name__ == "__main__":
    train_bpe(
        input_path="data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=25000,
        special_tokens=["<|endoftext|>"],
        output_path="data/tokenizers/tsv2-bpe/",
        verbose=True
    )
    
    train_bpe(
        input_path="data/owt_train.txt",
        vocab_size=32_000,
        special_tokens=["<|endoftext|>"],
        output_path="data/tokenizers/owt-bpe/",
        verbose=True
    )
