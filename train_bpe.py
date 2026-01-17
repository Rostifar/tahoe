import sys
import time
import regex as re
import numpy as np
from tokenizer import Tokenizer, get_chunk_boundaries

def test_tokenizer_compression(input: str) -> None:
    tokenizers = dict(
        owt=Tokenizer.from_files("data/tokenizers/owt-bpe/"),
        tsv2=Tokenizer.from_files("data/tokenizers/tsv2-bpe")
    )

    for id, tokenizer in tokenizers.items():
        input_bytes = input.encode("utf-8")
        tokens = tokenizer.encode(input)

        print(
            f"==={id}===\n"
            f"Source Text: {input.replace("\n", ";;")}\n"
            f"Compression Rate: {(len(input_bytes) - len(tokens)) / len(input_bytes) * 100:0.2f}\n"
        )

def test_throughput():
    total_bytes = 0

    def build_chunks():
        nonlocal total_bytes

        boundaries = get_chunk_boundaries(
            "data/owt_valid.txt", 
            parallelism=1,
            max_mapped_size_bytes=1_000_000
        )

        with open("data/owt_valid.txt", "rb") as f:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                data = f.read(end - start)
                for doc in re.split(re.escape(b"<|endoftext|>"), data):
                    yield doc.decode("utf-8", errors="replace")
                    total_bytes += len(doc)
    
    tokenizer = Tokenizer.from_files("data/tokenizers/owt-bpe/")
    start = time.perf_counter()
    for _ in tokenizer.encode_iterable(iterable=build_chunks()):
        pass
    end = time.perf_counter()

    print("===Summary===")
    print(f"Elapsed time: {end - start:0.04f}")
    print(f"Throughput: {total_bytes / (end - start)} bytes / sec")


def embed_training_set(path: str, out_path: str, tokenizer_path: str):
    def dataset_iter():
        boundaries = get_chunk_boundaries(
            path,
            parallelism=1,
            max_mapped_size_bytes=100_000_000
        )
        with open(path, "rb") as f:
            for start, end in zip(boundaries[:-1], boundaries[1:]):
                f.seek(start)
                data = f.read(end - start)
                for doc in re.split(re.escape(b"<|endoftext|>"), data):
                    yield doc.decode("utf-8", errors="replace")

    # Open training path, process along boundaries, and write to output
    tokens = []
    batch = 0
    tokenizer = Tokenizer.from_files(tokenizer_path)
    start = time.perf_counter()
    for batch_tokens in tokenizer.encode_iterable(iterable=dataset_iter()):
        batch += 1
        tokens.extend(list(batch_tokens))
        if batch % 1000 == 0:
            print(f"Processed batch {batch}; total tokens processed ({len(tokens)})")
    end = time.perf_counter()

    tokens = np.array(tokens, dtype=np.uint16)
    print(f"Saving tokens to path {out_path}.npy.")
    np.save(out_path + ".npy", tokens)

    print("===Summary===")
    print(f"Elapsed time: {end - start:0.04f}")
    print(f"Number of tokens: {len(tokens)}")
    print(f"Sample: {tokens[:100]}")


if __name__ == "__main__":
    if "--test-compression" in sys.argv:
        examples = [
            "Hello, world! This is a test: 你好，世界！",
            (
                "We faced a significant number of hardware failures in our compute cluster "
                "while training OPT-175B. In total, hardware failures contributed to at least "
                "35 manual restarts and the cycling of over 100 hosts over the course of 2 months. "
                "During manual restarts, the training run was paused, and a series of diagnostics "
                "tests were conducted to detect problematic nodes. Flagged nodes were then cordoned "
                "off and training was resumed from the last saved checkpoint. Given the difference "
                "between the number of hosts cycled out and the number of manual restarts, we "
                "estimate 70+ automatic restarts due to hardware failures."
            ),
            "The cat sat on the mat.",
            "Alice went to the library to borrow a book.",
            "It was a sunny day, so we had a picnic in the park.",
            "My favorite color is blue.",
            "He quickly finished his homework before dinner.",
            "The children laughed as they played with the puppy.",
            "Tomorrow we will visit Grandma's house.",
            "Please remember to bring your umbrella if it rains.",
            "The teacher explained the lesson clearly.",
            "We watched a movie together last night.",
            (
                "def quicksort(arr):\n"
                "    if len(arr) <= 1:\n"
                "        return arr\n"
                "    pivot = arr[len(arr) // 2]\n"
                "    left = [x for x in arr if x < pivot]\n"
                "    middle = [x for x in arr if x == pivot]\n"
                "    right = [x for x in arr if x > pivot]\n"
                "    return quicksort(left) + middle + quicksort(right)\n"
            )
        ]

        for example in examples:
            test_tokenizer_compression(example)
    
    if "--test-throughput" in sys.argv:
        test_throughput()

    if "--build-train" in sys.argv:

        for path, out_path, tokenizer_path in [
            #("data/owt_train.txt", "data/owt_train"), 
            ("data/TinyStoriesV2-GPT4-train.txt", "data/TinyStoriesV2-GPT4-train", "data/tokenizers/tsv2-bpe/")
        ]:
            embed_training_set(path, out_path, tokenizer_path)

    if "--build-val" in sys.argv:
        for path, out_path, tokenizer_path in [
            #("data/owt_valid.txt", "data/owt_valid"), 
            ("data/TinyStoriesV2-GPT4-valid.txt", "data/TinyStoriesV2-GPT4-valid", "data/tokenizers/tsv2-bpe/")
        ]:
            embed_training_set(path, out_path, tokenizer_path)
