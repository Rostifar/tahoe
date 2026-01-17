import numpy as np
import matplotlib.pyplot as plt

def get_vocab_stats(data: np.array) -> tuple[np.array, np.array]:
    counts = np.bincount(data)
    mass = np.cumsum(counts)
    return mass, mass / mass[-1]

if __name__ == "__main__":
    data = np.load("./data/TinyStoriesV2-GPT4-train.npy", mmap_mode='r').astype(np.uint16) 
    mass, cdf = get_vocab_stats(data)

    vocab = len(cdf)
    for i in range(0, vocab, 1000):
        lost_density = cdf[vocab - 1] - cdf[i]
        lost_tokens = mass[vocab - 1] - mass[i]
        print(f"> {i}: {cdf[i]}; density_loss={lost_density}; token_loss={lost_tokens}")

    plt.figure(figsize=(10, 6))
    plt.plot(np.arange(vocab), cdf)
    plt.xlabel("Token ID (k)")
    plt.ylabel("P(token ≤ k)")
    plt.title("Cumulative Distribution of Token IDs")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("vocab_cdf.png", dpi=150)
    plt.show()
