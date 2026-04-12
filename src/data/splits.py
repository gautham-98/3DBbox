import random
from pathlib import Path


def get_splits(data_dir: str, seed: int = 42) -> dict:
    samples = sorted(Path(data_dir).iterdir())
    samples = [s for s in samples if s.is_dir()]

    rng = random.Random(seed)
    shuffled = samples.copy()
    rng.shuffle(shuffled)

    n = len(shuffled)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)

    return {
        "train": shuffled[:n_train],
        "val": shuffled[n_train : n_train + n_val],
        "test": shuffled[n_train + n_val :],
    }


if __name__ == "__main__":
    splits = get_splits("data")
    for split, samples in splits.items():
        print(f"{split}: {len(samples)} frames")
