"""Prepare the Tiny Shakespeare character corpus.

Downloads the raw text (if not already present), builds a character vocabulary,
writes data/meta.json, and splits the encoded text 90% train / 10% val into
data/train.bin and data/val.bin (numpy uint16).

All paths are resolved relative to this script's own directory, so the script
can be invoked from any working directory.
"""

import json
import os
import urllib.request

import numpy as np

# Everything is anchored to the directory containing this script (data/).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_PATH = os.path.join(BASE_DIR, "input.txt")
META_PATH = os.path.join(BASE_DIR, "meta.json")
TRAIN_PATH = os.path.join(BASE_DIR, "train.bin")
VAL_PATH = os.path.join(BASE_DIR, "val.bin")

# The one fixed upstream source for the corpus.
URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


def load_text(path: str) -> str:
    # path: single file on disk. Returns: the full text as one str.
    if not os.path.exists(path):
        print(f"Downloading corpus from {URL} ...")
        with urllib.request.urlopen(URL) as resp:
            raw = resp.read()
        with open(path, "wb") as f:
            f.write(raw)
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def build_vocab(text: str) -> list[str]:
    # text: str. Returns: sorted list of unique characters (the vocabulary).
    return sorted(set(text))


def encode(text: str, stoi: dict[str, int]) -> np.ndarray:
    # text: str; stoi: mapping char -> int. Returns: (N,) uint16 token ids,
    # where N = number of characters.
    return np.array([stoi[c] for c in text], dtype=np.uint16)


def main() -> None:
    # No inputs; writes meta.json, train.bin, val.bin; prints summary.
    text = load_text(INPUT_PATH)

    chars = build_vocab(text)
    vocab_size = len(chars)
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {str(i): ch for i, ch in enumerate(chars)}

    meta = {"vocab_size": vocab_size, "itos": itos, "stoi": stoi}
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    data = encode(text, stoi)  # (N,) uint16
    n = data.shape[0]
    split = int(0.9 * n)
    train = data[:split]  # (split,) uint16
    val = data[split:]  # (n - split,) uint16

    train.tofile(TRAIN_PATH)
    val.tofile(VAL_PATH)

    print(f"length of text in characters: {n}")
    print(f"vocab_size: {vocab_size}")
    print(f"characters: {''.join(chars)}")
    print(f"train tokens: {train.shape[0]} -> {TRAIN_PATH}")
    print(f"val tokens: {val.shape[0]} -> {VAL_PATH}")


if __name__ == "__main__":
    main()
