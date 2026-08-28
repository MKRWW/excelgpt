"""Train the Tiny Shakespeare character model.

AdamW (weight_decay=0.1, betas=(0.9, 0.99)), gradient clipping at 1.0, a linear
warmup of 100 iterations followed by cosine decay to 10% of the base learning
rate. All seeds are set. Every --eval-interval steps the train and val losses
are estimated over --eval-iters batches. The final checkpoint and a short sample
of generated text are written at the end.

TF32 matmul is disabled so GPU and CPU stay in plain float32 (otherwise results
would drift in the fourth decimal place).
"""

import argparse
import dataclasses
import json
import math
import os

import numpy as np
import torch

from model import GPT, GPTConfig

# Plain float32 everywhere: no TF32 matmul, no cuDNN TF32.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Anchored to the project root (the directory containing this file).
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_PATH = os.path.join(BASE_DIR, "data", "train.bin")
VAL_PATH = os.path.join(BASE_DIR, "data", "val.bin")
META_PATH = os.path.join(BASE_DIR, "data", "meta.json")

WARMUP_ITERS = 100


def parse_args() -> argparse.Namespace:
    # Returns: argparse.Namespace with all CLI values.
    p = argparse.ArgumentParser(description="Train the Tiny Shakespeare model")
    p.add_argument("--iters", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-interval", type=int, default=250)
    p.add_argument("--eval-iters", type=int, default=100)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out", type=str, default=os.path.join(BASE_DIR, "out", "ckpt.pt"))
    return p.parse_args()


def get_lr(it: int, max_iters: int, base_lr: float) -> float:
    # it: current iteration (0-based); max_iters: total; base_lr: peak lr.
    # Returns: the learning rate for this step (float).
    if it < WARMUP_ITERS:
        return base_lr * (it / WARMUP_ITERS)
    if it >= max_iters:
        return base_lr * 0.1
    decay_ratio = (it - WARMUP_ITERS) / (max_iters - WARMUP_ITERS)
    ratio = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return base_lr * 0.1 + (base_lr - base_lr * 0.1) * ratio


def load_bin(path: str, device: str) -> torch.Tensor:
    # path: binary uint16 file. Returns: (N,) int64 tensor on `device`.
    data = np.fromfile(path, dtype=np.uint16).astype(np.int64)
    return torch.from_numpy(data).to(device)


def get_batch(data: torch.Tensor, batch_size: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    # data: (N,) int64 token ids on `device`; batch_size: B; block_size: T.
    # Returns: (x, y), each (batch_size, block_size) int64 on `device`, sampled
    # from batch_size independent start points; y is shifted by exactly one
    # position relative to x along the sequence axis.
    ix = torch.randint(len(data) - block_size - 1, (batch_size,), device=device)
    x = torch.stack([data[i : i + block_size] for i in ix])  # (batch_size, block_size)
    y = torch.stack([data[i + 1 : i + 1 + block_size] for i in ix])  # (batch_size, block_size)
    return x, y


def get_loss_estimates(model: GPT, train_data: torch.Tensor, val_data: torch.Tensor,
                       batch_size: int, block_size: int, eval_iters: int) -> tuple[float, float]:
    # Returns: (train_loss, val_loss) averaged over eval_iters random batches.
    model.eval()
    losses = {"train": [], "val": []}
    for name, data in (("train", train_data), ("val", val_data)):
        for _ in range(eval_iters):
            idx, targets = get_batch(data, batch_size, block_size, data.device)
            with torch.no_grad():
                _, loss = model(idx, targets)
                losses[name].append(loss.item())
    model.train()
    return float(np.mean(losses["train"])), float(np.mean(losses["val"]))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    with open(META_PATH, "r", encoding="utf-8") as f:
        meta = json.load(f)
    cfg = GPTConfig(vocab_size=meta["vocab_size"])

    model = GPT(cfg).to(device)
    print(f"Number of parameters: {sum(p.numel() for p in model.parameters()):,}")

    train_data = load_bin(TRAIN_PATH, device)
    val_data = load_bin(VAL_PATH, device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, betas=(0.9, 0.99), weight_decay=0.1
    )

    max_iters = args.iters
    for it in range(max_iters):
        for group in optimizer.param_groups:
            group["lr"] = get_lr(it, max_iters, args.lr)

        idx, targets = get_batch(train_data, args.batch_size, cfg.block_size, device)

        _, loss = model(idx, targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % args.eval_interval == 0:
            train_loss, val_loss = get_loss_estimates(
                model, train_data, val_data, args.batch_size, cfg.block_size, args.eval_iters
            )
            print(f"step {it}: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

    train_loss, val_loss = get_loss_estimates(
        model, train_data, val_data, args.batch_size, cfg.block_size, args.eval_iters
    )
    print(f"final: train loss {train_loss:.4f}, val loss {val_loss:.4f}")

    ckpt = {
        "model_state_dict": model.state_dict(),
        "config": dataclasses.asdict(cfg),
        "vocab_size": meta["vocab_size"],
        "itos": meta["itos"],
        "stoi": meta["stoi"],
        "iter": max_iters,
        "val_loss": val_loss,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(ckpt, args.out)
    print(f"Saved checkpoint to {args.out}")

    # Print 300 characters of generated text as a sanity check.
    model.eval()
    with torch.no_grad():
        prompt = torch.tensor([[0]], device=device)
        gen = model.generate(prompt, max_new_tokens=300, temperature=1.0)
    text = "".join(meta["itos"][str(tok)] for tok in gen[0].tolist())
    print("\n--- generated text (300 chars) ---")
    print(text)


if __name__ == "__main__":
    main()
