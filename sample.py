"""Sample text from a trained checkpoint.

Loads out/ckpt.pt (by default), rebuilds the model from its config, sets
model.eval(), and runs autoregressive generation under torch.no_grad(). The
random seed is applied through an explicit torch.Generator handed to
generate(), so the same --seed always produces byte-identical output.
"""

import argparse
import dataclasses
import os

import numpy as np
import torch

from model import GPT, GPTConfig

# Plain float32 everywhere: no TF32 matmul, no cuDNN TF32.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(BASE_DIR, "out", "ckpt.pt")


def parse_args() -> argparse.Namespace:
    # Returns: argparse.Namespace with all CLI values.
    p = argparse.ArgumentParser(description="Sample from a trained checkpoint")
    p.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    p.add_argument("--prompt", type=str, default="\n")
    p.add_argument("--tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=1337)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    with open(args.ckpt, "rb") as f:
        ckpt = torch.load(f, map_location="cpu")

    cfg_fields = [f.name for f in dataclasses.fields(GPTConfig)]
    cfg = GPTConfig(**{k: ckpt["config"][k] for k in cfg_fields})
    model = GPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    itos = ckpt["itos"]
    stoi = ckpt["stoi"]

    prompt_ids = [stoi[ch] for ch in args.prompt]
    idx = torch.tensor([prompt_ids], dtype=torch.long)  # (1, T)

    # Explicit generator so the sampling is reproducible for a fixed seed.
    gen = torch.Generator()
    gen.manual_seed(args.seed)

    with torch.no_grad():
        out = model.generate(idx, args.tokens, temperature=args.temperature, generator=gen)

    text = "".join(itos[str(tok)] for tok in out[0].tolist())
    print(text)


if __name__ == "__main__":
    main()
