"""Dump every intermediate of the forward pass as CSV reference data.

For one fixed prompt (default: "To be, or not to be") the model's single
forward path runs once on CPU with tracing enabled, and every intermediate
tensor is written to <out-dir>/ as a pure numeric CSV (no header, no index
column, comma separator, point decimal separator, %.10e for floats, plain
digits for token ids). A manifest.csv with a single header line lists every
written file in write order.

CSV conventions (binding):
  * rows = timesteps t (0..T-1, ascending); columns = feature index (ascending)
  * attention matrices: row = query position t, column = key position s
  * no nan / no inf / no -inf in any file: the causal mask value is written as
    -1.0e30 instead of -inf
  * the only file with a header is manifest.csv

Self-checks (any violation -> exit code 1):
  a) logits reconstructed from the dumped 90_ln_f and the lm_head weight/bias
     match the dumped 91_logits (max absolute deviation < 1e-5)
  b) every row of every attn_probs matrix sums to 1.0 (+/- 1e-5)
  c) every value strictly above the diagonal in 16_attn_probs is exactly 0.0
  d) no file contains nan or infinite values
"""

import argparse
import dataclasses
import os
import sys

import numpy as np
import torch

from model import GPT, GPTConfig

# Plain float32 everywhere: no TF32 matmul, no cuDNN TF32 (otherwise GPU and
# CPU values would drift in the fourth decimal place).
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# THE one fixed artifact this dump is compared against later.
PROMPT = "To be, or not to be"  # T = 19 characters

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(BASE_DIR, "out", "ckpt.pt")
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "reference")

MASK_VALUE = -1.0e30  # stands in for -inf inside the CSVs

# Canonical write order: global keys, then per-layer blocks for l = 0..n_layer-1
# (ln1, then the per-head group for h = 0..n_head-1, then the non-head keys),
# then the ending keys. The manifest is written in exactly this order.
GLOBAL_KEYS = ["00_tokens", "01_tok_emb", "02_pos_emb", "03_x_input"]
HEAD_KEY_BASES = [
    "11_q", "12_k", "13_v",
    "14_scores_scaled", "15_scores_masked", "16_attn_probs", "17_head_out",
]
NON_HEAD_KEYS = ["10_ln1", "18_attn_concat", "19_attn_proj", "20_resid_post_attn",
                 "30_ln2", "31_fc", "32_gelu", "33_mlp_proj", "34_resid_post_mlp"]
ENDING_KEYS = ["90_ln_f", "91_logits", "92_logits_last", "93_probs_last_temp1"]

DESCRIPTIONS = {
    "00_tokens": "Token-IDs of the prompt (integers)",
    "01_tok_emb": "Token embeddings: wte[idx]",
    "02_pos_emb": "Learned absolute position embeddings: wpe[0..T-1]",
    "03_x_input": "Sum of token embeddings and position embeddings",
    "10_ln1": "LayerNorm 1 output",
    "11_q": "Query of head h",
    "12_k": "Key of head h",
    "13_v": "Value of head h",
    "14_scores_scaled": "Q @ K^T / sqrt(head_dim), unmasked",
    "15_scores_masked": "Scaled scores with causal mask; masked entries = -1.0e30",
    "16_attn_probs": "Softmax attention probabilities (rows sum to 1)",
    "17_head_out": "Attention output of head h (probs @ V)",
    "18_attn_concat": "Attention heads h=0..n_head-1 concatenated",
    "19_attn_proj": "Output projection of attention, incl. bias",
    "20_resid_post_attn": "Residual connection after attention",
    "30_ln2": "LayerNorm 2 output",
    "31_fc": "First MLP linear layer, incl. bias",
    "32_gelu": "After GELU (tanh approximation)",
    "33_mlp_proj": "Second MLP linear layer, incl. bias",
    "34_resid_post_mlp": "Residual connection after MLP",
    "90_ln_f": "Final LayerNorm output",
    "91_logits": "Logits for all positions",
    "92_logits_last": "Logits of the last position",
    "93_probs_last_temp1": "Softmax of the last-position logits, temperature 1.0",
}


def ordered_trace_keys(n_layer: int, n_head: int) -> list[str]:
    # n_layer: int; n_head: int.
    # Returns: every trace key in the canonical write order.
    keys = list(GLOBAL_KEYS)
    for l in range(n_layer):
        keys.append(f"L{l}_10_ln1")
        for h in range(n_head):
            for base in HEAD_KEY_BASES:
                keys.append(f"L{l}_{base}_h{h}")
        for base in NON_HEAD_KEYS[1:]:
            keys.append(f"L{l}_{base}")
    keys.extend(ENDING_KEYS)
    return keys


def describe(key: str) -> str:
    # key: a trace key (without .csv). Returns: its manifest description.
    if key in DESCRIPTIONS:
        return DESCRIPTIONS[key]
    if key.startswith("L"):
        body = key[2:]  # strip the "L{layer}_" prefix
        if "_h" in body:
            body = body.split("_h")[0]  # drop the per-head suffix
        if body in DESCRIPTIONS:
            return DESCRIPTIONS[body]
    return key


def sanitize(a: np.ndarray) -> np.ndarray:
    # a: the in-memory array (may contain -inf from the causal mask).
    # Returns: a copy where -inf is replaced by MASK_VALUE (-1.0e30), so the
    # CSV is purely numeric (no "-inf" text, no nan, no inf) as required.
    if a.dtype == np.int64 or a.dtype == np.int32:
        return a
    return np.where(np.isfinite(a), a, MASK_VALUE)


def write_csv(path: str, tensor: torch.Tensor) -> None:
    # path: target .csv file; tensor: a trace tensor (no batch dimension).
    # Floats: %.10e with comma separator; -inf is written as -1.0e30. Integer
    # token ids: plain digit sequences, one per row. No header, no index column.
    a = tensor.numpy()
    if a.dtype == np.int64 or a.dtype == np.int32:
        np.savetxt(path, a.reshape(-1, 1), fmt="%d", delimiter=",")
    else:
        np.savetxt(path, sanitize(a), fmt="%.10e", delimiter=",")


def load_csv(path: str) -> np.ndarray:
    # path: a header-less .csv file. Returns: the 2-D array of parsed values.
    return np.genfromtxt(path, delimiter=",")


def die(msg: str) -> None:
    # msg: the violation message. Prints to stderr and exits with code 1.
    print(f"SELF-CHECK FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def self_check(trace: dict, lm_weight: np.ndarray, lm_bias: np.ndarray,
               n_head: int, n_layer: int) -> None:
    # trace: the dumped tensors; lm_weight: (V, C); lm_bias: (V,).
    # Any violation exits with code 1 (see the module docstring).
    # (a) logits reconstructed from 90_ln_f and lm_head match the dump.
    ln_f = trace["90_ln_f"].numpy()  # (T, C)
    reconstructed = ln_f @ lm_weight.T + lm_bias  # (T, V)
    dumped_logits = trace["91_logits"].numpy()  # (T, V)
    dev = float(np.max(np.abs(reconstructed - dumped_logits)))
    if not dev < 1e-5:
        die(f"logits reconstruction deviates by {dev:.3e} (>= 1e-5)")

    # (b) every row of every attn_probs matrix sums to 1.0 (+/- 1e-5).
    for l in range(n_layer):
        for h in range(n_head):
            probs = trace[f"L{l}_16_attn_probs_h{h}"].numpy()  # (T, T)
            row_sums = probs.sum(axis=1)
            dev = float(np.max(np.abs(row_sums - 1.0)))
            if not dev <= 1e-5:
                die(f"attn_probs row sum off by {dev:.3e} (layer {l}, head {h}, > 1e-5)")

    # (c) every value strictly above the diagonal in 16_attn_probs is 0.0.
    for l in range(n_layer):
        for h in range(n_head):
            probs = trace[f"L{l}_16_attn_probs_h{h}"].numpy()  # (T, T)
            upper = probs[np.triu_indices_from(probs, k=1)]
            if not np.all(upper == 0.0):
                die(f"16_attn_probs (layer {l}, head {h}) has non-zero upper triangle")

    # (d) no written file contains nan or infinite values. We test the
    # sanitized arrays — exactly what gets serialized to the CSVs — because the
    # in-memory causal-mask scores legitimately hold -inf and are replaced by
    # -1.0e30 on the way to disk.
    for name, tensor in trace.items():
        a = sanitize(tensor.numpy())
        if not np.all(np.isfinite(a)):
            die(f"non-finite value found in {name}.csv")


def main() -> None:
    p = argparse.ArgumentParser(description="Dump the forward-pass reference data")
    p.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    p.add_argument("--prompt", type=str, default=PROMPT)
    args = p.parse_args()

    with open(args.ckpt, "rb") as f:
        ckpt = torch.load(f, map_location="cpu")
    cfg_fields = [f.name for f in dataclasses.fields(GPTConfig)]
    cfg = GPTConfig(**{k: ckpt["config"][k] for k in cfg_fields})
    model = GPT(cfg)
    model.load_state_dict(ckpt["model_state_dict"])

    stoi = ckpt["stoi"]
    prompt = args.prompt

    # Every prompt character must exist in the vocabulary.
    missing = sorted({ch for ch in prompt if ch not in stoi})
    if missing:
        print(f"ERROR: prompt characters not in vocabulary: {missing}", file=sys.stderr)
        sys.exit(1)

    # The prompt must fit the context window.
    if len(prompt) > cfg.block_size:
        print(f"ERROR: prompt length {len(prompt)} exceeds block_size {cfg.block_size}",
              file=sys.stderr)
        sys.exit(1)

    # Reference values must be reproducible: CPU, eval mode, no gradients.
    model.eval()
    idx = torch.tensor([[stoi[ch] for ch in prompt]], dtype=torch.long)  # (1, T)
    trace: dict = {}
    with torch.no_grad():
        _, _ = model(idx, targets=None, trace=trace)

    os.makedirs(args.out_dir, exist_ok=True)
    keys = ordered_trace_keys(cfg.n_layer, cfg.n_head)
    for key in keys:
        if key not in trace:
            die(f"trace key {key} was not produced by the forward pass")
        write_csv(os.path.join(args.out_dir, key + ".csv"), trace[key])

    # manifest.csv — the only file with a header — in write order.
    with open(os.path.join(args.out_dir, "manifest.csv"), "w", newline="") as f:
        f.write("name,rows,cols,description\n")
        for key in keys:
            a = trace[key].numpy()
            f.write(f"{key},{a.shape[0]},{a.shape[1]},{describe(key)}\n")

    # Self-checks; any violation exits with code 1.
    lm_weight = model.lm_head.weight.detach().cpu().numpy()  # (V, C)
    lm_bias = model.lm_head.bias.detach().cpu().numpy()  # (V,)
    self_check(trace, lm_weight, lm_bias, cfg.n_head, cfg.n_layer)

    # File-level read-back: re-parse each CSV to prove the on-disk bytes are
    # purely numeric (no "-inf"/"nan" text), and that the masked scores file
    # really contains -1.0e30 rather than -inf.
    for key in keys:
        path = os.path.join(args.out_dir, key + ".csv")
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read()
        for bad in ("inf", "nan"):
            if bad in raw:
                die(f"{key}.csv contains the non-numeric text {bad!r}")
        a = load_csv(path)
        if not np.all(np.isfinite(a)):
            die(f"{key}.csv parsed with a non-finite value")
    masked = load_csv(os.path.join(args.out_dir, f"L0_15_scores_masked_h0.csv"))
    if not np.any(np.isclose(masked, MASK_VALUE)):
        die("L0_15_scores_masked_h0.csv does not contain the -1.0e30 mask value")

    # Quick visual check: the 5 most likely next characters.
    probs = trace["93_probs_last_temp1"].numpy().reshape(-1)  # (V,)
    top = np.argsort(probs)[::-1][:5]
    itos = ckpt["itos"]
    print(f"Wrote {len(keys)} CSV files plus manifest.csv to {args.out_dir}")
    print(f"Prompt: {prompt!r} (T = {len(prompt)})")
    print("Top-5 next-character probabilities:")
    for i in top:
        ch = itos.get(str(i), "?")
        print(f"  {ch!r} (index {int(i)}) : {probs[i]:.6f}")


if __name__ == "__main__":
    main()
