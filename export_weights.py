"""Export the 54 weight tensors of out/ckpt.pt as CSV files.

Every tensor of the checkpoint's model_state_dict is written to
<out-dir>/ as a pure numeric CSV (no header, no index column, comma
separator, point decimal separator, %.10e) — the same CSV conventions as
the reference dump. The file name is the contract's workbook-wide named
range in lowercase: wte.csv, l0_attn_w.csv, l0_ln1_b.csv, lm_w.csv, ...

Export conventions (binding):
  * nn.Linear weights are stored as (out_features, in_features) in
    PyTorch and are exported TRANSPOSED as (in_features, out_features),
    so the target environment can compute y = x @ W + b without a
    transpose. Affected: c_attn, c_proj, mlp.c_fc, mlp.c_proj, lm_head.
  * Embeddings are NOT transposed: wte stays (vocab_size, n_embd), wpe
    stays (block_size, n_embd) — row = token/position index.
  * Biases stay row vectors (1, out_features); LayerNorm weight/bias are
    row vectors (1, C).

Additionally written:
  * manifest.csv — header name,rows,cols,source,transposed; one line per
    tensor in contract order.
  * vocab.csv   — 65 lines, one column; each character as a JSON-encoded
    string so newline (ID 0) and space (ID 1) survive unambiguously.
  * config.csv  — header key,value; the seven CFG_ values.

Every entry of model_state_dict must be exported exactly once; a leftover
or unknown tensor exits with code 1 (a forgotten tensor must fail here,
not in Phase 4). Prints the number of written files and the total number
of exported elements, which must be 818241.
"""

import argparse
import dataclasses
import json
import os
import sys

import numpy as np
import torch

from model import GPT, GPTConfig

# Plain float32 everywhere: no TF32 matmul, no cuDNN TF32.
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CKPT = os.path.join(BASE_DIR, "out", "ckpt.pt")
DEFAULT_OUT_DIR = os.path.join(BASE_DIR, "export")

# The named-range contract in contract order: (name, source key in the
# state dict, transposed?). Workbook-wide names, uppercase, no sheet
# prefix. Embeddings are NOT transposed; the marked weights are.
NAMED_RANGES: list[tuple[str, str, bool]] = [
    ("WTE", "transformer.wte.weight", False),
    ("WPE", "transformer.wpe.weight", False),
]
for _l in range(4):
    NAMED_RANGES += [
        (f"L{_l}_LN1_W", f"transformer.h.{_l}.ln1.weight", False),
        (f"L{_l}_LN1_B", f"transformer.h.{_l}.ln1.bias", False),
        (f"L{_l}_ATTN_W", f"transformer.h.{_l}.attn.c_attn.weight", True),
        (f"L{_l}_ATTN_B", f"transformer.h.{_l}.attn.c_attn.bias", False),
        (f"L{_l}_PROJ_W", f"transformer.h.{_l}.attn.c_proj.weight", True),
        (f"L{_l}_PROJ_B", f"transformer.h.{_l}.attn.c_proj.bias", False),
        (f"L{_l}_LN2_W", f"transformer.h.{_l}.ln2.weight", False),
        (f"L{_l}_LN2_B", f"transformer.h.{_l}.ln2.bias", False),
        (f"L{_l}_FC_W", f"transformer.h.{_l}.mlp.c_fc.weight", True),
        (f"L{_l}_FC_B", f"transformer.h.{_l}.mlp.c_fc.bias", False),
        (f"L{_l}_FCPROJ_W", f"transformer.h.{_l}.mlp.c_proj.weight", True),
        (f"L{_l}_FCPROJ_B", f"transformer.h.{_l}.mlp.c_proj.bias", False),
    ]
NAMED_RANGES += [
    ("LNF_W", "transformer.ln_f.weight", False),
    ("LNF_B", "transformer.ln_f.bias", False),
    ("LM_W", "lm_head.weight", True),
    ("LM_B", "lm_head.bias", False),
]

# (key, checkpoint config field) of the seven CFG values, in contract
# order. HEAD_DIM is derived: n_embd / n_head.
CFG_KEYS = [
    ("N_LAYER", "n_layer"),
    ("N_HEAD", "n_head"),
    ("N_EMBD", "n_embd"),
    ("HEAD_DIM", None),
    ("BLOCK_SIZE", "block_size"),
    ("VOCAB_SIZE", "vocab_size"),
    ("MLP_HIDDEN", "mlp_hidden"),
]

# The contract's reference shapes. IMPORTANT: these are the shapes AFTER
# the export — for the transposed weights (ATTN_W, PROJ_W, FC_W,
# FCPROJ_W, LM_W) the pair already is (in_features, out_features). They
# are used verbatim as the expectation for the exported array; do NOT
# transpose them a second time (that double-transpose was the bug).
CONTRACT_SHAPES = {
    "WTE": (65, 128),
    "WPE": (64, 128),
    "LN1_W": (1, 128), "LN1_B": (1, 128),
    "ATTN_W": (128, 384), "ATTN_B": (1, 384),
    "PROJ_W": (128, 128), "PROJ_B": (1, 128),
    "LN2_W": (1, 128), "LN2_B": (1, 128),
    "FC_W": (128, 512), "FC_B": (1, 512),
    "FCPROJ_W": (512, 128), "FCPROJ_B": (1, 128),
    "LNF_W": (1, 128), "LNF_B": (1, 128),
    "LM_W": (128, 65), "LM_B": (1, 65),
}


def die(msg: str) -> None:
    # msg: the violation message. Prints to stderr and exits with code 1.
    print(f"EXPORT ERROR: {msg}", file=sys.stderr)
    sys.exit(1)


def export_shape(name: str) -> tuple[int, int]:
    # name: contract name. Returns: the exported (rows, cols) of this
    # tensor, verbatim from CONTRACT_SHAPES (already the post-export
    # shape — see the comment there).
    base = name.split("_", 1)[1] if name.startswith("L") and name[1].isdigit() else name
    return CONTRACT_SHAPES[base]


def write_tensor_csv(path: str, a: np.ndarray) -> None:
    # path: target .csv; a: the exported 2-D array (row-vector layout).
    # Pure numeric CSV: no header, no index column, %.10e.
    np.savetxt(path, a, fmt="%.10e", delimiter=",")


def main() -> None:
    p = argparse.ArgumentParser(description="Export the model weights as CSV")
    p.add_argument("--ckpt", type=str, default=DEFAULT_CKPT)
    p.add_argument("--out-dir", type=str, default=DEFAULT_OUT_DIR)
    args = p.parse_args()

    with open(args.ckpt, "rb") as f:
        ckpt = torch.load(f, map_location="cpu")
    cfg_fields = [f.name for f in dataclasses.fields(GPTConfig)]
    cfg = GPTConfig(**{k: ckpt["config"][k] for k in cfg_fields})
    state_dict: dict = ckpt["model_state_dict"]

    # The checkpoint must match the contract's reference shapes.
    contract_cfg = {
        "n_layer": 4, "n_head": 4, "n_embd": 128,
        "block_size": 64, "vocab_size": 65, "mlp_hidden": 512,
    }
    for key, value in contract_cfg.items():
        if getattr(cfg, key) != value:
            die(f"checkpoint config {key}={getattr(cfg, key)} differs from "
                f"the contract ({key}={value})")

    # Build the export entries: every state-dict tensor exactly once, in
    # contract order. 1-D parameters become row vectors (1, n); the marked
    # weights are transposed (out, in) -> (in, out).
    entries: list[dict] = []
    for name, source, transposed in NAMED_RANGES:
        if source not in state_dict:
            die(f"state dict has no entry {source!r} (contract name {name})")
        tensor = state_dict[source].detach().cpu().numpy()
        if tensor.ndim == 1:
            exported = tensor.reshape(1, -1)
        elif transposed:
            exported = tensor.T
        else:
            exported = np.asarray(tensor)
        expected = export_shape(name)
        if exported.shape != expected:
            die(f"{name} exports as {exported.shape}, the contract says {expected}")
        entries.append({"name": name, "source": source, "transposed": transposed,
                        "array": exported,
                        "shape": (exported.shape[0], exported.shape[1])})

    # Completeness: every state-dict entry exactly once, nothing unknown.
    exported_keys = [e["source"] for e in entries]
    missing = sorted(set(state_dict) - set(exported_keys))
    if missing:
        die(f"{len(missing)} state-dict tensors were not exported: {missing}")
    unknown = sorted(set(exported_keys) - set(state_dict))
    if unknown:
        die(f"{len(unknown)} exported tensors are not in the state dict: {unknown}")

    os.makedirs(args.out_dir, exist_ok=True)

    # One CSV per tensor: file name = contract name in lowercase.
    for entry in entries:
        write_tensor_csv(
            os.path.join(args.out_dir, entry["name"].lower() + ".csv"),
            entry["array"],
        )

    # manifest.csv — header name,rows,cols,source,transposed — in contract order.
    with open(os.path.join(args.out_dir, "manifest.csv"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write("name,rows,cols,source,transposed\n")
        for entry in entries:
            rows, cols = entry["shape"]
            f.write(f'{entry["name"]},{rows},{cols},{entry["source"]},'
                    f'{"1" if entry["transposed"] else "0"}\n')

    # vocab.csv: 65 lines, one column, each character as a JSON-encoded
    # string — newline (ID 0) and space (ID 1) stay unambiguous.
    itos = ckpt["itos"]
    with open(os.path.join(args.out_dir, "vocab.csv"), "w",
              encoding="utf-8", newline="\n") as f:
        for i in range(int(cfg.vocab_size)):
            ch = itos.get(str(i))
            if ch is None:
                die(f"itos has no entry for token ID {i}")
            f.write(json.dumps(ch, ensure_ascii=True) + "\n")

    # config.csv: header key,value; the seven CFG values in contract order.
    with open(os.path.join(args.out_dir, "config.csv"), "w",
              encoding="utf-8", newline="\n") as f:
        f.write("key,value\n")
        for key, field in CFG_KEYS:
            if field is None:  # HEAD_DIM = n_embd / n_head
                value = cfg.n_embd // cfg.n_head
            else:
                value = getattr(cfg, field)
            f.write(f"CFG_{key},{int(value)}\n")

    n_files = len(entries) + 3  # tensor CSVs plus three extra files: manifest.csv, vocab.csv, config.csv
    total = int(sum(e["array"].size for e in entries))
    print(f"Wrote {n_files} files to {args.out_dir} "
          f"({len(entries)} tensor CSVs plus manifest.csv, vocab.csv, config.csv)")
    print(f"Total elements exported: {total}")
    if total != 818241:
        die(f"element sum is {total}, expected 818241")


if __name__ == "__main__":
    main()
