"""Build the workbook from the exported CSVs.

Reads export/ (written by export_weights.py) and lays every tensor out as a
named range in a macro-enabled workbook: one sheet per layer, plus embeddings,
output head and a metadata sheet.

The named ranges are a contract. The forward pass implemented later addresses
weights exclusively through these names, never through cell addresses.

Conventions (binding, see README.md):
  * one assignment per tensor -- never a loop over single cells. At 818241
    values that is the difference between seconds and hours.
  * values only, no formulas, no formatting.
  * the named range covers the data block only, not its label row.
  * VOCAB holds Unicode code points as numbers, not the characters. Writing
    the characters is not an option: a cell value starting with an apostrophe
    is swallowed as a text-prefix marker, so token 5 would come back empty.
    Code points also make the two invisible entries legible -- 10 is the
    newline at token 0, 32 the space at token 1. The macro code decodes with
    Chr$().
  * saved as .xlsm (FileFormat 52) so macro code can be added later.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_EXPORT_DIR = os.path.join(BASE_DIR, "export")
DEFAULT_OUT = os.path.join(BASE_DIR, "build", "excelgpt.xlsm")

XL_OPEN_XML_MACRO_ENABLED = 52  # Excel's FileFormat code for .xlsm

SHEETS = ["00_LLM", "10_Embedding", "20_Layer0", "30_Layer1",
          "40_Layer2", "50_Layer3", "90_Output", "99_Meta"]

CFG_KEYS = ["N_LAYER", "N_HEAD", "N_EMBD", "HEAD_DIM",
            "BLOCK_SIZE", "VOCAB_SIZE", "MLP_HIDDEN"]


def layer_sheet(l: int) -> str:
    # l: layer index 0..3. Returns: its sheet name -- 20_Layer0, 30_Layer1,
    # 40_Layer2, 50_Layer3. The step is 10, not 20; the gap to 90_Output is
    # deliberate so further sheets can be slotted in.
    return f"{10 * l + 20}_Layer{l}"


def block_order() -> list[tuple[str, str]]:
    # Returns: (name, sheet) for all 62 blocks, in the order they are laid
    # out on the sheets.
    blocks = [("WTE", "10_Embedding"), ("WPE", "10_Embedding")]
    for l in range(4):
        sheet = layer_sheet(l)
        for suffix in ("LN1_W", "LN1_B", "ATTN_W", "ATTN_B", "PROJ_W", "PROJ_B",
                       "LN2_W", "LN2_B", "FC_W", "FC_B", "FCPROJ_W", "FCPROJ_B"):
            blocks.append((f"L{l}_{suffix}", sheet))
    blocks += [("LNF_W", "90_Output"), ("LNF_B", "90_Output"),
               ("LM_W", "90_Output"), ("LM_B", "90_Output")]
    blocks += [(f"CFG_{k}", "99_Meta") for k in CFG_KEYS]
    blocks.append(("VOCAB", "99_Meta"))
    return blocks


def die(msg: str) -> None:
    # msg: the violation message. Prints to stderr and exits with code 1.
    print(f"WORKBOOK BUILD FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def load_export(export_dir: str) -> tuple[dict, list, dict]:
    # export_dir: directory holding the CSVs from export_weights.py.
    # Returns: (tensors, vocab, config) -- tensors is {name: (rows, cols)
    #   -> float64 array} keyed by contract name, vocab a list of 65 single
    #   characters, config {CFG_NAME: int}.
    manifest = os.path.join(export_dir, "manifest.csv")
    if not os.path.exists(manifest):
        die(f"{manifest} is missing -- run export_weights.py first")

    with open(manifest, encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not lines or lines[0] != "name,rows,cols,source,transposed":
        die("manifest.csv is missing its header line")

    tensors = {}
    for line in lines[1:]:
        parts = line.split(",")
        if len(parts) != 5:
            die(f"malformed manifest line {line!r}")
        name, rows, cols = parts[0], int(parts[1]), int(parts[2])
        path = os.path.join(export_dir, name.lower() + ".csv")
        a = np.atleast_2d(np.genfromtxt(path, delimiter=","))
        if a.shape != (rows, cols):
            die(f"{name}.csv is {a.shape}, the manifest says ({rows}, {cols})")
        if not np.all(np.isfinite(a)):
            die(f"{name}.csv holds a non-finite value")
        tensors[name] = a.astype(np.float64)

    vocab = []
    with open(os.path.join(export_dir, "vocab.csv"), encoding="utf-8") as f:
        for line in f.read().splitlines():
            try:
                ch = json.loads(line)
            except json.JSONDecodeError as e:
                die(f"malformed vocab.csv line {line!r}: {e}")
            if not isinstance(ch, str) or len(ch) != 1:
                die(f"vocab.csv line {line!r} is not a single character")
            vocab.append(ch)

    config = {}
    with open(os.path.join(export_dir, "config.csv"), encoding="utf-8") as f:
        lines = f.read().splitlines()
    if not lines or lines[0] != "key,value":
        die("config.csv is missing its header line")
    for line in lines[1:]:
        key, value = line.split(",")
        config[key] = int(value)
    expected_cfg = [f"CFG_{k}" for k in CFG_KEYS]
    if list(config) != expected_cfg:
        die(f"config.csv holds {list(config)}, expected {expected_cfg}")

    if len(vocab) != config["CFG_VOCAB_SIZE"]:
        die(f"vocab.csv has {len(vocab)} entries, config says "
            f"{config['CFG_VOCAB_SIZE']}")
    return tensors, vocab, config


def block_values(name: str, tensors: dict, vocab: list, config: dict):
    # name: a contract name. Returns: (values, rows, cols) where values is a
    # list of rows ready to be assigned to a range in one go.
    if name in tensors:
        a = tensors[name]
        return a.tolist(), a.shape[0], a.shape[1]
    if name == "VOCAB":
        return [[float(ord(ch))] for ch in vocab], len(vocab), 1
    if name in config:
        return [[config[name]]], 1, 1
    die(f"unknown block {name}")


def main() -> None:
    p = argparse.ArgumentParser(description="Build the workbook from export/")
    p.add_argument("--export-dir", type=str, default=DEFAULT_EXPORT_DIR)
    p.add_argument("--out", type=str, default=DEFAULT_OUT)
    p.add_argument("--visible", action="store_true", default=False,
                   help="show the spreadsheet while it is being built")
    args = p.parse_args()

    import xlwings as xw

    tensors, vocab, config = load_export(args.export_dir)
    blocks = block_order()
    missing = [n for n, _ in blocks
               if n not in tensors and n not in config and n != "VOCAB"]
    if missing:
        die(f"{len(missing)} contract blocks are absent from the export: {missing}")

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    if os.path.exists(out_path):
        os.remove(out_path)

    start = time.time()
    app = xw.App(visible=args.visible, add_book=False)
    try:
        app.display_alerts = False
        app.screen_updating = False

        wb = app.books.add()
        # Only settable once a workbook exists -- on a bare application
        # instance the property raises.
        app.calculation = "manual"
        # Create the sheets in contract order, each after the previous one,
        # then drop whatever default sheets the new workbook came with.
        previous = wb.sheets[0]
        for sheet_name in SHEETS:
            previous = wb.sheets.add(name=sheet_name, after=previous)
        for s in list(wb.sheets):
            if s.name not in SHEETS:
                s.delete()

        cells = 0
        for sheet_name in SHEETS:
            sheet = wb.sheets[sheet_name]
            sheet.range("A1").value = sheet_name
            row = 3
            for name, block_sheet in blocks:
                if block_sheet != sheet_name:
                    continue
                values, rows, cols = block_values(name, tensors, vocab, config)
                sheet.range((row, 1)).value = f"{name}  ({rows} x {cols})"
                data = sheet.range((row + 1, 1), (row + rows, cols))
                data.value = values           # one assignment per tensor
                data.name = name              # workbook-wide named range
                cells += rows * cols
                row += rows + 3               # label + data + two blank rows

        wb.api.SaveAs(out_path, FileFormat=XL_OPEN_XML_MACRO_ENABLED)
        duration = time.time() - start

        # Read-back probe: every named range, straight out of the workbook.
        sheet_names = [s.name for s in wb.sheets]
        if sheet_names != SHEETS:
            die(f"sheet order is {sheet_names}, expected {SHEETS}")

        worst, worst_name = 0.0, "-"
        for name, block_sheet in blocks:
            rng = wb.names[name].refers_to_range
            if rng.sheet.name != block_sheet:
                die(f"{name} sits on {rng.sheet.name}, expected {block_sheet}")
            values, rows, cols = block_values(name, tensors, vocab, config)
            read = rng.options(ndim=2).value
            if (len(read), len(read[0])) != (rows, cols):
                die(f"{name}: read back {len(read)}x{len(read[0])}, "
                    f"expected {rows}x{cols}")
            if name == "VOCAB":
                # The two invisible characters are the ones worth naming:
                # token 0 is the newline, token 1 the space.
                if int(read[0][0]) != 10 or int(read[1][0]) != 32:
                    die(f"VOCAB starts with {read[0][0]}, {read[1][0]} -- "
                        f"expected 10 (newline) and 32 (space)")
            dev = float(np.max(np.abs(np.array(read, dtype=np.float64)
                                      - np.array(values, dtype=np.float64))))
            if dev > worst:
                worst, worst_name = dev, name
            if not dev < 1e-12:
                die(f"{name} on {block_sheet}: deviation {dev:.3e} (>= 1e-12)")

        n_names = len(list(wb.names))
        print(f"Built {out_path}")
        print(f"  sheets:        {len(sheet_names)}  {sheet_names}")
        print(f"  named ranges:  {n_names}")
        print(f"  cells written: {cells:,}")
        print(f"  read-back:     all {len(blocks)} ranges verified, "
              f"largest deviation {worst:.3e} ({worst_name})")
        print(f"  vocabulary:    {len(vocab)} code points, "
              f"token 0 = newline, token 1 = space")
        print(f"  duration:      {duration:.1f} s")
        if n_names != len(blocks):
            die(f"workbook holds {n_names} names, expected {len(blocks)}")
    finally:
        try:
            app.screen_updating = True
            app.calculation = "automatic"
        except Exception:
            pass
        app.quit()


if __name__ == "__main__":
    main()
