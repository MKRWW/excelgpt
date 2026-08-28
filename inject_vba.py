"""Inject the macro modules from vba/ into the built workbook.

The macro code lives in vba/*.bas as plain text so it is versionable and
reviewable like any other source. This script is the only thing that puts it
into the workbook -- nothing is typed into the editor by hand, so a rebuild
always reproduces the same state.

Requires "Trust access to the VBA project object model" to be enabled in the
spreadsheet's trust centre; without it the project is not reachable through
automation and the import fails with a permission error.
"""

import argparse
import os
import sys

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_WB = os.path.join(BASE_DIR, "build", "excelgpt.xlsm")
DEFAULT_SRC = os.path.join(BASE_DIR, "vba")

XL_OPEN_XML_MACRO_ENABLED = 52

# Sheets the macro code needs that the data build does not create. They are
# slotted in before 99_Meta so the numbering stays in reading order.
EXTRA_SHEETS = ["97_Trace", "98_Probe"]
EXTRA_SHEETS_BEFORE = "99_Meta"

# Module import order. Modules referencing each other by name do not need a
# particular order in the project, but a stable order keeps the project
# listing and any diff of it predictable.
MODULE_ORDER = ["Mat.bas", "Nn.bas", "Gpt.bas", "Sampler.bas", "Probe.bas"]


def die(msg: str) -> None:
    print(f"VBA INJECTION FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


def main() -> None:
    p = argparse.ArgumentParser(description="Inject the macro modules")
    p.add_argument("--workbook", type=str, default=DEFAULT_WB)
    p.add_argument("--src", type=str, default=DEFAULT_SRC)
    p.add_argument("--visible", action="store_true", default=False)
    args = p.parse_args()

    import xlwings as xw

    wb_path = os.path.abspath(args.workbook)
    if not os.path.exists(wb_path):
        die(f"{wb_path} is missing -- run build_workbook.py first")

    modules = []
    for name in MODULE_ORDER:
        path = os.path.join(args.src, name)
        if not os.path.exists(path):
            die(f"{path} is missing")
        modules.append((os.path.splitext(name)[0], path))
    extra = sorted(f for f in os.listdir(args.src)
                   if f.endswith(".bas") and f not in MODULE_ORDER)
    if extra:
        die(f"vba/ holds modules that are not in MODULE_ORDER: {extra}")

    app = xw.App(visible=args.visible, add_book=False)
    try:
        app.display_alerts = False
        wb = app.books.open(wb_path)

        try:
            project = wb.api.VBProject
            _ = project.VBComponents.Count
        except Exception as exc:  # noqa: BLE001 - the message matters, not the type
            die("the VBA project is not reachable through automation. Enable "
                "'trust access to the VBA project object model' in the trust "
                f"centre and run again. ({exc})")

        # Sheets the macro code expects.
        existing = [s.name for s in wb.sheets]
        for sheet_name in EXTRA_SHEETS:
            if sheet_name not in existing:
                wb.sheets.add(name=sheet_name,
                              before=wb.sheets[EXTRA_SHEETS_BEFORE])

        # Replace rather than add: importing twice would otherwise leave
        # Mat, Mat1, Mat2 behind and the wrong one would win.
        wanted = {name for name, _ in modules}
        for comp in list(project.VBComponents):
            if comp.Name in wanted:
                project.VBComponents.Remove(comp)

        for name, path in modules:
            project.VBComponents.Import(path)

        present = [c.Name for c in project.VBComponents]
        missing = sorted(wanted - set(present))
        if missing:
            die(f"modules missing after import: {missing}")

        wb.api.SaveAs(wb_path, FileFormat=XL_OPEN_XML_MACRO_ENABLED)

        print(f"Injected into {wb_path}")
        print(f"  modules:  {[n for n, _ in modules]}")
        print(f"  project:  {sorted(present)}")
        print(f"  sheets:   {[s.name for s in wb.sheets]}")
    finally:
        app.quit()


if __name__ == "__main__":
    main()
