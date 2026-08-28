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
import re
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
MODULE_ORDER = ["Mat.bas", "Nn.bas", "Gpt.bas", "Sampler.bas", "Ui.bas", "Probe.bas"]

# Built by a macro rather than by hand, so the layout is versioned with the
# code and comes out identical after every rebuild.
SETUP_MACRO = "Ui.SetupSheetSafe"

# Goes into the workbook's own code module. Gridlines are a property of the
# window, and the window the layout is built in is not the one anybody ends up
# looking at, so the view is set when the file is opened.
WORKBOOK_CODE = """Private Sub Workbook_Open()
    Ui.ApplyView
End Sub
"""


def die(msg: str) -> None:
    print(f"VBA INJECTION FAILED: {msg}", file=sys.stderr)
    sys.exit(1)


# Identifiers the macro language refuses as variable names. Not the full
# keyword list -- only the ones that are tempting and actually break.
RESERVED = {
    "scale", "line", "circle", "print", "base", "rem", "stop", "loop", "next",
    "step", "then", "else", "end", "exit", "to", "is", "mod", "new", "not",
    "in", "or", "and", "set", "let", "get", "put", "close", "open", "input",
    "type", "error", "call", "const", "dim", "do", "each", "for", "if", "sub",
    "function", "while", "with", "wend", "option", "redim", "select", "class",
}

PROC_RE = re.compile(
    r"^\s*(?:Public\s+|Private\s+|Friend\s+)?(?:Static\s+)?"
    r"(Sub|Function|Property\s+\w+)\s+(\w+)\s*\((.*?)\)", re.IGNORECASE)
END_RE = re.compile(r"^\s*End\s+(Sub|Function|Property)\b", re.IGNORECASE)
DECL_RE = re.compile(r"^\s*(?:Dim|Static|Const|Private|Public)\s+(.*)$",
                     re.IGNORECASE)
ASSIGN_RE = re.compile(r"^\s*(?:Set\s+)?([A-Za-z_]\w*)\s*(?:\([^)]*\))?\s*=(?!=)")
FOR_RE = re.compile(r"^\s*For\s+(?:Each\s+)?([A-Za-z_]\w*)\s", re.IGNORECASE)


def declared_names(text: str) -> set:
    # text: a declaration tail such as "a As Long, b() As Double".
    # Returns: the bare identifiers it introduces.
    names = set()
    for part in text.split(","):
        m = re.match(r"\s*(?:ByVal\s+|ByRef\s+|Optional\s+)*([A-Za-z_]\w*)",
                     part, re.IGNORECASE)
        if m:
            names.add(m.group(1).lower())
    return names


def lint_module(path: str) -> list:
    # path: a .bas file. Returns: a list of complaints.
    #
    # Catches the two mistakes that cost the most time here, because the
    # language compiles procedure by procedure and only reports them once the
    # procedure is first reached: a variable that was never declared, and a
    # reserved word used as a name.
    problems = []
    lines = open(path, encoding="utf-8").read().splitlines()

    module_level = set()
    in_proc = False
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("'"):
            continue
        if PROC_RE.match(line):
            in_proc = True
        elif END_RE.match(line):
            in_proc = False
        elif not in_proc:
            m = DECL_RE.match(line)
            if m and not re.match(r"^\s*(?:Private|Public)\s+(?:Type|Const|Declare)\b",
                                  line, re.IGNORECASE):
                module_level |= declared_names(m.group(1))

    i = 0
    while i < len(lines):
        m = PROC_RE.match(lines[i])
        if not m:
            i += 1
            continue
        proc_name = m.group(2)
        local = declared_names(m.group(3)) | {proc_name.lower()}
        body = []
        i += 1
        while i < len(lines) and not END_RE.match(lines[i]):
            body.append(lines[i])
            i += 1

        for line in body:
            d = DECL_RE.match(line)
            if d:
                local |= declared_names(d.group(1))

        known = local | module_level
        for line in body:
            stripped = line.strip()
            if not stripped or stripped.startswith("'") or stripped.startswith("."):
                continue
            if DECL_RE.match(line):
                for nm in declared_names(DECL_RE.match(line).group(1)):
                    if nm in RESERVED:
                        problems.append(
                            f"{os.path.basename(path)}: {proc_name} declares "
                            f"'{nm}', which is a reserved word")
                continue
            for rx in (ASSIGN_RE, FOR_RE):
                mm = rx.match(line)
                if mm:
                    nm = mm.group(1).lower()
                    if nm not in known and nm not in RESERVED:
                        problems.append(
                            f"{os.path.basename(path)}: {proc_name} assigns to "
                            f"'{mm.group(1)}', which is never declared")
                    break
    return problems


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

    # Pre-flight: cheaper to catch here than through a dialog in a hidden
    # spreadsheet instance.
    complaints = []
    for _, path in modules:
        complaints += lint_module(path)
    if complaints:
        for c in complaints:
            print(f"  {c}", file=sys.stderr)
        die(f"{len(complaints)} problem(s) in the macro sources")
    print(f"Checked {len(modules)} modules, no undeclared or reserved names")

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

        # The workbook's own code module, addressed through its code name so
        # this does not depend on the interface language.
        doc = project.VBComponents(wb.api.CodeName).CodeModule
        if doc.CountOfLines:
            doc.DeleteLines(1, doc.CountOfLines)
        doc.AddFromString(WORKBOOK_CODE)

        # The safe wrapper returns the error text instead of raising: an
        # unhandled macro error opens a dialog nobody can click away in a
        # hidden instance, and the call hangs instead of failing.
        try:
            problem = app.api.Run(SETUP_MACRO)
        except Exception as exc:  # noqa: BLE001 - the message matters
            die(f"{SETUP_MACRO} could not be called: {exc}")
        if problem:
            die(f"{SETUP_MACRO}: {problem}")

        wb.api.SaveAs(wb_path, FileFormat=XL_OPEN_XML_MACRO_ENABLED)

        ui_names = sorted(n.name for n in wb.names
                          if n.name.split("!")[-1].startswith("UI_"))
        print(f"Injected into {wb_path}")
        print(f"  modules:   {[n for n, _ in modules]}")
        print(f"  project:   {sorted(present)}")
        print(f"  sheets:    {[s.name for s in wb.sheets]}")
        print(f"  interface: {SETUP_MACRO} ran, {len(ui_names)} named controls")
    finally:
        app.quit()


if __name__ == "__main__":
    main()
