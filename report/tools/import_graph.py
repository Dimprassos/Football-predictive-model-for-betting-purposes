"""Εξαγωγή του πραγματικού γράφου εσωτερικών εξαρτήσεων (imports) του έργου.

Χρησιμεύει για να **επαληθεύονται** τα διαγράμματα σχεδίασης του `report/design.md`
αντί να γράφονται με βάση την εντύπωση: το διάγραμμα συστατικών (Δ2) και οι
εξαρτήσεις του διαγράμματος κλάσεων (Δ3) προκύπτουν από αυτή την έξοδο.

Όταν αλλάξει η δομή του κώδικα, ξανατρέξε το και σύγκρινε με τα διαγράμματα:

    python report/tools/import_graph.py            # πλήρης γράφος ανά μονάδα
    python report/tools/import_graph.py --fan-in   # πόσες μονάδες εισάγουν την καθεμία

Δεν έχει εξαρτήσεις πέρα από τη standard library και δεν αγγίζει το pipeline.
"""
from __future__ import annotations

import argparse
import ast
import os
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PACKAGES = ("src.", "scripts.")


def python_files() -> list[str]:
    """Every project .py file worth analysing (src/, scripts/, app.py)."""
    found: list[str] = []
    for base in ("src", "scripts"):
        for dirpath, _dirs, files in os.walk(os.path.join(ROOT, base)):
            if "__pycache__" in dirpath:
                continue
            found.extend(os.path.join(dirpath, f) for f in files if f.endswith(".py"))
    app = os.path.join(ROOT, "app.py")
    if os.path.exists(app):
        found.append(app)
    return sorted(found)


def build_graph() -> dict[str, set[str]]:
    """Map each file (repo-relative) to the set of internal modules it imports."""
    edges: dict[str, set[str]] = defaultdict(set)
    for path in python_files():
        rel = os.path.relpath(path, ROOT).replace(os.sep, "/")
        try:
            with open(path, encoding="utf-8") as fh:
                tree = ast.parse(fh.read())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom):
                if node.module and node.module.startswith(PACKAGES):
                    edges[rel].add(node.module)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(PACKAGES):
                        edges[rel].add(alias.name)
    return edges


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fan-in", action="store_true", help="count how many modules import each module")
    args = parser.parse_args()

    graph = build_graph()

    if args.fan_in:
        counts: dict[str, int] = defaultdict(int)
        for deps in graph.values():
            for dep in deps:
                counts[dep] += 1
        print("modules imported by N others (fan-in):")
        for mod, n in sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])):
            print("  %-38s %d" % (mod, n))
        return

    for src in sorted(graph):
        print(src)
        for dep in sorted(graph[src]):
            print("    ->", dep)


if __name__ == "__main__":
    main()
