#!/usr/bin/env python3
"""
Usage:
  python nb_keep_code_only.py input.ipynb output.ipynb [--keep-metadata]

Creates a copy of input.ipynb containing only code cells.
By default outputs and execution_count are cleared.
"""
import sys
import nbformat

def keep_code_only(src_path, dst_path, keep_metadata=False):
    nb = nbformat.read(src_path, as_version=4)
    new_nb = nbformat.v4.new_notebook()
    # Optionally copy top-level metadata
    if keep_metadata:
        new_nb.metadata = nb.metadata.copy()
    new_cells = []
    for cell in nb.cells:
        if cell.cell_type == "code":
            # copy code cell, clear outputs and execution count
            new_cell = nbformat.v4.new_code_cell(source=cell.source)
            if keep_metadata:
                new_cell.metadata = cell.metadata.copy()
            # clear outputs and execution_count to make a clean copy
            new_cell.outputs = []
            new_cell.execution_count = None
            new_cells.append(new_cell)
    new_nb.cells = new_cells
    nbformat.write(new_nb, dst_path)
    print(f"Wrote {len(new_cells)} code cells to {dst_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python nb_keep_code_only.py input.ipynb output.ipynb [--keep-metadata]")
        sys.exit(1)
    src = sys.argv[1]
    dst = sys.argv[2]
    keep_meta = "--keep-metadata" in sys.argv[3:]
    keep_code_only(src, dst, keep_meta)

