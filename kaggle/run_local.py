"""Run one of the Kaggle notebooks on this computer and save the executed copy.

The notebooks read their settings from environment variables (see kaggle/README.md), so set
those before calling this script. It runs every cell in a fresh Python kernel, writes the executed
notebook (with all outputs) even if a cell fails, and exits with status 1 if any cell raised.

    python kaggle/run_local.py <notebook.ipynb> <executed_copy.ipynb>
"""
import sys
import time

import nbformat
from nbclient import NotebookClient

src, dst = sys.argv[1], sys.argv[2]
nb = nbformat.read(src, as_version=4)
t0 = time.time()
try:
    # timeout=None: a full training cell runs for hours.
    NotebookClient(nb, timeout=None, kernel_name="python3",
                   resources={"metadata": {"path": "."}}).execute()
finally:
    nbformat.write(nb, dst)

errors = [o for c in nb.cells if c.cell_type == "code" for o in c.outputs if o.output_type == "error"]
print(f"{src} finished at {time.strftime('%H:%M')} after {(time.time() - t0) / 60:.0f} min; "
      f"errors: {len(errors)}; executed copy: {dst}")
for e in errors:
    print(e.ename, e.evalue)
sys.exit(1 if errors else 0)
