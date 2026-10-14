"""Checks every shared label file in shared/labels (see shared/labels/README.md). GitHub runs it on every proposed upload
and every change to that folder (.github/workflows/check-shared-labels.yml); the app runs the same checks
(newstrader/ml/shared_labels.py) before it uses a file it downloaded. It only reads the files - nothing in them is run.

    python scripts/check_shared_labels.py                   # shared/labels in this folder
    python scripts/check_shared_labels.py path/to/labels    # another folder (e.g. the files of a pull request)

Exit code 0 = every file passed (or there are none yet), 1 = at least one file was rejected.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from newstrader.ml.shared_labels import check_folder  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", nargs="?", default=str(ROOT / "shared" / "labels"))
    args = ap.parse_args(argv)
    folder = Path(args.folder)
    print("## Shared label files\n")
    if not folder.is_dir():
        print(f"There is no {folder} folder - nothing to check.")
        return 0
    results = check_folder(folder)
    if not results:
        print("No label files yet.")
        return 0
    for res in results:
        print(f"- {res.summary}")
    bad = sum(not r.ok for r in results)
    print(f"\n{len(results) - bad} of {len(results)} file(s) passed." + (" Rejected files must not be merged." if bad
                                                                         else ""))
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
