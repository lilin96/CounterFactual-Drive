#!/usr/bin/env python
"""Run MindDrive test with CF reranker replacing Decision Expert selection.

The script keeps MindDrive's Action Expert as the ego candidate generator. It
sets two environment switches before delegating to the original test entry:

  MINDDRIVE_CF_USE_ACTION_EXPERT_CANDIDATES=1
      Feed the 7 native Action Expert ego candidates into the CF head.

  MINDDRIVE_CF_REPLACE_DECISION=1
      Replace the output ``ego_fut_preds`` with the CF-selected candidate.

Usage mirrors ``adzoo/minddrive/test.py``.
"""

import os
import sys


REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    os.environ["MINDDRIVE_CF_USE_ACTION_EXPERT_CANDIDATES"] = "1"
    os.environ["MINDDRIVE_CF_REPLACE_DECISION"] = "1"

    from adzoo.minddrive.test import main as minddrive_test_main

    minddrive_test_main()


if __name__ == "__main__":
    main()
