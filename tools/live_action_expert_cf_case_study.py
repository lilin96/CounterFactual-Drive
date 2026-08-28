#!/usr/bin/env python
"""Live CF case study using MindDrive Action Expert ego candidates.

This is a thin wrapper around ``live_counterfactual_case_study.py``. It sets an
environment switch consumed by ``Minddrive.simple_test_pts`` so the
counterfactual head receives the ego candidate trajectories decoded by
MindDrive's Action Expert instead of the counterfactual module's rule-based
fallback generator.

Candidate semantics follow MindDrive's native meta-action heads:
  speed candidate 0: <maintain_moderate_speed>
  speed candidate 1: <stop>
  speed candidate 2: <maintain_slow_speed>
  speed candidate 3: <speed_up>
  speed candidate 4: <slow_down>
  speed candidate 5: <maintain_fast_speed>
  speed candidate 6: <slow_down_rapidly>

The path action remains the path selected by MindDrive's Decision Expert for the
current frame; MindDrive decodes path candidates through ``pw_ego_fut_decoder``
for path supervision/diagnostics, while the ego future used by the planner is
selected from the 7-way speed-conditioned trajectory decoder.
"""

import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def main():
    os.environ["MINDDRIVE_CF_USE_ACTION_EXPERT_CANDIDATES"] = "1"

    if "--candidate-indices" not in sys.argv:
        sys.argv.extend(["--candidate-indices", "0,1,2,3,4,5,6"])

    from tools.live_counterfactual_case_study import main as live_main

    live_main()


if __name__ == "__main__":
    main()
