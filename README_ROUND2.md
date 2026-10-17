# Round 2 workspace: file layout

Kit files (kit.py, evaluate.py, verify_data.py, data/, docs/) are the organisers'. Everything below is ours.

| Path | Purpose | Used by the submitted container? |
|---|---|---|
| `policy.py` | The submitted policy (standard library + `kit.py` only) | yes (with `kit.py`; `Dockerfile` unchanged) |
| `baselines.py` | The starter baselines, preserved. Run `evaluate.py --policy baselines.py --baseline greedy` | no |
| `test_policy.py`, `tests_hidden_isolation.py`, `tests_hard_constraints.py` | Policy tests, hidden-value isolation tests, after-the-fact hard-constraint audit | no |
| `harness.py`, `check_matching.py`, `run_final_exp.py`, `fit_exp.py` | In-process harness (reproduces the official evaluator exactly) and Round 1 experiments | no |
| `training/` | Rollout collection and model fitting (numpy, scikit-learn) | no |
| `analysis/` | ANALYSIS ONLY. Reads hidden simulator state after decisions are made. No policy may import it (tests enforce this) | no |
| `results/` | Machine-readable results and logs for every number in the technical report | no |
| `reproduce.sh`, `requirements-analysis.txt` | Re-run everything; pinned analysis dependencies | no |

Licence: the kit's own terms govern the kit files. Choose a licence for our code before making the repository public (for example MIT or Apache-2.0) and add a LICENSE file; this is the team's decision.


## Setup and evaluation commands (from the project root, after merging the kit files)

Requirements for the policy and container: Python 3.12 (container image `python:3.12.7-slim`), standard library only. Analysis and training tooling is optional and pinned in `requirements-analysis.txt` (numpy 2.4.4, scipy 1.17.1, scikit-learn 1.8.0, networkx 3.6.1).

```bash
python -m unittest -v
python verify_data.py
python evaluate.py --seeds 101,102,103 --variants all --output results/final.json
python evaluate.py --baseline greedy --seeds 101,102,103 --variants all --output results/greedy.json
python evaluate.py --baseline no_asks --seeds 101,102,103 --variants all --output results/no_asks.json
python evaluate.py --baseline random --seeds 101,102,103 --variants all --output results/random.json
docker build -t sequential-policy:submission .
python evaluate.py --image sequential-policy:submission --seeds 101 --output results/container_check.json
```

To reproduce every experiment in the report, run `bash reproduce.sh` (single CPU core; the times in its comments are approximate, and a full run takes a few hours).

## Seeds and randomness

- Training seeds: 3001–3030 (agreement model), 3001–3180 and 5001–5120 (rollout collection for the member model).
- Evaluation and comparison seeds are listed in Section 9 of `technical_report_round2.md`; training seeds are disjoint from all of them.
- Inference random seed: none. The submitted policy contains no random number generator and is deterministic given the request.

## Licence and provenance

Our code is released under the licence in `LICENSE`. The kit's own licence (`KIT_LICENSE`) and `DATA_LICENSE.md` continue to govern the organisers' files and the supplied synthetic records. No external datasets or pretrained models are used. See Section 9.2 of the report for dependency licences.
