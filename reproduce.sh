#!/bin/sh
# Reproduces the Round 2 evidence from the kit root. Single CPU core; wall-clock times are approximate.
# Policy/container: stdlib only (Dockerfile unchanged). Analysis: pip install -r requirements-analysis.txt
set -e
python3 -m unittest -v                                   # 22 kit tests + policy tests + hidden-state isolation tests (~25 s)
python3 verify_data.py                                   # public data integrity
python3 check_matching.py                                # stdlib exact matcher vs networkx on real daily states (~1 min)
# Official evaluator, supplied baselines (starter policy preserved in baselines.py) and the submitted policy
for b in greedy no_asks random; do python3 evaluate.py --policy baselines.py --baseline $b --seeds 101,102,103 --variants all --output results/$b.json; done
python3 evaluate.py --policy policy.py --seeds 101,102,103 --variants all --output results/final_official.json
# Edge-usage diagnostic: official assignments, in-process replay for edge counts (see research note, Section 1)
python3 evaluate.py --policy baselines.py --baseline greedy --seeds 101,102,103,104,105,106 --variants development --output results/edge_check_official_greedy.json
# Training data and models (development variant; seeds disjoint from every evaluation seed)
python3 training/collect_funnel.py 5001 5120             # ~10 min
python3 training/fit_funnel.py results/funnel_rows_5001_5120.json
# Question 1: timing vs selection vs model error (fresh cold_start seeds)
python3 analysis/exp_decompose.py cold_start 4001 4040   # ~12 min
# Question 2: learned vs simple scores on fresh populations (signal x0/x1/x2, pool size 100/200/300)
python3 analysis/exp_grid.py                             # ~25 min
# Round 2 additions: default-score comparison, calibration under shift/drift, equity, unified ablation, pooled, sparse/delayed
python3 analysis/exp_default_score.py development,shift,drift,cold_start   # ~20 min
python3 training/check_calibration.py shift 8001 8025                        # ~3 min
python3 training/check_calibration.py drift 8001 8025
python3 analysis/equity.py development,cold_start,shift                     # ~8 min
python3 analysis/equity_ci.py                                               # ~5 min
python3 analysis/exp_ablation.py development 7301 7310 full                 # ~11 min
python3 analysis/exp_pooled.py                                              # ~10 min
python3 analysis/exp_ablation.py sparse 7401 7410 small                     # ~4 min
python3 analysis/exp_ablation.py delayed 7501 7510 small                    # ~4 min
