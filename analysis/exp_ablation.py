"""Unified ablation on the FINAL policy (fresh seeds). Reference = shipped default. ANALYSIS ONLY (expected MSMI uses hidden truth)."""
import json, sys
sys.path.insert(0, '.')
from analysis.eval_expected import compare
FULL = {'FINAL (member, lean asks, exact matching)': ({}, False),
        'starter-equivalent (list-order asks, greedy agreement)': ({'asks': 'baseline', 'match': 'greedy_soft'}, False),
        'A1 no clarification': ({'asks': 'none'}, False),
        'A2 list-order asks': ({'asks': 'baseline'}, False),
        'A3 hard asks only (no soft asks)': ({'asks': 'hard'}, False),
        'A4 greedy instead of exact matching': ({'match': 'greedy_fit'}, False),
        'A5a score=uniform': ({'score': 'uniform'}, False),
        'A5b score=agree (simple count)': ({'score': 'agree'}, False),
        'A5c score=fit (agreement logistic)': ({'score': 'fit'}, False)}
SMALL = {k: v for k, v in FULL.items() if k.startswith(('FINAL', 'starter', 'A5b'))}
if __name__ == '__main__':
    variant, lo, hi, preset = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), sys.argv[4]
    cells = FULL if preset == 'full' else SMALL; seeds = list(range(lo, hi + 1))
    out, rows = compare(cells, seeds, variant, 200, 1.0)
    print(f'\n== {variant} n=200 ({len(seeds)} paired fresh seeds {lo}-{hi}); difference in expected MSMI/episode vs FINAL', flush=True)
    print(f"{'cell':56s} E[MSMI]  diff vs FINAL [95% CI]        E[mut] R[MSMI] cover  ask")
    for l in out:
        d = l.get('dE_ms'); dd = f"{d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]" if d else '(reference)'
        print(f"{l['cell']:56s} {l['E_ms']:.3f}  {dd:26s} {l['E_mut']:5.2f} {l['R_ms']:6.2f} {l['coverage']:.3f} {l['ask_cost']:5.0f}", flush=True)
    json.dump(out, open(f'results/ablation_{variant}.json', 'w'), indent=1, default=float)
