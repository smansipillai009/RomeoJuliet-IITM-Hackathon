"""Experiment 2 (ANALYSIS ONLY): with asking and allocation fixed, when do learned funnel scores beat simple scores
on FRESH populations with different preference signal (x0, x1, x2) and density (pool size)? Weights were trained at
signal 1, n=200 only. Metric: expected MSMI (true probabilities, luck removed) and realized, paired by seed."""
import json, sys
sys.path.insert(0, '.')
from analysis.eval_expected import compare

CELLS = [(200, 0.0), (200, 2.0), (200, 1.0), (100, 1.0), (300, 1.0)]
SEEDS = list(range(6001, 6009))
SCORES = {'uniform': ({'score': 'uniform'}, False), 'agree': ({'score': 'agree'}, False), 'fit': ({'score': 'fit'}, False),
          'member': ({'score': 'member'}, False), 'ORACLE(true P)': ({}, True)}
if __name__ == '__main__':
    which = [int(x) for x in sys.argv[1].split(',')] if len(sys.argv) > 1 else range(len(CELLS)); allres = {}
    for ci in which:
        n, sig = CELLS[ci]
        out, rows = compare(SCORES, SEEDS, 'development', n, sig)
        allres[f'n{n}_signal{sig}'] = out
        print(f'\n== pool n={n}, signal x{sig}  ({len(SEEDS)} paired fresh seeds)', flush=True)
        print(f"{'score':16s} E[MSMI]/ep  vs uniform [95% CI]         E[mutual] R[MSMI]  E[MSMI]/100 members")
        for l in out:
            d = l.get('dE_ms'); dd = f"{d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]" if d else '      (reference)       '
            print(f"{l['cell']:16s} {l['E_ms']:9.3f}   {dd:28s} {l['E_mut']:8.2f} {l['R_ms']:7.2f}  {l['per100_E_ms']:.3f}", flush=True)
        json.dump(allres, open('results/grid_results.json', 'w'), indent=1, default=float)
