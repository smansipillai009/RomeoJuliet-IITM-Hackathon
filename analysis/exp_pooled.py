"""Pooled evidence for the default edge score across changing conditions (n=200). ANALYSIS ONLY."""
import json, sys
sys.path.insert(0, '.')
import harness
from analysis.eval_expected import compare
CELLS = {'starter': ({'asks': 'baseline', 'match': 'greedy_soft'}, False),      # list-order asks + greedy agreement = supplied baseline behaviour
         'simple': ({'asks': 'baseline', 'score': 'agree'}, False),                 # list-order asks + exact matching + agreement count
         'agree': ({'score': 'agree'}, False),                                      # lean asks + exact matching + agreement count
         'final': ({}, False)}                                                      # shipped default (member score, lean asks, exact matching)
if __name__ == '__main__':
    allrows = {k: [] for k in CELLS}
    for v in ['development', 'shift', 'drift', 'cold_start']:
        _, rows = compare(CELLS, list(range(7101, 7107)), v, 200, 1.0)
        for k in CELLS: allrows[k] += [x['E_ms'] for x in rows[k]]
        print(v, 'done', flush=True)
    n = len(allrows['starter']); print(f'\nPOOLED over {n} paired seed-variant episodes (6 fresh seeds x development/shift/drift/cold_start): expected MSMI/episode')
    for k in CELLS: print(f"  {k:8s} mean {sum(allrows[k])/n:.3f}")
    for a, b in (('final', 'starter'), ('simple', 'starter'), ('agree', 'starter'), ('final', 'simple'), ('final', 'agree')):
        d, lo, hi = harness.paired_ci(allrows[b], allrows[a]); print(f'  {a} - {b}: {d:+.4f} [{lo:+.4f},{hi:+.4f}]')
    json.dump(allrows, open('results/pooled_scores.json', 'w'))
