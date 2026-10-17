"""Paired-by-seed intervals for the equity comparison (development, 20 fresh seeds). ANALYSIS ONLY (servable uses hidden truth)."""
import json, statistics, sys
import numpy as np
sys.path.insert(0, '.')
import harness
from analysis.equity import run, gini
if __name__ == '__main__':
    seeds = range(7201, 7217); per = {'base': [], 'final': []}
    for s in seeds:
        for name in ('base', 'final'):
            R = run(s, name, 'development'); w = [r['wait'] for r in R if r['served']]
            per[name].append(dict(n=len(R), never=sum(not r['served'] for r in R), mean_wait=statistics.mean(w), gini=gini(w)))
        print(s, flush=True)
    tot = lambda k, f: sum(x[f] for x in per[k])
    print(f"\n16 paired fresh seeds, development. servable members: base {tot('base','n')} final {tot('final','n')}")
    print(f"never served: base {tot('base','never')} ({tot('base','never')/tot('base','n'):.1%}) final {tot('final','never')} ({tot('final','never')/tot('final','n'):.1%})")
    for lab, f in (('never-served count per episode', 'never'), ('mean wait (days)', 'mean_wait'), ('wait Gini', 'gini')):
        d, lo, hi = harness.paired_ci([x[f] for x in per['base']], [x[f] for x in per['final']])
        print(f'  {lab}: final - base = {d:+.3f} [{lo:+.3f},{hi:+.3f}]')
    json.dump(per, open('results/equity_ci.json', 'w'))
