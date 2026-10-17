"""Fresh-seed replication: base vs submitted policy on one variant (seeds disjoint from everything used so far)."""
import json, statistics, sys
import harness, run_final_exp  # registers 'final' policies
variant, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); rows = []
for s in range(lo, hi + 1):
    for p in ('base', 'final'): rows.append(harness.run_episode(s, variant, p))
    if s % 10 == 0: print(variant, s, flush=True)
json.dump(rows, open(f'results/rep_{variant}_{lo}_{hi}.json', 'w'))
ra = sorted([r for r in rows if r['policy'] == 'base'], key=lambda r: r['seed']); rb = sorted([r for r in rows if r['policy'] == 'final'], key=lambda r: r['seed'])
print(f'== {variant} {len(ra)} paired fresh seeds')
for m in ['mutual', 'dates', 'msmi', 'coverage']:
    print(f"  {m}: base {statistics.mean(r[m] for r in ra):.3f} final {statistics.mean(r[m] for r in rb):.3f} diff %+.3f [%+.3f,%+.3f]" % harness.paired_ci([r[m] for r in ra], [r[m] for r in rb]))
