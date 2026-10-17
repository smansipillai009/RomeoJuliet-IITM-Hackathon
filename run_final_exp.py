"""Held-out comparison of the SUBMITTED policy.decide (via the harness) against the starter baseline, paired by seed."""
import json, os, statistics, sys
import harness, policy

def adapter(el):
    return lambda a, b: {'status': el.status(a, b)}

def register(name, cfg):
    def ask(state, el):   return policy.decide({'phase': 'ask', 'state': state, 'memory': None}, cfg, adapter(el))['asks']
    def match(state, el): return policy.decide({'phase': 'match', 'state': state, 'memory': None}, cfg, adapter(el))['pairs']
    harness.POLICIES[name] = (ask, match)

CFG = {'final': {}, 'final_hard': {'asks': 'hard'}, 'final_gfit': {'match': 'greedy_fit'},
       'final_gsoft': {'match': 'greedy_soft'}, 'final_none': {'asks': 'none'}}
for n, c in CFG.items(): register(n, c)

if __name__ == '__main__':
    variant, pols = sys.argv[1], sys.argv[2].split(',')
    seeds = list(range(2001, 2026)); rows = []
    cache = f'results/fit_exp_{variant}.json'
    base_rows = [r for r in json.load(open(cache)) if r['policy'] == 'base'] if os.path.exists(cache) else None
    if base_rows:
        rows += base_rows                                    # identical deterministic episodes, reuse
    else:
        for s in seeds: rows.append(harness.run_episode(s, variant, 'base'))
    for p in pols:
        for s in seeds: rows.append(harness.run_episode(s, variant, p))
        print(variant, p, 'done', flush=True)
    json.dump(rows, open(f'results/final_{variant}_{"_".join(pols)}.json', 'w'))
    print(f'\n== {variant} ({len(seeds)} paired held-out seeds)')
    for p in ['base'] + pols:
        rs = [r for r in rows if r['policy'] == p]
        print(f'{p:12s}', ' '.join(f"{m}={statistics.mean(r[m] for r in rs):.3f}" for m in ['assignments', 'mutual', 'dates', 'msmi', 'ask_cost', 'coverage']),
              f"max_s/episode={max(r['seconds'] for r in rs):.1f}")
    ra = sorted([r for r in rows if r['policy'] == 'base'], key=lambda r: r['seed'])
    for p in pols:
        rb = sorted([r for r in rows if r['policy'] == p], key=lambda r: r['seed'])
        print(f'  {p} - base:', '; '.join('%s %+.3f [%+.3f,%+.3f]' % ((m,) + harness.paired_ci([r[m] for r in ra], [r[m] for r in rb])) for m in ['mutual', 'dates', 'msmi', 'coverage', 'ask_cost']), flush=True)
