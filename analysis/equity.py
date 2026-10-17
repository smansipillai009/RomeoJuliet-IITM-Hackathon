"""Who gets left waiting (ANALYSIS ONLY: 'servable' uses hidden truth to define who COULD be served).
Servable = arrived, no declined hard field, and at least one truly feasible partner. Groups use PUBLIC fields only."""
import copy, itertools, json, statistics, sys
import numpy as np
sys.path.insert(0, '.')
import harness, run_final_exp
from kit import Simulator, generate, eligibility, HARD

def run(seed, name, variant):
    ask, match = harness.POLICIES[name]; sim = Simulator(generate(seed, 200, 'evaluation', variant)); el = harness.Elig()
    for _ in range(60):
        sim.resolve_asks(ask(sim.observe(), el)); sim.advance(match(sim.observe(), el))
    first = {}
    for i in sim.introductions:
        for u in (i['user_a'], i['user_b']): first.setdefault(u, i['assigned_day'])
    mem = {i: m for i, m in sim.members.items() if m['arrived_day'] <= 59}
    nd = [i for i, m in mem.items() if not any(m['field_status'][k] == 'declined' for k in HARD)]
    tm = {}
    for i in nd:
        x = copy.deepcopy(mem[i]); x['fields'] = dict(mem[i]['truth']); tm[i] = x
    serv = set()
    for a, b in itertools.combinations(nd, 2):
        if eligibility(tm[a], tm[b])['status'] == 'feasible': serv.update((a, b))
    return [dict(g=mem[i]['gender'], z=mem[i]['zone'], age='<30' if mem[i]['age'] < 30 else '30+', served=i in first,
                 wait=(first[i] - mem[i]['arrived_day']) if i in first else None) for i in serv]

def gini(x):
    x = np.sort(np.array(x, float)); n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum())) if x.sum() > 0 else 0.0

if __name__ == '__main__':
    variants = sys.argv[1].split(','); seeds = range(7001, 7011); out = {}
    for v in variants:
        for name in ('base', 'final'):
            R = [r for s in seeds for r in run(s, name, v)]; w = [r['wait'] for r in R if r['served']]
            res = dict(n=len(R), never_served=sum(not r['served'] for r in R) / len(R), mean_wait=statistics.mean(w), median_wait=statistics.median(w),
                       p90_wait=float(np.percentile(w, 90)), wait_gini=gini(w), groups={})
            for key in ('g', 'age', 'z'):
                for c in sorted({r[key] for r in R}, key=str):
                    rs = [r for r in R if r[key] == c]; ws = [r['wait'] for r in rs if r['served']]
                    res['groups'][f'{key}={c}'] = dict(n=len(rs), never_served=sum(not r['served'] for r in rs) / len(rs), mean_wait=statistics.mean(ws) if ws else None)
            out[f'{v}/{name}'] = res
            print(f"{v:12s} {name:5s} servable n={res['n']} never-served {res['never_served']:.1%} wait mean {res['mean_wait']:.1f} median {res['median_wait']:.0f} p90 {res['p90_wait']:.0f} Gini {res['wait_gini']:.2f}", flush=True)
            small = [k for k, g in res['groups'].items() if g['n'] < 100]
            worst = max((g['mean_wait'], k) for k, g in res['groups'].items() if g['n'] >= 100 and g['mean_wait'] is not None)
            print(f"             longest mean wait among groups with n>=100: {worst[1]} {worst[0]:.1f}d | groups too small to read: {small}", flush=True)
        json.dump(out, open('results/equity.json', 'w'), indent=1)
