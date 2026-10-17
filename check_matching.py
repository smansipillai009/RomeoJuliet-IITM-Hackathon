import time, itertools, statistics
import networkx as nx
import harness, policy
from kit import Simulator, generate

class E:  # memoised eligibility adapter returning the contract's dict shape
    def __init__(self): self.e = harness.Elig()
    def __call__(self, a, b): return {'status': self.e.status(a, b)}

worst_gap = 0.0; n_states = 0; max_edges = 0; max_comp = 0; t_max = 0.0; mismatch = 0
for seed in (2001, 2002, 2003, 2004):
    for variant in ('development', 'sparse'):
        sim = Simulator(generate(seed, 200, 'evaluation', variant)); el = E()
        for day in range(60):
            st = sim.observe()
            sim.resolve_asks(policy.decide({'phase': 'ask', 'state': st, 'memory': None}, elig=el)['asks'])
            st = sim.observe()
            cand = policy.feasible_pairs(st, el)
            edges = [(policy.pair_probability(a, b), k) for a, b, k in cand]
            if edges:
                t = time.perf_counter(); mine = policy.max_weight_matching(edges); t_max = max(t_max, time.perf_counter() - t)
                G = nx.Graph(); [G.add_edge(a, b, weight=w) for w, (a, b) in edges]
                ref = nx.max_weight_matching(G)
                wmap = {tuple(sorted(k)): w for w, k in edges}
                wm = sum(wmap[tuple(sorted(p))] for p in mine); wr = sum(wmap[tuple(sorted(p))] for p in ref)
                worst_gap = max(worst_gap, wr - wm); n_states += 1; max_edges = max(max_edges, len(edges))
                comps = list(nx.connected_components(G)); max_comp = max(max_comp, max(len(c) for c in comps))
                # validity of my pairs
                used = [x for p in mine for x in p]; assert len(used) == len(set(used))
            sim.advance(policy.decide({'phase': 'match', 'state': st, 'memory': None}, elig=el)['pairs'])
print(f'states with edges={n_states}  max edges={max_edges}  max component size={max_comp}  worst (networkx - mine) weight gap={worst_gap:.2e}  slowest matching call={t_max*1000:.1f} ms')
