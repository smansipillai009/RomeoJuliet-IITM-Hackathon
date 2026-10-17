"""ANALYSIS ONLY: evaluate a policy config in-process and report REALIZED and EXPECTED (true-probability) outcomes.
Expected values remove outcome luck, so differences between policies are measurable with far fewer episodes.
Hidden truth is used only to compute the expectation after decisions are made; it never enters any decision."""
import itertools, statistics, sys
sys.path.insert(0, '.')
import harness, policy
from analysis.diagnostics import true_probs
from analysis.scaled_sim import make_sim

def oracle_match(state, el, sim):
    """UPPER BOUND ONLY: exact max-weight matching on TRUE P(MSMI). Uses hidden truth, so it is not a policy."""
    day = state['day']
    edges = [(true_probs(sim, k[0], k[1], day)['p_msmi'], k) for _, _, k in policy.feasible_pairs(state, lambda a, b: {'status': el.status(a, b)})]
    return [sorted(p) for p in policy.max_weight_matching(edges)] if edges else []

def episode(seed, cfg, variant='development', n=200, signal=1.0, oracle=False):
    sim = make_sim(seed, variant, n, signal); el = harness.Elig(); adapt = lambda a, b: {'status': el.status(a, b)}
    for _ in range(60):
        st = sim.observe()
        sim.resolve_asks(policy.decide({'phase': 'ask', 'state': st, 'memory': None}, cfg, adapt)['asks']); st = sim.observe()
        pairs = oracle_match(st, el, sim) if oracle else policy.decide({'phase': 'match', 'state': st, 'memory': None}, cfg, adapt)['pairs']
        sim.advance(pairs)
    for _ in range(40): sim.advance([])
    by = {}
    for e in sim.events: by.setdefault(e['introduction_id'], []).append(e)
    Em = Es = Rm = 0.0
    for i in sim.introductions:
        t = true_probs(sim, i['user_a'], i['user_b'], i['assigned_day']); Em += t['p_mutual']; Es += t['p_msmi']
        rs = [e for e in by.get(i['introduction_id'], []) if e['event'] == 'introduction_response']
        Rm += len(rs) == 2 and all(e['value'] == 'yes' for e in rs)
    m = sim.metrics(); arrived = max(1, m['arrived_members'])
    served = len({u for i in sim.introductions for u in (i['user_a'], i['user_b'])})
    return dict(seed=seed, ask_cost=m['ask_cost'], coverage=served / arrived, n_intro=len(sim.introductions), R_ms=m['mutual_second_meeting_intention'], E_ms=Es, R_mut=Rm, E_mut=Em,
               per100_E_ms=100 * Es / arrived, per100_R_ms=100 * m['mutual_second_meeting_intention'] / arrived)

def compare(cells, seeds, variant, n, signal):
    """cells: {name: (cfg, oracle_flag)} -> paired table versus first cell."""
    rows = {k: [episode(s, c, variant, n, signal, o) for s in seeds] for k, (c, o) in cells.items()}
    names = list(cells); base = names[0]; out = []
    for k in names:
        r = rows[k]
        line = {'cell': k, 'E_ms': statistics.mean(x['E_ms'] for x in r), 'R_ms': statistics.mean(x['R_ms'] for x in r),
                'E_mut': statistics.mean(x['E_mut'] for x in r), 'R_mut': statistics.mean(x['R_mut'] for x in r),
                'per100_E_ms': statistics.mean(x['per100_E_ms'] for x in r),
                'ask_cost': statistics.mean(x['ask_cost'] for x in r), 'coverage': statistics.mean(x['coverage'] for x in r)}
        if k != base:
            d, lo, hi = harness.paired_ci([x['E_ms'] for x in rows[base]], [x['E_ms'] for x in r]); line['dE_ms'] = (d, lo, hi)
        out.append(line)
    return out, rows
