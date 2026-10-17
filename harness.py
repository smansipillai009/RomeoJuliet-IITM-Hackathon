"""Fast in-process experiment harness (development tool only, trusted code).

Mirrors evaluate.episode(): 60 decision days of ask -> observe -> match -> advance,
then 40 follow-up days. Policies only see sim.observe() output, exactly as the
subprocess protocol would. Differences from evaluate.py: no process spawn / JSON
round trip, and a pairwise eligibility cache (valid because a member's `fields`
only ever change from unknown to known, so the count of known fields is a
complete version stamp for a member's constraints).
"""
import argparse, itertools, json, random, statistics, sys, time
from kit import Simulator, generate, eligibility, baseline_asks, HARD, SOFT


class Elig:
    """Per-episode memo for eligibility(a, b)."""
    def __init__(self):
        self.memo = {}
        self.calls = 0

    @staticmethod
    def sig(m):
        return sum(v is not None for v in m['fields'].values())

    def status(self, a, b):
        ia, ib = a['member_id'], b['member_id']
        key = (ia, ib, self.sig(a), self.sig(b)) if ia < ib else (ib, ia, self.sig(b), self.sig(a))
        s = self.memo.get(key)
        if s is None:
            s = eligibility(a, b)['status']
            self.memo[key] = s
            self.calls += 1
        return s


def past_pairs(state):
    return {tuple(sorted((i['user_a'], i['user_b']))) for i in state['introductions']}


def declined_hard(m):
    return any(m['field_status'][k] == 'declined' for k in HARD)


def missing_hard(m):
    return any(m['fields'][k] is None for k in HARD)


# ---------------------------------------------------------------- matching
def match_greedy_soft(state, el):
    """Identical output to kit.baseline_match, using the eligibility cache."""
    members = [m for m in state['members'] if m['available']]
    past = past_pairs(state)
    edges = []
    for a, b in itertools.combinations(members, 2):
        key = tuple(sorted((a['member_id'], b['member_id'])))
        if key in past or el.status(a, b) != 'feasible':
            continue
        score = sum(a['fields'].get(k) is not None and a['fields'].get(k) == b['fields'].get(k) for k in SOFT)
        edges.append((-score, key))
    used, out = set(), []
    for _, pair in sorted(edges):
        if not used.intersection(pair):
            out.append(list(pair))
            used.update(pair)
    return out


# ---------------------------------------------------------------- ask policies
def ask_baseline(state, el):
    return baseline_asks(state)


def voi_ranked(state, el):
    """Askable members ranked by expected edges unlocked.

    value(m) = sum over available partners p (no declined hard field, pair not
    already used, pair not already infeasible given KNOWN fields) of 1.0 if p's
    hard fields are complete else 0.5 (p still needs its own ask).
    Members with a declined hard field are never asked (a bundle returns nothing).
    """
    avail = [m for m in state['members'] if m['available']]
    past = past_pairs(state)
    cands = [m for m in avail if missing_hard(m) and not declined_hard(m)]
    pool = [m for m in avail if not declined_hard(m)]
    scored = []
    for m in cands:
        v = 0.0
        for p in pool:
            if p['member_id'] == m['member_id']:
                continue
            if tuple(sorted((m['member_id'], p['member_id']))) in past:
                continue
            if el.status(m, p) == 'infeasible':
                continue
            v += 0.5 if missing_hard(p) else 1.0
        scored.append((-v, m['arrived_day'], m['member_id']))
    scored.sort()
    return [s[2] for s in scored]


def make_ask_voi(soft_reserve=0, soft_fields=None):
    """VOI hard asks; soft asks use `soft_reserve` units up front plus any spare
    units left after hard asks. soft_fields=None means every SOFT field in the
    contract's order (deliberately agnostic: no field is assumed informative)."""
    fields = list(SOFT if soft_fields is None else soft_fields)

    def ask(state, el):
        budget = state['ask_budget_remaining']
        by_id = {m['member_id']: m for m in state['members']}
        ranked = voi_ranked(state, el)
        n_hard = min(len(ranked), max(0, (budget - soft_reserve)) // 3)
        asks = [{'member_id': i, 'field': 'constraints'} for i in ranked[:n_hard]]
        left = budget - 3 * n_hard
        if left > 0:
            # soft asks for available members already in (or one ask from) a feasible edge
            avail = [m for m in state['members'] if m['available']]
            deg = {}
            for a, b in itertools.combinations([m for m in avail if not missing_hard(m)], 2):
                if el.status(a, b) == 'feasible':
                    for m in (a, b):
                        deg[m['member_id']] = deg.get(m['member_id'], 0) + 1
            for mid, _ in sorted(deg.items(), key=lambda x: (-x[1], x[0])):
                for f in fields:
                    if left > 0 and by_id[mid]['field_status'][f] == 'not_asked':
                        asks.append({'member_id': mid, 'field': f})
                        left -= 1
        return asks

    return ask


POLICIES = {
    'base':          (ask_baseline, match_greedy_soft),
    'voi':           (make_ask_voi(0, soft_fields=[]), match_greedy_soft),   # hard asks only, VOI order
    'voi_spare':     (make_ask_voi(0), match_greedy_soft),                    # + soft asks from spare units
    'voi_reserve4':  (make_ask_voi(4), match_greedy_soft),                    # + 4 units/day reserved for soft
}


# ---------------------------------------------------------------- episode
def run_episode(seed, variant, policy):
    ask_fn, match_fn = POLICIES[policy]
    sim = Simulator(generate(seed, 200, 'evaluation', variant))
    el = Elig()
    t0 = time.perf_counter()
    for _ in range(60):
        sim.resolve_asks(ask_fn(sim.observe(), el))
        sim.advance(match_fn(sim.observe(), el))
    arrived = {m['member_id']: m['arrived_day'] for m in sim.observe()['members'] if m['arrived_day'] <= 59}
    for _ in range(40):
        sim.advance([])
    r = sim.metrics()
    first = {}
    for intro in sim.introductions:
        for i in (intro['user_a'], intro['user_b']):
            first.setdefault(i, intro['assigned_day'])
    return dict(seed=seed, variant=variant, policy=policy,
                assignments=r['assignments'], mutual=r['mutual_acceptances'], dates=r['dates'],
                msmi=r['mutual_second_meeting_intention'], ask_cost=r['ask_cost'],
                missing=r['missing_feedback'], arrived=len(arrived),
                coverage=len(first) / max(1, len(arrived)), served=len(first),
                seconds=time.perf_counter() - t0)


# ---------------------------------------------------------------- stats
def paired_ci(a, b, reps=4000, rng=None):
    """Mean of (b - a) with a paired bootstrap 95% interval."""
    rng = rng or random.Random(0)
    d = [y - x for x, y in zip(a, b)]
    n = len(d)
    means = sorted(statistics.mean(rng.choice(d) for _ in range(n)) for _ in range(reps))
    return statistics.mean(d), means[int(.025 * reps)], means[int(.975 * reps)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--policies', default='base,voi,voi_spare,voi_reserve4')
    p.add_argument('--variants', default='development')
    p.add_argument('--seeds', default='1001-1020', help='a-b range or comma list')
    p.add_argument('--output', default='results/harness.json')
    a = p.parse_args()
    if '-' in a.seeds:
        lo, hi = map(int, a.seeds.split('-'))
        seeds = list(range(lo, hi + 1))
    else:
        seeds = [int(s) for s in a.seeds.split(',')]
    pols = a.policies.split(',')
    variants = a.variants.split(',')
    rows = []
    for v in variants:
        for s in seeds:
            for pol in pols:
                rows.append(run_episode(s, v, pol))
            print(v, s, 'done', flush=True)
    json.dump(rows, open(a.output, 'w'), indent=1)
    # report
    metrics = ['assignments', 'mutual', 'dates', 'msmi', 'ask_cost', 'coverage']
    for v in variants:
        print(f'\n== {v}  ({len(seeds)} paired seeds)')
        print('policy         ' + ''.join(f'{m:>13s}' for m in metrics))
        for pol in pols:
            rs = [r for r in rows if r['variant'] == v and r['policy'] == pol]
            print(f'{pol:14s} ' + ''.join(f'{statistics.mean(r[m] for r in rs):13.3f}' for m in metrics))
        base = pols[0]
        for pol in pols[1:]:
            ra = sorted([r for r in rows if r['variant'] == v and r['policy'] == base], key=lambda r: r['seed'])
            rb = sorted([r for r in rows if r['variant'] == v and r['policy'] == pol], key=lambda r: r['seed'])
            out = []
            for m in ['mutual', 'dates', 'msmi', 'coverage', 'ask_cost']:
                d, lo, hi = paired_ci([r[m] for r in ra], [r[m] for r in rb])
                out.append(f'{m} {d:+.3f} [{lo:+.3f},{hi:+.3f}]')
            print(f'  {pol} - {base}: ' + '; '.join(out))


if __name__ == '__main__':
    main()
