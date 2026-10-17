"""Scarce-supply allocation policy for the Sequential Matching Problem (stdlib only, offline, CPU).

Interface: docs/POLICY_INTERFACE.md (one JSON request on stdin, one JSON response on stdout).
Starter baselines are preserved in baselines.py.

Method (see research_note.md):
  ask phase   hard-constraint bundles ranked by expected feasible pairs unlocked (never for members with a
              declined hard field); spare budget buys the few soft fields the learned fit model weights.
  match phase feasible pairs scored by a learned P(mutual acceptance) and chosen by exact max-weight matching
              (component-wise search with a deterministic state cap and a greedy fallback).
  safety      every action is re-validated against the public contract before it is returned; any failure
              degrades to "wait" (empty list) instead of an invalid episode.
Only observable request fields are used. Nothing here reads simulator internals, seeds or hidden truth.
"""
import argparse
import itertools
import json
import math
import os
import sys

from kit import HARD, SOFT, eligibility

# --- learned fit model: logistic regression on soft-field agreement (+1 equal, -1 differ, 0 unknown) ---------
# Trained on own rollouts: development variant, training seeds 3001-3030, random-feasible exploration matching,
# 2,697 introductions / 398 mutual acceptances, grouped-by-seed CV AUC 0.647. Not copied from simulator source.
FIT_WEIGHTS = {'relationship_goal': 0.674, 'relationship_pace': 0.191, 'lifestyle': 0.235, 'conversations': 0.072,
               'emotional_availability': 0.017, 'space_for_relationship': -0.003, 'relocate': -0.097}
FIT_INTERCEPT = -1.7795819461565971
LEAN_MIN_ABS_WEIGHT = 0.05                       # only ask soft fields the model actually uses
LEAN_FIELDS = [k for k in sorted(SOFT, key=lambda k: (-abs(FIT_WEIGHTS[k]), k)) if abs(FIT_WEIGHTS[k]) >= LEAN_MIN_ABS_WEIGHT]
STATE_CAP = 200000                               # deterministic bound on exact-matching search states

# member-history pair model (training/fit_funnel.py, 120 development episodes, seeds 5001-5120): logistic on
# [7 soft agreements, logit(resp-rate_a)+logit(resp-rate_b), logit(yes-rate_a)+logit(yes-rate_b)] with shrunk rates.
MEMBER_MODEL = {'coef': [0.5449, 0.2878, 0.143, 0.1106, -0.0073, -0.0425, 0.0064, 0.0544, 0.2651], 'b0': -1.8686, 'rho0': 0.7649, 'eta0': 0.4678, 'alpha': 1, 'beta': 4}

DEFAULT_CONFIG = {'asks': 'lean', 'match': 'mw', 'score': 'member'}
# score: member (default; soft agreement + shrunk member history) | fit (soft-agreement logistic) | agree (starter-style count) | uniform
# Chosen from paired expected-MSMI comparisons: no score is reliably worse at n=200; member is best in denser pools (README of results).
# asks : lean (default) | hard (hard bundles only) | none | baseline (starter order)
# match: mw (default) | greedy_fit | greedy_soft (starter score)


def parse_config(text):
    cfg = dict(DEFAULT_CONFIG)
    for part in (text or '').split(','):
        if '=' in part:
            k, v = part.split('=', 1)
            if k.strip() in cfg:
                cfg[k.strip()] = v.strip()
    return cfg


def _declined_hard(m):
    return any(m['field_status'][k] == 'declined' for k in HARD)


def _missing_hard(m):
    return any(m['fields'][k] is None for k in HARD)


def _past_pairs(state):
    return {tuple(sorted((i['user_a'], i['user_b']))) for i in state.get('introductions', [])}


def _status(elig, a, b):
    return elig(a, b)['status']


# ------------------------------------------------------------------------------------------- fit model
def pair_features(a, b):
    out = []
    for k in SOFT:
        x, y = a['fields'].get(k), b['fields'].get(k)
        out.append(0.0 if x is None or y is None else (1.0 if x == y else -1.0))
    return out


def pair_probability(a, b):
    z = FIT_INTERCEPT + sum(FIT_WEIGHTS[k] * f for k, f in zip(SOFT, pair_features(a, b)))
    return 1.0 / (1.0 + math.exp(-z))


def _shrunk_logits(row):
    asked, responded, yes = row[0], row[1], row[2]
    mm = MEMBER_MODEL
    rho = (responded + mm['alpha'] * mm['rho0']) / (asked + mm['alpha'])
    eta = (yes + mm['beta'] * mm['eta0']) / (responded + mm['beta'])
    return math.log(rho / (1 - rho)), math.log(eta / (1 - eta))


def member_probability(a, b, hist):
    la, ea = _shrunk_logits(hist.get(a['member_id'], [0] * 5))
    lb, eb = _shrunk_logits(hist.get(b['member_id'], [0] * 5))
    x = pair_features(a, b) + [la + lb, ea + eb]
    z = MEMBER_MODEL['b0'] + sum(c * v for c, v in zip(MEMBER_MODEL['coef'], x))
    return 1.0 / (1.0 + math.exp(-z))


def soft_agreement_score(a, b):
    """Starter baseline's score (used for the greedy_soft ablation)."""
    return sum(a['fields'].get(k) is not None and a['fields'].get(k) == b['fields'].get(k) for k in SOFT)


# ------------------------------------------------------------------------------------------- member history (observable)
def member_history(state):
    """Per-member counts from feedback already visible in the request (no hidden values, no policy memory needed):
    {member_id: [asked, responded, yes, second_asked, second_yes]}."""
    h = {}
    for e in state.get('feedback', []):
        m = e.get('member_id')
        if m is None:
            continue
        row = h.setdefault(m, [0, 0, 0, 0, 0])
        if e['event'] == 'introduction_response':
            row[0] += 1
            if e.get('value') is not None:
                row[1] += 1
                row[2] += e['value'] == 'yes'
        elif e['event'] == 'second_meeting_intention':
            row[3] += 1
            row[4] += e.get('value') == 'yes'
    return h


# ------------------------------------------------------------------------------------------- feasible graph
def feasible_pairs(state, elig):
    members = [m for m in state['members'] if m['available']]
    past = _past_pairs(state)
    out = []
    for a, b in itertools.combinations(members, 2):
        key = tuple(sorted((a['member_id'], b['member_id'])))
        if key in past or _status(elig, a, b) != 'feasible':
            continue
        out.append((a, b, key))
    return out


# ------------------------------------------------------------------------------------------- ask phase
def _voi_ranking(state, elig):
    """Askable members (missing hard data, none declined), best expected unlock first.
    value = sum over available partners (no declined hard field, not yet introduced, not already infeasible
    from KNOWN fields) of 1.0 if the partner's hard fields are complete else 0.5."""
    avail = [m for m in state['members'] if m['available']]
    past = _past_pairs(state)
    pool = [m for m in avail if not _declined_hard(m)]
    scored = []
    for m in avail:
        if not _missing_hard(m) or _declined_hard(m):
            continue
        v = 0.0
        for p in pool:
            if p['member_id'] == m['member_id']:
                continue
            if tuple(sorted((m['member_id'], p['member_id']))) in past or _status(elig, m, p) == 'infeasible':
                continue
            v += 0.5 if _missing_hard(p) else 1.0
        scored.append((-v, m['arrived_day'], m['member_id']))
    scored.sort()
    return [s[2] for s in scored]


def plan_asks(state, cfg, elig):
    budget = state['ask_budget_remaining']
    mode = cfg['asks']
    if mode == 'none' or budget <= 0 or not state['members']:
        return []
    if mode == 'baseline':
        ids = [m['member_id'] for m in state['members'] if m['available'] and _missing_hard(m) and not _declined_hard(m)]
        return [{'member_id': i, 'field': 'constraints'} for i in ids[:budget // 3]]
    ranked = _voi_ranking(state, elig)
    n_hard = min(len(ranked), budget // 3)
    asks = [{'member_id': i, 'field': 'constraints'} for i in ranked[:n_hard]]
    left = budget - 3 * n_hard
    if mode == 'lean' and left > 0:
        by_id = {m['member_id']: m for m in state['members']}
        deg = {}
        ready = [m for m in state['members'] if m['available'] and not _missing_hard(m)]
        for a, b in itertools.combinations(ready, 2):
            if _status(elig, a, b) == 'feasible' and tuple(sorted((a['member_id'], b['member_id']))) not in _past_pairs(state):
                for m in (a, b):
                    deg[m['member_id']] = deg.get(m['member_id'], 0) + 1
        for mid, _ in sorted(deg.items(), key=lambda x: (-x[1], x[0])):
            for f in LEAN_FIELDS:
                if left > 0 and by_id[mid]['field_status'][f] == 'not_asked':
                    asks.append({'member_id': mid, 'field': f})
                    left -= 1
    return asks


# ------------------------------------------------------------------------------------------- allocation
def _greedy(edges):
    used, out = set(), []
    for w, (a, b) in sorted(edges, key=lambda e: (-e[0], e[1])):
        if a not in used and b not in used:
            out.append((a, b))
            used.update((a, b))
    return out


def _solve_component(nodes, adj):
    """Exact max-weight matching on one connected component by memoised branching on the lowest-degree node.
    Raises OverflowError if the state cap is exceeded (caller falls back to greedy for that component)."""
    memo = {}

    def best(rem):
        rem = frozenset(n for n in rem if any(u in rem for u in adj[n]))
        if not rem:
            return 0.0, None
        hit = memo.get(rem)
        if hit is not None:
            return hit
        if len(memo) >= STATE_CAP:
            raise OverflowError
        v = min(rem, key=lambda n: (sum(1 for u in adj[n] if u in rem), n))
        top = (best(rem - {v})[0], None)
        for u in sorted(adj[v]):
            if u in rem:
                val = adj[v][u] + best(rem - {v, u})[0]
                if val > top[0] + 1e-12:
                    top = (val, u)
        memo[rem] = (top[0], (v, top[1]))
        return memo[rem]

    pairs, rem = [], frozenset(nodes)
    while True:
        rem = frozenset(n for n in rem if any(u in rem for u in adj[n]))
        if not rem:
            return pairs
        _, choice = best(rem)
        v, u = choice
        if u is None:
            rem = rem - {v}
        else:
            pairs.append((v, u))
            rem = rem - {v, u}


def max_weight_matching(edges):
    """edges: list of (weight, (a, b)). Returns list of (a, b). Deterministic."""
    adj = {}
    for w, (a, b) in edges:
        adj.setdefault(a, {})[b] = w
        adj.setdefault(b, {})[a] = w
    seen, out = set(), []
    for start in sorted(adj):
        if start in seen:
            continue
        comp, stack = [], [start]
        seen.add(start)
        while stack:
            n = stack.pop()
            comp.append(n)
            for u in adj[n]:
                if u not in seen:
                    seen.add(u)
                    stack.append(u)
        try:
            out.extend(_solve_component(comp, adj))
        except OverflowError:
            cset = set(comp)
            out.extend(_greedy([(w, k) for w, k in edges if k[0] in cset]))
    return out


def plan_pairs(state, cfg, elig):
    cand = feasible_pairs(state, elig)
    if not cand:
        return []
    mode = cfg['match']
    if mode == 'greedy_soft':
        edges = [(soft_agreement_score(a, b), key) for a, b, key in cand]
        return [list(p) for p in _greedy(edges)]
    score = cfg.get('score', 'fit')
    if score == 'uniform':
        edges = [(1.0, key) for a, b, key in cand]
    elif score == 'agree':
        edges = [(1.0 + soft_agreement_score(a, b), key) for a, b, key in cand]
    elif score == 'member':
        hist = member_history(state)
        edges = [(member_probability(a, b, hist), key) for a, b, key in cand]
    else:
        edges = [(pair_probability(a, b), key) for a, b, key in cand]
    chosen = _greedy(edges) if mode == 'greedy_fit' else max_weight_matching(edges)
    return [sorted(p) for p in sorted(chosen)]


# ------------------------------------------------------------------------------------------- safety guard
def sanitize_asks(asks, state):
    by_id = {m['member_id']: m for m in state['members']}
    out, seen, spent = [], set(), 0
    for a in asks:
        mid, f = a['member_id'], a['field']
        m = by_id.get(mid)
        cost = 3 if f == 'constraints' else 1
        if m is None or not m['available'] or (mid, f) in seen or f not in ['constraints'] + list(SOFT):
            continue
        if spent + cost > state['ask_budget_remaining']:
            continue
        seen.add((mid, f))
        spent += cost
        out.append({'member_id': mid, 'field': f})
    return out


def sanitize_pairs(pairs, state, elig):
    by_id = {m['member_id']: m for m in state['members']}
    past = _past_pairs(state)
    out, used = [], set()
    for p in pairs:
        a, b = p
        if a == b or a in used or b in used or a not in by_id or b not in by_id:
            continue
        if not (by_id[a]['available'] and by_id[b]['available']) or tuple(sorted((a, b))) in past:
            continue
        if _status(elig, by_id[a], by_id[b]) != 'feasible':
            continue
        used.update((a, b))
        out.append([a, b])
    return out


# ------------------------------------------------------------------------------------------- entry points
def decide(request, config=None, elig=None):
    """Pure function of the request. `elig` lets trusted experiments inject a memoised eligibility check."""
    cfg = config if isinstance(config, dict) else parse_config(config)
    cfg = {**DEFAULT_CONFIG, **cfg}
    elig = elig or eligibility
    memory = request.get('memory') or {}
    phase = request.get('phase')
    key = 'asks' if phase == 'ask' else 'pairs'
    try:
        state = request['state']
        if not state.get('members'):
            return {key: [], 'memory': memory}
        if phase == 'ask':
            return {'asks': sanitize_asks(plan_asks(state, cfg, elig), state), 'memory': memory}
        return {'pairs': sanitize_pairs(plan_pairs(state, cfg, elig), state, elig), 'memory': memory}
    except Exception as error:                        # degrade to waiting; never emit an invalid episode
        print('policy fallback to empty action: %r' % (error,), file=sys.stderr)
        return {key: [], 'memory': memory}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--baseline', default='greedy', help='accepted for evaluate.py compatibility; ignored')
    parser.add_argument('--mode', default=os.environ.get('SMP_MODE', ''),
                        help="local ablations only, e.g. 'asks=hard,match=greedy_fit'; default is the submitted method")
    args = parser.parse_args()
    request = json.load(sys.stdin)
    print(json.dumps(decide(request, args.mode), allow_nan=False))
