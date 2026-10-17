"""Collect per-introduction, per-DIRECTION training rows from own rollouts (exploration: random feasible matching).
Features are pre-action observables only (soft-field agreement + member history from visible feedback);
labels come from the simulator's event log after the follow-up window (allowed for training, never used at inference)."""
import itertools, json, random, sys
sys.path.insert(0, '.')
import harness, policy
from kit import Simulator, generate, SOFT

def agree(a, b):
    out = []
    for k in SOFT:
        x, y = a['fields'].get(k), b['fields'].get(k)
        out.append(0.0 if x is None or y is None else (1.0 if x == y else -1.0))
    return out

def collect(seed, variant='development', n=200):
    ask = harness.make_ask_voi(0); rng = random.Random(seed)
    sim = Simulator(generate(seed, n, 'evaluation', variant)); el = harness.Elig(); rec = {}
    adapt = lambda a, b: {'status': el.status(a, b)}
    for _ in range(60):
        sim.resolve_asks(ask(sim.observe(), el)); st = sim.observe()
        hist = policy.member_history(st); mem = {m['member_id']: m for m in st['members']}
        edges = [k for _, _, k in policy.feasible_pairs(st, adapt)]
        rng.shuffle(edges); used = set(); pairs = []
        for e in edges:
            if not used.intersection(e): pairs.append(list(e)); used.update(e)
        for a, b in pairs:
            rec[(a, b, st['day'])] = (agree(mem[a], mem[b]), hist.get(a, [0] * 5), hist.get(b, [0] * 5))
        sim.advance(pairs)
    for _ in range(40): sim.advance([])
    by = {}
    for e in sim.events: by.setdefault(e['introduction_id'], []).append(e)
    rows = []
    for i in sim.introductions:
        ag, ha, hb = rec[(i['user_a'], i['user_b'], i['assigned_day'])]
        es = by.get(i['introduction_id'], [])
        resp = {e['member_id']: e['value'] for e in es if e['event'] == 'introduction_response'}
        mutual = len(resp) == 2 and all(v == 'yes' for v in resp.values())
        ds = [e for e in es if e['event'] == 'date_happened' and e['value'] is True]
        ss = [e for e in es if e['event'] == 'second_meeting_intention']
        msmi = bool(ds) and ds[0]['occurred_day'] - i['assigned_day'] <= 30 and len(ss) == 2 and all(e['value'] == 'yes' and e['occurred_day'] - ds[0]['occurred_day'] <= 3 for e in ss)
        for actor, h in ((i['user_a'], ha), (i['user_b'], hb)):
            v = resp.get(actor)
            rows.append([seed, i['assigned_day'], ag, h, int(v is not None), int(v == 'yes'), int(mutual), int(bool(ds)), int(msmi)])
    return rows

if __name__ == '__main__':
    lo, hi = int(sys.argv[1]), int(sys.argv[2]); data = []
    for s in range(lo, hi + 1):
        data += collect(s)
        if s % 20 == 0: print(s, len(data), flush=True)
    json.dump(data, open(f'results/funnel_rows_{lo}_{hi}.json', 'w'))
    print('done rows', len(data))
