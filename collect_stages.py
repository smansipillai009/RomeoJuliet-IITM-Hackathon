"""Collect per-introduction stage labels (mutual, date, MSMI) from OWN rollouts under random-feasible exploration.
Features are pre-action observable soft-field agreement only. MSMI label replicates the evaluator's definition."""
import itertools, json, random, sys
import harness
from fit_exp import pair_feats
from kit import Simulator, generate

def collect(seed, variant):
    ask = harness.make_ask_voi(0); rng = random.Random(seed)
    sim = Simulator(generate(seed, 200, 'evaluation', variant)); el = harness.Elig(); rec = {}
    for _ in range(60):
        sim.resolve_asks(ask(sim.observe(), el)); st = sim.observe()
        mem = {m['member_id']: m for m in st['members']}
        av = [m for m in st['members'] if m['available']]; past = harness.past_pairs(st)
        edges = [tuple(sorted((a['member_id'], b['member_id']))) for a, b in itertools.combinations(av, 2)
                 if tuple(sorted((a['member_id'], b['member_id']))) not in past and el.status(a, b) == 'feasible']
        rng.shuffle(edges); used = set(); pairs = []
        for e in edges:
            if not used.intersection(e): pairs.append(list(e)); used.update(e)
        for a, b in pairs: rec[(min(a, b), max(a, b), st['day'])] = pair_feats(mem[a], mem[b])
        sim.advance(pairs)
    for _ in range(40): sim.advance([])
    ev = sim.receive_feedback() if hasattr(sim, 'receive_feedback') else sim.events
    by = {}
    for e in sim.events: by.setdefault(e['introduction_id'], []).append(e)
    rows = []
    for intro in sim.introductions:
        es = by.get(intro['introduction_id'], [])
        rs = [e for e in es if e['event'] == 'introduction_response']
        mutual = len(rs) == 2 and all(e['value'] == 'yes' for e in rs)
        ds = [e for e in es if e['event'] == 'date_happened' and e['value'] is True]
        ss = [e for e in es if e['event'] == 'second_meeting_intention']
        msmi = bool(ds) and ds[0]['occurred_day'] - intro['assigned_day'] <= 30 and len(ss) == 2 and all(e['value'] == 'yes' and e['occurred_day'] - ds[0]['occurred_day'] <= 3 for e in ss)
        rows.append([seed, rec[(intro['user_a'], intro['user_b'], intro['assigned_day'])], int(mutual), int(bool(ds)), int(msmi)])
    return rows

if __name__ == '__main__':
    variant, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); data = []
    for s in range(lo, hi + 1):
        data += collect(s, variant)
        if s % 20 == 0: print(variant, s, len(data), flush=True)
    json.dump(data, open(f'results/stage_data_{variant}_{lo}_{hi}.json', 'w'))
    print('done', len(data), sum(r[2] for r in data), sum(r[3] for r in data), sum(r[4] for r in data))
