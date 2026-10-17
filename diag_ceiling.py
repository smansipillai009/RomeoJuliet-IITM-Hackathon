"""Analysis-only: uses hidden truth to explain WHY members go unserved. Not usable inside a policy."""
import itertools, statistics, copy
import harness
from kit import Simulator, generate, HARD

def run(seed, variant):
    ask, match = harness.POLICIES['base']
    sim = Simulator(generate(seed, 200, 'evaluation', variant)); el = harness.Elig()
    for _ in range(60):
        sim.resolve_asks(ask(sim.observe(), el)); sim.advance(match(sim.observe(), el))
    mem = {i: m for i, m in sim.members.items() if m['arrived_day'] <= 59}
    served = {i for x in sim.introductions for i in (x['user_a'], x['user_b'])}
    declined = {i for i, m in mem.items() if any(m['field_status'][k] == 'declined' for k in HARD)}
    # truth-based graph among members that are NOT declined-hard (they are the only ones introducible)
    ok = [i for i in mem if i not in declined]
    tm = {}
    for i in ok:
        t = copy.deepcopy(mem[i]); t['fields'] = dict(mem[i]['truth']); tm[i] = t
    partners = {i: 0 for i in ok}
    for a, b in itertools.combinations(ok, 2):
        if harness.eligibility(tm[a], tm[b])['status'] == 'feasible':
            partners[a] += 1; partners[b] += 1
    n = len(mem)
    isolated = [i for i in ok if partners[i] == 0]
    linked = [i for i in ok if partners[i] > 0]
    served_linked = [i for i in linked if i in served]
    return dict(n=n, declined=len(declined), isolated=len(isolated), linked=len(linked),
                linked_served=len(served_linked), served=len(served),
                served_declined=len([i for i in served if i in declined]),
                med_partners=statistics.median(partners[i] for i in linked) if linked else 0)

for v in ['development', 'cold_start', 'sparse']:
    rs = [run(s, v) for s in range(1001, 1006)]
    m = lambda k: statistics.mean(r[k] for r in rs)
    print(f"{v:11s} n={m('n'):.0f} declined-hard={m('declined'):.1f} isolated(no feasible partner)={m('isolated'):.1f} "
          f"linked={m('linked'):.1f} served={m('served'):.1f} (linked&served={m('linked_served'):.1f}, served-but-declined={m('served_declined'):.1f}) "
          f"median feasible partners per linked member={m('med_partners'):.1f}")
