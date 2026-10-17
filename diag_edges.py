"""Analysis-only (uses truth): what fraction of truly feasible pairs does the baseline ever introduce?"""
import itertools, statistics, copy
import harness
from kit import Simulator, generate, HARD

def run(seed, variant):
    ask, match = harness.POLICIES['base']
    sim = Simulator(generate(seed, 200, 'evaluation', variant)); el = harness.Elig()
    for _ in range(60):
        sim.resolve_asks(ask(sim.observe(), el)); sim.advance(match(sim.observe(), el))
    for _ in range(40): sim.advance([])
    mem = {i: m for i, m in sim.members.items() if m['arrived_day'] <= 59}
    ok = [i for i, m in mem.items() if not any(m['field_status'][k] == 'declined' for k in HARD)]
    tm = {}
    for i in ok:
        t = copy.deepcopy(mem[i]); t['fields'] = dict(mem[i]['truth']); tm[i] = t
    E = {tuple(sorted((a, b))) for a, b in itertools.combinations(ok, 2)
         if harness.eligibility(tm[a], tm[b])['status'] == 'feasible'}
    I = {tuple(sorted((x['user_a'], x['user_b']))) for x in sim.introductions}
    m = sim.metrics()
    return len(E), len(I), len(I & E), m['mutual_second_meeting_intention'], m['mutual_acceptances']

for v in ['development', 'cold_start', 'sparse']:
    rs = [run(s, v) for s in range(1001, 1009)]
    E = statistics.mean(r[0] for r in rs); I = statistics.mean(r[1] for r in rs); IE = statistics.mean(r[2] for r in rs)
    print(f"{v:11s} feasible edges E={E:.1f}  introduced={I:.1f}  introduced∩E={IE:.1f}  -> {100*IE/E:.0f}% of edges used;  "
          f"mutual={statistics.mean(r[4] for r in rs):.1f}  msmi={statistics.mean(r[3] for r in rs):.2f}")
