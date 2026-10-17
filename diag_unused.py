"""Analysis-only (truth): why are some truly feasible pairs never introduced by the baseline?"""
import itertools, copy, collections
import harness
from kit import Simulator, generate, HARD

tot = collections.Counter(); n_ep = 0
for seed in range(2001, 2009):
    ask, match = harness.POLICIES['base']
    sim = Simulator(generate(seed, 200, 'evaluation', 'development')); el = harness.Elig()
    for _ in range(60):
        sim.resolve_asks(ask(sim.observe(), el)); sim.advance(match(sim.observe(), el))
    paused = set()
    for e in sim.events:
        if e['event'] == 'pause_after_mutual_interest':
            i = next(x for x in sim.introductions if x['introduction_id'] == e['introduction_id'])
            paused.update((i['user_a'], i['user_b']))
    mem = {i: m for i, m in sim.members.items() if m['arrived_day'] <= 59}
    ok = [i for i, m in mem.items() if not any(m['field_status'][k] == 'declined' for k in HARD)]
    tm = {}
    for i in ok:
        t = copy.deepcopy(mem[i]); t['fields'] = dict(mem[i]['truth']); tm[i] = t
    I = {tuple(sorted((x['user_a'], x['user_b']))) for x in sim.introductions}
    for a, b in itertools.combinations(ok, 2):
        if harness.eligibility(tm[a], tm[b])['status'] != 'feasible': continue
        key = tuple(sorted((a, b)))
        tot['edges'] += 1
        if key in I: tot['used'] += 1; continue
        ea, eb = mem[a]['exit_day'], mem[b]['exit_day']
        if a in paused or b in paused: tot['unused: endpoint paused after mutual interest (pair succeeded elsewhere)'] += 1
        elif min(ea, eb) <= 59: tot['unused: endpoint exited before day 60'] += 1
        elif max(mem[a]['arrived_day'], mem[b]['arrived_day']) > 40: tot['unused: endpoint arrived after day 40'] += 1
        else: tot['unused: both present & idle-eligible (queue/time-limited)'] += 1
    n_ep += 1
E = tot['edges']
print(f"{n_ep} episodes, {E/n_ep:.1f} feasible edges/episode, used {100*tot['used']/E:.0f}%")
for k, v in sorted(tot.items(), key=lambda x: -x[1]):
    if k.startswith('unused'): print(f"  {k}: {v/n_ep:.1f}/episode ({100*v/E:.0f}% of edges)")
