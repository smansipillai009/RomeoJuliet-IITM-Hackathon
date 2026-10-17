"""Experiment 1: why can acceptance gains fail to become MSMI gains? ANALYSIS ONLY (uses hidden truth for diagnosis).
For each policy and fresh seed we compute, per introduction, the TRUE success probabilities, then decompose
  realized difference = volume + selection + luck    (luck = realized - expected, mean zero by construction)
and check how well the policy's own score ranks candidate pairs against the true probabilities."""
import itertools, json, statistics, sys
import numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, '.')
import harness, policy, run_final_exp  # registers 'final'
from kit import Simulator, generate
from analysis.diagnostics import true_probs

def episode(seed, variant, name, collect_candidates=False):
    ask, match = harness.POLICIES[name]
    sim = Simulator(generate(seed, 200, 'evaluation', variant)); el = harness.Elig(); cands = []
    adapt = lambda a, b: {'status': el.status(a, b)}
    for _ in range(60):
        sim.resolve_asks(ask(sim.observe(), el)); st = sim.observe()
        if collect_candidates:
            for a, b, key in policy.feasible_pairs(st, adapt):
                t = true_probs(sim, key[0], key[1], st['day'])
                cands.append((st['day'], policy.pair_probability(a, b), t['p_mutual'], t['p_msmi']))
        sim.advance(match(st, el))
    for _ in range(40): sim.advance([])
    by = {}
    for e in sim.events: by.setdefault(e['introduction_id'], []).append(e)
    E_mut = E_ms = R_mut = R_ms = 0.0; ps = []
    for i in sim.introductions:
        t = true_probs(sim, i['user_a'], i['user_b'], i['assigned_day']); E_mut += t['p_mutual']; E_ms += t['p_msmi']; ps.append(t['p_msmi'])
        es = by.get(i['introduction_id'], []); rs = [e for e in es if e['event'] == 'introduction_response']
        R_mut += len(rs) == 2 and all(e['value'] == 'yes' for e in rs)
    R_ms = sim.metrics()['mutual_second_meeting_intention']
    days = [i['assigned_day'] for i in sim.introductions]
    return dict(seed=seed, policy=name, n=len(sim.introductions), E_mut=E_mut, E_ms=E_ms, R_mut=R_mut, R_ms=R_ms,
                mean_day=float(np.mean(days)) if days else 0.0, cands=cands)

if __name__ == '__main__':
    variant, lo, hi = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]); rows = []
    for s in range(lo, hi + 1):
        rows.append(episode(s, variant, 'base')); rows.append(episode(s, variant, 'final', collect_candidates=True))
        if s % 5 == 0: print(variant, s, flush=True)
    json.dump(rows, open(f'results/decompose_{variant}_{lo}_{hi}.json', 'w'))
    B = sorted([r for r in rows if r['policy'] == 'base'], key=lambda r: r['seed']); F = sorted([r for r in rows if r['policy'] == 'final'], key=lambda r: r['seed'])
    m = lambda rs, k: statistics.mean(r[k] for r in rs)
    print(f'\n== {variant}: {len(B)} paired fresh seeds ({lo}-{hi})')
    print('                base    final   diff [95% paired bootstrap]')
    for k, lab in [('n', 'introductions'), ('R_mut', 'realized mutual'), ('E_mut', 'expected mutual'), ('R_ms', 'realized MSMI'), ('E_ms', 'expected MSMI')]:
        d, a, b = harness.paired_ci([r[k] for r in B], [r[k] for r in F])
        print(f'{lab:18s}{m(B,k):7.3f} {m(F,k):7.3f}  {d:+.3f} [{a:+.3f},{b:+.3f}]')
    pb = sum(r['E_ms'] for r in B) / sum(r['n'] for r in B); pf = sum(r['E_ms'] for r in F) / sum(r['n'] for r in F)
    nb, nf = m(B, 'n'), m(F, 'n')
    print(f'mean true MSMI-prob per introduction: base {pb:.5f}  final {pf:.5f}')
    print(f'expected-MSMI gap decomposition: volume {(nf-nb)*pb:+.3f}  selection {nf*(pf-pb):+.3f}  (sum {(nf-nb)*pb+nf*(pf-pb):+.3f} vs expected diff {m(F,"E_ms")-m(B,"E_ms"):+.3f})')
    print(f'luck (realized - expected MSMI): base {m(B,"R_ms")-m(B,"E_ms"):+.3f}  final {m(F,"R_ms")-m(F,"E_ms"):+.3f}  (per-episode SD base {np.std([r["R_ms"]-r["E_ms"] for r in B]):.2f}, final {np.std([r["R_ms"]-r["E_ms"] for r in F]):.2f})')
    print(f'mean assignment day: base {m(B,"mean_day"):.1f} final {m(F,"mean_day"):.1f}')
    C = np.array([c for r in F for c in r['cands']])
    if len(C):
        print(f'\ncandidate pairs scored by the submitted policy: n={len(C)}')
        print('  Spearman(policy score, true P(mutual)) = %.3f' % spearmanr(C[:, 1], C[:, 2])[0])
        print('  Spearman(policy score, true P(MSMI))   = %.3f' % spearmanr(C[:, 1], C[:, 3])[0])
        print('  Spearman(true P(mutual), true P(MSMI)) = %.3f' % spearmanr(C[:, 2], C[:, 3])[0])
