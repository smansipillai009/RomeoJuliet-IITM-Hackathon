"""Learn a per-stage fit model from OWN rollouts (pre-action observable features only), then test it in the harness.
Training tooling (numpy/sklearn) is NOT used at inference: the policy only needs the weight dict."""
import itertools, json, random, statistics, sys
import numpy as np
import harness
from kit import Simulator, generate, SOFT, HARD

def pair_feats(a, b):
    """+1 both known & equal, -1 both known & differ, 0 unknown. Observable fields only."""
    out = []
    for k in SOFT:
        x, y = a['fields'].get(k), b['fields'].get(k)
        out.append(0.0 if x is None or y is None else (1.0 if x == y else -1.0))
    return out

def collect(seed, variant):
    """Exploration policy: VOI hard asks + soft asks, RANDOM feasible matching (keeps labels less selection-biased)."""
    ask = harness.make_ask_voi(0)
    rng = random.Random(seed)
    sim = Simulator(generate(seed, 200, 'evaluation', variant)); el = harness.Elig()
    rec = {}
    for _ in range(60):
        sim.resolve_asks(ask(sim.observe(), el))
        st = sim.observe()
        mem = {m['member_id']: m for m in st['members']}
        av = [m for m in st['members'] if m['available']]
        past = harness.past_pairs(st)
        edges = [tuple(sorted((a['member_id'], b['member_id']))) for a, b in itertools.combinations(av, 2)
                 if tuple(sorted((a['member_id'], b['member_id']))) not in past and el.status(a, b) == 'feasible']
        rng.shuffle(edges); used = set(); pairs = []
        for e in edges:
            if not used.intersection(e): pairs.append(list(e)); used.update(e)
        for a, b in pairs:
            rec[(min(a, b), max(a, b), st['day'])] = (pair_feats(mem[a], mem[b]), st['day'])
        sim.advance(pairs)
    for _ in range(40): sim.advance([])
    rows = []
    by_intro = {}
    for e in sim.events: by_intro.setdefault(e['introduction_id'], []).append(e)
    for intro in sim.introductions:
        key = (intro['user_a'], intro['user_b'], intro['assigned_day'])
        feats, day = rec[key]
        es = by_intro.get(intro['introduction_id'], [])
        resp = [e for e in es if e['event'] == 'introduction_response']
        mutual = len(resp) == 2 and all(e['value'] == 'yes' for e in resp)
        rows.append((seed, feats, int(mutual)))
    return rows

if __name__ == '__main__':
    variant = sys.argv[1] if len(sys.argv) > 1 else 'development'
    seeds = range(3001, 3031)
    data = []
    for s in seeds: data += collect(s, variant)
    X = np.array([r[1] for r in data]); y = np.array([r[2] for r in data]); g = np.array([r[0] for r in data])
    print(f'{variant}: {len(y)} introductions, {y.sum()} mutual acceptances ({y.mean():.3f}); known-feature share per field:',
          [round(float((X[:, j] != 0).mean()), 2) for j in range(len(SOFT))])
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import GroupKFold
    from sklearn.metrics import roc_auc_score
    aucs = []
    for tr, te in GroupKFold(5).split(X, y, g):
        m = LogisticRegression(C=1.0, max_iter=1000).fit(X[tr], y[tr])
        aucs.append(roc_auc_score(y[te], m.predict_proba(X[te])[:, 1]))
    print('grouped-by-seed CV AUC (mutual acceptance): %.3f ± %.3f' % (np.mean(aucs), np.std(aucs)))
    m = LogisticRegression(C=1.0, max_iter=1000).fit(X, y)
    w = dict(zip(SOFT, [round(float(c), 3) for c in m.coef_[0]]))
    print('coefficients:', w, 'intercept', round(float(m.intercept_[0]), 3))
    json.dump({'variant': variant, 'weights': w, 'intercept': float(m.intercept_[0]), 'n': int(len(y)), 'positives': int(y.sum()),
               'cv_auc': float(np.mean(aucs))}, open(f'results/fit_{variant}.json', 'w'), indent=1)
