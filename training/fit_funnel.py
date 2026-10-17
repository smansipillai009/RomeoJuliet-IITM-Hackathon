"""Fit funnel models from collected rows; grouped-by-seed CV; calibration (reliability, Brier, ECE).
Targets: DIRECTIONAL  P(actor says yes) ;  PAIR  P(mutual acceptance) with and without member-history features."""
import json, sys, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score, brier_score_loss, log_loss

rows = json.load(open(sys.argv[1]))
seed = np.array([r[0] for r in rows]); AG = np.array([r[2] for r in rows]); H = np.array([r[3] for r in rows], float)
resp = np.array([r[4] for r in rows]); yes = np.array([r[5] for r in rows]); mutual = np.array([r[6] for r in rows]); msmi = np.array([r[8] for r in rows])
rho0 = resp.mean(); eta0 = yes.sum() / max(1, resp.sum())
print(f'directional rows {len(rows)} | response rate {rho0:.3f} | yes rate among responded {eta0:.3f} | pair mutual rate {mutual[::2].mean():.3f}')
logit = lambda p: np.log(p / (1 - p))

def member_feats(alpha, beta):
    rho = (H[:, 1] + alpha * rho0) / (H[:, 0] + alpha); eta = (H[:, 2] + beta * eta0) / (H[:, 1] + beta)
    return logit(rho), logit(eta)

def cv_pred(X, y, g, C=1.0):
    p = np.zeros(len(y))
    for tr, te in GroupKFold(5).split(X, y, g):
        p[te] = LogisticRegression(C=C, max_iter=3000).fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
    return p

def calib(p, y, bins=5):
    qs = np.unique(np.quantile(p, np.linspace(0, 1, bins + 1))); out = []; ece = 0
    for i in range(len(qs) - 1):
        m = (p >= qs[i]) & ((p <= qs[i + 1]) if i == len(qs) - 2 else (p < qs[i + 1]))
        if m.sum(): out.append((int(m.sum()), float(p[m].mean()), float(y[m].mean()))); ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return out, float(ece)

# pair rows = consecutive direction rows (a then b of one introduction)
pg = seed[::2]; ymut = mutual[::2]
best = None
for alpha in (1, 2, 4, 8):
    for beta in (1, 2, 4, 8):
        lr, le = member_feats(alpha, beta)
        Xp = np.column_stack([AG[::2], lr[::2] + lr[1::2], le[::2] + le[1::2]])
        auc = roc_auc_score(ymut, cv_pred(Xp, ymut, pg))
        if best is None or auc > best[0]: best = (auc, alpha, beta)
auc_m, alpha, beta = best
auc_0 = roc_auc_score(ymut, cv_pred(AG[::2], ymut, pg))
print(f'\nPAIR mutual acceptance, grouped-by-seed CV AUC: agreement-only {auc_0:.3f} | + member history {auc_m:.3f} (alpha={alpha}, beta={beta})')
lr, le = member_feats(alpha, beta)
Xp = np.column_stack([AG[::2], lr[::2] + lr[1::2], le[::2] + le[1::2]])
pp = cv_pred(Xp, ymut, pg); p0 = cv_pred(AG[::2], ymut, pg)
print(f'  Brier agreement-only {brier_score_loss(ymut, p0):.4f} | member {brier_score_loss(ymut, pp):.4f} | constant {brier_score_loss(ymut, np.full(len(ymut), ymut.mean())):.4f}')
fit_pair = LogisticRegression(C=1.0, max_iter=3000).fit(Xp, ymut)

# directional: P(actor says yes) from actor's own history and pair agreement
Xd = np.column_stack([AG, lr, le]); pd = cv_pred(Xd, yes, seed); pd0 = cv_pred(AG, yes, seed)
print(f'\nDIRECTIONAL P(actor says yes), grouped CV AUC: agreement-only {roc_auc_score(yes, pd0):.3f} | + actor history {roc_auc_score(yes, pd):.3f}')
cal, ece = calib(pd, yes)
print('  reliability (n, mean predicted, observed) with actor history:', [(n, round(a, 3), round(b, 3)) for n, a, b in cal], f'ECE {ece:.4f} Brier {brier_score_loss(yes, pd):.4f} (constant {brier_score_loss(yes, np.full(len(yes), yes.mean())):.4f})')
cal0, ece0 = calib(pd0, yes); print(f'  agreement-only: ECE {ece0:.4f} Brier {brier_score_loss(yes, pd0):.4f}')
fit_dir = LogisticRegression(C=1.0, max_iter=3000).fit(Xd, yes)
# pair-level calibration of the directional product
prod = (pd[::2] * pd[1::2]); cp, ecep = calib(prod, ymut)
print('  product of directional probabilities vs observed mutual:', [(n, round(a, 3), round(b, 3)) for n, a, b in cp], f'ECE {ecep:.4f} Brier {brier_score_loss(ymut, prod):.4f}')
# unequal-service-relevant check: calibration by actor history depth
for lab, m in (('no history', H[:, 0] == 0), ('has history', H[:, 0] > 0)):
    print(f'  directional calibration, {lab}: n={m.sum()} predicted {pd[m].mean():.3f} observed {yes[m].mean():.3f}')
json.dump({'rho0': float(rho0), 'eta0': float(eta0), 'alpha': alpha, 'beta': beta,
           'pair': {'coef': fit_pair.coef_[0].tolist(), 'b0': float(fit_pair.intercept_[0])},
           'dir': {'coef': fit_dir.coef_[0].tolist(), 'b0': float(fit_dir.intercept_[0])},
           'cv': {'auc_agree': float(auc_0), 'auc_member': float(auc_m)}}, open('results/funnel_models.json', 'w'), indent=1)
