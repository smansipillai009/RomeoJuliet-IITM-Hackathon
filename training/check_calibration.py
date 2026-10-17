"""Out-of-distribution calibration: the directional model (trained on development rows) scored on rows from another variant."""
import json, sys, numpy as np
sys.path.insert(0, '.')
from training.collect_funnel import collect
M = json.load(open('results/funnel_models.json')); variant = sys.argv[1]; lo, hi = int(sys.argv[2]), int(sys.argv[3])
rows = [r for s in range(lo, hi + 1) for r in collect(s, variant)]
AG = np.array([r[2] for r in rows]); H = np.array([r[3] for r in rows], float); yes = np.array([r[5] for r in rows]); mutual = np.array([r[6] for r in rows])
rho = (H[:, 1] + M['alpha'] * M['rho0']) / (H[:, 0] + M['alpha']); eta = (H[:, 2] + M['beta'] * M['eta0']) / (H[:, 1] + M['beta'])
X = np.column_stack([AG, np.log(rho / (1 - rho)), np.log(eta / (1 - eta))]); z = X @ np.array(M['dir']['coef']) + M['dir']['b0']; p = 1 / (1 + np.exp(-z))
qs = np.unique(np.quantile(p, np.linspace(0, 1, 6))); ece = 0; bins = []
for i in range(len(qs) - 1):
    m = (p >= qs[i]) & ((p <= qs[i + 1]) if i == len(qs) - 2 else (p < qs[i + 1]))
    if m.sum(): ece += m.mean() * abs(p[m].mean() - yes[m].mean()); bins.append((int(m.sum()), round(float(p[m].mean()), 3), round(float(yes[m].mean()), 3)))
brier = float(np.mean((p - yes) ** 2)); const = float(np.mean((yes.mean() - yes) ** 2))
prod = p[::2] * p[1::2]
print(f'{variant}: directional rows {len(rows)} | observed yes {yes.mean():.3f} predicted {p.mean():.3f} | ECE {ece:.4f} Brier {brier:.4f} (constant {const:.4f}) | pair product predicted {prod.mean():.3f} vs observed mutual {mutual[::2].mean():.3f}')
print('  bins (n, predicted, observed):', bins)
