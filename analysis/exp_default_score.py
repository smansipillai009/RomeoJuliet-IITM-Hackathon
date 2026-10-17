"""Decide the default edge score on the evaluation-sized population (n=200) across changing conditions.
Asking and allocation fixed; paired by seed; reference = shipped default ('fit'). ANALYSIS ONLY (expected MSMI uses hidden truth)."""
import json, sys
sys.path.insert(0, '.')
from analysis.eval_expected import compare
CELLS = {'fit': ({'score': 'fit'}, False), 'member': ({'score': 'member'}, False), 'agree': ({'score': 'agree'}, False)}
SEEDS = list(range(7001, 7013))
if __name__ == '__main__':
    res = {}
    for v in sys.argv[1].split(','):
        out, _ = compare(CELLS, SEEDS, v, 200, 1.0); res[v] = out
        print(f'\n== {v} n=200 ({len(SEEDS)} paired fresh seeds): expected MSMI/episode, difference vs shipped default "fit"', flush=True)
        for l in out:
            d = l.get('dE_ms'); dd = f"{d[0]:+.3f} [{d[1]:+.3f},{d[2]:+.3f}]" if d else '(reference)'
            print(f"  {l['cell']:7s} E[MSMI] {l['E_ms']:.3f}  {dd:26s} E[mutual] {l['E_mut']:.2f}  realized MSMI {l['R_ms']:.2f}", flush=True)
        json.dump(res, open('results/default_score.json', 'w'), indent=1, default=float)
