"""ANALYSIS/EXPERIMENT ONLY: fresh populations with different preference-signal strength and density.
`signal` scales the soft-field fit term of the acceptance model (0 = fit is irrelevant, 1 = kit default, 2 = doubled);
density is varied through pool size n (more members -> more feasible partners per member) or the 'sparse' variant."""
import math
from kit import Simulator, generate

class ScaledSim(Simulator):
    def __init__(self, world, signal=1.0):
        super().__init__(world); self.signal = signal

    def _prob(self, a, b, shared):
        f, g = a['truth'], b['truth']; variant = self.world['variant']
        weights = {'relationship_goal': .7, 'relationship_pace': .4, 'lifestyle': .25, 'conversations': .2}
        if variant == 'shift':
            weights = {'relationship_goal': .25, 'relationship_pace': .8, 'lifestyle': -.25, 'conversations': .5}
        fit = self.signal * sum(w * (1 if f[k] == g[k] else -.5) for k, w in weights.items())
        drift = -.5 if variant == 'drift' and self.day >= 35 else 0
        return 1 / (1 + math.exp(-(-.25 + a['bias'] + fit + shared + drift)))

def make_sim(seed, variant='development', n=200, signal=1.0):
    return ScaledSim(generate(seed, n, 'evaluation', variant), signal)
