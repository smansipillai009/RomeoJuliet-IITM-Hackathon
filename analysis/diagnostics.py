"""True (hidden-state) outcome probabilities for an introduction. ANALYSIS ONLY - never inside a policy decision.

For an introduction of pair (a, b) on day d the simulator draws a pair-level 'shared' ~ N(0, .45), then per person:
response (rate r_i), acceptance sigmoid(z_i + shared), and, after a date (0.78), second-meeting answer (rate r_i),
yes sigmoid(.15 + second_bias_i + shared + .4*[goal match]) and on-time (delay <= 3 of 1..5 => 0.6).
Expected values integrate `shared` out with Gauss-Hermite quadrature; acceptance probabilities call the simulator's own
_prob so variant-specific weights and drift are used exactly as in the simulator, not re-implemented."""
import math
import numpy as np

_X, _W = np.polynomial.hermite.hermgauss(24)
_NODES = [math.sqrt(2) * .45 * x for x in _X]
_WEIGHTS = [w / math.sqrt(math.pi) for w in _W]
DATE_RATE, ON_TIME = .78, .6


def _sig(z):
    return 1 / (1 + math.exp(-z))


def true_probs(sim, a_id, b_id, day):
    """Return dict(p_mutual, p_msmi) for introducing a_id/b_id on `day`, integrating the pair-level draw out."""
    ma, mb = sim.members[a_id], sim.members[b_id]
    old_day, sim.day = sim.day, day                   # _prob reads self.day for drift
    try:
        pm = pms = 0.0
        goal = ma['truth']['relationship_goal'] == mb['truth']['relationship_goal']
        for s, w in zip(_NODES, _WEIGHTS):
            mutual = 1.0
            second = 1.0
            for actor, other in ((ma, mb), (mb, ma)):
                mutual *= actor['response_rate'] * sim._prob(actor, other, s)
                second *= actor['response_rate'] * ON_TIME * _sig(.15 + actor['second_bias'] + s + .4 * goal)
            pm += w * mutual
            pms += w * mutual * DATE_RATE * second
        return {'p_mutual': pm, 'p_msmi': pms}
    finally:
        sim.day = old_day
