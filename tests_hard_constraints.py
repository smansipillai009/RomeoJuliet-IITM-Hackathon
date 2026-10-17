"""After-the-fact audit with hidden truth (analysis-side test): every introduction the submitted policy makes must satisfy the
members' stated hard constraints in BOTH directions under their TRUE field values, involve no member who declined a hard field,
never repeat a pair, and never ask a declined or already-observed field."""
import copy, unittest
import policy
from kit import HARD, SOFT, Simulator, eligibility, generate

def audit(variant, seed, days=45):
    sim = Simulator(generate(seed, 140, 'evaluation', variant)); seen_asks = set(); bad = []
    for _ in range(days):
        st = sim.observe(); by = {m['member_id']: m for m in st['members']}
        asks = policy.decide({'phase': 'ask', 'state': st, 'memory': None})['asks']
        for a in asks:
            m = by[a['member_id']]
            if a['field'] != 'constraints' and m['field_status'][a['field']] != 'not_asked': bad.append(('ask on resolved field', a))
            if a['field'] == 'constraints' and any(m['field_status'][k] == 'declined' for k in HARD): bad.append(('bundle for declined member', a))
        sim.resolve_asks(asks)
        sim.advance(policy.decide({'phase': 'match', 'state': sim.observe(), 'memory': None})['pairs'])
    pairs = set()
    for i in sim.introductions:
        a, b = sim.members[i['user_a']], sim.members[i['user_b']]
        ta, tb = copy.deepcopy(a), copy.deepcopy(b); ta['fields'] = dict(a['truth']); tb['fields'] = dict(b['truth'])
        if eligibility(ta, tb)['status'] != 'feasible': bad.append(('violates hard constraint under TRUE fields', i['user_a'], i['user_b']))
        for m in (a, b):
            if any(m['field_status'][k] == 'declined' for k in HARD): bad.append(('introduced a member who declined a hard field', m['member_id']))
        key = tuple(sorted((i['user_a'], i['user_b'])))
        if key in pairs: bad.append(('repeated pair', key))
        pairs.add(key)
    return bad, len(sim.introductions)

class HardConstraintAudit(unittest.TestCase):
    def test_no_violations_across_variants(self):
        total = 0
        for variant in ('development', 'sparse', 'cold_start', 'shift'):
            bad, n = audit(variant, 31); total += n
            self.assertEqual(bad, [], f'{variant}: {bad[:3]}')
        self.assertGreater(total, 20)          # the audit must actually exercise introductions

if __name__ == '__main__':
    unittest.main()
