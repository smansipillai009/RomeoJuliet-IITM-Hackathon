"""Tests for the submitted policy: protocol shape, validity, optimality, fallbacks, determinism, stdlib-only."""
import ast, itertools, json, random, subprocess, sys, unittest
from pathlib import Path
import policy
from evaluate import validate_response
from kit import HARD, Simulator, generate


def request(state, phase, memory=None):
    return {'schema_version': '1.0.0', 'phase': phase, 'state': state, 'memory': memory}


def brute_force(edges):
    best = 0.0
    for r in range(len(edges) + 1):
        for subset in itertools.combinations(edges, r):
            nodes = [x for _, (a, b) in subset for x in (a, b)]
            if len(nodes) == len(set(nodes)):
                best = max(best, sum(w for w, _ in subset))
    return best


class PolicyTests(unittest.TestCase):
    def test_empty_state_both_phases(self):
        state = Simulator(generate(1, 0)).observe()
        self.assertEqual(policy.decide(request(state, 'ask')), {'asks': [], 'memory': {}})
        self.assertEqual(policy.decide(request(state, 'match')), {'pairs': [], 'memory': {}})

    def test_memory_passed_through_unchanged(self):
        state = Simulator(generate(1, 0)).observe()
        self.assertEqual(policy.decide(request(state, 'match', {'x': [1, 2]}))['memory'], {'x': [1, 2]})

    def test_matching_is_optimal_on_random_small_graphs(self):
        rng = random.Random(5)
        for _ in range(60):
            nodes = ['n%d' % i for i in range(rng.randint(2, 8))]
            edges = [(rng.random() + .01, tuple(sorted(p))) for p in itertools.combinations(nodes, 2) if rng.random() < .5]
            if not edges:
                continue
            got = policy.max_weight_matching(edges)
            used = [x for p in got for x in p]
            self.assertEqual(len(used), len(set(used)))
            wmap = {k: w for w, k in edges}
            self.assertAlmostEqual(sum(wmap[tuple(sorted(p))] for p in got), brute_force(edges), places=9)

    def test_state_cap_falls_back_to_valid_greedy(self):
        edges = [(.3 + .01 * i, ('a%d' % i, 'a%d' % (i + 1))) for i in range(12)]
        old, policy.STATE_CAP = policy.STATE_CAP, 0
        try:
            got = policy.max_weight_matching(edges)
        finally:
            policy.STATE_CAP = old
        used = [x for p in got for x in p]
        self.assertTrue(got)
        self.assertEqual(len(used), len(set(used)))

    def test_full_episode_actions_are_valid_and_respect_declined(self):
        for variant in ('development', 'sparse', 'cold_start'):
            sim = Simulator(generate(77, 120, 'evaluation', variant))
            for _ in range(60):
                state = sim.observe()
                asks = policy.decide(request(state, 'ask'))['asks']
                by_id = {m['member_id']: m for m in state['members']}
                self.assertLessEqual(sum(3 if a['field'] == 'constraints' else 1 for a in asks), state['ask_budget_remaining'])
                self.assertEqual(len({(a['member_id'], a['field']) for a in asks}), len(asks))
                for a in asks:
                    self.assertTrue(by_id[a['member_id']]['available'])
                    if a['field'] == 'constraints':
                        self.assertFalse(any(by_id[a['member_id']]['field_status'][k] == 'declined' for k in HARD))
                sim.resolve_asks(asks)               # raises ValueError on any invalid ask
                sim.advance(policy.decide(request(sim.observe(), 'match'))['pairs'])   # raises on any invalid pair

    def test_deterministic(self):
        sim = Simulator(generate(9, 100, 'evaluation', 'development'))
        for _ in range(10):
            sim.resolve_asks(policy.decide(request(sim.observe(), 'ask'))['asks'])
            state = sim.observe()
            self.assertEqual(policy.decide(request(state, 'match')), policy.decide(request(state, 'match')))
            sim.advance(policy.decide(request(state, 'match'))['pairs'])

    def test_malformed_state_degrades_to_empty_action(self):
        self.assertEqual(policy.decide({'phase': 'match', 'state': {'members': [{'broken': 1}]}, 'memory': None}), {'pairs': [], 'memory': {}})
        self.assertEqual(policy.decide({'phase': 'ask', 'state': {'members': [{'broken': 1}]}, 'memory': None}), {'asks': [], 'memory': {}})

    def test_policy_only_imports_stdlib_and_kit(self):
        tree = ast.parse(Path('policy.py').read_text())
        mods = {n.names[0].name.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.Import)}
        mods |= {n.module.split('.')[0] for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)}
        self.assertLessEqual(mods, set(sys.stdlib_module_names) | {'kit'})

    def test_real_process_midepisode_output_is_valid_json_protocol(self):
        sim = Simulator(generate(3, 100, 'evaluation', 'development'))
        for _ in range(8):
            sim.resolve_asks(policy.decide(request(sim.observe(), 'ask'))['asks'])
            sim.advance(policy.decide(request(sim.observe(), 'match'))['pairs'])
        for phase in ('ask', 'match'):
            out = subprocess.run([sys.executable, 'policy.py', '--baseline', 'greedy'],
                                 input=json.dumps(request(sim.observe(), phase)).encode(), capture_output=True, timeout=10)
            self.assertEqual(out.returncode, 0)
            validate_response(json.loads(out.stdout), phase)


if __name__ == '__main__':
    unittest.main()
