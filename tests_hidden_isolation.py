"""No policy decision may use hidden simulator state: policy.py must not touch analysis code, truth, biases, response
rates, the simulator, or random streams; from kit it may import only the public contract helpers."""
import ast, unittest
from pathlib import Path

FORBIDDEN_NAMES = {'truth', 'bias', 'second_bias', 'response_rate', 'exit_day', 'rand_for', 'Simulator', 'generate', 'analysis', 'diagnostics', '_prob'}
ALLOWED_KIT = {'HARD', 'SOFT', 'eligibility'}


class HiddenIsolation(unittest.TestCase):
    def test_policy_does_not_reference_hidden_state(self):
        src = Path('policy.py').read_text(); tree = ast.parse(src)
        used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
        used |= {n.value for n in ast.walk(tree) if isinstance(n, ast.Constant) and isinstance(n.value, str)}
        self.assertEqual(used & FORBIDDEN_NAMES, set())

    def test_policy_imports_only_public_kit_helpers(self):
        tree = ast.parse(Path('policy.py').read_text())
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom) and n.module == 'kit':
                self.assertLessEqual({a.name for a in n.names}, ALLOWED_KIT)
            if isinstance(n, ast.Import):
                self.assertNotIn('analysis', [a.name.split('.')[0] for a in n.names])
            if isinstance(n, ast.ImportFrom) and n.module:
                self.assertNotEqual(n.module.split('.')[0], 'analysis')


if __name__ == '__main__':
    unittest.main()
