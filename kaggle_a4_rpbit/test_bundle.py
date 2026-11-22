"""Delivery and controller-rule tests of the Kaggle A4 held-out bundle; no planner is launched."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tools')]
import rp_bit_protocol as protocol
from kaggle_a4_rpbit import bootstrap, fairtime, package_notebook
from corridor_v4_execution import read


class NotebookTests(unittest.TestCase):
    def test_fairtime_notebook(self):
        notebook = package_notebook.create_notebook('payload.b64', 'b' * 64, package_notebook.PROFILES['fairtime'])
        code = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
        for index, source in enumerate(code):
            compile(source, f'cell{index}', 'exec')
        self.assertIn('map seeds 71001-71010', code[1])
        self.assertIn("'kaggle_a4_rpbit.fairtime'", code[2])
        self.assertIn("reports/heldout_fairtime/REPORT.md", code[2])
        self.assertNotIn('CONTROLLER_MODULE', code[2])
        self.assertIn('500 lượt', ''.join(notebook['cells'][0]['source']))

    def test_selfstop_notebook(self):
        notebook = package_notebook.create_notebook('payload.b64', 'c' * 64, package_notebook.PROFILES['selfstop'])
        code = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
        for index, source in enumerate(code):
            compile(source, f'cell{index}', 'exec')
        self.assertIn('map seeds 81001-81010', code[1])
        self.assertIn("'kaggle_a4_rpbit.selfstop'", code[2])
        self.assertIn("reports/heldout_selfstop/REPORT.md", code[2])
        intro = ''.join(notebook['cells'][0]['source'])
        self.assertIn('500 lượt', intro)
        self.assertIn('4 000 vòng lặp', intro)

    def test_putn_notebook(self):
        notebook = package_notebook.create_notebook('payload.b64', 'd' * 64, package_notebook.PROFILES['putn'])
        code = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
        for index, source in enumerate(code):
            compile(source, f'cell{index}', 'exec')
        self.assertIn('map seeds 101001-101010', code[1])
        self.assertIn("'kaggle_a4_rpbit.putn'", code[2])
        self.assertIn("reports/heldout_putn/REPORT.md", code[2])
        intro = ''.join(notebook['cells'][0]['source'])
        self.assertIn('1 000 lượt', intro)
        self.assertIn('10 s', intro)
        self.assertIn('34 nhóm CTest', intro)

    def test_a5_notebook(self):
        notebook = package_notebook.create_notebook('payload.b64', 'e' * 64, package_notebook.PROFILES['a5'])
        code = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
        for index, source in enumerate(code):
            compile(source, f'cell{index}', 'exec')
        self.assertIn('map seeds 111001-111010', code[1])
        self.assertIn("'kaggle_a4_rpbit.a5'", code[2])
        self.assertIn("reports/heldout_a5_kaggle/REPORT.md", code[2])
        intro = ''.join(notebook['cells'][0]['source'])
        for text in ('1 000 lượt', '20 s', '34 nhóm CTest', package_notebook.A5_REGISTRATION):
            self.assertIn(text, intro)

    def test_a5_10s_notebook(self):
        notebook = package_notebook.create_notebook('payload.b64', 'f' * 64, package_notebook.PROFILES['a5_10s'])
        code = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
        for index, source in enumerate(code):
            compile(source, f'cell{index}', 'exec')
        self.assertIn('map seeds 121001-121010', code[1])
        self.assertIn("'kaggle_a4_rpbit.a5_10s'", code[2])
        self.assertIn("reports/heldout_a5_10s/REPORT.md", code[2])
        intro = ''.join(notebook['cells'][0]['source'])
        for text in ('500 lượt', '10 s', '34 nhóm CTest', package_notebook.A5_10S_REGISTRATION):
            self.assertIn(text, intro)

    def test_a5_ablation_notebook(self):
        notebook = package_notebook.create_notebook('payload.b64', '0' * 64, package_notebook.PROFILES['a5_ablation'])
        code = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
        for index, source in enumerate(code):
            compile(source, f'cell{index}', 'exec')
        self.assertIn('map seeds 131001-131010', code[1])
        self.assertIn("'kaggle_a4_rpbit.a5_ablation'", code[2])
        self.assertIn("reports/heldout_a5_abl/REPORT.md", code[2])
        intro = ''.join(notebook['cells'][0]['source'])
        for text in ('2 550 lượt', '20 s', '34 nhóm CTest', 'A5_ABLATION.md', package_notebook.A5_ABLATION_REGISTRATION):
            self.assertIn(text, intro)

    def test_cells_compile_and_carry_the_registered_identity(self):
        notebook = package_notebook.create_notebook('payload.b64', 'a' * 64)
        code = [''.join(cell['source']) for cell in notebook['cells'] if cell['cell_type'] == 'code']
        self.assertEqual(len(code), 4)
        for index, source in enumerate(code):
            compile(source, f'cell{index}', 'exec')
        # The embedded transport is the tested module, byte for byte.
        self.assertEqual(code[0], (ROOT / 'kaggle_pri_segment/transport.py').read_text(encoding='utf-8'))
        self.assertIn("PAYLOAD_SHA256 = '" + 'a' * 64 + "'", code[1])
        self.assertIn('ALLOW_HELDOUT = True', code[1])
        self.assertIn('RUNTIME_ROOTS = ' + repr(bootstrap.RUNTIME_ROOTS), code[1])
        self.assertIn("argv.append('--allow-heldout')", code[2])
        self.assertIn("'kaggle_a4_rpbit.bootstrap'", code[2])
        self.assertEqual(len({cell['id'] for cell in notebook['cells']}), len(notebook['cells']))


class PayloadTests(unittest.TestCase):
    REQUIRED = ('configs/a4_rpbitstar_v1_kaggle.json', 'configs/a4_rpbitstar_v1.json', 'tools/run_rp_bit.py',
                'tools/rp_bit_protocol.py', 'tools/analyze_rp_bit.py', 'tools/rp_bit_validation.py',
                'kaggle_a4_rpbit/bootstrap.py', 'kaggle_a4_rpbit/test_bundle.py', 'kaggle_pri_segment/transport.py',
                'kaggle_pri_pathsearch/bootstrap.py', 'logs/a4_rpbit_20261116/lock_manifest.json', 'src/RpBit.cpp',
                'include/terrain25d/RpBit.hpp', 'apps/RpBitWorker.inc', 'cmake/RpBitTargets.cmake', 'CMakeLists.txt',
                'tests/test_rp_bit_tooling.py', '.deps/uv-0.9.7-linux.whl', '.deps/ompl-1.7.0.tar.gz',
                # same-time run with the C++ SEM-RRT* port
                'configs/a4_rpbitstar_v1_fairtime_kaggle.json', 'kaggle_a4_rpbit/fairtime.py', 'tools/sem_cpp_equivalence.py',
                'tools/sem_native_cpp.py', 'tools/native_paper_worker.py', 'src/SemRrtStar.cpp', 'src/SemRrtStarC.cpp',
                'include/terrain25d/SemRrtStar.hpp', 'include/terrain25d/SemRrtStarC.h', 'cmake/SemCppTargets.cmake',
                'tests/test_sem_native_cpp.py', package_notebook.FAIRTIME_REGISTRATION,
                # both stop on convergence
                'configs/a4_rpbitstar_v1_selfstop_kaggle.json', 'kaggle_a4_rpbit/selfstop.py',
                package_notebook.SELFSTOP_REGISTRATION,
                # PUTN beside A4 and the self-stop SEM-RRT*
                'configs/a4_rpbitstar_v1_putn_kaggle.json', 'kaggle_a4_rpbit/putn.py', package_notebook.PUTN_REGISTRATION,
                'src/PutnStatic.cpp', 'include/terrain25d/PutnStatic.hpp', 'src/TerrainC.cpp', 'include/terrain25d/TerrainC.h',
                'tools/terrain_compare_bridge.py', 'tests/test_putn_paper.cpp', 'tests/test_putn_static.cpp',
                'tests/test_terrain_capi.py',
                # A5 beside A4 and the self-stop SEM-RRT*
                'configs/a4_rpbitstar_v1_a5_kaggle.json', 'kaggle_a4_rpbit/a5.py', package_notebook.A5_REGISTRATION,
                'src/RpBitA5.cpp', 'include/terrain25d/RpBitA5.hpp', 'apps/RpBitA5Worker.inc', 'cmake/RpBitA5Targets.cmake',
                'tools/rp_bit_a5_field.py', 'tools/analyze_rp_bit_a5.py', 'tests/integration/rp_bit_a5_tests.cpp',
                'tests/test_rp_bit_a5_worker.py', 'tests/unit/terrain_tests.cpp', 'docs/A5_RP_BITSTAR.md',
                # A5 against SEM-RRT* only, 10 s cap
                'configs/a4_rpbitstar_v1_a5_10s_kaggle.json', 'kaggle_a4_rpbit/a5_10s.py',
                package_notebook.A5_10S_REGISTRATION, 'tools/analyze_rp_bit_a5_sem.py',
                # A5 ablation: 2^4 factorial of C1-C4 beside SEM-RRT*
                'configs/a4_rpbitstar_v1_a5_ablation_kaggle.json', 'kaggle_a4_rpbit/a5_ablation.py',
                package_notebook.A5_ABLATION_REGISTRATION, 'tools/analyze_rp_bit_a5_ablation.py')

    def test_payload_holds_the_registered_sources_only(self):
        catalog = ROOT / '_bundle_catalog.json'
        if catalog.is_file():  # inside the extracted Kaggle payload: membership only, CTest may add files
            names = set(json.loads(catalog.read_text(encoding='utf-8'))['files'])
        else:  # the current delivery; the spent held-out delivery predates the C++ port
            names = {path.relative_to(ROOT).as_posix() for path in package_notebook.source_files(package_notebook.PROFILES['a5_ablation'])}
        self.assertFalse(set(self.REQUIRED) - names)
        self.assertFalse([name for name in names if name.startswith(('results/', 'build', 'dist/', 'runs/'))])


class GateTests(unittest.TestCase):
    def setUp(self):
        self.config = protocol.load_config(protocol.KAGGLE_CONFIG)

    def write(self, folder, rows, *, collection='COMPLETE'):
        directory = Path(folder) / 'kaggle_gate'
        directory.mkdir(parents=True, exist_ok=True)
        entries = protocol.inventory(self.config, 'kaggle_gate')
        manifest = dict(entries=entries, sessions=[dict(collection_status=collection)], config_sha256='c', binary_sha256='b')
        (directory / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
        summary = dict(planned=len(entries), validated=len(rows), success=sum(r['success'] for r in rows),
                       correctness_failures=sum(bool(r['correctness_errors']) for r in rows), rows=rows)
        (directory / 'summary_full.json').write_text(json.dumps(summary), encoding='utf-8')
        return directory

    def rows(self):
        out = []
        for entry in protocol.inventory(self.config, 'kaggle_gate'):
            native = entry['method'] == protocol.NATIVE
            out.append(dict(trial_id=entry['trial_id'], method=entry['method'], success=True, correctness_errors=[],
                            iterations=7425 if native else None, solve_time_s=50. if native else 3.))
        return out

    def verdict(self, rows, **kwargs):
        with tempfile.TemporaryDirectory() as folder:
            return bootstrap.gate_verdict(self.write(folder, rows, **kwargs), self.config, 'kaggle_gate')

    def test_gate_rules(self):
        good = self.rows()
        self.assertEqual(len(good), 30)
        verdict = self.verdict(good)
        self.assertTrue(verdict['passed'], verdict['problems'])
        self.assertEqual((verdict['sem_work_iterations'], verdict['sem_iterations']), (7425, [7425] * 5))
        find = lambda rows, method: next(r for r in rows if r['method'] == method)
        tolerated = copy.deepcopy(good)
        find(tolerated, 'a3t_sem_exposure_l33')['success'] = False  # planner failures of baselines are results
        find(tolerated, protocol.NATIVE)['success'] = False
        self.assertTrue(self.verdict(tolerated)['passed'])
        for method, change in (('a4_rpbit_l33_m20', dict(success=False)), ('a4_rpbit_l33', dict(success=False)),
                               (protocol.NATIVE, dict(iterations=7000)), ('a3t_sem_exposure_l33', dict(correctness_errors=['x']))):
            rows = copy.deepcopy(good)
            find(rows, method).update(change)
            self.assertFalse(self.verdict(rows)['passed'], (method, change))
        self.assertFalse(self.verdict(good[:-1])['passed'])
        self.assertFalse(self.verdict(good, collection='DEADLINE')['passed'])


class ControllerTests(unittest.TestCase):
    def test_registered_constants_match_the_kaggle_config(self):
        config = protocol.load_config(protocol.KAGGLE_CONFIG)
        self.assertEqual(len(set(bootstrap.REQUIRED_CTEST)), 30)
        self.assertTrue(set(bootstrap.GATES) | {bootstrap.HELDOUT} <= set(config['phases']))
        deviation = {item['item']: item for item in bootstrap.DEVIATIONS}
        self.assertEqual(deviation['execution.cores']['kaggle'], config['execution']['cores'])
        phase = config['phases'][bootstrap.HELDOUT]
        self.assertIn(f"{phase['sem_work_iterations']} iterations", deviation['SEM-RRT* budget']['kaggle'])
        self.assertIn(f"{phase['sem_budget_cap_s']:g} s", deviation['SEM-RRT* budget']['kaggle'])
        self.assertEqual(len(protocol.inventory(config, bootstrap.HELDOUT)), 1500)

    def test_refuses_a_foreign_host_or_budget_before_any_side_effect(self):
        # Outside the Kaggle scratch repository the host check fails; inside it the budget check does.
        with self.assertRaises((RuntimeError, ValueError)):
            bootstrap.main(['--started-epoch', '0', '--session-seconds', '10'])
        with self.assertRaises((RuntimeError, ValueError)):
            fairtime.main(['--started-epoch', '0', '--session-seconds', '10'])
        with self.assertRaises((RuntimeError, ValueError)):
            fairtime.main(['--started-epoch', '0', '--session-seconds', '10'], experiment='selfstop')
        with self.assertRaises((RuntimeError, ValueError)):
            fairtime.main(['--started-epoch', '0', '--session-seconds', '10'], experiment='putn')
        with self.assertRaises((RuntimeError, ValueError)):
            fairtime.main(['--started-epoch', '0', '--session-seconds', '10'], experiment='a5_10s')
        with self.assertRaises((RuntimeError, ValueError)):
            fairtime.main(['--started-epoch', '0', '--session-seconds', '10'], experiment='a5_ablation')


class FairtimeControllerTests(unittest.TestCase):
    def setUp(self):
        self.config = protocol.load_config(protocol.FAIRTIME_CONFIG)

    def test_registration_and_constants(self):
        registered = fairtime.check_registration(protocol.FAIRTIME_CONFIG, self.config)
        self.assertEqual(registered['planned_trials'], {'smoke_fairtime': 14, 'gate_fairtime': 10, 'heldout_fairtime': 500})
        self.assertEqual(len(set(fairtime.REQUIRED_CTEST)), 31)
        self.assertIn('NAME sem_native_cpp', (ROOT / 'cmake/SemCppTargets.cmake').read_text(encoding='utf-8'))
        gate = fairtime.IDENTITY_GATE
        self.assertIn(gate['phase'], fairtime.GATES)
        self.assertIn(self.config['phases'][gate['phase']]['planner_seeds'][0], gate['seeds'])
        self.assertIn(gate['core'], self.config['execution']['cores'])
        self.assertEqual(self.config['phases'][fairtime.HELDOUT]['primary_method'],
                         json.loads(bootstrap.LOCK.read_text(encoding='utf-8-sig'))['primary_method'])

    def test_a_changed_config_is_refused(self):
        changed = copy.deepcopy(self.config)
        changed['phases'][fairtime.HELDOUT]['planner_seeds'] = [716001]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'differs from the registration'):
                fairtime.check_registration(path, changed)

    def test_gate_rules(self):
        entries = protocol.inventory(self.config, 'gate_fairtime')
        good = [dict(trial_id=e['trial_id'], method=e['method'], success=True, correctness_errors=[],
                     iterations=30000 if e['method'] == protocol.NATIVE else None, solve_time_s=20.) for e in entries]

        def verdict(rows):
            with tempfile.TemporaryDirectory() as folder:
                directory = Path(folder) / 'gate_fairtime'
                directory.mkdir()
                manifest = dict(entries=entries, sessions=[dict(collection_status='COMPLETE')], config_sha256='c', binary_sha256='b')
                (directory / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                summary = dict(planned=len(entries), validated=len(rows), success=sum(r['success'] for r in rows),
                               correctness_failures=sum(bool(r['correctness_errors']) for r in rows), rows=rows)
                (directory / 'summary_full.json').write_text(json.dumps(summary), encoding='utf-8')
                return bootstrap.gate_verdict(directory, self.config, 'gate_fairtime')
        self.assertTrue(verdict(good)['passed'])
        for method, change, passed in ((protocol.NATIVE, dict(success=False), True),
                                       (protocol.NATIVE, dict(correctness_errors=['sem_engine_mismatch']), False),
                                       ('a4_rpbit_l33_m20', dict(success=False), False)):
            rows = copy.deepcopy(good)
            next(r for r in rows if r['method'] == method).update(change)
            self.assertEqual(verdict(rows)['passed'], passed, (method, change))


class SelfstopControllerTests(unittest.TestCase):
    """Same controller with the self-stop registration: SEM-RRT* (C++) also stops on convergence."""

    def setUp(self):
        self.experiment = fairtime.EXPERIMENTS['selfstop']
        self.config = protocol.load_config(protocol.SELFSTOP_CONFIG)

    def test_registration_and_constants(self):
        registered = fairtime.check_registration(protocol.SELFSTOP_CONFIG, self.config, self.experiment)
        self.assertEqual(registered['planned_trials'], {'smoke_selfstop': 14, 'gate_selfstop': 10, 'heldout_selfstop': 500})
        gate, heldout = self.experiment['identity_gate'], self.experiment['heldout']
        self.assertEqual({k: v for k, v in gate.items() if k != 'phase'}, {k: v for k, v in fairtime.IDENTITY_GATE.items() if k != 'phase'})
        self.assertIn(gate['phase'], self.experiment['gates'])
        self.assertIn(self.config['phases'][gate['phase']]['planner_seeds'][0], gate['seeds'])
        self.assertIn(gate['core'], self.config['execution']['cores'])
        self.assertEqual(self.config['phases'][heldout]['primary_method'],
                         json.loads(bootstrap.LOCK.read_text(encoding='utf-8-sig'))['primary_method'])
        self.assertEqual(self.config['methods'][protocol.NATIVE]['convergence_stop'],
                         dict(relative_eps=1e-3, window_iterations=4000))
        self.assertEqual(self.experiment['module'], package_notebook.PROFILES['selfstop']['module'])

    def test_a_changed_config_is_refused(self):
        changed = copy.deepcopy(self.config)
        changed['methods'][protocol.NATIVE]['convergence_stop']['window_iterations'] = 2000
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'differs from the registration'):
                fairtime.check_registration(path, changed, self.experiment)

    def test_a_stop_rule_error_fails_the_gate(self):
        entries = protocol.inventory(self.config, 'gate_selfstop')
        good = [dict(trial_id=e['trial_id'], method=e['method'], success=True, correctness_errors=[],
                     iterations=9000 if e['method'] == protocol.NATIVE else None, solve_time_s=4.) for e in entries]
        with tempfile.TemporaryDirectory() as folder:
            directory = Path(folder) / 'gate_selfstop'
            directory.mkdir()
            manifest = dict(entries=entries, sessions=[dict(collection_status='COMPLETE')], config_sha256='c', binary_sha256='b')
            (directory / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
            for rows, passed in ((good, True), ([dict(r, correctness_errors=['sem_convergence_inconsistent'])
                                                 if r['method'] == protocol.NATIVE else r for r in good], False)):
                summary = dict(planned=len(entries), validated=len(rows), success=sum(r['success'] for r in rows),
                               correctness_failures=sum(bool(r['correctness_errors']) for r in rows), rows=rows)
                (directory / 'summary_full.json').write_text(json.dumps(summary), encoding='utf-8')
                self.assertEqual(bootstrap.gate_verdict(directory, self.config, 'gate_selfstop')['passed'], passed)


class PutnControllerTests(unittest.TestCase):
    """Same controller with the PUTN registration: four methods, 10 s cap, validation on four cores."""

    def setUp(self):
        self.experiment = fairtime.EXPERIMENTS['putn']
        self.config = protocol.load_config(protocol.PUTN_KAGGLE_CONFIG)

    def test_registration_and_constants(self):
        registered = fairtime.check_registration(protocol.PUTN_KAGGLE_CONFIG, self.config, self.experiment)
        self.assertEqual(registered['planned_trials'], {'smoke_putn': 28, 'gate_putn': 20, 'heldout_putn': 1000})
        tests = self.experiment['ctest']
        self.assertEqual(len(set(tests)), 34)
        self.assertEqual(tests[:31], fairtime.REQUIRED_CTEST)
        for name, script in (('putn_paper', 'NativePaperTargets'), ('putn_static', 'TerrainCompareTargets'),
                             ('terrain_capi', 'TerrainCompareTargets')):
            self.assertIn(f'NAME {name} ', (ROOT / 'cmake' / (script + '.cmake')).read_text(encoding='utf-8'))
        gate, heldout = self.experiment['identity_gate'], self.experiment['heldout']
        self.assertEqual({k: v for k, v in gate.items() if k != 'phase'}, {k: v for k, v in fairtime.IDENTITY_GATE.items() if k != 'phase'})
        self.assertIn(gate['phase'], self.experiment['gates'])
        self.assertIn(self.config['phases'][gate['phase']]['planner_seeds'][0], gate['seeds'])
        execution = self.config['execution']
        self.assertEqual((execution['cores'], execution['validation_cores']), ([0], [0, 1, 2, 3]))
        self.assertIn(gate['core'], execution['cores'])
        self.assertEqual(self.config['phases'][heldout]['primary_method'],
                         json.loads(bootstrap.LOCK.read_text(encoding='utf-8-sig'))['primary_method'])
        self.assertEqual({phase['planning_budget_s'] for name, phase in self.config['phases'].items() if name != 'smoke_putn'}, {10.})
        self.assertEqual(self.config['methods'], protocol.load_config(protocol.PUTN_DEV_CONFIG)['methods'])
        self.assertEqual(self.experiment['module'], package_notebook.PROFILES['putn']['module'])

    def test_a_changed_config_is_refused(self):
        changed = copy.deepcopy(self.config)
        changed['methods']['putn_slope35']['tau_slope_cap_deg'] = 30.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'differs from the registration'):
                fairtime.check_registration(path, changed, self.experiment)

    def test_gate_rules(self):
        entries = protocol.inventory(self.config, 'gate_putn')
        good = [dict(trial_id=e['trial_id'], method=e['method'], success=True, correctness_errors=[],
                     iterations=9000 if self.config['methods'][e['method']]['kind'] != 'rp_bit' else None, solve_time_s=4.)
                for e in entries]

        def verdict(rows):
            with tempfile.TemporaryDirectory() as folder:
                directory = Path(folder) / 'gate_putn'
                directory.mkdir()
                manifest = dict(entries=entries, sessions=[dict(collection_status='COMPLETE')], config_sha256='c', binary_sha256='b')
                (directory / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                summary = dict(planned=len(entries), validated=len(rows), success=sum(r['success'] for r in rows),
                               correctness_failures=sum(bool(r['correctness_errors']) for r in rows), rows=rows)
                (directory / 'summary_full.json').write_text(json.dumps(summary), encoding='utf-8')
                return bootstrap.gate_verdict(directory, self.config, 'gate_putn')
        self.assertTrue(verdict(good)['passed'])
        # A PUTN planner failure is a result; a PUTN correctness error or an A4 failure stops the run.
        for method, change, passed in (('putn_source', dict(success=False), True),
                                       ('putn_slope35', dict(correctness_errors=['putn_convergence_inconsistent']), False),
                                       ('a4_rpbit_l33_m20', dict(success=False), False)):
            rows = copy.deepcopy(good)
            next(r for r in rows if r['method'] == method).update(change)
            self.assertEqual(verdict(rows)['passed'], passed, (method, change))


class A5ControllerTests(unittest.TestCase):
    """Same controller with the A5 registration: A5 primary, its C1 + C2 arm, A4 and SEM-RRT*, 20 s cap."""

    def setUp(self):
        self.experiment = fairtime.EXPERIMENTS['a5']
        self.config = protocol.load_config(protocol.A5_KAGGLE_CONFIG)

    def test_registration_and_constants(self):
        registered = fairtime.check_registration(protocol.A5_KAGGLE_CONFIG, self.config, self.experiment)
        self.assertEqual(registered['planned_trials'], {'smoke_a5_kaggle': 28, 'gate_a5_kaggle': 20, 'heldout_a5_kaggle': 1000})
        tests = self.experiment['ctest']
        self.assertEqual(len(set(tests)), 34)
        self.assertEqual(tests[:31], fairtime.REQUIRED_CTEST)
        for name, script in (('rp_bit_a5_core', 'cmake/RpBitA5Targets.cmake'), ('rp_bit_a5_worker', 'cmake/RpBitA5Targets.cmake'),
                             ('terrain_unit', 'CMakeLists.txt')):
            self.assertRegex((ROOT / script).read_text(encoding='utf-8'), rf'NAME {name}\b')
        gate, heldout = self.experiment['identity_gate'], self.experiment['heldout']
        self.assertEqual({k: v for k, v in gate.items() if k != 'phase'}, {k: v for k, v in fairtime.IDENTITY_GATE.items() if k != 'phase'})
        self.assertIn(gate['phase'], self.experiment['gates'])
        self.assertIn(self.config['phases'][gate['phase']]['planner_seeds'][0], gate['seeds'])
        execution = self.config['execution']
        self.assertEqual((execution['cores'], execution['validation_cores']), ([0], [0, 1, 2, 3]))
        self.assertIn(gate['core'], execution['cores'])
        self.assertEqual(self.config['phases'][heldout]['primary_method'], 'a5_rpbit_l33_w4000')
        self.assertEqual({phase['planning_budget_s'] for name, phase in self.config['phases'].items()
                          if name != 'smoke_a5_kaggle'}, {20.})
        self.assertEqual(self.experiment['module'], package_notebook.PROFILES['a5']['module'])
        # The registration freezes the planner, runner, validation and analysis sources, and they are unchanged.
        for name in ('src/RpBitA5.cpp', 'src/RpBit.cpp', 'src/SemRrtStar.cpp', 'src/Terrain.cpp', 'tools/rp_bit_validation.py',
                     'tools/analyze_rp_bit.py', 'tools/analyze_rp_bit_a5.py'):
            self.assertIn(name, registered['source_sha256'])

    def test_a_changed_config_or_source_is_refused(self):
        changed = copy.deepcopy(self.config)
        changed['methods']['a5_rpbit_l33_w4000']['batch_min'] = 400
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'differs from the registration'):
                fairtime.check_registration(path, changed, self.experiment)
        real = fairtime.file_hash
        with mock.patch.object(fairtime, 'file_hash', lambda p: '0' * 64 if Path(p).name == 'RpBitA5.cpp' else real(p)):
            with self.assertRaisesRegex(ValueError, 'source differs from the registration: src/RpBitA5.cpp'):
                fairtime.check_registration(protocol.A5_KAGGLE_CONFIG, self.config, self.experiment)

    def test_gate_rules(self):
        entries = protocol.inventory(self.config, 'gate_a5_kaggle')
        good = [dict(trial_id=e['trial_id'], method=e['method'], success=True, correctness_errors=[],
                     iterations=9000 if e['method'] == protocol.NATIVE else None, solve_time_s=2.) for e in entries]

        def verdict(rows):
            with tempfile.TemporaryDirectory() as folder:
                directory = Path(folder) / 'gate_a5_kaggle'
                directory.mkdir()
                manifest = dict(entries=entries, sessions=[dict(collection_status='COMPLETE')], config_sha256='c', binary_sha256='b')
                (directory / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                summary = dict(planned=len(entries), validated=len(rows), success=sum(r['success'] for r in rows),
                               correctness_failures=sum(bool(r['correctness_errors']) for r in rows), rows=rows)
                (directory / 'summary_full.json').write_text(json.dumps(summary), encoding='utf-8')
                return bootstrap.gate_verdict(directory, self.config, 'gate_a5_kaggle')
        self.assertTrue(verdict(good)['passed'])
        # A SEM-RRT* planner failure is a result; an A4 or A5 failure, or any correctness error, stops the run.
        for method, change, passed in ((protocol.NATIVE, dict(success=False), True),
                                       ('a5_rpbit_l33_w4000', dict(success=False), False),
                                       ('a5_rpbit_l33_w4000_eager', dict(success=False), False),
                                       ('a4_rpbit_l33_m20', dict(success=False), False),
                                       ('a5_rpbit_l33_w4000', dict(correctness_errors=['a5_seed_event_inconsistent']), False)):
            rows = copy.deepcopy(good)
            next(r for r in rows if r['method'] == method).update(change)
            self.assertEqual(verdict(rows)['passed'], passed, (method, change))



class A5TenSecondControllerTests(unittest.TestCase):
    """Same controller with the 10 s registration: only the A5 primary and the self-stop SEM-RRT*, fresh seeds."""

    def setUp(self):
        self.experiment = fairtime.EXPERIMENTS['a5_10s']
        self.config = protocol.load_config(self.experiment['config'])

    def test_registration_and_constants(self):
        registered = fairtime.check_registration(self.experiment['config'], self.config, self.experiment)
        self.assertEqual(registered['planned_trials'], {'smoke_a5_10s': 14, 'gate_a5_10s': 10, 'heldout_a5_10s': 500})
        # Same CTest set and sources as the 20 s A5 held-out, plus the verdict tool of this experiment.
        earlier = fairtime.EXPERIMENTS['a5']
        self.assertEqual(self.experiment['ctest'], earlier['ctest'])
        self.assertEqual(set(registered['source_sha256']) - set(read(earlier['registration'])['source_sha256']),
                         {'tools/analyze_rp_bit_a5_sem.py'})
        self.assertEqual(self.experiment['a5_verdict'], 'tools/analyze_rp_bit_a5_sem.py')
        gate, heldout = self.experiment['identity_gate'], self.experiment['heldout']
        self.assertEqual({k: v for k, v in gate.items() if k != 'phase'}, {k: v for k, v in fairtime.IDENTITY_GATE.items() if k != 'phase'})
        self.assertIn(gate['phase'], self.experiment['gates'])
        self.assertIn(self.config['phases'][gate['phase']]['planner_seeds'][0], gate['seeds'])
        self.assertIn(gate['core'], self.config['execution']['cores'])
        self.assertEqual(self.config['phases'][heldout]['primary_method'], 'a5_rpbit_l33_w4000')
        profile = package_notebook.PROFILES['a5_10s']
        self.assertEqual((profile['module'], profile['config'], profile['heldout']),
                         (self.experiment['module'], self.experiment['config'], heldout))

    def test_a_changed_config_or_source_is_refused(self):
        changed = copy.deepcopy(self.config)
        changed['phases'][self.experiment['heldout']]['planning_budget_s'] = 20.
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'differs from the registration'):
                fairtime.check_registration(path, changed, self.experiment)
        real = fairtime.file_hash
        with mock.patch.object(fairtime, 'file_hash', lambda p: '0' * 64 if Path(p).name == 'analyze_rp_bit_a5_sem.py' else real(p)):
            with self.assertRaisesRegex(ValueError, 'source differs from the registration: tools/analyze_rp_bit_a5_sem.py'):
                fairtime.check_registration(self.experiment['config'], self.config, self.experiment)

    def test_gate_rules(self):
        entries = protocol.inventory(self.config, 'gate_a5_10s')
        good = [dict(trial_id=e['trial_id'], method=e['method'], success=True, correctness_errors=[],
                     iterations=9000 if e['method'] == protocol.NATIVE else None, solve_time_s=2.) for e in entries]

        def verdict(rows):
            with tempfile.TemporaryDirectory() as folder:
                directory = Path(folder) / 'gate_a5_10s'
                directory.mkdir()
                manifest = dict(entries=entries, sessions=[dict(collection_status='COMPLETE')], config_sha256='c', binary_sha256='b')
                (directory / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                summary = dict(planned=len(entries), validated=len(rows), success=sum(r['success'] for r in rows),
                               correctness_failures=sum(bool(r['correctness_errors']) for r in rows), rows=rows)
                (directory / 'summary_full.json').write_text(json.dumps(summary), encoding='utf-8')
                return bootstrap.gate_verdict(directory, self.config, 'gate_a5_10s')
        self.assertEqual(len(good), 10)
        self.assertTrue(verdict(good)['passed'])
        # A SEM-RRT* planner failure is a result; an A5 failure or any correctness error stops the run.
        for method, change, passed in ((protocol.NATIVE, dict(success=False), True),
                                       ('a5_rpbit_l33_w4000', dict(success=False), False),
                                       (protocol.NATIVE, dict(correctness_errors=['sem_convergence_inconsistent']), False)):
            rows = copy.deepcopy(good)
            next(r for r in rows if r['method'] == method).update(change)
            self.assertEqual(verdict(rows)['passed'], passed, (method, change))



class A5AblationControllerTests(unittest.TestCase):
    """Same controller with the ablation registration: 16 factorial cells and SEM-RRT*, A4 in the gate only."""

    def setUp(self):
        self.experiment = fairtime.EXPERIMENTS['a5_ablation']
        self.config = protocol.load_config(self.experiment['config'])

    def test_registration_and_constants(self):
        registered = fairtime.check_registration(self.experiment['config'], self.config, self.experiment)
        self.assertEqual(registered['planned_trials'], {'smoke_a5_abl': 119, 'gate_a5_abl': 90, 'heldout_a5_abl': 2550})
        earlier = fairtime.EXPERIMENTS['a5']
        self.assertEqual(self.experiment['ctest'], earlier['ctest'])
        self.assertEqual(set(registered['source_sha256']) - set(read(earlier['registration'])['source_sha256']),
                         {'tools/analyze_rp_bit_a5_ablation.py'})
        self.assertEqual((self.experiment['a5_verdict'], self.experiment['path_identity']),
                         ('tools/analyze_rp_bit_a5_ablation.py', True))
        gate, heldout = self.experiment['identity_gate'], self.experiment['heldout']
        self.assertEqual({k: v for k, v in gate.items() if k != 'phase'}, {k: v for k, v in fairtime.IDENTITY_GATE.items() if k != 'phase'})
        self.assertIn(gate['phase'], self.experiment['gates'])
        self.assertIn(self.config['phases'][gate['phase']]['planner_seeds'][0], gate['seeds'])
        self.assertIn(gate['core'], self.config['execution']['cores'])
        self.assertEqual(self.config['phases'][heldout]['primary_method'], 'a5_rpbit_l33_w4000')
        profile = package_notebook.PROFILES['a5_ablation']
        self.assertEqual((profile['module'], profile['config'], profile['heldout']),
                         (self.experiment['module'], self.experiment['config'], heldout))

    def test_a_changed_config_or_source_is_refused(self):
        changed = copy.deepcopy(self.config)
        changed['methods']['a5_abl_0110']['lazy_refine'] = False
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'config.json'
            path.write_text(json.dumps(changed), encoding='utf-8')
            with self.assertRaisesRegex(ValueError, 'differs from the registration'):
                fairtime.check_registration(path, changed, self.experiment)
        real = fairtime.file_hash
        with mock.patch.object(fairtime, 'file_hash', lambda p: '0' * 64 if Path(p).name == 'analyze_rp_bit_a5_ablation.py' else real(p)):
            with self.assertRaisesRegex(ValueError, 'source differs from the registration: tools/analyze_rp_bit_a5_ablation.py'):
                fairtime.check_registration(self.experiment['config'], self.config, self.experiment)

    def test_gate_rules_with_final_path_identity(self):
        entries = protocol.inventory(self.config, 'gate_a5_abl')
        good = [dict(trial_id=e['trial_id'], method=e['method'], map_id=e['map_id'], planner_seed=e['planner_seed'],
                     success=True, correctness_errors=[], iterations=9000 if e['method'] == protocol.NATIVE else None,
                     solve_time_s=1., termination_reason='CONVERGED_NO_IMPROVEMENT', final_j_sem=100., final_cost_terrain_m=30.,
                     final_length_xy_m=29., final_pri=1., final_vertex_count=9, final_sharp_turn_count=0) for e in entries]

        def verdict(rows):
            with tempfile.TemporaryDirectory() as folder:
                directory = Path(folder) / 'gate_a5_abl'
                directory.mkdir()
                manifest = dict(protocol_id=protocol.PROTOCOL_ID, phase='gate_a5_abl', config=self.config, entries=entries,
                                sessions=[dict(collection_status='COMPLETE')], config_sha256='c', binary_sha256='b')
                (directory / 'manifest.json').write_text(json.dumps(manifest), encoding='utf-8')
                summary = dict(planned=len(entries), validated=len(rows), success=sum(r['success'] for r in rows),
                               correctness_failures=sum(bool(r['correctness_errors']) for r in rows), rows=rows)
                (directory / 'summary_full.json').write_text(json.dumps(summary), encoding='utf-8')
                return fairtime.path_identity(bootstrap.gate_verdict(directory, self.config, 'gate_a5_abl'), directory)
        self.assertEqual(len(good), 90)
        passed = verdict(good)
        self.assertTrue(passed['passed'], passed['problems'])
        self.assertEqual({k: v['checked'] for k, v in passed['path_identity'].items() if k != 'basis'},
                         {'c2_pairs': 40, 'cell_0000_vs_a4': 5})
        # A SEM-RRT* planner failure is a result; an A5 or A4 failure, or a different final path where the C2 shortcut
        # or cell 0000 must reproduce the same decisions, stops the run.
        for method, change, ok in ((protocol.NATIVE, dict(success=False), True),
                                   ('a5_abl_1011', dict(success=False), False),
                                   ('a4_rpbit_l33_m20', dict(final_length_xy_m=29.5), False),
                                   ('a5_abl_0100', dict(final_j_sem=100.5), False),
                                   (protocol.NATIVE, dict(final_j_sem=150.), True)):
            rows = copy.deepcopy(good)
            next(r for r in rows if r['method'] == method).update(change)
            self.assertEqual(verdict(rows)['passed'], ok, (method, change))


if __name__ == '__main__':
    unittest.main()
