"""Publication contract: run from the repository root with unittest discovery."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

ROOT = Path(__file__).resolve().parents[1]


class PublicationContract(unittest.TestCase):
    def test_path_relocation_does_not_rewrite_its_own_output(self):
        spec = importlib.util.spec_from_file_location('reproduce', ROOT / 'tools/reproduce.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        root = Path('C:/PRISM/outputs/reproduction')
        self.assertEqual(module.relocate('D:/PRISM/Logs/a.json', root), root.as_posix() + '/Logs/a.json')
        self.assertEqual(module.relocate('C:\\PRISM\\Logs\\a.json', root), root.as_posix() + '/Logs/a.json')
        self.assertEqual(module.relocate({'value': 46.13}, root), {'value': 46.13})

    def test_all_planner_aggregate_inputs_are_released(self):
        manifest = json.loads((ROOT / 'release_assets/MANIFEST.json').read_text())
        files = {f['path'] for b in manifest['bundles'].values() for f in b['files']}
        for p in (ROOT / 'results/planner').glob('*.json'):
            d = json.loads(p.read_text())
            for arm in d.get('inputs', {}).values():
                for path in arm.get('paths', []):
                    relative = path.replace('\\', '/').split('/PRISM/', 1)[-1]
                    self.assertTrue(relative in files, f'{p.name}: missing {relative}')

    def test_version_and_scope(self):
        self.assertEqual((ROOT / 'VERSION').read_text().strip(), 'PRISM-revision-v3.2')
        for p in ROOT.rglob('*'):
            if not p.is_file() or any(x in p.parts for x in ('.git', '__pycache__', 'outputs', 'downloads', '.venv', 'venv')):
                continue
            self.assertNotIn(p.suffix.lower(), ('.docx', '.pt', '.pkl', '.zip'))
            self.assertFalse(any(x in p.name.lower() for x in ('response_r', 'revision_handbook')))
        self.assertFalse((ROOT / 'dataGen3D').exists())

    def test_final_figures_are_byte_identical(self):
        manifest = json.loads((ROOT / 'paper/figures/manifest.json').read_text(encoding='utf-8'))
        self.assertEqual(len(manifest), 21)
        for f in manifest:
            for im in f['images']:
                p = ROOT / im['file']
                self.assertEqual(hashlib.sha256(p.read_bytes()).hexdigest(), im['sha256'])

    def test_assets_and_checkpoint(self):
        m = json.loads((ROOT / 'release_assets/MANIFEST.json').read_text(encoding='utf-8'))
        all_files = [f for b in m['bundles'].values() for f in b['files']]
        paths = [f['path'] for f in all_files]
        self.assertEqual(len(paths), len(set(paths)))
        for f in all_files:
            self.assertFalse(Path(f['path']).is_absolute())
            self.assertNotIn('..', Path(f['path']).parts)
            self.assertEqual(len(f['sha256']), 64)
            self.assertNotIn('r3_ood8/', f['path'])
            self.assertNotIn('r3_sref_pilot/', f['path'])
        headline = next(f for f in all_files if f['path'].endswith('20260926_141629_r3_k3_s1/final/model_final.pt'))
        self.assertEqual(headline['sha256'], '1bbb693397af69141a563df5eb7b8300138b93c58f2322b4ad1a701dcca841ba')

    def test_installer_rejects_unverified_or_traversal_files(self):
        spec = importlib.util.spec_from_file_location('reproduce', ROOT / 'tools/reproduce.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            archive = td / 'bad.zip'
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('../outside', b'bad')
            with self.assertRaises(ValueError):
                module.install_bundle(archive, td / 'out', {'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'files': [{'path': '../outside', 'sha256': hashlib.sha256(b'bad').hexdigest(), 'bytes': 3}]})
            self.assertFalse((td / 'outside').exists())
            with zipfile.ZipFile(archive, 'w') as z:
                z.writestr('Logs/a.txt', b'good')
            valid = {'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'files': [{'path': 'Logs/a.txt', 'sha256': hashlib.sha256(b'good').hexdigest(), 'bytes': 4}]}
            module.install_bundle(archive, td / 'out', valid)
            self.assertEqual((td / 'out/Logs/a.txt').read_bytes(), b'good')
            (td / 'out/Logs/a.txt').write_bytes(b'local-change')
            with self.assertRaises(FileExistsError):
                module.install_bundle(archive, td / 'out', valid)
            self.assertEqual((td / 'out/Logs/a.txt').read_bytes(), b'local-change')
            valid['sha256'] = '0' * 64
            with self.assertRaises(ValueError):
                module.install_bundle(archive, td / 'new', valid)


if __name__ == '__main__':
    unittest.main()
