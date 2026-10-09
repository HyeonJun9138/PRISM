"""Public correction package consistency checks; no dataset downloads needed."""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import unittest
ROOT=Path(__file__).resolve().parents[1]

class Round2Contract(unittest.TestCase):
    def test_corrected_regressions_in_public_suite(self):
        result=subprocess.run([sys.executable,str(ROOT/'modelVal/tests/test_corrected_endpoints.py')],capture_output=True,text=True)
        self.assertEqual(result.returncode,0,result.stdout+result.stderr)

    def test_manifest_duplicates_equal_frozen_aggregates(self):
        manifest=json.loads((ROOT/'results/round2/numbers_manifest_round2.json').read_text(encoding='utf-8'))
        self.assertEqual(manifest['coverage']['validated_rows'],12300)
        self.assertTrue(manifest['ready'])
        self.assertEqual(len(manifest['aggregates']),20)
        for name,record in manifest['aggregates'].items():
            path=ROOT/'results/round2/aggregates'/(name+'.json')
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(),record['sha256'])
            self.assertEqual(json.loads(path.read_text(encoding='utf-8')),manifest['results'][name])

    def test_corrected_asset_row_hashes_match_frozen_analysis(self):
        assets=json.loads((ROOT/'release_assets/MANIFEST.json').read_text(encoding='utf-8'))
        rows={f['path']:f for f in assets['bundles']['corrected-endpoints']['files']}
        analysis=json.loads((ROOT/'results/round2/numbers_manifest_round2.json').read_text(encoding='utf-8'))
        for name,item in analysis['coverage']['inputs'].items():
            self.assertEqual(rows['round2/rescored/'+name+'.jsonl']['sha256'],item['sha256'])
        for name,bundle in assets['bundles'].items():
            expected='PRISM-revision-v3.1' if name=='corrected-endpoints' else 'PRISM-revision-v3.0'
            self.assertEqual(bundle['source_release'],expected)
            self.assertTrue(bundle['download_url'].endswith('/'+expected+'/'+bundle['archive']))

if __name__=='__main__':unittest.main()
