"""Prepare an isolated paper-reproduction workspace and verify release assets."""
from pathlib import Path
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
VERSION = 'PRISM-revision-v3.2'


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def install_bundle(archive, root, bundle):
    """Verify all bytes before extraction; never replace a different local file."""
    archive, root = Path(archive), Path(root).resolve()
    if digest(archive) != bundle['sha256']:
        raise ValueError(f'Archive checksum mismatch: {archive.name}')
    expected = {f['path']: f for f in bundle['files']}
    if len(expected) != len(bundle['files']):
        raise ValueError('Duplicate manifest paths')
    with zipfile.ZipFile(archive) as z:
        if len(z.namelist()) != len(expected) or set(z.namelist()) != set(expected):
            raise ValueError('Archive members differ from the manifest')
        for name, record in expected.items():
            target = (root / name).resolve()
            if '\\' in name or ':' in name or Path(name).is_absolute() or '..' in Path(name).parts or not target.is_relative_to(root):
                raise ValueError(f'Unsafe archive path: {name}')
            if z.getinfo(name).file_size != record['bytes']:
                raise ValueError(f'Size mismatch: {name}')
            h = hashlib.sha256()
            with z.open(name) as src:
                for block in iter(lambda: src.read(1024 * 1024), b''):
                    h.update(block)
            if h.hexdigest() != record['sha256']:
                raise ValueError(f'File checksum mismatch: {name}')
            if target.exists() and digest(target) != record['sha256']:
                raise FileExistsError(f'Refusing to replace a local file: {target}')
        for name in expected:
            target = root / name
            if not target.exists():
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(name) as src, target.open('xb') as dst:
                    shutil.copyfileobj(src, dst)


def relocate(value, root):
    """Relocate historical path strings only; preserve numerical values."""
    if isinstance(value, dict):
        return {k: relocate(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [relocate(v, root) for v in value]
    if isinstance(value, str):
        # One pass: the destination itself may start with C:/PRISM.
        if re.match(r'^[CD]:[/\\]PRISM(?:[/\\]|$)', value):
            return root.as_posix() + value[8:].replace('\\', '/')
        return re.sub(r'\b[CD]:[/\\]PRISM(?=[/\\]|$)', lambda _: root.as_posix(), value)
    return value


def copy_tree(src, dst, paths=False):
    for p in sorted(src.rglob('*')):
        if not p.is_file() or '__pycache__' in p.parts or p.suffix in ('.pyc', '.pyo'):
            continue
        out = dst / p.relative_to(src)
        out.parent.mkdir(parents=True, exist_ok=True)
        if paths and p.suffix == '.json':
            value = json.loads(p.read_text(encoding='utf-8-sig'))
            out.write_text(json.dumps(relocate(value, WORKSPACE), ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        else:
            shutil.copyfile(p, out)


def prepare(root):
    global WORKSPACE
    WORKSPACE = root
    if root.exists() and any(root.iterdir()):
        raise FileExistsError('Choose an empty workspace; existing results are never overwritten by prepare.')
    root.mkdir(parents=True, exist_ok=True)
    for package in ('dataGen', 'modelTraining', 'modelVal', 'baselines'):
        copy_tree(REPO / package, root / package)
    copy_tree(REPO / 'metadata/Logs/Models', root / 'Logs/Models', paths=True)
    copy_tree(REPO / 'metadata/Logs/Analysis', root / 'Logs/Analysis', paths=True)
    copy_tree(REPO / 'results', root / 'Paper/rebuttal/r3', paths=True)
    copy_tree(REPO / 'reproduction', root / 'Paper/rebuttal/r3')
    copy_tree(REPO / 'paper/figure_sources', root / 'Paper/figures')
    # Case input bytes stay frozen because the plotting scripts validate their provenance.
    for p in (REPO / 'paper/figure_data').iterdir():
        if not p.is_file(): continue
        dst = root / 'Paper/figures_r3'
        if p.name not in ('case_rule_r3.txt', 'fig_hdr_manifest_v3_s1.json', 'fig_case_r3_manifest.json', 'r3fig_manifest.json'):
            dst /= 'data'
        dst.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(p, dst / p.name)
    shutil.copyfile(root / 'Paper/rebuttal/r3/param_sensitivity_results.json', root / 'Paper/rebuttal/param_sensitivity_results.json')
    (root / '.prism-reproduction.json').write_text(json.dumps({'version': VERSION, 'source': str(REPO)}), encoding='utf-8')
    print(f'Prepared isolated workspace: {root}')


def run(root, script, *args):
    env = dict(os.environ, PRISM_ROOT=root.as_posix(), PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', MPLBACKEND='Agg')
    subprocess.run([sys.executable, str(root / script), *args], cwd=root / script.parent, env=env, check=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('action', choices=('prepare','install','verify','tables','figures','figure03','figure06','figure07','figure12','figure16'))
    ap.add_argument('--root', type=Path, default=REPO / 'outputs/reproduction')
    ap.add_argument('--archives', type=Path, default=REPO / 'downloads')
    ap.add_argument('--bundle', action='append', help='Install or verify only this named bundle; may be repeated.')
    ap.add_argument('--only', help='Figure names for make_r3_figs, e.g. rollout,planner_forest')
    args = ap.parse_args(); root = args.root.resolve()
    if args.action == 'prepare':
        prepare(root); return
    marker = root / '.prism-reproduction.json'
    if not marker.is_file() or json.loads(marker.read_text())['version'] != VERSION:
        raise SystemExit('Run prepare first in an empty directory.')
    manifest = json.loads((REPO / 'release_assets/MANIFEST.json').read_text())
    if args.action in ('install','verify'):
        names = args.bundle or list(manifest['bundles'])
        for name in names:
            b = manifest['bundles'][name]
            if args.action == 'install':
                install_bundle(args.archives / b['archive'], root, b)
            else:
                for f in b['files']:
                    if digest(root / f['path']) != f['sha256']:
                        raise ValueError(f"Checksum mismatch: {f['path']}")
            print(f'{args.action}: {name} ({len(b["files"])} verified files)')
        return
    r3 = Path('Paper/rebuttal/r3')
    figs = Path('Paper/figures')
    if args.action == 'tables':
        for script in ('analysis_economy.py','analysis_branch_radius.py','make_r3_tables.py','make_r3_results.py','framing_numbers.py'):
            run(root, r3 / script)
    elif args.action == 'figures':
        run(root, figs / 'make_r3_figs.py', *(['--only', args.only] if args.only else []))
    elif args.action == 'figure16':
        run(root, figs / 'make_fig_case_avoid_r3_sel.py')
    elif args.action == 'figure07':
        run(root, figs / 'make_fig_hdr.py', '--model', str(root / 'Logs/Models/20260926_141629_r3_k3_s1/final/model_final.pt'), '--suffix', '_v3_s1')
    else:
        run(root, figs / ('make_' + args.action + '.py'))


if __name__ == '__main__':
    main()
