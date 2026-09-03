#!/usr/bin/env python3
"""Generate demars.yaml from assets/demars.yaml.example, filling the machine-specific keys.

    python3 assets/setup/write_config.py --python /abs/envs/demars/bin/python \
        --checkpoint-dir /data/sevennet_ckpt [--icsd-db ~/icsd_db] [--mp-key KEY] [--out demars.yaml]

Stdlib only, so it runs under any python3 -- the demars env may not exist yet. The example's
comments are kept: keys are replaced line by line, not re-serialised, because that file's comments
are the documentation a new deployment reads.

What is checked, not guessed:
  --python          must exist, be executable, and import demars_core. This is `paths.python`, the
                    ONE place the env is written down (`tools/py` reads it). Take it from
                    make_env.sh's PYTHON= line, never from `which python`.
  --checkpoint-dir  the nano checkpoint is a LOCAL FILE the user downloads (SevenNet-nano weights);
                    `checkpoint_7net_nano_5.5.pth` is looked for there and pinned by absolute path.
                    Missing -> written as a placeholder and reported, never silently pointed at
                    a name that may not resolve.
  --icsd-db         optional. Absent is honest: the record then says the sibling search did NOT
                    run ("unchecked"), which is different from "no ordered form exists".
  --mp-key          optional. Absent -> gates.hull is `not_run`. demars.yaml is gitignored, but the
                    key still sits in a file; $MP_API_KEY overrides it at run time.
"""
import argparse
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EXAMPLE = os.path.join(ROOT, 'assets', 'demars.yaml.example')
NANO_FILE = 'checkpoint_7net_nano_5.5.pth'


def _yaml_scalar(v):
    if v is None:
        return 'null'
    return v if re.fullmatch(r'[A-Za-z0-9_./~+-]+', v) else repr(v)


def _sub(lines, key, value, section=None):
    """Replace the value of `key:` (optionally only inside `section:`), keeping the trailing
    comment. Returns True iff a line was rewritten."""
    pat = re.compile(r'^(\s*)' + re.escape(key) + r':\s*([^#]*?)(\s*#.*)?$')
    in_section = section is None
    for i, line in enumerate(lines):
        if section is not None and re.match(r'^\S', line):
            in_section = line.startswith(section + ':')
        if not in_section:
            continue
        m = pat.match(line)
        if m:
            lines[i] = f'{m.group(1)}{key}: {value}{m.group(3) or ""}'
            return True
    return False


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--python', required=True, help='absolute path of the demars env interpreter')
    ap.add_argument('--checkpoint-dir', help='directory holding the SevenNet-nano checkpoint(s)')
    ap.add_argument('--nano-spec', help='override: nano checkpoint path or sevenn name')
    ap.add_argument('--icsd-db', help='ICSD_DB_DIR (icsd.sqlite + icsd_cif.zip); omit = unchecked')
    ap.add_argument('--mp-key', help='Materials Project API key; omit = hull gate not_run')
    ap.add_argument('--accelerator', default='none', choices=['none', 'cueq', 'oeq', 'flash'])
    ap.add_argument('--out', default=os.path.join(ROOT, 'demars.yaml'))
    ap.add_argument('--force', action='store_true', help='overwrite an existing --out')
    a = ap.parse_args(argv)

    problems = []
    py = os.path.abspath(os.path.expanduser(a.python))
    if not os.access(py, os.X_OK):
        sys.exit(f'write_config.py: --python {py} is not an executable file')
    r = subprocess.run([py, '-c', 'import demars_core, yaml'], capture_output=True, text=True)
    if r.returncode:
        sys.exit(f'write_config.py: {py} cannot import demars_core/yaml -- run make_env.sh first\n'
                 + r.stderr.strip())

    ckpt_dir = os.path.abspath(os.path.expanduser(a.checkpoint_dir)) if a.checkpoint_dir else None
    nano = a.nano_spec
    if nano is None and ckpt_dir:
        cand = os.path.join(ckpt_dir, NANO_FILE)
        if os.path.isfile(cand):
            nano = cand
        else:
            problems.append(f'{NANO_FILE} not found in {ckpt_dir}; models.nano.spec left as a '
                            'placeholder -- download the SevenNet-nano checkpoint there and rerun, '
                            'or pass --nano-spec')
    elif nano is None:
        problems.append('no --checkpoint-dir: models.nano.spec left as a placeholder')
    if ckpt_dir and not os.path.isdir(ckpt_dir):
        problems.append(f'checkpoint dir {ckpt_dir} does not exist (written anyway)')

    icsd = os.path.abspath(os.path.expanduser(a.icsd_db)) if a.icsd_db else None
    if icsd:
        for f in ('icsd.sqlite', 'icsd_cif.zip'):
            if not os.path.isfile(os.path.join(icsd, f)):
                problems.append(f'{icsd}/{f} missing -- the sibling search needs both; '
                                'build_icsd.sh lays them out')

    if os.path.exists(a.out) and not a.force:
        sys.exit(f'write_config.py: {a.out} exists; pass --force to overwrite it')

    with open(EXAMPLE, encoding='utf-8') as fh:
        lines = fh.read().split('\n')
    ok = [
        _sub(lines, 'python', _yaml_scalar(py), 'paths'),
        _sub(lines, 'checkpoint_dir', _yaml_scalar(ckpt_dir), 'paths') if ckpt_dir else True,
        _sub(lines, 'icsd_db', _yaml_scalar(icsd), 'paths'),
        _sub(lines, 'api_key', _yaml_scalar(a.mp_key), 'materials_project'),
        _sub(lines, 'accelerator', a.accelerator, 'compute'),
        _sub(lines, 'nano', '{spec: %s}' % _yaml_scalar(nano), 'models') if nano else True,
    ]
    if not all(ok):
        sys.exit('write_config.py: assets/demars.yaml.example no longer has the expected keys')
    lines[0] = '# DeMARS deployment config -- generated by assets/setup/write_config.py; edit freely.'
    with open(a.out, 'w', encoding='utf-8') as fh:
        fh.write('\n'.join(lines))

    # Round-trip through the env's own loader so what was written is what the engine will read.
    chk = subprocess.run([py, '-c',
                          'import sys, yaml; c = yaml.safe_load(open(sys.argv[1])); '
                          'print(c["paths"]["python"])', a.out], capture_output=True, text=True)
    if chk.returncode or chk.stdout.strip() != py:
        sys.exit(f'write_config.py: wrote {a.out} but it does not read back cleanly:\n{chk.stderr}')

    print(f'wrote {a.out}')
    print(f'  paths.python     = {py}')
    print(f'  models.nano.spec = {nano or "(placeholder)"}')
    print(f'  paths.icsd_db    = {icsd or "null  (sibling search UNCHECKED)"}')
    print(f'  materials_project.api_key = {"set" if a.mp_key else "null  (gates.hull not_run)"}')
    for p in problems:
        print(f'  WARN {p}')
    return 1 if problems else 0


if __name__ == '__main__':
    sys.exit(main())
