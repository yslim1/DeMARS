"""Command-line entry points:

    demars run <structure> [opts]        de-average one disordered structure
    demars gallery <results_dir> [opts]  build a browsable gallery of results

ICSD-free: arguments are files/folders the caller owns, never ids.
"""
import argparse
import sys


def _add_run(sub):
    r = sub.add_parser('run', help='de-average one disordered structure file')
    r.add_argument('structure', help='CIF / POSCAR / xyz path (disorder preserved from CIF)')
    r.add_argument('--calculator', default='sevennet',
                   help="'sevennet' | '7net-nano' | '7net-omni' | <checkpoint.pth> "
                        '(ASE-instance backends are Python-API only)')
    r.add_argument('-o', '--out', default=None, help='output dir for ensemble + deaverage_output.json')
    r.add_argument('--n-samples', type=int, default=30)
    r.add_argument('--min-cell', type=float, default=15.0, help='min supercell edge (A)')
    r.add_argument('--rounds', type=int, default=3, help='self-driving rounds')
    r.add_argument('--no-final', action='store_true', help='skip the omni-mpa final tier')
    r.add_argument('--d3', action='store_true', help='enable D3 (default OFF -- DeMARS uses no dispersion)')


def _add_gallery(sub):
    g = sub.add_parser('gallery', help='build a browsable web gallery from result folders')
    g.add_argument('results', help='directory of de-average result folders (each with deaverage_output.json)')
    g.add_argument('-o', '--out', default=None, help='gallery output dir (default <results>/_gallery)')
    g.add_argument('--title', default='DeMARS results', help='page title')
    g.add_argument('--serve', action='store_true', help='serve the gallery over HTTP after building')
    g.add_argument('--port', type=int, default=8000)


def _cmd_run(args):
    from . import deaverage
    # keep stdout PURE JSON: route the engine/backend chatter to stderr during the run.
    _real = sys.stdout
    sys.stdout = sys.stderr
    try:
        rec = deaverage(args.structure, calculator=args.calculator, n_samples=args.n_samples,
                        min_cell=args.min_cell, max_rounds=args.rounds, final=not args.no_final,
                        d3=args.d3, out_dir=args.out)
    finally:
        sys.stdout = _real
    print(rec.summary(), file=sys.stderr)
    import json
    print(json.dumps(rec.to_dict(), indent=1, default=str))
    return 0 if rec.status != 'error' else 1


def _cmd_gallery(args):
    from .gallery import build_gallery, serve
    out = build_gallery(args.results, out_dir=args.out, title=args.title)
    print(f'gallery built: {out}/index.html', file=sys.stderr)
    if args.serve:
        serve(out, port=args.port)
    else:
        print(f'view it:  demars gallery {args.results} --serve   '
              f'(or: python -m http.server --directory {out})', file=sys.stderr)
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog='demars', description='De-average disordered crystals into MARs.')
    sub = ap.add_subparsers(dest='cmd', required=True)
    _add_run(sub)
    _add_gallery(sub)
    args = ap.parse_args(argv)
    return _cmd_gallery(args) if args.cmd == 'gallery' else _cmd_run(args)


if __name__ == '__main__':
    raise SystemExit(main())
