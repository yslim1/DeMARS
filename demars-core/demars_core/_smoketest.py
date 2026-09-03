"""End-to-end smoke test: de-average a disordered structure built from scratch
(NO ICSD anywhere), proving the ICSD-free entry point + pluggable calculator.

    python -m demars_core._smoketest

Constructs a rock-salt (Mg,Ni)O solid solution -- a 50/50 mixed cation site,
oxidation states attached so evidence() reads them directly -- then de-averages
it with the native SevenNet path. Small cell + few samples so it runs in ~1 min.
"""
import sys
import tempfile


def _disordered_rocksalt():
    from pymatgen.core import Structure, Lattice, Species
    a = 4.20
    latt = Lattice.cubic(a)
    # conventional rock salt: 4 cation (0,0,0)+fcc, 4 anion (1/2,0,0)+fcc
    cation = {Species('Mg', 2): 0.5, Species('Ni', 2): 0.5}      # mixed site
    anion = {Species('O', -2): 1.0}
    coords = [[0, 0, 0], [0, .5, .5], [.5, 0, .5], [.5, .5, 0],           # cations
              [.5, .5, .5], [.5, 0, 0], [0, .5, 0], [0, 0, .5]]           # anions
    species = [cation] * 4 + [anion] * 4
    return Structure(latt, species, coords)


def main():
    import demars_core as D
    struct = _disordered_rocksalt()
    print(f'[smoke] built disordered {struct.composition.reduced_formula} '
          f'({len(struct)} sites, mixed Mg/Ni cation)', flush=True)

    out = tempfile.mkdtemp(prefix='demars_smoke_')
    # small + fast: 8 A cell, 6 samples, 1 round, skip omni-final. Production
    # defaults are min_cell=15, n_samples=30, final=True.
    rec = D.deaverage(struct, calculator='7net-nano', n_samples=6, min_cell=8.0,
                      max_rounds=1, final=False, out_dir=out)
    print('\n' + rec.summary(), flush=True)
    print(f'\n[smoke] status={rec.status}  ensemble_files={rec.ensemble_files}', flush=True)
    ok = rec.status == 'de-averaged' and rec.distribution.get('n_relaxed', 0) >= 1
    print('[smoke] PASS' if ok else '[smoke] FAIL', flush=True)
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
