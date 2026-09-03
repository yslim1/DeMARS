"""MARRecord -- the structured result of a de-averaging run.

Mirrors what mar_engine.main() computes for the distribution, but decoupled from
ICSD ids and JSON-first so a third party gets a clean object + files.
"""
import json
import os
from dataclasses import dataclass, field, asdict

import numpy as np

__all__ = ['MARRecord', 'write_custom_ensemble']

_ENGINE_INFO_KEYS = ('supercell', 'n_atoms_template', 'axes_A', 'exclusion_merge',
                     'n_groups_merged', 'couple_cut', 'coupling_signal')
_ENGINE_FLAG_KEYS = ('broad_spread_flag', 'coordination_integrity_flag',
                     'exclusion_merge')


def _ox(ev):
    return {el: (sum(v) / len(v) if isinstance(v, list) else v)
            for el, v in ev.get('oxidation_states', {}).items()}


@dataclass
class MARRecord:
    source: str
    status: str                          # 'de-averaged' | 'no-disorder' | 'error'
    calculator: str
    formula: object = None               # reduced (empirical) formula of the representative
    oxidation_states: dict = field(default_factory=dict)
    supercell: object = None
    n_atoms_template: object = None
    distribution: dict = field(default_factory=dict)
    engine: dict = field(default_factory=dict)
    self_driving: list = field(default_factory=list)
    final_MAR: object = None
    ensemble_files: object = None
    mlip: object = None                  # compute provenance: weights, hash, versions, device
    error: object = None
    _built: object = field(default=None, repr=False, compare=False)

    # ---- construction from the engine's (ev, built, info, trace) ----------
    @classmethod
    def from_engine(cls, ev, built, info, trace, source, calculator):
        ox = _ox(ev)
        if built is None:
            status = 'no-disorder' if info.get('error') == 'no disordered sites' else 'error'
            return cls(source=source, status=status, calculator=calculator,
                       oxidation_states=ox, error=info.get('error'),
                       engine={'ordered_sibling': ev.get('ordered_sibling')})

        confs, rel, val, samples, einfo = built
        ok = [s for s in samples if s.get('valid')]
        if not ok:
            return cls(source=source, status='error', calculator=calculator,
                       oxidation_states=ox, error='no valid relaxations')
        from ._engine.mar_engine import bracket_margins, ship_candidates
        Es = sorted(s['E_per_atom'] for s in ok)
        # The DISTRIBUTION describes every valid sample -- bounding it is what the bracket probes are
        # for. The REPRESENTATIVE comes from the ship candidates, which excludes them: a designed
        # extreme winning take-lowest is an artifact, not the phase. One definition, shared with
        # `mar_engine`'s own distribution, `api._final_omni` and the written `representative.*`.
        ship = ship_candidates(samples)
        lo = min(ship, key=lambda x: x['E_per_atom'])
        hi = max(ok, key=lambda x: x['E_per_atom'])
        widx = samples.index(lo)                         # representative decoration
        try:
            formula = rel[widx].get_chemical_formula(mode='hill', empirical=True) if val[widx] else None
        except Exception:
            formula = None
        dist = {
            'n_relaxed': len(ok), 'n_total': len(samples),
            'ground_E_per_atom': Es[0], 'highest_E_per_atom': Es[-1],
            'spread_meV': round(1000 * (Es[-1] - Es[0]), 2),
            'std_meV': round(1000 * float(np.std([s['E_per_atom'] for s in ok])), 2),
            'energies_sorted': Es,
            'representative': lo,            # the TYPICAL/lowest SHIP CANDIDATE = the MAR
            'highest': hi,
            'n_ship_candidates': len(ship),
            'bracket_probes': bracket_margins(samples),
        }
        eng = {k: einfo[k] for k in _ENGINE_INFO_KEYS if k in einfo}
        eng.update({k: einfo[k] for k in _ENGINE_FLAG_KEYS if k in einfo})
        rec = cls(source=source, status='de-averaged', calculator=calculator,
                  formula=formula, oxidation_states=ox, supercell=einfo.get('supercell'),
                  n_atoms_template=einfo.get('n_atoms_template'),
                  distribution=dist, engine=eng, self_driving=trace)
        rec._built = built
        return rec

    # ---- the deliverable ensemble (energy-marked, OVITO-native) -----------
    def write_ensemble(self, out_dir, final_struct=None):
        if self._built is None:
            return None
        from ase.io import write as asewrite
        confs, rel, val, samples, einfo = self._built
        os.makedirs(out_dir, exist_ok=True)
        frames = []
        for i, smp in enumerate(samples):
            if smp.get('valid'):
                at = rel[i].copy()
                at.info.update(label=smp.get('label'), mode=smp.get('mode'),
                               E_per_atom=smp.get('E_per_atom'),
                               charge=smp.get('charge'))
                frames.append((smp['E_per_atom'], at))
        frames.sort(key=lambda x: x[0])
        paths = {}
        asewrite(f'{out_dir}/ensemble.xyz', [a for _, a in frames], format='extxyz')
        # the written representative must be the sample `distribution.representative` names and the
        # one the final tier relaxed -- a bracket probe stays in the ensemble but never becomes the
        # deliverable by winning take-lowest.
        from ._engine.mar_engine import ship_candidates
        _ship = {s.get('label') for s in ship_candidates(samples)}
        _rep = next((a for _, a in frames if a.info.get('label') in _ship), frames[0][1])
        asewrite(f'{out_dir}/representative.cif', _rep, format='cif')
        asewrite(f'{out_dir}/representative.vasp', _rep, format='vasp', sort=True)
        paths.update(ensemble=f'{out_dir}/ensemble.xyz', n_frames=len(frames),
                     representative=f'{out_dir}/representative.cif')
        if final_struct is not None:
            asewrite(f'{out_dir}/representative_final.cif', final_struct, format='cif')
            paths['representative_final'] = f'{out_dir}/representative_final.cif'
        self.ensemble_files = paths
        # NOT `record.json`: that name belongs to the stage-6 mar-1.0 record, which is written into
        # this same tree by the tools layer and has a different schema (mechanism / gates / review).
        # Two files with one name meant whichever ran last defined what "the record" was -- and a
        # consumer reading the wrong schema gets nulls where it expects judgement. The name says
        # what produced it instead. (The dict key stays 'record' -- it is a published contract that
        # engine.json consumers read.)
        dea = f'{out_dir}/deaverage_output.json'
        with open(dea, 'w', encoding='utf-8') as fh:
            json.dump(self.to_dict(), fh, indent=1, default=str)
        paths['record'] = dea
        return paths

    # ---- serialisation ----------------------------------------------------
    def to_dict(self):
        d = asdict(self)
        d.pop('_built', None)
        return d

    def summary(self):
        if self.status != 'de-averaged':
            return f'[{self.status}] {self.source}  ({self.error or "ordered"})'
        d = self.distribution
        rep = d['representative']
        return (f'MAR de-averaged  ({self.calculator})\n'
                f'  supercell {self.supercell} -> {self.n_atoms_template} atoms\n'
                f'  ensemble: {d["n_relaxed"]}/{d["n_total"]} relaxed  '
                f'spread {d["spread_meV"]} meV/at  std {d["std_meV"]} meV/at\n'
                f'  representative: {rep.get("label")} '
                f'E={rep.get("E_per_atom")} eV/at  q={rep.get("charge")}  '
                f'min_dist={rep.get("min_dist_A")} A  n_atoms={rep.get("n_atoms")}'
                + _final_line(self.final_MAR))


def _final_line(fm):
    """-> the summary's FINAL line. Says 'not available' out loud rather than printing E=None.

    Three states, and a reader must be able to tell them apart: absent = never requested,
    available False = requested and could not run, otherwise = it ran. See api._final_unavailable.
    """
    if not fm:
        return ''
    if fm.get('available') is False:
        return f'\n  FINAL (omni-mpa): NOT AVAILABLE -- {fm.get("reason")}'
    return f'\n  FINAL (omni-mpa): E={fm.get("E_final_per_atom")} eV/at'


def _sample(at, e_total, ox, label, mode):
    """One ensemble member, in the SAME schema mar_engine writes -- see its relax loop."""
    from collections import Counter

    from ase.geometry import get_distances

    pos = at.get_positions()
    D = get_distances(pos, pos, cell=np.array(at.get_cell()), pbc=True)[1]
    np.fill_diagonal(D, 9e9)
    syms = at.get_chemical_symbols()
    ij = np.unravel_index(int(D.argmin()), D.shape)
    smp = {'label': label, 'mode': mode, 'valid': True,
           'E_per_atom': round(float(e_total) / len(at), 4),
           'charge': round(float(sum(ox.get(e, 0.0) for e in syms)), 2),
           'min_dist_A': round(float(D.min()), 2),
           'min_dist_pair': '-'.join(sorted([syms[ij[0]], syms[ij[1]]])),
           'n_atoms': len(at), 'composition': dict(Counter(syms))}
    try:
        from ._engine.mar_engine import config_descriptors
        smp['descriptors'] = config_descriptors(at)
    except Exception:                    # a descriptor is a nicety; the contract is not
        pass
    return smp


def write_custom_ensemble(out_dir, source, calculator, relaxed, E, valid, *,
                          oxidation_states=None, labels=None, mode='custom-build',
                          final_struct=None, final_MAR=None, mlip=None,
                          engine_info=None, self_driving=None):
    """Write the stage-3 file contract for a build the stock engine could not produce. -> MARRecord

    `relaxed, E, valid` is exactly the relaxer's own (rel, E, val) triple, reordered to read as
    "these structures, these energies, these are the good ones".

    WHY THIS IS NOT OPTIONAL. When `deaverage()` cannot handle a structure, the analyst writes a
    bespoke build script -- and that script inherits none of the engine's guarantees. On ICSD
    one such script printed its energies to stdout and saved nothing, so a finished 21-minute
    relaxation left no ensemble, no representative, and nothing to run the gates against; the
    whole relaxation had to be repeated. Printing is not persisting, and the difference is
    invisible until the structures are needed.

    So a custom build writes the same three files under the same names as the stock engine --
    `ensemble.xyz`, `representative.cif`/`.vasp`, `deaverage_output.json` -- with samples in the
    engine's own schema. Downstream (gates, stage-6 record, gallery, the reference check) then
    cannot tell the two apart, which is the point: a custom build is a different way to REACH the
    ensemble, not a different kind of result.

    Pass `mlip=` (the provenance dict from `api._make_relax`) so the record still says which
    weights produced it. Without it the record is silent about its own potential.
    """
    ox = dict(oxidation_states or {})
    relaxed, E, valid = list(relaxed), list(E), [bool(v) for v in valid]
    samples = []
    for i, at in enumerate(relaxed):
        label = labels[i] if labels else f'custom{i}'
        samples.append(_sample(at, E[i], ox, label, mode) if valid[i]
                       else {'label': label, 'mode': mode, 'valid': False, 'E_per_atom': None})
    einfo = dict(engine_info or {})
    rec = MARRecord.from_engine({'oxidation_states': ox},
                                (relaxed, relaxed, valid, samples, einfo),
                                einfo, list(self_driving or []), source, calculator)
    rec.mlip, rec.final_MAR = mlip, final_MAR
    rec.write_ensemble(out_dir, final_struct=final_struct)
    return rec
