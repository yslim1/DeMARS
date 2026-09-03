"""Polyanion / rigid-unit CONNECTIVITY gate.

The charge & fidelity gates prove COMPOSITION (charge-neutral, right element counts) and
OCCUPANCY faithfulness -- they are BLIND to which centre a ligand attaches to. A build can
pass both and still ship a chemically broken unit (an SO3 where an SO4 was intended: right
O count, one O on the wrong S / floating free). This gate closes that hole: it checks that
every unit centre carries its expected ligand count.

GENERAL BY CONSTRUCTION -- no per-compound table. Centres, bonds and the expected
coordination all come from `_engine.mar_engine.former_coordination`:

  * which elements can centre a unit -> COORD_FORMERS (the same set the engine's
    coordination-integrity report and the rigid-unit constraint use);
  * a bond is `d < (r_cov_i + r_cov_j) * tol`, so B-O (~1.4 A) and W-O (~2.2 A) are both
    found without a table of absolute cutoffs;
  * the expected coordination is the MODE observed in THIS structure, so BO3 -> 3,
    SO4/PO4/SiO4/ClO4 -> 4, MoO6 -> 6, NH4 -> 4 fall out of the geometry -- taken per
    ROLE, not per element (`mar_engine.centre_expectations`): one element can centre two
    unit types in one structure, and cyanide C beside methylated-cation C cost 160 false
    defects on a cyanide/methylated-cation entry when they shared one expectation.

Sharing one definition with the constraint and the descriptors means the three cannot
disagree about what "a unit" is. The role split refines the EXPECTATION only -- the bond
rule, the former set and the prevalence gate stay where they were, so the constraint and
the descriptors are untouched (moving them would move relaxation energies).

This lives in the package rather than in the CLI that calls it: it is the one gate that
catches what charge and fidelity cannot, so a standalone `demars-core` install needs it,
and the tests need it without reaching outside the package to find it.
"""
from collections import defaultdict

from ._engine.mar_engine import centre_expectations

__all__ = ['audit_frame']


def audit_frame(atoms, tol=1.25, expect=None):
    """-> (summary, defects) for one frame.

    A centre deviating from its element's target coordination is a defect. Formers that
    ended up with NO ligand bond at all are included at coordination 0 -- a fully stripped
    centre is the worst outcome and must not vanish from the audit just because it has no
    bonds left to detect. Elements that never coordinate a ligand anywhere in the structure
    are not units here and are skipped entirely (a chloride's Cl is not a ClO4 centre).

    `expect` ({'S': 4, ...}) ENFORCES a coordination instead of inferring it -- for a build
    that is supposed to be tetrahedral but relaxed into a mixture, where the mode would
    follow the majority and hide the defect.
    """
    ce = centre_expectations(atoms, tol=tol, expect=expect)
    bonds, expected, roles = ce['bonds'], ce['expected'], ce['roles']
    med, ligand_role = ce['pair_median'], ce['ligand_role']
    syms = atoms.get_chemical_symbols()

    # Grouped by ROLE, not by element. One element can centre two different unit types in the same
    # structure -- cyanide C ({N}, CN 1) beside methylated-cation C ({N,H}, CN 4) -- and judging both
    # against one element-wide mode produced 160 false defects on such an entry. `centre_expectations`
    # owns that split (and the ligand-role exclusion that goes with it); this loop only reports it.
    # A single-role element keeps the plain element label, so structures with one unit type per
    # element -- 14 of the frozen run's 15 -- read exactly as they did before.
    by_role = defaultdict(list)
    for i in expected:
        by_role[roles[i]].append((int(i), len(bonds.get(i, ()))))

    summary, defects = {}, []
    for label in sorted(by_role):
        items = sorted(by_role[label])
        coords = [c for _, c in items]
        el = syms[items[0][0]]
        target = int(expected[items[0][0]])      # constant within a role, by construction
        ligand_els = sorted({syms[j] for i, _c in items for j, _d in bonds.get(i, ())})
        summary[label] = {
            'element': el, 'role': label,
            'ligands': ligand_els, 'tol': tol, 'expected': target,
            'expected_from': 'expect override' if (expect or {}).get(el) is not None else 'observed mode',
            'n_centres': len(coords), 'n_ok': sum(1 for c in coords if c == target),
            'coord_hist': {int(c): coords.count(c) for c in sorted(set(coords))},
            'median_bond_A': {f'{el}-{le}': round(med[(el, le)], 3)
                              for le in ligand_els if (el, le) in med},
        }
        for i, c in items:
            if c != target:
                defects.append({'centre': el, 'role': label, 'atom_index': i,
                                'coordination': c, 'expected': target})

    # WHAT THIS GATE DID NOT LOOK AT. Centres come from COORD_FORMERS, a fixed 21-element set, so a
    # cation outside it is never examined and a clean verdict says nothing about it. One phosphate
    # mineral read as "connectivity clean apart from 9/48 Mn" while 16 of 62 Fe were under-coordinated
    # -- Fe is not in the set, and the mechanism under test lived on exactly that sublattice.
    # Listing the skipped elements does not make the gate more capable; it stops "clean" from being
    # read as "every cation checked", which is the claim that was actually wrong.
    from ._engine.mar_engine import COORD_ANIONS
    examined = set(summary)
    examined_els = {v['element'] for v in summary.values()}
    skipped = sorted({s for s in syms} - examined_els - set(COORD_ANIONS))
    summary['_coverage'] = {
        'examined_centres': sorted(examined),
        'examined_elements': sorted(examined_els),
        'not_examined': skipped,
        # Atoms of a dual-membership element (N, S, Se, Te, halogens are in COORD_FORMERS AND
        # COORD_ANIONS) that carry no ligands of their own but ARE a ligand of another centre. They
        # were being re-attached as 0-coordination "fully stripped centres" -- 96 cyanide N on ICSD
        # that entry -- so they are excluded, and reported here rather than dropped in silence.
        'ligand_role': ligand_role,
        'note': ('elements listed in not_examined are absent from COORD_FORMERS, so their '
                 'coordination was never checked -- a clean result does not cover them')
        if skipped else 'every non-ligand element present was examined',
    }
    return summary, defects
