"""Orbit-level structural-disorder classification, after Antypov, Collins, Dyer, Claridge &
Rosseinsky, "Classification and statistical analysis of structural disorder in crystalline
materials", J. Appl. Cryst. (2025) 58, 659-677 (Table 1).

Pure CIF-derived -- no MLIP, no relaxation, no external service. Each crystallographic orbit is
labelled O / S / V / P / SV / SP / VP / SVP from:
 - number of distinct elements on the orbit's site,
 - total site occupancy,
 - whether sites INTERSECT (r_ij < max(1, 0.5*(R_i+R_j)), Shannon ionic radius at the CIF
   oxidation state, atomic-radius fallback) -- internal (within one orbit) or external (between
   orbits) -- and the summed occupancy of the resulting COMBINED site.

A compound's "disorder set" is the set of unique orbit labels (e.g. {O, V, P}).

This reports the PUBLISHED descriptor; it is NOT the DeMARS mechanism call (A-F). The record keeps
it under `disorder_descriptor`, and `strategy.md` reads `no_full_backbone` out of it as a triage
signal -- so a build that cannot compute it says `{"available": false, "reason": ...}` rather than
leaving `null`, which in that schema means an optional step did not run.

Usage:  python -m demars_core.disorder_class <structure.cif>
"""
import json
import sys
import warnings

__all__ = ['classify', 'classify_text']

TOL = 0.011                       # rounding tolerance: occ >= 1-TOL counts as "1"


def _radius(el, ox):
    """smallest Shannon ionic radius at oxidation state ox; atomic-radius fallback; 1.0 last resort."""
    from pymatgen.core import Element, Species
    try:
        sp = Species(el, int(round(ox)))
        r = sp.ionic_radius
        if r:
            return float(r)
    except Exception:
        pass
    try:
        r = Element(el).atomic_radius
        if r:
            return float(r)
    except Exception:
        pass
    return 1.0


def _site_occ(site):
    return sum(oc for _, oc in site.species.items())


def _label(ne_self, occ_self, internal, external, comb_occ, comb_ne):
    def close1(x):
        return x >= 1 - TOL
    if not internal and not external:                 # lines 1-4: no intersection
        if ne_self == 1:
            return 'O' if close1(occ_self) else 'V'
        return 'S' if close1(occ_self) else 'SV'
    if internal and not external:                     # lines 5-8: internal only -> combined occ
        if ne_self == 1:
            return 'P' if close1(comb_occ) else 'VP'
        return 'SP' if close1(comb_occ) else 'SVP'
    if comb_ne == 1:                                  # lines 9-12: external -> combined elems + occ
        return 'P' if close1(comb_occ) else 'VP'
    return 'SP' if close1(comb_occ) else 'SVP'


def _structure_from_text(text):
    """Tolerant CIF -> Structure for classification. Refinements frequently report a site occupancy
    that sums to slightly >1 (rounding, e.g. 1.0003-1.04); pymatgen's default occupancy_tolerance=1.0
    then raises 'Invalid CIF file with no structures!', leaving the disorder-set column empty.
    occupancy_tolerance=1.15 rescales those tiny overages to 1.0 while PRESERVING the partial
    occupancies (the disorder), so the orbit O/S/V/P labels stay faithful -- the same tolerance
    stage 1a parses with. Falls back to the strict loader if the tolerant parse fails."""
    from pymatgen.io.cif import CifParser
    try:
        return CifParser.from_str(text, occupancy_tolerance=1.15).parse_structures(primitive=False)[0]
    except Exception:
        return CifParser.from_str(text).parse_structures(primitive=False)[0]


def classify_text(text, oxidation_states=None, iid=None):
    """The descriptor from CIF TEXT. `oxidation_states` is the stage-1a bundle's own map (values may
    be a list for a mixed-valence element -- averaged here); omitted, it is derived from the same
    text, which costs a second parse. `iid` only stamps the result."""
    warnings.filterwarnings('ignore')
    from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
    if oxidation_states is None:
        from ._engine import mar_evidence as MEV
        oxidation_states = MEV.evidence_from_text(text, iid=iid,
                                                  search_siblings=False).get('oxidation_states')
    oxi = {k: (sum(v) / len(v) if isinstance(v, list) else v)
           for k, v in (oxidation_states or {}).items()}
    s = _structure_from_text(text)
    n = len(s)
    rad = []
    for site in s:
        rs = [_radius(sp.symbol, oxi.get(sp.symbol, 0) or 0) for sp in site.species]
        rad.append(max(rs) if rs else 1.0)
    # intersection graph on the full cell (pymatgen distance_matrix = min-image distances)
    D = s.distance_matrix
    adj = [set() for _ in range(n)]
    for i in range(n):
        for j in range(i + 1, n):
            if D[i, j] < max(1.0, 0.5 * (rad[i] + rad[j])):
                adj[i].add(j)
                adj[j].add(i)
    # connected components = combined sites
    comp = [-1] * n
    c = 0
    for i in range(n):
        if comp[i] < 0:
            stack = [i]
            comp[i] = c
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if comp[v] < 0:
                        comp[v] = c
                        stack.append(v)
            c += 1
    members = {}
    for i in range(n):
        members.setdefault(comp[i], []).append(i)
    # crystallographic orbits via symmetry; fall back to per-site
    try:
        groups = SpacegroupAnalyzer(s, symprec=1e-2).get_symmetrized_structure().equivalent_indices
    except Exception:
        groups = [[i] for i in range(n)]
    labels = []
    for g in groups:
        i0 = g[0]
        gi = set(g)
        elems = {sp.symbol for sp in s[i0].species}
        ne_self = len(elems)
        occ_self = _site_occ(s[i0])
        internal = any((j in gi) for i in g for j in adj[i])
        external = any((j not in gi) for i in g for j in adj[i])
        mem = members[comp[i0]]
        comb_occ = sum(_site_occ(s[k]) for k in mem)
        comb_elems = {sp.symbol for k in mem for sp in s[k].species}
        labels.append(_label(ne_self, occ_self, internal, external, comb_occ, len(comb_elems)))
    multiset = sorted(labels)
    disorder_set = sorted(set(labels))
    # no_full_backbone: EVERY orbit is vacancy-bearing (label contains 'V') -> no fully-occupied
    # (O / full-S / full-P) scaffold anywhere. A triage flag for "framework unclear": the deposited
    # cell is usually a high-symmetry average of a correlated/ordered superstructure (a defect
    # compound whose stoichiometry mandates a fixed cation-vacancy fraction) or a cell-doubling
    # artifact masking a full backbone (a manganite whose formula wants the site FULL while the CIF
    # lists it at occ 0.5) -> do NOT random-decorate. Corroborated geometrically by high ensemble
    # cell-scatter (see _engine.mar_record.cell_scatter, which ranks the same two cases at the top).
    no_full_backbone = bool(labels) and all('V' in lab for lab in labels)
    return {"icsd_id": iid, "orbit_labels": labels, "multiset": multiset,
            "disorder_set": disorder_set, "n_orbits": len(labels),
            "no_full_backbone": no_full_backbone}


def classify(source, oxidation_states=None, iid=None):
    """The descriptor for any accepted input form -- a CIF path, a structure-file path, CIF text, a
    pymatgen Structure or an ase.Atoms. The FILE is the key here, not an ICSD id."""
    from .io import to_cif_text
    return classify_text(to_cif_text(source), oxidation_states=oxidation_states, iid=iid)


if __name__ == '__main__':
    print(json.dumps(classify(sys.argv[1])))
