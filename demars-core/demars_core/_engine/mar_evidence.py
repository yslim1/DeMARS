"""MAR pipeline stage ①a -- EVIDENCE bundle.

Emits NEUTRAL signatures mined from the CIF itself (principle 8: mine the CIF before any web
search). It reports facts, never conclusions: it says "sites 2,3 are 1.02 A apart, different
orbits" and "Li 0.857 / Mn 0.143 share one position" -- it does NOT say "split site",
"slaved", or "antisite". The screener (①b) and analyst (②) draw the mechanism call.

Deliberately lightweight: pure CIF parse + pymatgen geometry + a composition-matched ordered-
sibling search. No MLIP / torch / GPU import, so it is cheap enough to run on every entry.

Usage:  python mar_evidence.py <icsd_id> [--summary]
        stdout = JSON evidence bundle ; --summary also prints a human digest to stderr.
"""
import sys, os, json, re, shlex, warnings
warnings.filterwarnings('ignore')
import numpy as np
from collections import defaultdict, Counter
from pymatgen.io.cif import CifParser
from ase.geometry import get_distances

COUPLE_A = 2.6        # cross-orbit contact below this = a coupling/slaving SIGNATURE (neutral)
PARTIAL = 0.98        # occupancy sum below this on a position = vacancy signature
LIGHT = {'H', 'B', 'C', 'N', 'O', 'F'}   # light ligand atoms of molecular/orientational units
R_ALT = 1.6           # same-element ligand sites closer than this = orientation alternates (can't coexist)

# ---- slaved-pair / orientation-unit detection (ported from the research constructor's build_entry;
# its skill kept that path alive for exactly this detection on tricky cases, and the enumeration
# diagnostic REFUSED to run when either count was non-zero -- independent per-site decoration breaks
# a bonded or orientational unit). Both emit neutral facts; taxonomy class E is the analyst's call.
SLAVE_SCORE = {'H': 3, 'O': 2}   # higher = more likely to be the slaved FOLLOWER of a heavier parent
SLAVE_BOND = (0.3, 1.9)          # parent-follower separation that counts as the bond carrying it
SLAVE_OCC_TOL = 0.15             # |occ(parent trigger species) - occ(follower orbit)| to pair them
ORIENT_OCC = (0.25, 0.75)        # a partial orbit outside this is a vacancy sublattice, not a rotor
ORIENT_CUT = 3.4                 # anchor-to-alternative and alternative-to-alternative window

# ---- PROVENANCE: how a derived parameter got its value -----------------------------------------
# The engine->analyst interface used to be one-way and advisory: a free-text `basis` string that
# nothing downstream could act on, and no way to say "I could not derive this". Worse, one string
# covered several different situations -- `couple_cut`'s "no contact in evidence" was TRUE for one
# set-A entry and FALSE for three (contacts existed; no rule matched them).
#
# A CLOSED state set makes that machine-readable. Keep it closed: a free-text state is the bug this
# replaces. `confident` is derived, so a consumer needs to know only that one boolean.
PROV_STATES = (
    'derived',        # the data determined the value
    'overridden',     # the analyst set it explicitly (--same-excl, --couple-cut)
    'vacuous',        # the derivation ran, zero candidates, and none were expected -> default is right
    'ambiguous',      # it ran but the evidence is weak (the old "; CHECK" suffix)
    'out_of_window',  # data exists but lies outside the range the derivation inspects  [defect (4)]
    'unmatched',      # candidates exist but no rule matched them                        [defect (5)]
    'not_run',        # the derivation never executed (capability absent, or it errored)
)
# NOT confident: ambiguous / out_of_window / unmatched / not_run. `vacuous` IS confident and that
# distinction carries the design: with a coarser derived/defaulted/underivable split, `couple_cut`
# comes back unconfident on 4 of 4 set-A entries and the signal is useless.
PROV_CONFIDENT = ('derived', 'overridden', 'vacuous')


def provenance(state, detail, evidence=None, source=None):
    """Build the standard provenance block for one derived parameter.

    `evidence` carries the NUMBERS an analyst needs to decide, so they do not have to re-mine
    evidence.json. `source` says where the value came from when it was not derived.
    """
    if state not in PROV_STATES:
        raise ValueError(f'unknown provenance state {state!r}; must be one of {PROV_STATES}')
    return {'state': state, 'confident': state in PROV_CONFIDENT, 'detail': detail,
            'evidence': evidence or {}, 'source': source}


def provenance_summary(blocks):
    """Roll up {name: provenance-block} so a gate can check one place instead of walking the tree.

    Nested dicts of blocks (exclusion_merge is per element) flatten to "parent.child" names.
    """
    flat = {}
    for name, blk in (blocks or {}).items():
        if isinstance(blk, dict) and 'confident' in blk:
            flat[name] = blk
        elif isinstance(blk, dict):
            for sub, b in blk.items():
                if isinstance(b, dict) and 'confident' in b:
                    flat[f'{name}.{sub}'] = b
    bad = sorted(n for n, b in flat.items() if not b['confident'])
    return {'n_unconfident': len(bad), 'unconfident': bad,
            'states': {n: b['state'] for n, b in sorted(flat.items())}}

# ---- reuse the ordered-sibling search from mar_ordered_lookup (lightweight: db + zip only) --
_sib = None
try:
    _src = open(os.environ.get('DEMARS_ORDERED_LOOKUP', '')).read()
    _ns = {'__name__': 'mar_ordered_lib'}
    exec(_src[:_src.index('\nledger = json.load')], _ns)
    _sib = _ns                                    # has is_ordered, db, z, Composition
except Exception as _e:
    _sib = None

# ---------------------------------------------------------------- CIF text parsing helpers
def _unq(v):
    if v is None: return None
    v = v.strip().strip("'").strip('"').strip()
    return None if v in ('', '.', '?') else v

def _num(v):
    v = _unq(v)
    if v is None: return None
    m = re.match(r'^[-+]?[0-9]*\.?[0-9]+', v.replace('(', ' ').split()[0] if v else '')
    try: return float(m.group(0)) if m else None
    except Exception: return None

def cif_scalar(lines, key):
    """value of a `_key value` line; handles next-line quoted value and ;...; text blocks."""
    for idx, l in enumerate(lines):
        st = l.strip()
        if st == key or st.startswith(key + ' ') or st.startswith(key + '\t'):
            rest = st[len(key):].strip()
            if rest:
                return _unq(rest)
            j = idx + 1
            while j < len(lines) and not lines[j].strip(): j += 1
            if j >= len(lines): return None
            nx = lines[j].strip()
            if nx == ';':
                buf, j = [], j + 1
                while j < len(lines) and lines[j].strip() != ';':
                    buf.append(lines[j].strip()); j += 1
                return ' '.join(x for x in buf if x).strip() or None
            if nx.startswith('_') or nx == 'loop_': return None
            return _unq(nx)
    return None

def cif_loops(lines):
    """yield (headers, [rows]) for every loop_ block (quote-aware tokenisation)."""
    i, n = 0, len(lines)
    while i < n:
        if lines[i].strip() == 'loop_':
            i += 1; hdr = []
            while i < n and lines[i].strip().startswith('_'):
                hdr.append(lines[i].strip()); i += 1
            rows = []
            while i < n:
                s = lines[i].strip()
                if (not s) or s == 'loop_' or s.startswith(('_', '#', 'data_', ';')):
                    break
                try: toks = shlex.split(s)
                except ValueError: toks = s.split()
                rows.append(toks); i += 1
            yield hdr, rows
        else:
            i += 1

def loop_with(loops, tag):
    for hdr, rows in loops:
        if tag in hdr: return hdr, rows
    return None, None

def _el(sym):
    m = re.match(r'^([A-Z][a-z]?)', sym or '')
    e = m.group(1) if m else sym
    return 'H' if e in ('D', 'T') else e

def _occ_orbits(s):
    """-> {signature: [(site_index, position, {el: occ})]} plus {signature: total occupancy}.

    Grouped by occupancy SIGNATURE, the same key the evidence bundle's `disorder.orbits` uses, so a
    count reported here refers to an orbit the analyst can find in that list. One entry per POSITION:
    the research constructor first merged mutually-exclusive same-element alternates into one group,
    and without that merge a split site is counted as its separate images. Where that matters the
    orbit is a split site and `rigid_unit_signal` / `--same-excl` already say so.
    """
    orb, occ_tot = defaultdict(list), {}
    for i, site in enumerate(s):
        d = {}
        for sp, o in site.species.items():
            d[sp.symbol] = d.get(sp.symbol, 0.0) + o
        sig = tuple(sorted((el, round(o, 3)) for el, o in d.items()))
        orb[sig].append((i, site.coords, d))
        occ_tot[sig] = sum(o for _el, o in sig)
    return orb, occ_tot


def slaved_units(s):
    """Parent orbit -> FOLLOWER orbit pairs: a partial orbit whose atoms only exist where a partial
    parent orbit is occupied (a hydroxide H following its O, a BO3 oxygen following its boron).

    The criteria are the research constructor's: the follower must outrank the parent on
    `SLAVE_SCORE` (H follows O follows a metal, never the reverse); the parent must carry a species
    whose occupancy MATCHES the follower orbit's within `SLAVE_OCC_TOL` and is itself partial -- a
    full parent encodes no choice, which is the false-positive guard that stops a 0.88-occupied
    orbit being read as slaved to a full one; and EVERY parent position must find a follower inside
    `SLAVE_BOND`, since a parent left without one means the pairing is not what holds the orbit.

    Follower count per parent comes from occupancy bookkeeping, so a carbonate's three oxygens are
    reported as k=3 rather than the nearest one only.

    Non-empty means per-site decoration is NOT valid on those orbits -- place the unit, not the atoms.
    """
    orb, occ_tot = _occ_orbits(s)
    if len(orb) < 2:
        return []
    cell = np.array(s.lattice.matrix)

    def score(sig):
        return max((SLAVE_SCORE.get(el, 0) for el, _o in sig), default=0)

    out = []
    partials = [sig for sig in orb if occ_tot[sig] < 0.99]
    for S in partials:
        if score(S) == 0:
            continue
        for P in orb:
            if P == S or score(P) >= score(S):
                continue
            trig = [el for el, o in P if abs(o - occ_tot[S]) < SLAVE_OCC_TOL and o < 0.99]
            if not trig:
                continue
            occ_t = next(o for el, o in P if el == trig[0])
            k_fol = max(1, int(round((len(orb[S]) * occ_tot[S]) /
                                     max(1e-9, len(orb[P]) * occ_t))))
            sel = max(S, key=lambda x: x[1])[0]              # the follower orbit's majority element
            spos = np.array([p for _i, p, d in orb[S] if sel in d])
            if not len(spos):
                continue
            ppos = np.array([p for _i, p, _d in orb[P]])
            D = get_distances(ppos, spos, cell=cell, pbc=True)[1]
            lo, hi = SLAVE_BOND
            n_matched = int(sum(1 for r in range(len(ppos)) if ((D[r] >= lo) & (D[r] <= hi)).any()))
            if n_matched == len(ppos):
                out.append({'follower_signature': [list(t) for t in S],
                            'parent_signature': [list(t) for t in P],
                            'trigger_element': trig[0],
                            'n_parent_positions': len(ppos),
                            'followers_per_parent': k_fol,
                            'follower_element': sel})
                break
    return out


def orientation_units(s, slaved=None):
    """A partial single-element orbit whose positions cluster into DISCRETE alternatives around
    full-occupancy anchors of the SAME element -- the water/rotor case, where the choice is which
    orientation, not which atom.

    Criteria, again the research constructor's: one element only; occupancy inside `ORIENT_OCC` (a
    0.1 or a 0.9 orbit is a vacancy sublattice, not a rotor); anchors are the full orbits of that
    same element; each anchor claims the alternatives within `ORIENT_CUT`, those claims must be
    DISJOINT and must cover the whole orbit, each anchor's occupancy must pick a proper subset
    (0 < k < n), and there must be at least two distinct such subsets or there is no choice to make.

    This is a different signature from `rigid_units`, which keys on a FORMER centre and its partial
    light ligand shell. Here there is no distinct centre element -- the anchor is the same element as
    the alternatives -- so that detector cannot see these, and this one cannot see a BO3.
    """
    orb, occ_tot = _occ_orbits(s)
    cell = np.array(s.lattice.matrix)
    # `slaved` is passed in by the evidence bundle so the detection runs once per structure, not
    # once per consumer.
    su = slaved_units(s) if slaved is None else slaved
    taken = ({tuple(map(tuple, u['follower_signature'])) for u in su}
             | {tuple(map(tuple, u['parent_signature'])) for u in su})
    out = []
    for sig in orb:
        if occ_tot[sig] >= 0.99 or sig in taken:
            continue
        els = {el for el, _o in sig}
        if len(els) != 1:
            continue
        el = next(iter(els)); c = occ_tot[sig]
        if not (ORIENT_OCC[0] <= c <= ORIENT_OCC[1]):
            continue
        anchors = [p for f2 in orb if occ_tot[f2] >= 0.99 and {e for e, _o in f2} == {el}
                   for _i, p, _d in orb[f2]]
        if not anchors:
            continue
        gpos = np.array([p for _i, p, _d in orb[sig]])
        Dag = get_distances(np.array(anchors), gpos, cell=cell, pbc=True)[1]
        Dgg = get_distances(gpos, gpos, cell=cell, pbc=True)[1]
        assigned, ks, ok = set(), [], True
        for a in range(len(anchors)):
            cand = [j for j in range(len(gpos)) if Dag[a, j] < ORIENT_CUT]
            k = int(round(c * len(cand)))
            if k <= 0 or k >= len(cand) or (assigned & set(cand)):
                ok = False; break
            assigned |= set(cand)
            subs = []
            for st in cand:                                   # distinct k-subsets, mutually close
                sub = [st]
                for q in cand:
                    if q != st and q not in sub and all(Dgg[q, x] < ORIENT_CUT for x in sub):
                        sub.append(q)
                    if len(sub) == k:
                        break
                if len(sub) == k and sorted(sub) not in subs:
                    subs.append(sorted(sub))
            if len(subs) < 2:
                ok = False; break
            ks.append((k, len(subs)))
        if ok and assigned == set(range(len(gpos))):
            out.append({'signature': [list(t) for t in sig], 'element': el,
                        'occupancy': round(c, 3), 'n_anchors': len(anchors),
                        'positions_per_anchor': ks[0][0] if ks else None,
                        'alternatives_per_anchor': min(n for _k, n in ks) if ks else None})
    return out


def rigid_units(s):
    """Orientational rigid-unit SIGNAL (neutral geometry). For each candidate center: gather its
    PARTIAL LIGHT-atom ligand shell (within COUPLE_A). Flag it when (A) there are MORE ligand
    sites than atoms (excess >= ~1 -- a complete shell smeared over orientations) AND (B) >=1
    pair of SAME-element ligand sites is closer than R_ALT (alternates that cannot coexist).
    Signal B is the discriminator vs a plain vacancy sublattice (whose partial sites are spaced
    apart, no same-element close alternates). Same-element-only avoids mistaking a real bond
    (e.g. C#O 1.15 A, different elements) for an alternate. Emits neutral facts; the LLM judges."""
    n = len(s)
    if n < 2: return []
    pos = np.array([site.coords for site in s]); cell = np.array(s.lattice.matrix)
    el, occ, light = [], [], []
    for site in s:
        d = {}
        for sp, o in site.species.items(): d[sp.symbol] = d.get(sp.symbol, 0.0) + o
        maj = max(d, key=d.get); el.append(maj); occ.append(sum(d.values())); light.append(maj in LIGHT)
    D = get_distances(pos, pos, cell=cell, pbc=True)[1]
    raw = []
    for c in range(n):
        shell = [j for j in range(n) if j != c and light[j] and occ[j] < PARTIAL and D[c, j] < COUPLE_A]
        if len(shell) < 2: continue
        occ_sum = sum(occ[j] for j in shell)
        excess = len(shell) - occ_sum
        alt = sum(1 for a in range(len(shell)) for b in range(a + 1, len(shell))
                  if el[shell[a]] == el[shell[b]] and D[shell[a], shell[b]] < R_ALT)
        if excess >= 0.8 and alt >= 1:
            raw.append((el[c], len(shell), round(occ_sum, 2),
                        tuple(sorted(set(el[j] for j in shell))), alt))
    cnt = Counter(raw); out = []
    for (cel, nsites, osum, ligels, alt), k in cnt.items():
        out.append({"center": cel, "ligand_elements": list(ligels), "n_ligand_sites": nsites,
                    "occ_sum": osum, "est_coordination": round(osum),
                    "est_orientations": round(nsites / max(1, osum)),
                    "n_alt_pairs": alt, "n_equivalent_centers": k})
    return out

# ---------------------------------------------------------------- main
def _icsd_module():
    """Lazy import of the ICSD database module (only the id-based wrapper needs it)."""
    try:
        import icsd_db
    except ModuleNotFoundError:
        if os.path.isdir(os.environ.get('ICSD_DB_DIR', '')):
            sys.path.insert(0, os.environ.get('ICSD_DB_DIR', ''))
        import icsd_db
    return icsd_db


def evidence(iid):
    """ICSD entry point: fetch CIF text by id, then derive evidence (with the
    local-DB ordered-sibling search). Thin wrapper over evidence_from_text."""
    return evidence_from_text(_icsd_module().get_cif_text(iid), iid=iid)


def evidence_from_text(text, iid=None, search_siblings=None):
    """Derive evidence from CIF TEXT -- ICSD-FREE. `iid` only stamps the record and
    self-excludes in the sibling search; `search_siblings` defaults to True only
    when an iid is given (an external CIF has no ICSD sibling context, so the
    local-DB lookup is skipped and 'ordered_sibling' is left unsearched)."""
    if search_siblings is None:
        search_siblings = iid is not None
    lines = text.split('\n')
    loops = list(cif_loops(lines))
    s = CifParser.from_str(text, occupancy_tolerance=1.15).parse_structures(primitive=False)[0]
    lat = s.lattice

    # ---- provenance / intrinsic metadata (principle 8) ----
    chdr, crows = loop_with(loops, '_citation_journal_full')
    journal = year = None
    if crows:
        r = dict(zip(chdr, crows[0] + [None] * len(chdr)))
        journal = _unq(r.get('_citation_journal_full')); year = _num(r.get('_citation_year'))
    ahdr, arows = loop_with(loops, '_citation_author_name')
    authors = []
    if arows:
        ai = ahdr.index('_citation_author_name')
        authors = [r[ai] for r in arows if len(r) > ai]
    prov = {
        'chemical_name_common':    cif_scalar(lines, '_chemical_name_common'),
        'chemical_name_mineral':   cif_scalar(lines, '_chemical_name_mineral'),
        'structure_type':          cif_scalar(lines, '_chemical_name_structure_type'),
        'formula_structural':      cif_scalar(lines, '_chemical_formula_structural'),
        'formula_sum':             cif_scalar(lines, '_chemical_formula_sum'),
        'citation_title':          (cif_scalar(lines, '_citation_title')
                                    or cif_scalar(lines, '_publ_section_title')),
        'journal': journal, 'year': int(year) if year else None,
        'authors': authors,
        'measurement_temperature_K': (_num(cif_scalar(lines, '_cell_measurement_temperature'))
                                      or _num(cif_scalar(lines, '_diffrn_ambient_temperature'))),
        'measurement_pressure_kPa':  (_num(cif_scalar(lines, '_cell_measurement_pressure'))
                                      or _num(cif_scalar(lines, '_diffrn_ambient_pressure'))),
        'R_factor': (_num(cif_scalar(lines, '_refine_ls_R_factor_gt'))
                     or _num(cif_scalar(lines, '_refine_ls_R_factor_all'))),
        'density_diffrn': _num(cif_scalar(lines, '_exptl_crystal_density_diffrn')),
    }

    # ---- oxidation states straight from the CIF (read, don't infer) ----
    ohdr, orows = loop_with(loops, '_atom_type_oxidation_number')
    type_ox = {}                                   # type_symbol -> oxidation number
    if orows:
        si = ohdr.index('_atom_type_symbol'); oi = ohdr.index('_atom_type_oxidation_number')
        for r in orows:
            if len(r) > max(si, oi):
                try: type_ox[r[si]] = float(r[oi])
                except Exception: pass
    elem_ox = defaultdict(list)
    for ts, ox in type_ox.items(): elem_ox[_el(ts)].append(ox)
    oxidation = {el: (vals[0] if len(set(vals)) == 1 else sorted(set(vals)))
                 for el, vals in elem_ox.items()}
    mixed_valence = sorted(el for el, vals in elem_ox.items() if len(set(vals)) > 1)

    # ---- atom_site loop -> raw rows -> coincident positions ----
    shdr, srows = loop_with(loops, '_atom_site_label')
    def col(tag): return shdr.index(tag) if shdr and tag in shdr else None
    cL, cT = col('_atom_site_label'), col('_atom_site_type_symbol')
    cM, cW = col('_atom_site_symmetry_multiplicity'), col('_atom_site_Wyckoff_symbol')
    cX, cY, cZ = col('_atom_site_fract_x'), col('_atom_site_fract_y'), col('_atom_site_fract_z')
    cO = col('_atom_site_occupancy')
    cB, cU = col('_atom_site_B_iso_or_equiv'), col('_atom_site_U_iso_or_equiv')
    has_aniso = any('_atom_site_aniso_U_11' in h for h, _ in loops)

    def g(r, c): return r[c] if (c is not None and len(r) > c) else None
    pos = defaultdict(list)
    for r in (srows or []):
        x, y, z = (_num(g(r, cX)) or 0) % 1, (_num(g(r, cY)) or 0) % 1, (_num(g(r, cZ)) or 0) % 1
        ts = g(r, cT) or g(r, cL); el = _el(ts)
        adp = _num(g(r, cB)); adp_t = 'B'
        if adp is None: adp, adp_t = _num(g(r, cU)), 'U'
        pos[(round(x, 4), round(y, 4), round(z, 4))].append({
            'label': g(r, cL), 'element': el, 'type': ts,
            'oxidation': type_ox.get(ts), 'occ': _num(g(r, cO)) if cO is not None else 1.0,
            'wyckoff': g(r, cW), 'multiplicity': g(r, cM),
            'adp': adp, 'adp_type': (adp_t if adp is not None else None)})

    sites = []
    for xyz, mem in pos.items():
        occ_sum = round(sum((m['occ'] or 0) for m in mem), 3)
        elems = {m['element'] for m in mem}
        sig = tuple(sorted((m['element'], round(m['occ'] or 0, 3)) for m in mem))
        sites.append({
            'frac': list(xyz), 'members': mem, 'occ_sum': occ_sum,
            'wyckoff': mem[0]['wyckoff'], 'multiplicity': mem[0]['multiplicity'],
            'is_mixed': len(elems) > 1, 'is_partial': occ_sum < PARTIAL, 'signature': sig})

    # ---- orbits = positions grouped by occupancy signature ----
    orb = defaultdict(list)
    for k, st in enumerate(sites): orb[st['signature']].append(k)
    orbits = []
    for sig, idxs in orb.items():
        st0 = sites[idxs[0]]
        if st0['is_mixed'] or st0['is_partial']:
            orbits.append({'signature': [list(t) for t in sig], 'n_positions': len(idxs),
                           'mixed': st0['is_mixed'], 'partial': st0['is_partial'],
                           'wyckoff': st0['wyckoff']})
    n_partial = sum(st['is_partial'] for st in sites)
    n_mixed = sum(st['is_mixed'] for st in sites)

    # ---- cross-orbit close contacts on the FULL cell (coupling/slaving SIGNATURE) ----
    dis_idx, dis_sig, dis_pos = [], [], []
    for i, site in enumerate(s):
        d = defaultdict(float)
        for sp, oc in site.species.items(): d[sp.symbol] += oc
        if len(d) > 1 or sum(d.values()) < 0.99:
            dis_idx.append(i); dis_pos.append(site.coords)
            dis_sig.append(tuple(sorted((e, round(o, 3)) for e, o in d.items())))
    contacts = []
    if len(dis_pos) > 1:
        P = np.array(dis_pos)
        D = get_distances(P, P, cell=lat.matrix, pbc=True)[1]
        agg = {}                                   # (orbit-pair, rounded d) -> count
        for a in range(len(P)):
            for b in range(a + 1, len(P)):
                if dis_sig[a] != dis_sig[b] and D[a, b] < COUPLE_A:
                    pair = tuple(sorted([dis_sig[a], dis_sig[b]]))
                    k = (pair, round(float(D[a, b]), 2))
                    agg[k] = agg.get(k, 0) + 1
        contacts = [{'d_A': d, 'count': c, 'orbit_a': dict(pair[0]), 'orbit_b': dict(pair[1])}
                    for (pair, d), c in sorted(agg.items(), key=lambda kv: kv[0][1])]

    # ---- nominal cell charge (using the CIF oxidation states) ----
    def ox_of(el):
        v = oxidation.get(el)
        return (sum(v) / len(v)) if isinstance(v, list) else v
    q = 0.0; q_ok = bool(type_ox)
    for site in s:
        for sp, oc in site.species.items():
            o = ox_of(sp.symbol)
            if o is None: q_ok = False
            else: q += oc * o
    cell_charge = round(q, 3) if q_ok else None

    # ---- composition: formula_sum (x Z) vs what is actually deposited in the structure ----
    # neutral fact; catches missing N (AuCN: N in formula, not deposited) and unlocated H alike
    Z = int(_num(cif_scalar(lines, '_cell_formula_units_Z')) or 1)
    fsum = {}
    for m in re.finditer(r'([A-Z][a-z]?)([0-9]*\.?[0-9]*)', (prov['formula_sum'] or '')):
        fsum[_el(m.group(1))] = fsum.get(_el(m.group(1)), 0.0) + (float(m.group(2)) if m.group(2) else 1.0)
    located = defaultdict(float)
    for site in s:
        for sp, oc in site.species.items(): located[sp.symbol] += oc
    comp_check = []
    for el in sorted(set(fsum) | set(located)):
        exp = round(fsum.get(el, 0.0) * Z, 2); loc = round(located.get(el, 0.0), 2)
        if abs(exp - loc) > 0.05:
            comp_check.append({'element': el, 'formula_expected': exp, 'located': loc,
                               'deficit': round(exp - loc, 2),
                               'missing_from_structure': loc == 0.0 and exp > 0})
    H_exp = round(fsum.get('H', 0.0) * Z, 2); H_loc = round(located.get('H', 0.0), 2)
    h_info = {'located': H_loc, 'formula_expected': H_exp,
              'deficit': round(H_exp - H_loc, 2)} if (H_exp or H_loc) else None

    # ---- ordered-sibling search (reused from mar_ordered_lookup) ----
    sibling = 'not searched'; sibling_ids = []; sibling_key = None
    sibling_prov = provenance('not_run', 'sibling search disabled or no local ICSD DB configured',
                              source='ICSD_DB_DIR / DEMARS_ORDERED_LOOKUP unset')
    if search_siblings and _sib is not None:
        try:
            Comp = _sib['Composition']; comp = Comp(prov['formula_sum'])
            f0 = comp.fractional_composition
            chemsys = '-'.join(sorted({e.symbol for e in comp}))
            rows = _sib['db'].execute(
                'SELECT icsd_id, filename, formula_sum, spacegroup_number FROM cifs WHERE chemsys=?',
                (chemsys,)).fetchall()
            sg = int(_num(cif_scalar(lines, '_space_group_IT_number')) or 0)
            # NOMINAL reduced formula of the disordered target: round each element's partial-occupancy
            # count to its full-occupancy integer. A candidate is a sibling ONLY if its reduced formula
            # EQUALS this. This distinguishes "an ordered version of the SAME phase" from a DIFFERENT nearby
            # compound that a flat composition tolerance wrongly conflates (a cation-deficient spinel and a
            # genuinely different nearby oxide of the same chemsys): a flat tolerance can't separate them because the legit
            # disordered<->ordered gap can exceed the different-compound gap. Bonus: it auto-finds the true
            # ordered parent (the stoichiometric spinel) instead of an unrelated high-pressure phase.
            def _nrf(c):                               # nominal (partial-occ rounded to full) reduced formula
                nom = Comp({e.symbol: round(c[e]) for e in c if round(c[e]) > 0})
                return nom.reduced_formula if len(nom) else None
            target_rf = _nrf(comp)                      # round BOTH sides: candidates can be slightly off-integer too
            # RECORD THE KEY WE ACTUALLY SEARCHED ON. The key is the CIF's own `_chemical_formula_sum`, so a
            # MODEL REPAIR that rewrites that field silently changes it -- and an empty result then reads as
            # "checked and absent" when it really means "searched under a different formula". (Ga2Te3 repaired
            # to Ga2.667Te4 searched for Ga3Te4 and found nothing.) With the key in the bundle the three
            # states are distinguishable after the fact: unchecked / checked-and-absent / key-was-mutated.
            sibling_key = {'formula_sum': prov['formula_sum'], 'nominal_reduced_formula': target_rf,
                           'chemsys': chemsys}
            same, other = [], []
            for cid, fn, fs2, sgn in rows:
                if cid == iid or target_rf is None: continue
                try:
                    if _nrf(Comp(fs2)) != target_rf: continue
                except Exception: continue
                if _sib['is_ordered'](fn): (same if sgn == sg else other).append((cid, sgn))
            # full ranked candidate list (same-SG first, then other-SG) -- so a downstream screen can relax
            # them and pick the LOWEST-energy ordered sibling, not just the first match.
            sibling_ids = [cid for cid, _ in same] + [cid for cid, _ in other]
            if same:   sibling = f'ORDERED same-SG: ICSD {same[0][0]} (+{len(same)-1+len(other)} more)'
            elif other: sibling = f'ORDERED other-SG {other[0][1]}: ICSD {other[0][0]} (+{len(other)-1} more)'
            else:       sibling = 'none — no fully-ordered ICSD entry of this composition'
            # "checked and absent" is `derived`, NOT `vacuous`: the search swept the whole chemsys and
            # a real negative is an actionable answer. `vacuous` is for "nothing was expected".
            # The key-was-mutated third state (defect (3)) cannot be detected HERE -- at search time we
            # do not know whether a model repair rewrote `_chemical_formula_sum`. The key travels in
            # `evidence` so a downstream check can compare it against the deposited formula.
            sibling_prov = provenance(
                'derived', f'searched {len(rows)} {chemsys} entries; {len(sibling_ids)} ordered match(es)',
                evidence={**sibling_key, 'n_chemsys_rows': len(rows), 'n_matches': len(sibling_ids)})
        except Exception as e:
            sibling = f'lookup error: {e}'
            sibling_prov = provenance('not_run', f'lookup raised: {type(e).__name__}: {e}',
                                      source='local ICSD sibling DB')

    _slaved = slaved_units(s)          # shared by the orientation detector below

    # ---- scope (settled policy = code may decide this one thing) ----
    n_el = len({e for e in fsum})
    p = prov['measurement_pressure_kPa']
    scope = {'n_elements': n_el, 'n_atoms_cell': len(s), 'excluded': False, 'reason': None}
    if n_el > 6: scope.update(excluded=True, reason='gt6-elements')   # census extended to include hexanary (≤6)
    elif p and p > 1000: scope.update(excluded=True, reason='high-pressure')
    elif len(s) < 1.5: scope.update(excluded=True, reason='partial-structure')

    return {
        'icsd_id': iid,
        'provenance': prov,
        'cell': {'a': round(lat.a, 4), 'b': round(lat.b, 4), 'c': round(lat.c, 4),
                 'alpha': round(lat.alpha, 2), 'beta': round(lat.beta, 2),
                 'gamma': round(lat.gamma, 2), 'volume': round(lat.volume, 2), 'Z': Z,
                 'spacegroup': cif_scalar(lines, '_space_group_name_H-M_alt'),
                 'spacegroup_number': int(_num(cif_scalar(lines, '_space_group_IT_number')) or 0),
                 'n_atoms_cell': len(s)},
        'oxidation_states_in_cif': bool(type_ox),
        'oxidation_states': oxidation,
        'mixed_valence_elements': mixed_valence,
        'nominal_cell_charge': cell_charge,
        'composition_vs_formula': comp_check,
        'sites': sites,
        'disorder': {'n_partial_positions': n_partial, 'n_mixed_positions': n_mixed,
                     'n_disordered_orbits': len(orbits), 'orbits': orbits,
                     'cross_orbit_contacts_below_2.6A': contacts,
                     'rigid_unit_signal': rigid_units(s),
                     # non-empty => per-SITE decoration is invalid on those orbits (taxonomy class E)
                     'slaved_units': _slaved, 'orientation_units': orientation_units(s, _slaved),
                     'hydrogen': h_info},
        'ordered_sibling': sibling,
        'ordered_sibling_ids': sibling_ids,
        'ordered_sibling_key': sibling_key,       # what we searched on -- see the comment at the lookup
        'ordered_sibling_provenance': sibling_prov,   # supersedes reading the string above
        'scope': scope,
    }

def _digest(ev):
    p = ev['provenance']; d = ev['disorder']
    L = [f"ICSD {ev['icsd_id']}  {p['formula_sum']}  SG {ev['cell']['spacegroup_number']}  Z={ev['cell']['Z']}",
         f"  name: {p['chemical_name_common']}   structure-type: {p['structure_type']}",
         f"  title: {p['citation_title']}",
         f"  T={p['measurement_temperature_K']} K  P={p['measurement_pressure_kPa']} kPa  R={p['R_factor']}",
         f"  oxidation (from CIF={ev['oxidation_states_in_cif']}): {ev['oxidation_states']}"
         + (f"  MIXED-VALENCE {ev['mixed_valence_elements']}" if ev['mixed_valence_elements'] else ''),
         f"  nominal cell charge: {ev['nominal_cell_charge']}",
         *([f"  composition_vs_formula: " + '; '.join(
             f"{c['element']} formula {c['formula_expected']} / located {c['located']}"
             + (' MISSING' if c['missing_from_structure'] else f" (deficit {c['deficit']})")
             for c in ev['composition_vs_formula'])] if ev['composition_vs_formula'] else []),
         f"  disorder: {d['n_mixed_positions']} mixed + {d['n_partial_positions']} partial positions, "
         f"{d['n_disordered_orbits']} orbits; cross-orbit <2.6A contacts: {len(d['cross_orbit_contacts_below_2.6A'])}"]
    for o in d['orbits']:
        L.append(f"     orbit {o['signature']}  x{o['n_positions']}  Wyckoff {o['wyckoff']}"
                 f"  {'mixed' if o['mixed'] else ''}{' partial' if o['partial'] else ''}")
    for c in d['cross_orbit_contacts_below_2.6A'][:8]:
        L.append(f"     contact {c['d_A']} A (x{c['count']}) : {c['orbit_a']} -- {c['orbit_b']}")
    for u in d.get('rigid_unit_signal', []):
        L.append(f"     RIGID-UNIT: {u['center']}-{u['ligand_elements']}  {u['n_ligand_sites']} sites / "
                 f"~{u['occ_sum']} atoms -> ~{u['est_orientations']} orientations  (x{u['n_equivalent_centers']})")
    if d['hydrogen']: L.append(f"  hydrogen: {d['hydrogen']}")
    L.append(f"  ordered sibling: {ev['ordered_sibling']}")
    L.append(f"  scope: {ev['scope']}")
    return '\n'.join(L)

if __name__ == '__main__':
    args = sys.argv[1:]
    iid = int([a for a in args if not a.startswith('-')][0])
    ev = evidence(iid)
    if '--summary' in args:
        print(_digest(ev), file=sys.stderr)
    txt = json.dumps(ev, indent=2, ensure_ascii=False)
    if '--out' in args:                              # persist evidence.json into the rundir (the page reads it)
        od = args[args.index('--out') + 1]; os.makedirs(od, exist_ok=True)
        with open(os.path.join(od, 'evidence.json'), 'w') as f: f.write(txt)
        print(f"wrote {os.path.join(od, 'evidence.json')}", file=sys.stderr)
    print(txt)                                       # also to stdout (backward-compatible)
