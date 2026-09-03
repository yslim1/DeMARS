"""MAR pipeline stage ③ -- the ENGINE (uniform, every entry).

Build a faithful >=1.5 nm supercell, sample diverse occupancy-preserving decorations, relax with
7net-nano(+D3), and emit the FULL energy distribution + per-sample (energy, arrangement
descriptor, features) so the LLM (stage ④) can correlate energy with occupation -- to classify
open-endedly AND to discover constraints the setup missed. The lowest sample is the MAR; it is
recomputed with the omni mpa modal (+D3) as the final tier (self-consistent with mar_hull).

COUPLING (geometric, primary): before enumerating, disordered sites that are too close to coexist
(< EXCL_A, e.g. a sub-Angstrom transition-metal split site) are EXCLUSION-MERGED into a single GROUP that
holds at most one atom (in one alternate). This is the load-bearing fix: relaxation HEALS such
clashes, so they would NOT show up as eV-outliers in the distribution -- split-site coupling must
be caught at setup, not discovered post-hoc. Softer bonded coupling (e.g. As->Co alternate
preference at 1.8-2.5 A) survives as a real energy signal and is left for the distribution loop
(stage ④) to discover.

This is the CODE half ("code proves"): it computes the distribution, it does not judge it.

Usage:  python mar_engine.py <icsd_id> [--nr N] [--min-nm 1.5] [--excl 1.1] [--d3] [--final] [--summary]
        self-driving loop runs by default ([--rounds N], [--no-auto]); cutoffs are DATA-DERIVED & reported
        (engine.exclusion_merge / engine.couple_cut) and overridable: [--same-excl A] [--couple-cut A].
        stdout = JSON distribution ; --final also runs the omni-mpa recompute of the winner.
"""
import sys, json, warnings
warnings.filterwarnings('ignore')
import numpy as np
from collections import defaultdict, Counter
from ase import Atoms
from ase.geometry import get_distances
from ase.data import covalent_radii, atomic_numbers
from ase.neighborlist import neighbor_list
from pymatgen.io.cif import CifParser
from . import mar_evidence as MEV
from .. import _torchsim as S      # batched relaxation backend: use_model / batched_fire_relax

MAXAT = 500
BROAD_SPREAD_MEV = 200   # spread > this (eV/at-scale) -> the decorations vary wildly = a likely missed
                         # coupled/rigid unit (per-atom decoration shattered it) OR strong ordering: INVESTIGATE.
COUPLE_CUT = 2.0         # former-anion bond cutoff for constructive co-placement (covers e.g. B-O 1.35 A)
MIN_UNIT_PREVALENCE = 0.2  # an element centres a UNIT only if >=20% of its atoms actually do. Below that the
                           # "unit" is an artifact of the covalent-radius cutoff, not a structural motif.
                           # Trade-off: a genuine minority motif (e.g. sulfate S diluted in a sulfide matrix
                           # below 1:5) is missed -- pass --expect / min_prevalence when that is the case.

# Normal covalent bond count per acceptor -> capacity = valence - n_neighbours. O/N are the common
# Bronsted acceptors; the halides and chalcogenides come from the research pipeline's H-allocation
# stage (its PROTON_CAP), which covered them because a hydrogen bifluoride, an ammonium chloride or a
# hydrosulfide has nowhere else to put its proton -- with O/N only, such an entry got no placement at
# all and fell through to `custom`.
# The two extremes the enumeration always appends. They are built by `decorate(mode)` at maximum /
# minimum similarity to BOUND the distribution -- `strategy.md`'s SQS principle is random-only and
# `mar_record.sqs_gate` fails a record that ships one. They stay in the ensemble, where bounding the
# distribution is their job; they are just not candidates for the representative.
BRACKET_MODES = ('clustered', 'dispersed')

H_VALENCE = {'O': 2, 'N': 3, 'F': 1, 'Cl': 1, 'S': 2, 'Se': 2}
H_NBR_TOL = 1.25                      # per-PAIR cutoff d < (r_cov_i + r_cov_j)*tol -- the same bond rule
                                      # former_coordination and coordination_integrity already use.
                                      # A single per-element number (it was O:1.8, N:1.7) cannot see a
                                      # metal-anion bond: a Fe-O bond runs 1.9-2.3 A, so an O that had
                                      # just LOST its metal partner looked identical to a satisfied one and
                                      # the protons went to the wrong oxygens.
# X-H equilibrium lengths. O/N as before; the rest from the research pipeline's XH_BOND. The MLIP
# relax refines them, but starting a Cl-H at 0.98 A instead of 1.28 puts the proton inside the anion.
H_BOND_LEN = {'O': 0.98, 'N': 1.01, 'F': 0.93, 'Cl': 1.28, 'S': 1.34, 'Se': 1.46}

# Nothing may be placed closer than this to an existing atom or to a proton placed earlier in the
# same pass. The research pipeline used 1.15 A and let the relax repair mild crowding.
H_MIN_SEP = 1.15

# Ceiling on protons per anion in the BOND-VALENCE fallback, from the research pipeline's PROTON_CAP.
# N is 4 here and 3 in H_VALENCE on purpose: ammonium.
PROTON_CAP = {'O': 2, 'N': 4, 'F': 1, 'Cl': 1, 'S': 2, 'Se': 2}
BV_NBR_CUT = 2.85      # cation-anion cutoff for the coordination sum, as in that pipeline
BV_MIN_DEFICIT = 0.15  # below this an anion is treated as saturated (also its greedy tie window)

def _h_nbr_cuts(syms):
    """Per-pair neighbour cutoff matrix for the H-capacity model."""
    rc = np.array([covalent_radii[atomic_numbers[s]] for s in syms])
    return (rc[:, None] + rc[None, :]) * H_NBR_TOL

def ship_candidates(samples):
    """The valid samples eligible to be the REPRESENTATIVE, bracket probes excluded.

    Take-lowest over the whole pool lets a probe win, and it does: a designed extreme is not a member
    of the random distribution, so its energy being lowest says nothing about the phase -- typically
    it wins by a margin at the relaxation noise floor. `strategy.md` has said "sample random only and
    pick the representative from the random configs" all along, and the research tree wrote the flaw
    down next to it (*"the ENGINE currently has the same flaw ... a separate engine fix to make
    later"*) without making the fix; the record layer then grew a gate to DETECT it and a judgment
    field to work AROUND it by hand. This is the fix.

    THREE places choose the lowest and they must agree, or the record, the final tier and the written
    `representative.*` point at different structures: this function is the single definition.

    Falls back to the whole valid pool when no random sample survived relaxation -- there is nothing
    else to ship then, and `sqs_gate` catches that the shipped frame is a probe.
    """
    ok = [s for s in samples if s.get('valid')]
    return [s for s in ok if s.get('mode') not in BRACKET_MODES] or ok


def bracket_margins(samples):
    """Each probe's energy against the lowest SHIP candidate, in meV/atom.

    Reported because excluding the probes silently would erase a signal `strategy.md` asks for in the
    same breath: *"a clustered config sitting FAR below the random ones is a FLAG that the material
    may genuinely ORDER (reconsider the class -> D+F / averaged-superstructure), not a config to
    ship"*. Negative margin = the probe is below the random draws. A few meV is noise; tens of meV on
    `clustered` is the analyst's cue to re-read the mechanism class.
    """
    ship = ship_candidates(samples)
    if not ship:
        return {}
    base = min(s['E_per_atom'] for s in ship)
    out = {}
    for smp in samples:
        if smp.get('mode') in BRACKET_MODES and smp.get('valid'):
            out[smp['mode']] = {
                'E_per_atom': round(float(smp['E_per_atom']), 5),
                'margin_meV_vs_lowest_ship': round(1000 * (smp['E_per_atom'] - base), 2),
                'label': smp.get('label')}
    return out


def _place_h(c, P, C, exclude, k, blen, occupied=None, rng=None):
    """Place k H around centre c at length blen, spread AWAY from the coordinating crowd into a void
    (k=1 along the void bisector; k=2 at ~109 deg; k>=3 tripod). The MLIP relax fixes exact angles.

    Every candidate is CLASH-CHECKED against `P` and against `occupied` (protons already placed in
    this pass), and a rejected direction is retried at a random tilt about the void axis. This used to
    return the ideal directions unchecked, which puts a proton on top of a neighbouring atom whenever
    the void is not actually free -- the relax cannot undo an overlap it starts inside of. Returns the
    positions it could place, which may be FEWER than k; the caller decides what that means.
    """
    r = rng if rng is not None else np.random.default_rng(0)
    vecs = get_distances([c], P, cell=C, pbc=True)[0][0]              # MIC vectors c -> all atoms
    d = np.linalg.norm(vecs, axis=1)
    near = [j for j in range(len(P)) if j not in exclude and 0.1 < d[j] < 3.2]
    if near:
        crowd = sum(vecs[j] / d[j] for j in near); nrm = np.linalg.norm(crowd)
        bis = -crowd / nrm if nrm > 1e-6 else np.array([0., 0., 1.])
    else:
        bis = np.array([0., 0., 1.])
    a = np.array([1., 0., 0.]) if abs(bis[0]) < 0.9 else np.array([0., 1., 0.])
    u = a - bis * float(a @ bis); u /= np.linalg.norm(u); v = np.cross(bis, u)
    if k == 1:
        dirs = [bis]
    elif k == 2:
        h = np.radians(54.75)
        dirs = [bis * np.cos(h) + u * np.sin(h), bis * np.cos(h) - u * np.sin(h)]
    else:
        t = np.radians(70.5)
        dirs = [bis * np.cos(t) + np.sin(t) * (u * np.cos(2 * np.pi * j / k) + v * np.sin(2 * np.pi * j / k))
                for j in range(k)]

    placed = list(occupied) if occupied else []
    out = []
    for dn in dirs:
        dn = np.asarray(dn, dtype=float)
        for attempt in range(40):
            if attempt == 0:
                dd = dn
            else:                                  # tilt about the ideal direction and retry
                axis = r.normal(size=3); axis -= float(axis @ dn) * dn
                nrm = np.linalg.norm(axis)
                if nrm < 1e-6:
                    continue
                axis /= nrm
                ang = r.uniform(-1.4, 1.4)
                dd = np.cos(ang) * dn + np.sin(ang) * axis
            cand = c + blen * dd / np.linalg.norm(dd)
            others = np.vstack([P, np.array(placed)]) if placed else P
            sep = get_distances([cand], others, cell=C, pbc=True)[1][0]
            keep = np.ones(len(others), dtype=bool)
            for j in exclude:                      # the host bond itself is not a clash
                if j < len(P):
                    keep[j] = False
            if (sep[keep] > H_MIN_SEP).all():
                out.append(cand); placed.append(cand)
                break
    return out

def _bv_deficits(syms, P, C, ox):
    """Pauling bond-strength deficit per candidate anion, the research pipeline's H-allocation model:

        deficit(X) = |q_X| - sum over cation neighbours M of q_M / CN_M

    where CN_M counts M's anion neighbours within BV_NBR_CUT. It is a CONTINUOUS undersaturation
    measure, which is why it sees what the engine's capacity model cannot: a bridging O bonded to two
    Si is fully coordinated -- `capacity = valence - n_neighbours` is 0 -- yet its formal charge is
    not satisfied by two half-bonds from Si, so it can still take a proton. Hydrous silicates and
    phosphates live in exactly that gap.

    Needs oxidation states; returns `None` when they are absent, because without them this model has
    no input at all and a fabricated one would be worse than declining.
    """
    ox = {k: (sum(v) / len(v) if isinstance(v, (list, tuple)) else v)
          for k, v in (ox or {}).items() if v is not None}
    if not ox:
        return None
    n = len(syms)
    cat = [i for i in range(n) if ox.get(syms[i], 0) > 0.5]
    ani = [i for i in range(n) if ox.get(syms[i], 0) < -0.5 and syms[i] in PROTON_CAP]
    if not ani:
        return {}
    D = get_distances(P, P, cell=C, pbc=True)[1]
    np.fill_diagonal(D, 9e9)
    CN = {m: max(1, int(((D[m, ani] > 0.01) & (D[m, ani] < BV_NBR_CUT)).sum())) for m in cat}
    out = {}
    for x in ani:
        sat = sum(ox[syms[m]] / CN[m] for m in cat if 0.01 < D[m, x] < BV_NBR_CUT)
        sat += sum(1.0 for h in range(n) if syms[h] == 'H' and 0.01 < D[x, h] < 1.3)  # located H
        d = abs(ox[syms[x]]) - sat
        if d > BV_MIN_DEFICIT:
            out[x] = d
    return out


def _bv_fallback_ok(syms, P, C, ox, n_target):
    """Would the bond-valence fallback actually reconcile `n_target` protons on this skeleton?
    Asked BEFORE committing the plan, so a build is never promised protons it cannot place."""
    bv = _bv_deficits(syms, P, C, ox)
    if not bv:
        return False
    room = sum(min(PROTON_CAP[syms[i]], max(1, int(round(d)))) for i, d in bv.items())
    return 0 < n_target <= room


def _complete_h(at, n_target, ox=None, mode='capacity'):
    """Valence-driven protonation: each O/N acceptor's capacity = valence - n covalent neighbours; fill
    COMPLETE capacity-tiers (most under-saturated first: free O / amine N before hydroxyl/carboxylate)
    until n_target H are placed. No species lookup. Returns (atoms_with_H, n_H_added).

    `mode='bond-valence'` swaps the ranking for the research pipeline's Pauling bond-strength deficit
    (see `_bv_deficits`, which needs `ox`), keeping everything else -- the greedy fill, the
    clash-checked placement, the blocked-host re-offer -- identical. It is the FALLBACK the engine
    reaches for only where the capacity model finds nothing at all: a fully coordinated bridging
    anion has capacity 0 and a real bond-strength deficit, which is what a hydrous silicate is made
    of. Per-anion ceiling is PROTON_CAP, not H_VALENCE.
    """
    P = at.get_positions(); C = np.array(at.get_cell()); syms = at.get_chemical_symbols()
    Dall = get_distances(P, P, cell=C, pbc=True)[1]; np.fill_diagonal(Dall, 9e9)
    tiers = defaultdict(list)
    if mode == 'bond-valence':
        bv = _bv_deficits(syms, P, C, ox)
        if not bv:
            return at, 0
        # Rank by deficit, coarse-binned so the greedy round-robin below behaves as it does for
        # capacities: a bin is "one proton's worth" of undersaturation.
        cap_of = {i: min(PROTON_CAP[syms[i]], max(1, int(round(d)))) for i, d in bv.items()}
        order = {i: d for i, d in bv.items()}
    else:
        cuts = _h_nbr_cuts(syms)
        for i in range(len(at)):
            el = syms[i]
            if el not in H_VALENCE: continue
            cap = H_VALENCE[el] - int((Dall[i] < cuts[i]).sum())
            if cap > 0: tiers[cap].append(i)
        cap_of = order = None
    # GREEDY round-robin: place n_target H, 1 per acceptor per pass, most-under-saturated first, up to
    # each acceptor's capacity. Equals the old complete-tier fill when n_target fills whole tiers (no
    # regression); also handles PARTIAL tiers (1 H per hydroxide O when deficit = n_OH < 2*n_OH capacity).
    if cap_of is None:
        cap_of = {i: c for c in tiers for i in tiers[c]}
        order = {i: float(cap_of[i]) for i in cap_of}
    count = defaultdict(int); remaining = n_target
    while remaining > 0:
        avail = sorted((i for i in cap_of if count[i] < cap_of[i]),
                       key=lambda i: (order[i] - count[i], -i), reverse=True)
        if not avail: break
        for i in avail:
            if remaining <= 0: break
            count[i] += 1; remaining -= 1
    # Placement can refuse a host whose void is not actually free. A host that takes fewer protons
    # than it was allotted must not cost the structure those protons: re-offer them to the acceptors
    # that still have capacity, and give up only when every remaining host is blocked. Without this
    # one crowded oxygen silently reduces the H count the formula demands.
    rng = np.random.default_rng(0)
    add = []
    blocked = set()
    def _fill(i, k):
        got = _place_h(P[i], P, C, {i}, k, H_BOND_LEN[syms[i]], occupied=add, rng=rng)
        add.extend(got)
        return len(got)
    for i, k in list(count.items()):
        if k > 0:
            done = _fill(i, k)
            if done < k:
                blocked.add(i)
                count[i] = done
    short = n_target - len(add)
    while short > 0:
        avail = [i for i in cap_of if i not in blocked and count[i] < cap_of[i]]
        if not avail:
            break
        avail.sort(key=lambda i: (order[i] - count[i], -i), reverse=True)
        i = avail[0]
        got = _fill(i, 1)
        if got:
            count[i] += 1; short -= 1
        else:
            blocked.add(i)
    if add:
        at = at + Atoms('H' * len(add), positions=add)
    return at, len(add)

def _place_compensating_h(at, sub_el, n):
    """Fix C: place n charge-compensating Bronsted protons on the bridging anions (O/N) nearest the
    aliovalent substituent atoms (one H per anion; capacity-independent -- a framework bridging O is
    fully 2-coordinate so the valence-capacity model never sees it, but B3+-for-Si4+ leaves it under-
    bonded by exactly one proton). Proton oriented away from the coordinating crowd by _place_h."""
    sym = at.get_chemical_symbols(); P = at.get_positions(); C = np.array(at.get_cell())
    subs = [i for i, e in enumerate(sym) if e == sub_el]
    anions = [i for i, e in enumerate(sym) if e in COORD_ANIONS]
    if not subs or not anions or n <= 0:
        return at, 0
    Dsa = get_distances(P[anions], P[subs], cell=C, pbc=True)[1]
    order = sorted(range(len(anions)), key=lambda a: float(Dsa[a].min()))
    add = []
    for a in order:
        if len(add) >= n:
            break
        oi = anions[a]
        add += _place_h(P[oi], P, C, {oi}, 1, 0.97, occupied=add)
    if add:
        at = at + Atoms('H' * len(add), positions=add)
    return at, len(add)

# central elements whose covalent coordination DEFINES a discrete unit (molecular ion / oxo-anion /
# polyhedron). Ionic cations (Na, Eu, Ca, ...) are excluded -- their coordination varies legitimately.
COORD_FORMERS = {'B', 'C', 'N', 'P', 'S', 'Si', 'As', 'Se', 'Te', 'Ge', 'Cl', 'Br', 'I',
                 'V', 'Cr', 'Mo', 'W', 'Mn', 'Re', 'Nb', 'Ta'}
COORD_ANIONS = {'O', 'N', 'S', 'F', 'Cl', 'Br', 'I', 'Se', 'Te'}

def unexplained_dangling(atoms, formers=None, tol=1.25):
    """Split "uncoordinated anion" into anions with NO possible centre and anions whose centre is
    simply not in `formers` (defect D12).

    `COORD_FORMERS` is a fixed list, so a structure whose former site is SHARED by a listed and an
    unlisted element reports every anion hanging off the unlisted one as uncoordinated. Li4(Ge,Sn)Se4
    is the case that found this -- Ge is listed, Sn is not, so 100 of 192 Se read as dangling and
    three self-driving rounds tried to remove a defect that was never there. `--expect` cannot help:
    it retargets the CN of an element already selected, it cannot ADD one.

    WHAT NOT TO DO, measured. The first attempt asked which elements BEHAVE like formers
    (prevalence + a uniform coordination number). In an ordered periodic crystal that describes
    ionic cations too: across the 9 COD fixtures and 6 set-B representatives it fired on La, Mg, Fe,
    Ti, Zr, Ca, Cs, Er, Ni, Zn, Gd, Cu, Ba and a K with CN 27. Uniform CN does not separate a
    network former from a counter-cation, so a warning built on it would fire on nearly everything.

    The trigger has to be the doubt itself: only ask where a defect is actually being CLAIMED. An
    anion is `unexplained` when nothing at all bonds it, and `explained` when some non-former
    element does -- then the count is a statement about the element table, not about the structure.
    The engine still does not promote anything; whether Sn centres a unit here is chemistry, and its
    duty ends at asking with the numbers attached (the report-don't-decide rule the other
    diagnostics follow: state the doubt with its numbers, leave the chemistry to the analyst).

    An anion usually has SEVERAL non-former neighbours -- in the Li4(Ge,Sn)Se4 case both Sn (which
    IS the former here) and Li (which is not). No single number separates them: measured on that
    structure, Sn-Se 2.55 A and Li-Se 2.56 A are indistinguishable, and a uniform coordination
    number describes ordinary ionic cations too. So every candidate is reported with the numbers an
    analyst needs -- how many of the disputed anions it bonds, how far, its coordination number and
    how constant that is -- and the engine decides nothing.

    -> (n_unexplained, {element: {"n_anions", "median_d_A", "modal_cn", "cn_uniformity"}})
    """
    formers = set(formers) if formers else COORD_FORMERS
    syms = atoms.get_chemical_symbols(); n = len(syms)
    Z = [atomic_numbers[s] for s in syms]
    cut = [covalent_radii[z] * tol for z in Z]
    I, J, D = neighbor_list('ijd', atoms, cut)
    anion_formers = defaultdict(set); has_H = defaultdict(bool); other = defaultdict(dict)
    cn_other = defaultdict(int)
    for a, b, d in zip(I, J, D):
        if syms[b] == 'H':
            has_H[a] = True
        if syms[b] not in COORD_ANIONS:
            continue
        if syms[a] in formers:
            anion_formers[b].add(a)
        elif syms[a] not in COORD_ANIONS:      # a possible centre that the table does not list
            e = syms[a]
            other[b][e] = min(float(d), other[b].get(e, 9e9))
            cn_other[a] += 1
    bonding = {syms[b] for b in anion_formers}
    explained = defaultdict(list); n_unexplained = 0
    for i in range(n):
        if syms[i] not in bonding or anion_formers.get(i) or has_H[i]:
            continue                            # not counted as dangling in the first place
        if other.get(i):
            for el, d in other[i].items():
                explained[el].append(d)
        else:
            n_unexplained += 1
    out = {}
    for el, ds in explained.items():
        cns = [c for i, c in cn_other.items() if syms[i] == el]
        modal, n_modal = Counter(cns).most_common(1)[0] if cns else (None, 0)
        out[el] = {"n_anions": len(ds), "median_d_A": round(float(np.median(ds)), 2),
                   "modal_cn": (int(modal) if modal is not None else None),
                   "cn_uniformity": (round(n_modal / len(cns), 2) if cns else None)}
    return n_unexplained, out


def _bond_populations(lens, stretch, min_frac=0.2):
    """One element PAIR can carry two chemically different bonds: cyanide C-N (~1.15 A) and
    methyl-amine C-N (~1.46 A) can sit in one structure. A single median lands in whichever cluster
    is larger and every bond of the other kind is then reported `stretched`, in every config.

    So calibrate per population, not per pair. But split ONLY where each side is big enough to be a
    bond type: a lone long bond is the shattered unit this report exists to catch, and letting it
    become its own median would silence exactly the defect we are looking for."""
    s = sorted(lens); n = len(s)
    cuts = [i for i in range(1, n) if s[i] > s[i - 1] * stretch]
    if not cuts:
        return [s]
    groups, prev = [], 0
    for c in cuts + [n]:
        groups.append(s[prev:c]); prev = c
    if min(len(g) for g in groups) < max(3, min_frac * n):
        return [s]
    return groups


def coordination_integrity(atoms, formers=None, expect=None, tol=1.25, stretch=1.15, clash=0.6):
    """Deterministic coordination-integrity report on a RELAXED structure. Catches shattered/defective
    covalent units -- the shattered-oxo-anion failure mode (heterogeneous former coordination + stretched bonds +
    bridging anions) that the energy spread can miss. A FLAG, not a verdict: heterogeneous CN can be
    legitimate (polyborates, mixed frameworks), so it routes to investigation / verifies a re-plan.

      formers : central elements whose CN defines a unit (default COORD_FORMERS; pass e.g. {'B'} to focus).
      expect  : optional {element: target_CN} -> explicit `violations` + `matches_expected` (VERIFY mode).
      bonds via covalent_radii*tol; `stretch` = factor over the per-pair MEDIAN bond (self-calibrating);
      `clash` = factor of the covalent sum below which a contact is an unhealed clash.
    Returns a report dict incl. cn_histogram, heterogeneous_CN, stretched_bonds, clashes,
    bridging_anions, bond_len ranges, and integrity_flag (True == investigate)."""
    formers = set(formers) if formers else COORD_FORMERS
    syms = atoms.get_chemical_symbols(); n = len(syms)
    Z = [atomic_numbers[s] for s in syms]
    cut = [covalent_radii[z] * tol for z in Z]                # per-atom; bond iff d < cut_i + cut_j
    I, J, D = neighbor_list('ijd', atoms, cut)
    cn = defaultdict(int); pairs = defaultdict(list); anion_formers = defaultdict(set); clashes = []
    for a, b, d in zip(I, J, D):
        if a < b and d < (covalent_radii[Z[a]] + covalent_radii[Z[b]]) * clash:
            clashes.append((syms[a], syms[b], round(float(d), 2)))
        if syms[a] in formers and syms[b] in COORD_ANIONS:   # former CN = ANION bonds only (not M-M / M-cation)
            cn[a] += 1; pairs[(syms[a], syms[b])].append(float(d)); anion_formers[b].add(a)
    cn_hist = {}
    for el in formers:
        cns = [cn[i] for i in range(n) if syms[i] == el and cn[i] > 0]
        if cns: cn_hist[el] = dict(Counter(cns))
    hetero = {el: h for el, h in cn_hist.items() if len(h) > 1}
    stretched = []
    for (fe, an), lens in pairs.items():
        for grp in _bond_populations(lens, stretch):
            med = float(np.median(grp))
            stretched += [(fe, an, round(d, 2), round(med, 2)) for d in grp if d > med * stretch]
    bridging = sum(1 for fs in anion_formers.values() if len(fs) >= 2)
    rep = {"cn_histogram": cn_hist, "heterogeneous_CN": hetero,
           "stretched_bonds": stretched[:20], "n_stretched": len(stretched),
           "clashes": clashes[:20], "n_clashes": len(clashes), "bridging_anions": bridging,
           "bond_len": {f"{k[0]}-{k[1]}": [round(min(v), 2), round(max(v), 2)] for k, v in pairs.items()},
           "integrity_flag": bool(hetero or stretched or clashes)}
    if expect:
        viol = [{"element": el, "index": int(i), "cn": cn[i], "expected": tcn}
                for el, tcn in expect.items() for i in range(n) if syms[i] == el and cn[i] != tcn]
        rep.update(expected_cn=expect, violations=viol[:30], n_violations=len(viol),
                   matches_expected=(len(viol) == 0))
    return rep

# H completes MOLECULAR ions whose centre is a former (NH4+, BH4-, CH3-, PH4+). It is deliberately
# NOT an "anion" for coordination purposes -- O is not a former, so a hydroxyl O-H never enters here.
COORD_LIGANDS = COORD_ANIONS | {'H'}

def former_coordination(atoms, formers=None, ligands=None, tol=1.25, min_prevalence=MIN_UNIT_PREVALENCE):
    """RAW former-ligand coordination of a structure -- the single definition of "a discrete unit"
    shared by the rigid-unit constraint, the connectivity gate and the coordination report.

    Nothing here is per-compound. Which elements can centre a unit comes from COORD_FORMERS (the same
    set the coordination gate uses); a bond is `d < (r_cov_i + r_cov_j) * tol`, so B-O (~1.4 A) and
    W-O (~2.2 A) are both found without a table of cutoffs; and each element's coordination number is
    the MODE observed in THIS structure, so BO3 -> 3, SO4/PO4/SiO4/ClO4 -> 4, MoO6 -> 6, NH4 -> 4 all
    fall out of the geometry rather than a lookup.

    Returns (bonds, modal_cn, pair_median):
      bonds        {centre_index: [(ligand_index, d_A), ...]}   -- every detected former-ligand bond
      modal_cn     {former_element: most common coordination number in this structure}
      pair_median  {(former_el, ligand_el): median bond length}  -- self-calibrating length scale
    Empty dicts when the structure holds no former-ligand unit at all (the common case for alloys,
    simple oxides with no oxo-anion, intermetallics) -- callers degrade to their unconstrained path.
    """
    formers = set(formers) if formers else COORD_FORMERS
    ligands = set(ligands) if ligands else COORD_LIGANDS
    syms = atoms.get_chemical_symbols()
    Z = [atomic_numbers[s] for s in syms]
    cut = [covalent_radii[z] * tol for z in Z]
    I, J, D = neighbor_list('ijd', atoms, cut)
    bonds = defaultdict(list); pair_len = defaultdict(list)
    seen = set()
    for a, b, d in zip(I, J, D):
        if syms[a] not in formers or syms[b] not in ligands:
            continue
        # A few elements (N, S, Se, Te, halogens) are in BOTH sets, so an X-X bond arrives twice with
        # the roles swapped. For `bonds` that is NOT a duplicate: an X-X bond belongs to the
        # coordination of BOTH its atoms, and keeping one orientation credits it to one endpoint only
        # -- half the true CN. On a layered polytelluride that turned a clean bimodal 0/4 topology
        # into a scattered histogram whose mode was 3, and 141 correct centres into defects.
        # `pair_len` IS duplicated by the second arrival, so the de-dup belongs there alone.
        bonds[int(a)].append((int(b), float(d)))
        key = (min(int(a), int(b)), max(int(a), int(b)))
        if key not in seen:
            seen.add(key)
            pair_len[(syms[a], syms[b])].append(float(d))
    # PREVALENCE GATE. The modal CN is taken over BONDED centres only, so without this a handful of
    # accidental contacts define a "unit" for the whole element -- and every unbonded atom of it then
    # reads as a stripped centre. Ga2Te3 is the case that found this: Te is in BOTH formers and ligands,
    # 3 chance Te-Te contacts among 108 Te set the target CN to 1, and the other 105 were reported as
    # defects. If fewer than MIN_UNIT_PREVALENCE of an element's atoms centre a unit, it is not a unit
    # type in this structure. Element-agnostic, so it also catches cross-element accidents.
    modal = {}
    for el in formers:
        cns = [len(v) for i, v in bonds.items() if syms[i] == el]
        n_el = syms.count(el)
        if cns and n_el and (len(cns) / n_el) >= min_prevalence:
            modal[el] = Counter(cns).most_common(1)[0][0]
    bonds = {i: v for i, v in bonds.items() if syms[i] in modal}       # drop the rejected centres' bonds too,
    pair_len = {k: v for k, v in pair_len.items() if k[0] in modal}    # so all three consumers agree
    pair_median = {k: float(np.median(v)) for k, v in pair_len.items()}
    return dict(bonds), modal, pair_median

def centre_expectations(atoms, formers=None, ligands=None, tol=1.25,
                        min_prevalence=MIN_UNIT_PREVALENCE, expect=None):
    """Per-CENTRE expected coordination, split by the ROLE each centre actually plays.

    `former_coordination` derives one modal CN per ELEMENT. That is wrong whenever one element
    centres two different unit types in the same structure, and it is wrong in two distinct ways.
    A cyanide entry with methylated cations showed both and produced
    **160 false defects** on a structure that is chemically fine:

      * 64: C centres both a cyanide (ligands {N}, CN 1, x96) and a methylated cation
        (ligands {N,H}, CN 4, x64). The element-wide mode is 1, so every methyl C read as
        over-coordinated.
      * 96: the cyanide N is a LIGAND of that C, not a centre of its own. N sits in both
        COORD_FORMERS and COORD_ANIONS, so it was re-attached as a 0-coordination "fully stripped
        centre" -- the worst verdict this gate has -- for doing exactly its job.

    Two role distinctions fix both, and neither needs a per-compound table:
      (A) LIGAND ROLE -- an atom with no ligand bonds of its own that IS a ligand in another
          centre's unit is playing the ligand role, not a stripped centre.
      (B) UNIT-TYPE ROLE -- among the rest, group by the SET OF LIGAND ELEMENTS and take the mode
          within each group. Cyanide C and methyl C separate because {N} != {N,H}. An SO4 that lost
          an O does NOT separate from its siblings: losing one ligand of an element already present
          leaves the signature unchanged. That is what keeps this from hiding the defects the gate
          exists to find.

    A centre that lost EVERY ligand has no signature to group on, so it is judged against its
    element's DOMINANT role. It must stay a defect -- letting it form an empty-signature role of its
    own would give it an expectation of zero and turn the worst outcome into a pass.

    NO NEW THRESHOLD, and that is measured rather than assumed. Over the 60 shipped structures of
    the frozen 15-entry run, role size as a fraction of an element's centre population lands at
    0.137 0.186 0.217 0.217 0.4 0.4 0.4 0.4 0.6 0.6 0.6 0.6 0.783 0.783 0.814 0.863 -- **8 of 16
    inside (0.05, 0.5)**. Unlike the element-level prevalence distribution behind
    `MIN_UNIT_PREVALENCE` (bimodal, empty between 0.05 and 0.5), this is not bimodal, so any role-size cutoff would be load-bearing and tuned to one
    corpus. The element-level MIN_UNIT_PREVALENCE gate inside `former_coordination` is left to do
    the Ga2Te3 job it was measured for, and it still runs first: every element the split changes on
    that corpus (that entry's C and N) is one that gate already accepts.

    Deliberately does NOT feed `rigid_unit_bonds`. That function's decision is per centre
    (`min_cn <= len(lig) <= max_cn`) and never reads the modal CN, so nothing is lost by leaving it
    -- and changing the constraint would change relaxation, hence energies, hence comparability with
    the frozen run. A gate getting smarter must not silently move the numbers it is checking.

    -> dict with
      bonds        as `former_coordination`
      expected     {centre_index: expected_CN}
      roles        {centre_index: label}  'C[N]' / 'C[H+N]'; plain 'C' when C has one role only
      pair_median  as `former_coordination`
      modal_cn     as `former_coordination` (element view, for reports)
      ligand_role  {element: n_atoms} excluded by (A) -- report it; never drop it silently
    """
    bonds, modal, med = former_coordination(atoms, formers=formers, ligands=ligands, tol=tol,
                                            min_prevalence=min_prevalence)
    syms = atoms.get_chemical_symbols()
    ligand_idx = {j for lig in bonds.values() for j, _ in lig}

    # (A) split the element's atoms into centres and ligand-role atoms
    sig_of = {}; stripped = []; ligand_role = defaultdict(int)
    for i, s in enumerate(syms):
        if s not in modal:
            continue
        lig = bonds.get(i)
        if lig:
            sig_of[i] = tuple(sorted({syms[j] for j, _ in lig}))
        elif i in ligand_idx:
            ligand_role[s] += 1
        else:
            stripped.append(i)

    # (B) mode within each (element, ligand-signature) role
    groups = defaultdict(list)
    for i, sig in sig_of.items():
        groups[(syms[i], sig)].append(i)
    n_roles = Counter(el for el, _s in groups)
    expected, roles, dom = {}, {}, {}
    for (el, sig), idxs in sorted(groups.items()):
        cn = Counter(len(bonds[i]) for i in idxs).most_common(1)[0][0]
        label = el if n_roles[el] == 1 else f'{el}[{"+".join(sig)}]'
        for i in idxs:
            expected[i] = int(cn); roles[i] = label
        if el not in dom or len(idxs) > dom[el][2]:
            dom[el] = (label, int(cn), len(idxs))
    for i in stripped:                        # judged against the element's dominant role
        label, cn, _n = dom[syms[i]]
        expected[i] = cn; roles[i] = label
    if expect:                                # an explicit target overrides every role of that element
        for i in list(expected):
            if syms[i] in expect:
                expected[i] = int(expect[syms[i]])
    return {"bonds": bonds, "expected": expected, "roles": roles, "pair_median": med,
            "modal_cn": modal, "ligand_role": dict(ligand_role)}


def rigid_unit_bonds(atoms, formers=None, ligands=None, tol=1.25, stretch=1.15,
                     min_cn=2, max_cn=9):
    """Former-ligand bonds worth HOLDING RIGID through a constrained pre-relaxation.

    A freshly decorated cell puts the ligands of a discrete unit at their built offsets while the
    surrounding cations still sit at averaged CIF positions -- sometimes well inside bonding range. A
    free relax from that state lets a neighbouring cation pull a ligand off its centre: the unit
    breaks (an SO4 becomes SO3 + a free O) and only partly heals. Freezing the unit's internal bond
    LENGTHS through the violent early steps, then releasing, keeps it intact; see
    `calculators.ase_relax_batch(constrain=...)`.

    Deliberately conservative about WHICH bonds to freeze -- a frozen wrong length is worse than an
    unfrozen one:
      * centres whose coordination is degenerate (< min_cn) or implausible (> max_cn) are skipped;
      * a bond more than `stretch` away from its element-pair MEDIAN is skipped, because an outlier
        length on a freshly built cell is usually an averaging artifact rather than a real bond.
    Both bounds are measured from this structure, never assumed.

    Returns (pairs, report):
      pairs   [(i, j), ...] index pairs for ase.constraints.FixBondLengths; EMPTY when the structure
              has no discrete former-ligand unit, so the caller simply does its normal free relax.
      report  the basis of the decision (modal CN per element, median lengths, what was skipped),
              for the record -- same "defaults with a reason" discipline as exclusion_merge/couple_cut.
    """
    bonds, modal, med = former_coordination(atoms, formers=formers, ligands=ligands, tol=tol)
    syms = atoms.get_chemical_symbols()
    pairs = set(); skipped_cn = 0; skipped_len = 0
    for i, lig in bonds.items():
        if not (min_cn <= len(lig) <= max_cn):
            skipped_cn += 1
            continue
        for j, d in lig:
            m = med.get((syms[i], syms[j]))
            if m and (d > m * stretch or d < m / stretch):
                skipped_len += 1
                continue
            pairs.add((min(i, j), max(i, j)))
    report = {"n_bonds": len(pairs), "n_centres": len(bonds),
              "formers_present": sorted({syms[i] for i in bonds}),
              "modal_cn": modal,
              "median_bond_A": {f"{a}-{b}": round(v, 3) for (a, b), v in med.items()},
              "skipped_out_of_cn_range": skipped_cn, "skipped_length_outlier": skipped_len,
              "basis": (f"bonds from covalent_radii*{tol}; coordination = mode observed in this "
                        f"structure; lengths within +/-{round((stretch - 1) * 100)}% of the "
                        f"element-pair median")}
    return sorted(pairs), report

def config_descriptors(atoms, formers=None, tol=1.25, clash=0.6, stretch=1.15):
    """Per-config DEFECT descriptor vector (deterministic; every key is 'higher == more suspect', so a
    POSITIVE correlation with energy = a defect to suppress). This is the panel the spectrum-diagnosis
    scans to DISCOVER the constraint -- nothing here is unit- or chemistry-specific; the energy decides
    which descriptor matters (e.g. for a shattered planar oxo-anion, dangling_anions tracks energy while
    over-coordination of the former does NOT).
    Computable pre-relax (cheap proxy used to bias constrained re-sampling) or post-relax (diagnosis)."""
    formers = set(formers) if formers else COORD_FORMERS
    syms = atoms.get_chemical_symbols(); n = len(syms); Z = [atomic_numbers[s] for s in syms]
    cut = [covalent_radii[z] * tol for z in Z]
    I, J, D = neighbor_list('ijd', atoms, cut)
    cn = defaultdict(int); anion_formers = defaultdict(set); pairs = defaultdict(list)
    has_H = defaultdict(bool); clashes = 0
    for a, b, d in zip(I, J, D):
        if a < b and d < (covalent_radii[Z[a]] + covalent_radii[Z[b]]) * clash: clashes += 1
        if syms[b] == 'H': has_H[a] = True
        if syms[a] in formers and syms[b] in COORD_ANIONS:
            cn[a] += 1; anion_formers[b].add(a); pairs[(syms[a], syms[b])].append(d)
    modal = {}
    for el in formers:
        cns = [cn[i] for i in range(n) if syms[i] == el and cn[i] > 0]
        if cns: modal[el] = Counter(cns).most_common(1)[0][0]
    under = sum(1 for i in range(n) if syms[i] in modal and 0 < cn[i] < modal[syms[i]])
    over = sum(1 for i in range(n) if syms[i] in modal and cn[i] > modal[syms[i]])
    bonding_anions = {syms[b] for b in anion_formers}        # anion ELEMENTS that bond a former somewhere
    dangling = sum(1 for i in range(n) if syms[i] in bonding_anions
                   and not anion_formers.get(i) and not has_H[i])   # uncoordinated anion (no former, no H)
    bridging = sum(1 for fs in anion_formers.values() if len(fs) >= 2)
    stretch_ct = sum(sum(1 for d in lens if d > float(np.median(lens)) * stretch) for lens in pairs.values())
    # D12: an anion whose only neighbour is a former the TABLE does not list is not a defect of the
    # structure -- it is a gap in COORD_FORMERS. Kept separate rather than subtracted, so the raw
    # count stays comparable and the doubt is visible; the self-driving loop chases the unexplained
    # ones only, because the others cannot be removed by re-decorating.
    n_unexpl, expl = unexplained_dangling(atoms, formers=formers, tol=tol) if dangling else (0, {})
    return {"dangling_anions": dangling, "under_coord_formers": under, "over_coord_formers": over,
            "bridging_anions": bridging, "clashes": clashes, "stretched_bonds": stretch_ct,
            "dangling_unexplained": n_unexpl if dangling else dangling,
            "dangling_explained_by_unlisted_former": expl}

def _spearman(x, y):
    """rank correlation, robust to nonlinearity/outliers (small-N safe)."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    rx = np.argsort(np.argsort(x)).astype(float); ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    den = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / den) if den > 1e-12 else 0.0

def diagnose_spectrum(energies, descriptors, min_rho=0.5):
    """Find the DEFECT descriptor that best (monotonically) tracks energy across the ensemble -- the
    discovered origin of the high-energy configs. Returns ranked [(descriptor, rho, low_E_val, high_E_val)]
    and `top` = the strongest POSITIVE correlate above min_rho with variance, else None (-> remaining
    spread is genuine SRO, not a removable defect: STOP). Robust rank-correlation, so small-N safe."""
    E = np.asarray(energies, float)
    keys = set().union(*[set(d) for d in descriptors]) if descriptors else set()
    order = np.argsort(E)
    ranked = []
    for k in sorted(keys):
        vals = [d.get(k, 0) for d in descriptors]
        # descriptors are a NUMERIC feature vector here. `config_descriptors` also carries
        # explanatory payloads (D12's per-element breakdown is a dict), and float() on one of those
        # raises inside the loop -- i.e. adding a non-numeric descriptor would take out the whole
        # self-driving diagnosis. Rank-correlating a dict is meaningless anyway, so skip it.
        if not all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vals):
            continue
        x = np.array(vals, float)
        if x.std() < 1e-9: continue                          # constant -> explains no spread, skip
        ranked.append((k, round(_spearman(x, E), 3), float(x[order[0]]), float(x[order[-1]])))
    ranked.sort(key=lambda t: -t[1])                         # most positive (defect rises with energy) first
    top = next(({"descriptor": k, "rho": rho, "low_E_value": lo, "high_E_value": hi}
                for k, rho, lo, hi in ranked if rho >= min_rho), None)
    return {"ranked": ranked, "top": top, "min_rho": min_rho}

def dedup_coincident(s, tol=0.05):
    """Remove duplicate FULL-OCCUPANCY, single-species sites that pymatgen symmetry-expansion generates when a
    site is rounded a hair off a special position (near-coincident images at d ~ 0.001-0.01 A; typically an
    anion on a high-multiplicity special position). Two full-occ atoms of the SAME element within `tol` can never both be real,
    so keeping one is provably non-destructive -- and it prevents the template from double-counting that
    species. TARGETED (full-occ single-species only) so it never touches legitimate partial split-sites
    (those have occ<1 and are handled by the exclusion-merge). Returns (clean_structure, n_removed)."""
    cell = np.array(s.lattice.matrix)
    full = [k for k, site in enumerate(s)
            if len(site.species) == 1 and abs(sum(site.species.values()) - 1) < 0.02]
    if len(full) < 2: return s, 0
    el = [next(iter(s[k].species)).symbol for k in full]
    P = np.array([s[k].frac_coords for k in full]) @ cell
    D = get_distances(P, P, cell=cell, pbc=True)[1]; np.fill_diagonal(D, 9e9)
    remove = set()
    for a in range(len(full)):
        if full[a] in remove: continue
        for b in range(a + 1, len(full)):
            if full[b] not in remove and el[a] == el[b] and D[a, b] < tol:
                remove.add(full[b])
    if not remove: return s, 0
    s2 = s.copy(); s2.remove_sites(sorted(remove))
    return s2, len(remove)

def full_occ_clash(s, rho_cut=0.5, dup_floor=0.2):
    """TRIAGE signal (not a gate): impossible contacts among the FULLY-OCCUPIED, single-species sites of
    the DEPOSITED cell. A short contact on full-occupancy orbits cannot be relieved by ANY occupancy
    decoration, so it is not ordinary split-site disorder (which the exclusion-merge resolves). Two distinct
    causes, kept separate because the diagnosis differs:

      * SUPERPOSITION  (dup_floor <= d, rho < rho_cut): distinct atoms at an impossible-but-NONZERO
        separation -> average over a modulated / twinned / doubled-orbit structure where the two images are
        never simultaneously occupied in a real cell (fingerprint: a sub-Angstrom heavy-atom self-contact
        on occ-1.0 orbits, the 3D projection of a (3+1)D modulation). -> info['...'].flag
      * COINCIDENT     (d < dup_floor ~ 0): two full-occ atoms on the SAME point -> duplicate atoms from
        pymatgen symmetry expansion when a site is rounded a hair off a special position, OR a genuine site
        overlap. A dedup / stoichiometry concern (template double-counts these), NOT modulation. -> dup_flag

    ELEMENT-NORMALIZED so it is one cutoff for the whole periodic table: the signal is the dimensionless
    ratio rho = d/(r_i+r_j), NOT a fixed Angstrom number (1.8 A is a fine B-O bond but an impossible La-Te
    contact). Real bonds sit at rho ~ 0.9-1.1; even short/hypervalent/M-M contacts rarely dip below ~0.7,
    so rho < ~0.5 is 'no chemistry can do this'. Reports the offending pairs for LLM judgment of the CAUSE;
    it does NOT reject or modify the build."""
    syms, fr = [], []
    for site in s:
        d = defaultdict(float)
        for sp, oc in site.species.items(): d[sp.symbol] += oc
        if len(d) == 1 and sum(d.values()) >= 0.98:           # fully occupied, single species
            syms.append(next(iter(d))); fr.append(site.frac_coords)
    if len(syms) < 2:
        return {"rho_cut": rho_cut, "dup_floor_A": dup_floor, "n_full_occ_sites": len(syms),
                "rho_min": None, "rho_min_pair": None, "n_impossible": 0, "pairs": [], "flag": None,
                "n_coincident": 0, "coincident": [], "dup_flag": None}
    a = Atoms(syms, scaled_positions=fr, cell=np.array(s.lattice.matrix), pbc=True)
    P = a.get_positions(); C = np.array(a.get_cell()); Z = a.get_atomic_numbers()
    D = get_distances(P, P, cell=C, pbc=True)[1]; np.fill_diagonal(D, 9e9)
    n = len(syms); pairs = []; coincident = []; rmin, rmin_pair = 9e9, None
    for i in range(n):
        for j in range(i + 1, n):
            d_ij = float(D[i, j])
            rs = covalent_radii[Z[i]] + covalent_radii[Z[j]]
            rho = d_ij / rs
            if d_ij < dup_floor:                              # atoms on the same point -> duplicate/overlap
                coincident.append({"pair": f"{syms[i]}-{syms[j]}", "d_A": round(d_ij, 4)})
                continue
            if rho < rmin: rmin, rmin_pair = rho, f"{syms[i]}-{syms[j]}"   # min over the SUPERPOSITION regime
            if rho < rho_cut:
                pairs.append({"pair": f"{syms[i]}-{syms[j]}", "d_A": round(d_ij, 3),
                              "r_cov_sum_A": round(float(rs), 3), "rho": round(rho, 3)})
    pairs.sort(key=lambda p: p["rho"]); coincident.sort(key=lambda p: p["d_A"])
    flag = None
    if pairs:
        flag = (f"{len(pairs)} impossible contact(s) on FULLY-OCCUPIED orbits (rho<{rho_cut}, min "
                f"{rmin_pair} rho={rmin:.2f}) -> no occupancy decoration can relieve these; likely a "
                f"SUPERPOSITION artifact (average over a modulated/twinned/doubled-orbit structure), NOT "
                f"decoratable disorder. Judge the cause from the evidence before enumerating.")
    dup_flag = None
    if coincident:
        dup_flag = (f"{len(coincident)} pair(s) of full-occ atoms within {dup_floor} A (near-coincident) "
                    f"-> duplicate atoms from symmetry expansion (site rounded off a special position) or a "
                    f"genuine overlap; the template DOUBLE-COUNTS these. Dedup / verify stoichiometry; this "
                    f"is a parse/dedup issue, not modulation.")
    return {"rho_cut": rho_cut, "dup_floor_A": dup_floor, "n_full_occ_sites": n,
            "rho_min": (round(rmin, 3) if rmin_pair else None), "rho_min_pair": rmin_pair,
            "n_impossible": len(pairs), "pairs": pairs[:8], "flag": flag,
            "n_coincident": len(coincident), "coincident": coincident[:8], "dup_flag": dup_flag}

def strip_dummy_species(s):
    """Remove non-element 'dummy' species and treat their sites as VACANCIES. Crystallographers sometimes
    label a vacancy as a CHARGED DUMMY species so the deposited CIF formula balances -- e.g. a
    vacancy-bearing superionic conductor whose CIF puts a charged 'L1+' at partial occupancy on an
    alkali site as a cation-vacancy placeholder; 'L' is not a real element. pymatgen parses these as DummySpecies. They are NOT
    atoms (the MLIP has no such element and would crash), so strip the dummy component per site: the
    real-element occupancy is the truth and the dummy fraction IS the vacancy. A site that is purely a dummy
    is dropped entirely. Returns (clean_structure, report). Non-destructive to real atoms. NOTE: stripping
    the dummy reveals the REAL (often uncompensated) charge -- the dummy's formal charge was the CIF's
    bookkeeping trick to fake neutrality; the charge gate then honestly reflects the vacancy."""
    from pymatgen.core import Structure, Composition, Element, DummySpecies
    from collections import defaultdict
    def real(sp):
        sym = getattr(sp, 'symbol', None)
        if sym == 'D': return True                       # deuterium -> H (a real isotope label, handled elsewhere)
        return (not isinstance(sp, DummySpecies)) and bool(sym) and Element.is_valid_symbol(sym)
    detected = []; keep_sp = []; keep_xyz = []; cvac = []
    for site in s:
        clean = {}; had_dummy = False
        for sp, occ in site.species.items():
            if real(sp):
                clean[sp] = occ
            else:
                had_dummy = True
                detected.append({'symbol': str(getattr(sp, 'symbol', sp)), 'occ': round(float(occ), 3),
                                 'frac': [round(float(x), 3) for x in site.frac_coords],
                                 'co_occupants': sorted({str(getattr(k, 'symbol', k)) for k in site.species if k is not sp})})
        if clean:                                        # site keeps its real species (now partially vacant)
            keep_sp.append(Composition(clean)); keep_xyz.append(site.frac_coords)
            if had_dummy:                                # CONSTITUTIONAL vacancy: occupancy is the deliberate truth
                d = defaultdict(float)
                for sp, occ in clean.items(): d[sp.symbol] += occ
                if len(d) > 1 or sum(d.values()) < 0.99:
                    cvac.append({el: round(o, 3) for el, o in d.items()})
    if not detected:
        return s, {'detected': False}
    s2 = Structure(s.lattice, keep_sp, keep_xyz)
    seen = set(); cvac_u = []
    for c in cvac:
        k = tuple(sorted(c.items()))
        if k not in seen: seen.add(k); cvac_u.append(c)
    return s2, {'detected': True, 'symbols': sorted({d['symbol'] for d in detected}),
                'n_dummy_sites': len(detected), 'sites': detected[:8], 'constitutional_occ': cvac_u,
                'note': "CIF dummy/vacancy-placeholder species removed -> sites treated as CONSTITUTIONAL "
                        "vacancies (target fixed, NOT filled for charge); real (often uncompensated) charge reported."}

def _element_candidates(sym):
    """Real elements of which `sym` is a prefix (truncation candidates: 'L' -> Li/La/Lu/Lr/Lv). A non-element
    label is most often a truncated/typo'd element OR a deliberate dummy; an empty candidate set => not near
    any element => more likely a genuine dummy/vacancy placeholder."""
    from pymatgen.core import Element
    return sorted({e.symbol for e in Element if e.symbol.startswith(sym) and e.symbol != sym})

def detect_dummy_species(s, ev=None, ox=None):
    """Detect non-element 'dummy' species and SURFACE the disambiguation -- do NOT silently assume vacancy.
    A non-element label (e.g. 'L1+') may be a VACANCY placeholder OR a truncated/mislabeled real
    element (L->Li). Structure + charge alone CANNOT decide -- the charge can even FAVOUR the element
    reading (filling the site balances the cell exactly where the vacancy reading leaves a residual).
    The decisive evidence is the paper:
    CIF citation/name first, then a PUBLIC-REFERENCE web search (never the structure). So this flags the
    species with (a) element candidates by prefix, (b) the per-cell charge under each reading, (c) the CIF
    citation/name hint, and a recommendation + escalate. The engine auto-resolves to vacancy ONLY when
    unambiguous (no near-element candidate, OR the source explicitly says vacancy/deficient); otherwise it
    defaults to vacancy to proceed but sets escalate=True for the LLM to confirm (CIF-first, then web)."""
    from pymatgen.core import Element, DummySpecies
    def real(sp):
        sym = getattr(sp, 'symbol', None)
        if sym == 'D': return True
        return (not isinstance(sp, DummySpecies)) and bool(sym) and Element.is_valid_symbol(sym)
    sites = []; tot_occ = 0.0
    for site in s:
        for sp, occ in site.species.items():
            if not real(sp):
                sites.append({'symbol': str(getattr(sp, 'symbol', sp)), 'occ': round(float(occ), 3),
                              'frac': [round(float(x), 3) for x in site.frac_coords],
                              'co_occupants': sorted({str(getattr(k, 'symbol', k)) for k in site.species if k is not sp})})
                tot_occ += float(occ)
    if not sites:
        return {'detected': False}
    syms = sorted({d['symbol'] for d in sites})
    cand = {sym: _element_candidates(sym) for sym in syms}
    near = any(cand[sym] for sym in syms)
    # per-cell formal charge under each reading (vacancy = strip; element X = dummy occ * X's common ox state)
    charges = {}
    if ox is not None:
        vac_q = sum(ox.get(getattr(sp, 'symbol', ''), 0.0) * float(occ)
                    for site in s for sp, occ in site.species.items() if real(sp))
        charges['vacancy'] = round(vac_q, 2)
        for sym in syms:
            for el in cand[sym]:
                try:
                    oss = Element(el).common_oxidation_states or Element(el).oxidation_states
                    oxe = oss[0] if oss else 0
                except Exception:
                    oxe = 0
                charges[f'{sym}={el}({oxe:+d})'] = round(vac_q + tot_occ * oxe, 2)
    # CIF citation/name hint (CIF-first)
    hint = None; leaning = None
    if ev:
        prov = ev.get('provenance', {}) or {}
        txt = ' '.join(str(prov.get(k, '') or '') for k in ('citation_title', 'chemical_name_common', 'chemical_name_mineral')).lower()
        if any(w in txt for w in ('vacanc', 'deficien', 'defect', 'non-stoich', 'nonstoich')):
            leaning, hint = 'vacancy', 'CIF source mentions vacancy/deficiency'
        elif any(w in txt for w in (' dop', '-dop', 'substitut', 'solid solution', 'solid-solution')):
            leaning, hint = 'element', 'CIF source mentions doping/substitution'
    if leaning == 'vacancy' or not near:
        rec, escalate = 'vacancy', False
    elif leaning == 'element':
        rec, escalate = 'remap-to-element (confirm which)', True
    else:
        rec, escalate = 'vacancy (default — AMBIGUOUS)', True
    return {'detected': True, 'symbols': syms, 'n_dummy_sites': len(sites), 'sites': sites[:8],
            'element_candidates': cand, 'near_element': near, 'charge_under_readings': charges,
            'citation_hint': hint, 'leaning': leaning, 'recommendation': rec, 'escalate': escalate,
            'note': "non-element species = vacancy placeholder OR mislabeled element; structure+charge can't "
                    "decide (charge may favour the element). Resolve CIF-first, then public-reference web "
                    "search (never the structure). Engine default per 'recommendation'; escalate=True => LLM must confirm."}

def _icsd_module():
    """Lazy import of the ICSD database module. Only the ICSD-id wrappers
    (load/evidence by id) need it, so the engine imports fine without the DB
    present -- the CIF-text / structure entry points never touch it."""
    try:
        import icsd_db
    except ModuleNotFoundError:
        import os
        if os.path.isdir(os.environ.get('ICSD_DB_DIR', '')):
            sys.path.insert(0, os.environ.get('ICSD_DB_DIR', ''))
        import icsd_db
    return icsd_db


def structure_from_text(text):
    """Parse a (possibly disordered) pymatgen Structure from CIF TEXT. ICSD-free."""
    return CifParser.from_str(text, occupancy_tolerance=1.15).parse_structures(primitive=False)[0]


def load(iid):
    return structure_from_text(_icsd_module().get_cif_text(iid))

def lr_states(n, probs):
    """largest-remainder integer counts over states with given probabilities (rest = vacancy)."""
    raw = {k: p * n for k, p in probs.items()}
    tgt = {k: int(np.floor(v)) for k, v in raw.items()}
    cap = min(n, int(round(sum(raw.values()))))
    for k in sorted(raw, key=lambda k: raw[k] - tgt[k], reverse=True):
        if sum(tgt.values()) >= cap: break
        tgt[k] += 1
    return {k: c for k, c in tgt.items() if c > 0}

def _charge_balance_provenance(st, go_tgt, faithful, nsite):
    """Provenance for the charge balancer, in the closed `PROV_STATES` vocabulary.

    Why it needs one at all: `balance_charge` may move site counts AWAY from the CIF's refined
    occupancies -- by design, since a refined occupancy can itself be charged. But on one layered
    oxide it
    filled a Li orbit refined at **0.95(1)** -- a ~5 sigma sub-stoichiometry, i.e. the disorder the
    entry exists to represent -- back to 100 % to satisfy the absolute `|q| < 0.3` gate, and
    `provenance_summary` still reported **`n_unconfident = 0`**. The one place a gate is supposed to
    be able to look did not see the step that erased the phenomenon; the analyst caught it only by
    reading `engine.json` by hand. `charge_balancing` was reported, but never as PROVENANCE, so
    nothing counted it.

    The discriminator is CATEGORICAL, so no new threshold is invented (the D2/D28 rule -- an
    unmeasured cutoff is a cutoff tuned to one corpus): did a move take a partially occupied orbit to
    FULL, or to EMPTY? Then that orbit is not disordered any more, and a human has to decide whether
    the refined vacancy was real or the charge model is missing a lever -> `ambiguous`, which is NOT
    confident, which is what `n_unconfident` counts. Nudges that leave every disordered orbit partial
    are the designed behaviour and stay `derived`.
    """
    if not st.get('adjusted'):
        # `adjusted: False` has TWO causes and they are not alike: the balancer ran and found the
        # cell already neutral, or it never moved anything and the cell is STILL charged (a driver
        # disabled it, or it declined). Reporting the second as `vacuous` contradicts the evidence
        # in the same object -- "already charge-neutral" next to q_per_cell = -13 -- and `vacuous`
        # is CONFIDENT, so `n_unconfident` stays 0 and the summary a gate reads sees nothing. That
        # is the blind spot this function exists to close, one branch over.
        _q = st.get('q_initial')
        if _q:
            return MEV.provenance('ambiguous', f'nothing was moved but the cell still carries '
                                  f'q={_q} per cell: the balancer did not run (disabled or '
                                  'overridden), so the deficit is unexplained here -- gates.charge '
                                  'reads it directly', evidence={'q_per_cell': _q})
        return MEV.provenance('vacuous', 'the realized occupancies were already charge-neutral; '
                              'nothing was moved', evidence={'q_per_cell': _q})
    saturated, emptied = [], []
    for sig in go_tgt:
        n = nsite.get(sig)
        if n is None:
            continue
        was = sum(faithful.get(sig, {}).values())
        now = sum(go_tgt[sig].values())
        if was < n <= now:
            saturated.append(str(sig))
        elif was > 0 == now:
            emptied.append(str(sig))
    ev = {'q_initial': st.get('q_initial'), 'q_final': st.get('q_final'),
          'deviation_sites': st.get('deviation'),
          'orbits_filled_to_full': saturated, 'orbits_emptied': emptied}
    if saturated or emptied:
        what = ' · '.join(x for x in (f'filled to FULL: {saturated}' if saturated else '',
                                      f'emptied: {emptied}' if emptied else '') if x)
        return MEV.provenance(
            'ambiguous',
            f'charge balancing removed the disorder on an orbit to reach neutrality ({what}). '
            'Decide whether the refined occupancy is real -- if it is, the charge model needs the '
            'lever instead (e.g. a non-integer oxidation state on the counter-cation), not the '
            'vacancy filled in.', evidence=ev)
    return MEV.provenance('derived', 'nudged site counts toward neutrality; every disordered orbit '
                          'stayed partial', evidence=ev)


def balance_charge(go, go_tgt, fixed_comp, oxidation_states, fixed_sigs=frozenset()):
    """Branch-2: joint charge-constrained rounding. Greedily reassign single sites (species->vacancy,
    vacancy->species, species<->species) across ALL disordered orbits to drive the cell toward
    neutrality, valence-float aware (Branch 1) and minimising deviation from the faithful targets.
    Returns (adjusted go_tgt, status). CIF occupancy is a reference, not ground truth -- so it will
    nudge composition to reach neutral when the refined occupancies are themselves charged. EXCEPTION:
    orbits in `fixed_sigs` are CONSTITUTIONAL (e.g. dummy-derived cation vacancies) -- their
    occupancy is the deliberate truth, so they are NOT touched and any residual charge is reported honestly
    rather than papered over by filling the vacancy."""
    from .mar_record import valence_aware_charge
    go_tgt = {s: dict(t) for s, t in go_tgt.items()}
    faithful = {s: dict(t) for s, t in go_tgt.items()}
    nsite = {s: len(g) for s, g in go.items()}
    def comp_of(tgt):
        c = dict(fixed_comp)
        for s, t in tgt.items():
            for (role, el), cnt in t.items(): c[el] = c.get(el, 0) + cnt
        return c
    def dev(tgt):
        return sum(abs(tgt[s].get(k, 0) - faithful[s].get(k, 0)) for s in tgt for k in faithful[s])
    g0 = valence_aware_charge(comp_of(go_tgt), oxidation_states)
    st = {"adjusted": False, "q_initial": g0['q_per_cell'], "q_final": g0['q_per_cell'],
          "pass_initial": g0['pass'], "pass_final": g0['pass'], "deviation": 0}
    if g0['pass']:
        st['provenance'] = _charge_balance_provenance(st, go_tgt, faithful, nsite)
        return go_tgt, st
    best = abs(g0['q_per_cell'])
    for _ in range(80):
        scored = []
        for sig in go_tgt:
            if sig in fixed_sigs: continue            # constitutional (dummy-derived) vacancy -- do not fill
            keys = list(faithful[sig].keys()); vac = nsite[sig] - sum(go_tgt[sig].values())
            cur = {**go_tgt[sig], 'VAC': vac}
            for a in keys + ['VAC']:
                if cur.get(a, 0) <= 0: continue
                for b in keys + ['VAC']:
                    if b == a: continue
                    t2 = {s: dict(go_tgt[s]) for s in go_tgt}
                    if a != 'VAC': t2[sig][a] -= 1
                    if b != 'VAC': t2[sig][b] = t2[sig].get(b, 0) + 1
                    if any(v < 0 for v in t2[sig].values()): continue
                    g2 = valence_aware_charge(comp_of(t2), oxidation_states)
                    scored.append((abs(g2['q_per_cell']), dev(t2), g2['pass'], t2, g2))
        if not scored: break
        scored.sort(key=lambda x: (x[0], x[1]))
        bq, bdev, bpass, t2, g2 = scored[0]
        if bq < best - 1e-9:
            go_tgt = t2; best = bq
            st.update(adjusted=True, q_final=g2['q_per_cell'], pass_final=g2['pass'], deviation=bdev)
            if g2['pass']: break
        else:
            break
    st['provenance'] = _charge_balance_provenance(st, go_tgt, faithful, nsite)
    return go_tgt, st

def orient_units(s, R_bond=2.6, R_alt=1.6, max_sites=10, max_coord=6):
    """Handler core (scoped to CLEANLY-partitionable discrete units). For each rigid-unit center,
    try to partition its partial light-ligand shell into discrete ORIENTATIONS = complete
    coordination sets (size m = round(occ_sum)) that are internally compatible (no two sites closer
    than R_alt) and together TILE the shell. Returns per-center {status: clean|ambiguous, ...}.
    'clean' -> the engine can enumerate one orientation per center; 'ambiguous' -> route to custom.
    Deliberately conservative: anything not a clean tiling of equal complete sets is left to custom."""
    from itertools import combinations
    LIGHT_ = {'H', 'B', 'C', 'N', 'O', 'F'}
    n = len(s); pos = np.array([si.coords for si in s]); cell = np.array(s.lattice.matrix)
    el, occ, light = [], [], []
    for si in s:
        d = {}
        for sp, o in si.species.items(): d[sp.symbol] = d.get(sp.symbol, 0.0) + o
        maj = max(d, key=d.get); el.append(maj); occ.append(sum(d.values())); light.append(maj in LIGHT_)
    D = get_distances(pos, pos, cell=cell, pbc=True)[1]
    out = []
    for c in range(n):
        shell = [j for j in range(n) if j != c and light[j] and occ[j] < 0.98 and D[c, j] < R_bond]
        if len(shell) < 2: continue
        occ_sum = sum(occ[j] for j in shell)
        if len(shell) - occ_sum < 0.8: continue
        if not any(el[a] == el[b] and D[a, b] < R_alt for a in shell for b in shell if a < b): continue
        m = max(1, round(occ_sum))
        rec = {"center_el": el[c], "ligand_el": sorted(set(el[j] for j in shell)),
               "n_sites": len(shell), "occ_sum": round(occ_sum, 2), "coord_m": m}
        if len(shell) > max_sites or m > max_coord or m < 2:
            rec["status"] = "ambiguous (size/coord)"; out.append(rec); continue
        incompat = {(min(a, b), max(a, b)) for a in shell for b in shell
                    if a < b and el[a] == el[b] and D[a, b] < R_alt}
        compat = [set(cmb) for cmb in combinations(shell, m)
                  if all((min(a, b), max(a, b)) not in incompat for a in cmb for b in cmb if a < b)]
        covered = set().union(*compat) if compat else set()
        # clean = enough complete sets to tile the shell into K=n_sites/m disjoint orientations
        K = round(len(shell) / m)
        rec.update(n_complete_sets=len(compat), K_orientations=K, covers_all=(covered == set(shell)))
        # CLEAN only if the orientations are essentially UNIQUE: ~K complete sets that tile the shell.
        # Many complete sets (>> K) = combinatorially ambiguous -> not auto-handleable -> custom.
        rec["status"] = "clean" if (compat and covered == set(shell) and K >= 2
                                    and K <= len(compat) <= K + 1) else "ambiguous (combinatorial)"
        out.append(rec)
    return out

def build(iid=None, NR=30, MIN_A=15.0, EXCL=1.1, d3=False, seed=0, select=None, oversample=6, couple=False,
          same_excl=None, couple_cut=None, structure=None, evidence=None, relax=None, max_atoms=None,
          couple_formers=None, couple_anions=None):
    # ICSD id is now just ONE way to obtain (evidence, structure): pass them
    # directly -- with a `relax` callable for a pluggable MLIP -- to run ICSD-free.
    ev = evidence if evidence is not None else MEV.evidence(iid)
    ox = {el: (sum(v) / len(v) if isinstance(v, list) else v)
          for el, v in ev['oxidation_states'].items()}
    s = structure if structure is not None else load(iid)
    cell = np.array(s.lattice.matrix)

    # ---- dummy/non-element species (TRIAGE; flag, do not silently assume vacancy) ----
    # A non-element label (e.g. 'L1+') may be a VACANCY placeholder OR a mislabeled/truncated element
    # (L->Li). Structure+charge can't decide it (charge may even favour the element) -- the paper does
    # (CIF-first, then a public-reference web search), so SURFACE the candidates/charges/citation hint and an
    # escalate flag. pymatgen parses these as DummySpecies which would crash downstream, so the engine still
    # strips them (treats as vacancy) to PROCEED, but escalate=True means the LLM must confirm (and remap if needed).
    dummy = detect_dummy_species(s, ev, ox); strip_rep = {}
    if dummy.get('detected'):
        s, strip_rep = strip_dummy_species(s); cell = np.array(s.lattice.matrix)

    # ---- full-occupancy clash diagnose (TRIAGE), then dedup symmetry-expansion duplicates ----
    # Diagnose on the RAW parsed cell so the report names the SUPERPOSITION (modulation/twin; rho<0.5 on
    # distinct full-occ orbits -- the superposition fingerprint) and the COINCIDENT duplicates separately. Then
    # remove the coincident duplicates BEFORE building, so the template does not double-count a species
    # (provably safe -- two full-occ same-element atoms at ~0 A are never both real). The SUPERPOSITION flag
    # is left for the LLM; we do not auto-resolve a modulation.
    foc = full_occ_clash(s)
    if foc['n_coincident']:
        s, _ndup = dedup_coincident(s); cell = np.array(s.lattice.matrix)
        foc['n_coincident_removed'] = _ndup

    # ---- disordered orbits (by species-occupancy signature) ----
    psyms, pfr, dis_orbit, sig2id = [], [], [], {}
    for site in s:
        d = defaultdict(float)
        for sp, oc in site.species.items(): d[sp.symbol] += oc
        psyms.append(max(d, key=d.get)); pfr.append(site.frac_coords)
        if len(d) > 1 or sum(d.values()) < 0.99:
            sig = tuple(sorted((el, round(o, 3)) for el, o in d.items()))
            dis_orbit.append(sig2id.setdefault(sig, len(sig2id)))
        else:
            dis_orbit.append(-1)
    occ_of = {v: dict(k) for k, v in sig2id.items()}
    if not sig2id:
        return ev, None, {"error": "no disordered sites"}
    base = Atoms(psyms, scaled_positions=pfr, cell=cell, pbc=True)

    # ---- supercell >= MIN_A on every axis (back off if over budget) ----
    L = np.linalg.norm(base.cell, axis=1)
    mult = mult_req = tuple(int(np.ceil(MIN_A / x)) for x in L)
    nbase = len(base); sc = base.repeat(mult)
    # The back-off below is silent otherwise: `min_cell_A` keeps reporting what was REQUESTED, so a
    # cell that shipped under the minimum reads as a contract kept (defect D13). What it actually
    # reached is stamped next to the request further down (`min_cell_A_achieved` + provenance).
    budget = int(max_atoms) if max_atoms else int(MAXAT * 1.6)
    while len(sc) > budget and max(mult) > 1:
        ax = int(np.argmax(mult)); m = list(mult); m[ax] -= 1; mult = tuple(m); sc = base.repeat(mult)
    nrep = int(np.prod(mult))
    P = sc.get_positions(); C = np.array(sc.get_cell())

    # ---- all disordered supercell sites (global idx + occ dict) ----
    dsites = []
    for r in range(nrep):
        for bi in range(nbase):
            if dis_orbit[bi] >= 0:
                dsites.append((r * nbase + bi, occ_of[dis_orbit[bi]]))
    gpos = P[[g for g, _ in dsites]]

    # ---- EXCLUSION-MERGE: sites too close to coexist -> one group, one atom ----
    # ELEMENT-AWARE + DATA-DERIVED. Cross-element pairs use the tight EXCL (a 1.2-1.5 A cross-element contact
    # may be a real bond). For SAME-element disordered sites the cutoff is READ FROM THE STRUCTURE: alternates
    # (split/orientational images of ONE site) cluster well below a real same-element bond, then a GAP to real
    # neighbours -> set the per-element cutoff IN that gap. This is a DEFAULT-WITH-A-REASON (reported in
    # info['exclusion_merge'] so the LLM can audit it; --same-excl overrides). NOT a rigid rule: a close
    # same-element pair can be a REAL bond (peroxide O-O ~1.49, metal-metal cluster) rather than an alternate,
    # so the report flags suspicious merges (group occ-sum > ~1.2) for LLM judgment.
    DEFAULT_SAME = max(EXCL, 1.3)
    ALT_FRAC, ALT_GAP_MIN = 0.9, 0.5        # see the impossibility split below
    ALT_CERTAIN_FRAC = 0.6                  # below this * 2 r_cov no real homoatomic bond exists;
                                            # between the two the merge is right but not certain
    nD = len(dsites); parent = list(range(nD))
    site_el = [max(dsites[i][1], key=dsites[i][1].get) for i in range(nD)]
    occ_site = [sum(dsites[i][1].values()) for i in range(nD)]
    def find(x):
        while parent[x] != x: parent[x] = parent[parent[x]]; x = parent[x]
        return x
    Dg = get_distances(gpos, gpos, cell=C, pbc=True)[1] if nD > 1 else None
    same_excl_el, merge_basis, merge_prov = {}, {}, {}
    if nD > 1:
        for el in sorted(set(site_el)):
            ii = [i for i in range(nD) if site_el[i] == el]
            if len(ii) < 2: continue
            ds = sorted(Dg[a][b] for k, a in enumerate(ii) for b in ii[k + 1:] if Dg[a][b] < 3.0)
            _ev = {"n_pairs_below_3A": len(ds), "closest_A": round(ds[0], 2) if ds else None,
                   "window_A": DEFAULT_SAME}
            # IMPOSSIBILITY SPLIT, tried before the widest-gap heuristic below. Two atoms of the same
            # element cannot sit closer than a bond of that element, so every pair under
            # ALT_FRAC * 2 r_cov is an alternate no matter where the widest gap happens to fall.
            # Widest-gap alone reads the geometry and not the chemistry, and misses two shapes: a
            # split pair sitting ABOVE the 1.3 A window (no cut set at all), and a run of alternates
            # whose widest internal gap falls in the MIDDLE of the run (cut too low, the outer
            # alternates left free to co-occupy). A face-sharing cation channel or a cation on two
            # sublattices stays out -- those sit above the threshold, which is the point of using it.
            _dw = sorted(Dg[a][b] for k, a in enumerate(ii) for b in ii[k + 1:] if Dg[a][b] < 5.0)
            _rcov2 = 2 * covalent_radii[atomic_numbers[el]]
            _alt_max = _rcov2 * ALT_FRAC
            _below = [d for d in _dw if d < _alt_max]
            # nothing else within 5 A means there is no competing neighbour to protect, so the
            # absence of an upper pair is itself the gap -- not a reason to leave the cut unset
            _above = [d for d in _dw if d >= _alt_max] or [5.0]
            _imposs = (round((_below[-1] + _above[0]) / 2, 2)
                       if _below and _above and _above[0] - _below[-1] > ALT_GAP_MIN else None)
            # r_cov is a poor bound for heavy metals that bond to themselves: a Mo-Mo quadruple bond
            # (2.61 A) or a W-W dimer (2.30 A) is SHORTER than 0.9*2 r_cov, so a genuinely bonded
            # partial-occupancy pair would merge as if it were one split site. Distance alone cannot
            # separate that from alternates -- which is what `ambiguous` means here. The merge still
            # happens (leaving it unmerged is the worse failure); the record says to check it.
            _alt_certain = bool(_below) and _below[-1] < _rcov2 * ALT_CERTAIN_FRAC
            _all_full = all(occ_site[i] >= MEV.PARTIAL for i in ii)
            if same_excl is not None:
                same_excl_el[el] = same_excl; merge_basis[el] = f"override --same-excl {same_excl}"
                merge_prov[el] = MEV.provenance('overridden', merge_basis[el], _ev, source='--same-excl')
            elif _imposs is not None and _all_full:
                # D43. The impossibility split WOULD have fired here -- and this branch is the
                # only place it is suppressed, so a pair the split never reached still falls
                # through to `out_of_window` and stays an open chemistry question.
                # An alternate needs a vacancy to alternate INTO. Where every {el} site's
                # occupancies sum to full, all of them are really occupied, so no two can be
                # mutually exclusive images of one position. A 0.5/0.5 split pair is NOT caught
                # here: each of its sites sums to 0.5, not 1, so real split images still merge.
                # This guard exists because the impossibility split reads chemistry through
                # 2 r_cov -- a COVALENT single-bond radius. Metallic nearest-neighbour spacing at
                # CN 8-12 sits under 0.9 * 2 r_cov for much of the transition-metal and rare-earth
                # block, so on a fully-occupied MIXED sublattice (an intermetallic solid solution)
                # the split fired, union-find swallowed the whole sublattice into one exclusion
                # network, and every decoration on every seed came back short-filled.
                # `suspicious_merges` below already computed this same occupancy sum and said in
                # its own note that nothing branched on it. Now something does -- but only here,
                # where the covalent-radius premise is what would have driven the merge.
                merge_basis[el] = (f"{el} sites are fully occupied (occ sum >= {MEV.PARTIAL}); "
                                   f"alternates are impossible on a full orbit -- no cut set")
                merge_prov[el] = MEV.provenance('derived', merge_basis[el],
                                                {**_ev, "all_sites_full": True,
                                                 "n_sites": len(ii)})
            elif _imposs is not None:
                same_excl_el[el] = _imposs
                merge_basis[el] = (f"pairs below {_alt_max:.2f} A cannot be a real {el}-{el} bond "
                                   f"(alternates); clear gap {_below[-1]:.2f}->{_above[0]:.2f} A"
                                   + ("" if _alt_certain else
                                      f"; but {_below[-1]:.2f} A is within reach of a real {el}-{el} "
                                      f"bond -- CHECK whether this is a bonded pair, not alternates"))
                merge_prov[el] = MEV.provenance('derived' if _alt_certain else 'ambiguous',
                                                merge_basis[el],
                                                {**_ev, "impossible_below_A": round(_alt_max, 2),
                                                 "gap_lo_A": round(_below[-1], 2),
                                                 "gap_hi_A": round(_above[0], 2),
                                                 "certain_below_A": round(_rcov2 * ALT_CERTAIN_FRAC, 2)})
            elif not ds:
                # NOTHING within 3.0 A. Nothing was expected -> defaulting to "no exclusion" is right.
                merge_basis[el] = f"no same-element pair below {DEFAULT_SAME} A (none within 3.0 A)"
                merge_prov[el] = MEV.provenance('vacuous', merge_basis[el], _ev)
            elif ds[0] >= DEFAULT_SAME:
                # DEFECT (4). Pairs EXIST but all sit above the 1.3 A window this derivation inspects, so
                # no cut is set -- and a face-sharing cation pair at ~2.9 A reads identically to a
                # genuinely isolated one. Distinct from `vacuous`: there IS data, we just cannot judge it
                # from distance alone. Whether such a pair is a split image or a real bond is chemistry.
                merge_basis[el] = f"no same-element pair below {DEFAULT_SAME} A (closest {ds[0]:.2f} A)"
                merge_prov[el] = MEV.provenance('out_of_window', merge_basis[el], _ev,
                                                source=f'no exclusion set (window {DEFAULT_SAME} A)')
            else:                                          # data-derive: cut in the gap above the close cluster
                glo, ghi, best = ds[0], ds[0], 0.0
                for a in range(len(ds) - 1):
                    if ds[a] > 2.0: break
                    if ds[a + 1] - ds[a] > best: best, glo, ghi = ds[a + 1] - ds[a], ds[a], ds[a + 1]
                if best > 0.4:
                    same_excl_el[el] = round((glo + ghi) / 2, 2)
                    merge_basis[el] = f"gap {glo:.2f}->{ghi:.2f} A (alternates below, real neighbours above)"
                    merge_prov[el] = MEV.provenance('derived', merge_basis[el],
                                                    {**_ev, "gap_lo_A": round(glo, 2), "gap_hi_A": round(ghi, 2),
                                                     "gap_width_A": round(best, 2)})
                else:
                    same_excl_el[el] = DEFAULT_SAME
                    merge_basis[el] = f"no clear gap (closest {ds[0]:.2f}) -> default {DEFAULT_SAME} A; CHECK"
                    # The "; CHECK" suffix was already here -- someone felt the doubt but nothing could
                    # read it. That is exactly what `ambiguous` promotes to a first-class field.
                    merge_prov[el] = MEV.provenance('ambiguous', merge_basis[el],
                                                    {**_ev, "widest_gap_A": round(best, 2)},
                                                    source=f'DEFAULT_SAME {DEFAULT_SAME} A')
        for i in range(nD):
            for j in range(i + 1, nD):
                cut = same_excl_el.get(site_el[i], DEFAULT_SAME) if site_el[i] == site_el[j] else EXCL
                if Dg[i, j] < cut: parent[find(i)] = find(j)
    grp = defaultdict(list)
    for i in range(nD): grp[find(i)].append(i)
    groups = list(grp.values())                       # each = list of local indices into dsites
    # audit: same-element merged groups whose occupancies sum well above 1 may be a REAL bonded pair
    # (peroxide / metal-metal cluster), NOT alternates -> surface for LLM judgment (exception to the rule).
    suspicious_merges = []
    for g in groups:
        if len(g) > 1 and len({site_el[m] for m in g}) == 1:
            osum = sum(occ_site[m] for m in g)
            if osum > 1.25:
                suspicious_merges.append({"element": site_el[g[0]], "n_sites": len(g),
                                          "occ_sum": round(osum, 2),
                                          "note": "occ-sum>1 -> may be a real same-element bond (peroxide/M-M), "
                                                  "not alternates; verify. REPORT ONLY: this flag is computed on "
                                                  "PRE-refinement groups and no code branches on it -- the merge "
                                                  "was NOT reverted, and a run is not invalid for carrying one"})
    # ---- CONFLICT-GRAPH refinement: exclusion is PAIRWISE, not transitive ----
    # Union-find merges any chain A-B-C (A-B, B-C below cut) into ONE component and places ONE atom, which
    # wrongly forbids occupying A and C together when A-C is wide enough to coexist (1D ion channels,
    # tunnels, percolating interstitial networks -> collapse). Exclusion is a pairwise relation, so the
    # valid occupations are the INDEPENDENT SETS of the conflict graph, not "one per connected component".
    # Keep CLIQUE components merged (genuine split-site alternates -> still exactly one), but SPLIT a
    # non-clique component into singletons and carry its pairwise edges in excl_adj; decorate then fills
    # the occupancy target as an independent set (A+C allowed). Reduces EXACTLY to the old behaviour on
    # cliques (a clique's only independent sets are the empty set and singletons).
    def _cut(i, j):
        return same_excl_el.get(site_el[i], DEFAULT_SAME) if site_el[i] == site_el[j] else EXCL
    excl_adj = defaultdict(set); n_chain_split = 0; refined = []
    for g in groups:
        if len(g) > 1 and not all(Dg[a, b] < _cut(a, b) for k, a in enumerate(g) for b in g[k + 1:]):
            n_chain_split += 1                                  # a chain/network, NOT a clique of alternates
            for k, a in enumerate(g):
                for b in g[k + 1:]:
                    if Dg[a, b] < _cut(a, b): excl_adj[a].add(b); excl_adj[b].add(a)
            refined.extend([m] for m in g)                     # -> singletons; coexistence set by independent-set
        else:
            refined.append(g)
    groups = refined
    exclusion_merge_report = {"cross_element_cut_A": EXCL, "same_element_cut_per_element": same_excl_el,
                              "basis": merge_basis,        # kept: downstream still parses it (non-breaking)
                              "provenance": merge_prov,    # per element -- states defined in mar_evidence.PROV_STATES
                              "suspicious_merges": suspicious_merges,
                              "n_chain_components_split": n_chain_split}

    # canonical member order (by occ-signature) + centroid per group
    def occsig(m): return tuple(sorted(dsites[m][1].items()))
    g_members = [sorted(g, key=occsig) for g in groups]
    g_centroid = [P[[dsites[m][0] for m in g]].mean(axis=0) for g in g_members]

    # ---- group-orbits: groups with identical structure share targets ----
    def gsig(g): return tuple(occsig(m) for m in g)
    go = defaultdict(list)                             # group-signature -> [group index]
    for gi, g in enumerate(g_members): go[gsig(g)].append(gi)
    go_ids = {sig: k for k, sig in enumerate(go)}      # short ids
    state_probs, go_tgt, go_minor, go_nn = {}, {}, {}, {}
    for sig, gids in go.items():
        rep = g_members[gids[0]]
        sp = {}                                        # (role, species) -> prob
        for role, m in enumerate(rep):
            for el, o in dsites[m][1].items(): sp[(role, el)] = o
        state_probs[sig] = sp
        go_tgt[sig] = lr_states(len(gids), sp)
        go_minor[sig] = (min(go_tgt[sig], key=go_tgt[sig].get) if go_tgt[sig] else None)
        cent = np.array([g_centroid[g] for g in gids])
        if len(cent) > 1:
            Dc = get_distances(cent, cent, cell=C, pbc=True)[1]; np.fill_diagonal(Dc, 9e9)
            _nn = float(Dc.min()); go_nn[sig] = _nn if _nn > 1e-3 else 1.0   # guard degenerate centroids
        else:
            go_nn[sig] = 1.0

    # ---- Fix A: rescue dilute substituents that per-signature largest-remainder zeroed out ----
    # A declared minority spread thinly across many one-site orbit-signatures floors to 0 in each
    # (lr_states(1,..) always keeps the majority) -> the species vanishes (e.g. dilute B-for-Si -> false
    # SiO2). Guarantee each declared element keeps its GLOBAL expected integer count (>=1 if present at all),
    # demoting the majority in the single-role signatures where it is most under-represented. Substitutional/
    # partial orbits only; coupled/merged units are left to the coupling machinery. Under-allocation only
    # (never reduces an already-met species -> a no-op on the working path).
    minority_rescue = {}
    _exp_el = defaultdict(float)
    for _g, _occ in dsites:
        for _el, _o in _occ.items(): _exp_el[_el] += _o
    def _realized_el(el):
        return sum(c for sg in go_tgt for (r, e), c in go_tgt[sg].items() if e == el)
    _single = [sig for sig in go if len(g_members[go[sig][0]]) == 1]   # unmerged single-role signatures
    for _el, _ex in sorted(_exp_el.items()):
        _want = int(round(_ex))
        if _ex > 1e-6 and _want < 1: _want = 1        # never silently drop a declared species
        _g = 0
        while _realized_el(_el) < _want and _g < 2000:
            _g += 1
            _cand = []
            for sig in _single:
                ng = len(go[sig])
                if not any(e == _el for (_r, e) in state_probs[sig]): continue
                if go_tgt[sig].get((0, _el), 0) >= ng: continue
                _maj = max(((k, v) for k, v in go_tgt[sig].items() if k[1] != _el and v > 0),
                           key=lambda kv: kv[1], default=None)
                if _maj is None: continue
                _cand.append((state_probs[sig].get((0, _el), 0) * ng - go_tgt[sig].get((0, _el), 0), sig, _maj[0]))
            if not _cand: break
            _cand.sort(reverse=True)
            _, sig, _majk = _cand[0]
            go_tgt[sig][_majk] -= 1
            if go_tgt[sig][_majk] == 0: del go_tgt[sig][_majk]
            go_tgt[sig][(0, _el)] = go_tgt[sig].get((0, _el), 0) + 1
        if 0 < _ex < 0.5 and _realized_el(_el) >= 1:
            minority_rescue[_el] = {"expected_count": round(_ex, 3), "placed": _realized_el(_el),
                                    "note": "dilute species kept at >=1 (deviation quoted in fidelity)"}
        elif _realized_el(_el) < _want:
            minority_rescue[_el] = {"expected_count": round(_ex, 3), "placed": _realized_el(_el),
                                    "note": "could not reach expected count (no demotable single-role host)"}

    # precompute for Fix B (per-acceptor safe-acceptor gate): supercell symbols
    _scsym = sc.get_chemical_symbols()

    # ---- H-RESTORE plan: valence-driven completion of formula-declared, X-ray-invisible H ----
    # Many CIFs list H in the formula but give no H coordinates (X-ray cannot locate water/OH/amine H).
    # The engine builds from coordinates, so those H are missing -> the cell is wrongly charged and
    # routes to 'custom'. GENERAL (no species lookup): (a) deficit = formula H scaled by an ORDERED
    # reference element, minus H already present; (b) each O/N acceptor has capacity = valence (O=2,N=3)
    # minus its located covalent neighbours; (c) reconcile the deficit against COMPLETE capacity-tiers
    # of SAFE acceptors. An acceptor is SAFE if it is isolated (water) or bonded only to an ORDERED
    # skeleton; one bonded to a DISORDERED partner is a coupled unit whose orientation the per-atom
    # decoration would shatter -> NOT completed here, FLAGGED for the (future) orientation builder /
    # custom. If the deficit fills complete safe tiers exactly, add that many H per decoration.
    _disidx = {g for g, _ in dsites}
    import re as _re
    h_plan = {"apply": False}
    _fs = (ev.get('provenance') or {}).get('formula_sum') or ''
    _fd = {mm.group(1): float(mm.group(2) or 1) for mm in
           _re.finditer(r'([A-Z][a-z]?)(\d+\.?\d*)?', str(_fs).replace(' ', '')) if mm.group(1)}
    _fH = _fd.get('H', 0.0)
    _present_H = sum(1 for s in sc.get_chemical_symbols() if s == 'H')
    if _fH <= 0:
        h_restore = {"applied": False, "reason": "no H in formula"}
    else:
        _dis_els = {e for orb in occ_of.values() for e in orb}
        _refs = [(e, psyms.count(e)) for e in ({psyms[bi] for bi in range(nbase) if dis_orbit[bi] == -1}
                                               - _dis_els) if _fd.get(e, 0) > 0]
        if not _refs:
            h_restore = {"applied": False, "reason": "no ordered reference element to scale formula H"}
        else:
            _ref_el, _ref_n = max(_refs, key=lambda x: x[1])
            _declared_H = int(round(_fH * (_ref_n / _fd[_ref_el]) * nrep))
            _deficit = _declared_H - _present_H
            def _orient_dis(k):
                # a DISORDERED former/anion site (O/F mixing, amine N, Zundel water) is an orientational
                # risk: per-atom H placement would shatter the coupled orientation. A disordered CATION
                # (Sc/In, Zn/Al layer) is NOT -- the O-H points outward regardless of which cation sits there.
                return (k in _disidx) and (_scsym[k] in COORD_FORMERS or _scsym[k] in COORD_ANIONS)
            def _accept(idx, excl, el):                   # (capacity, is_safe) for an O/N acceptor
                # Fix B (per-acceptor, not global): SAFE unless the acceptor itself is an orientational
                # disordered site, or it is bonded to one. An ordered SO4/PO4 elsewhere no longer blocks an
                # unrelated hydroxide H (the old global rigid flag did). Hydroxide O bonded to mixed cations
                # -> safe; XeO2F/NH3/Zundel/ice acceptors -> unsafe -> orientation builder / custom.
                cut = _h_nbr_cuts(_scsym)[idx]
                vv = get_distances([P[idx]], P, cell=C, pbc=True)[1][0]
                nbrs = [k for k in range(len(P)) if k != idx and k not in excl and vv[k] < cut[k]]
                safe = (not _orient_dis(idx)) and not any(_orient_dis(k) for k in nbrs)
                return max(0, H_VALENCE[el] - len(nbrs)), safe
            safe_tier = defaultdict(int); unsafe_cap = 0
            for bi in range(nbase):                       # ordered acceptors
                if dis_orbit[bi] == -1 and psyms[bi] in H_VALENCE:
                    cap, safe = _accept(bi, {bi}, psyms[bi])
                    if cap > 0 and safe: safe_tier[cap] += nrep
                    elif cap > 0: unsafe_cap += cap * nrep
            for sig, gids in go.items():                  # disordered acceptors (split-collapsed)
                rep = g_members[gids[0]]; excl = {dsites[m][0] for m in rep}
                for (role, el), cnt in go_tgt[sig].items():
                    if el in H_VALENCE:
                        cap, safe = _accept(dsites[rep[role]][0], excl, el)
                        if cap > 0 and safe: safe_tier[cap] += cnt
                        elif cap > 0: unsafe_cap += cap * cnt
            # the formula deficit is the authoritative H count; placeable iff it fits within the SAFE
            # acceptor capacity (greedy partial-tier fill in _complete_h does the actual placement --
            # e.g. 1 H per hydroxide O when deficit = n_OH < 2*n_OH total capacity).
            _total_safe = sum(cap * cnt for cap, cnt in safe_tier.items())
            clean = 0 < _deficit <= _total_safe
            # Fix C: aliovalent FORMER substituent (e.g. B3+-for-Si4+) -> the declared H are charge-
            # compensating Bronsted protons on bridging anions (the capacity model can't see them: a
            # bridging O is 2-coordinate, cap 0). Detect a mixed-former orbit with differing ox states.
            _sub_el, _sub_def = None, 0
            for _sig in go_tgt:
                _fe = {e for (_r, e) in state_probs[_sig] if e in COORD_FORMERS}
                _oxs = {e: ox.get(e, 0) for e in _fe}
                if len(_fe) > 1 and len(set(_oxs.values())) > 1:
                    _maj = max(_oxs, key=_oxs.get)
                    for (_r, e), c in go_tgt[_sig].items():
                        if e in _oxs and e != _maj:
                            _sub_el = e; _sub_def += int(round(_oxs[_maj] - _oxs[e])) * c
            if _deficit > 0 and clean:
                h_plan = {"apply": True, "n_target": _deficit}
                ev['oxidation_states'].setdefault('H', 1.0); ox['H'] = 1.0
                h_restore = {"applied": True, "method": "valence-driven completion", "ref_element": _ref_el,
                             "declared_H_supercell": _declared_H, "present_H_in_cif": _present_H,
                             "n_H_added": _deficit, "acceptor_capacities": dict(safe_tier)}
            elif _deficit > 0 and _sub_el and abs(_sub_def - _deficit) <= 1:
                h_plan = {"apply": True, "n_target": _deficit, "mode": "compensating", "substituent": _sub_el}
                ev['oxidation_states'].setdefault('H', 1.0); ox['H'] = 1.0
                h_restore = {"applied": True, "method": "charge-compensating Bronsted (aliovalent former)",
                             "substituent": _sub_el, "declared_H_supercell": _declared_H,
                             "present_H_in_cif": _present_H, "n_H_added": _deficit}
            elif _deficit > 0 and unsafe_cap == 0 and _bv_fallback_ok(_scsym, P, C, ox, _deficit):
                # FALLBACK, only where the capacity model found nothing placeable and no orientational
                # risk was flagged: rank acceptors by Pauling bond-strength deficit instead. A fully
                # coordinated bridging anion has capacity 0 and a real charge undersaturation, which is
                # what a hydrous silicate or phosphate is -- those used to fall through to `custom`
                # with the deficit unrestored. The capacity model still wins wherever it applies; this
                # never overrides it, and it declines outright when the CIF carries no oxidation states.
                h_plan = {"apply": True, "n_target": _deficit, "mode": "bond-valence"}
                ev['oxidation_states'].setdefault('H', 1.0); ox['H'] = 1.0
                h_restore = {"applied": True, "method": "bond-valence deficit (fallback)",
                             "ref_element": _ref_el, "declared_H_supercell": _declared_H,
                             "present_H_in_cif": _present_H, "n_H_added": _deficit,
                             "note": "the valence-capacity model found no placeable acceptor; protons "
                                     "ranked by Pauling bond-strength deficit instead"}
            elif _deficit > 0:
                h_restore = {"applied": False, "declared_H_supercell": _declared_H, "n_H_deficit": _deficit,
                             "unsafe_acceptor_capacity": unsafe_cap,
                             "reason": ("deficit needs coupled/disordered-unit acceptors -> orientation "
                                        "builder / custom" if unsafe_cap > 0 else
                                        f"deficit {_deficit} not reconciled by safe acceptors, and the "
                                        "bond-valence fallback could not either -> custom")}
            else:
                h_restore = {"applied": False, "declared_H_supercell": _declared_H, "reason": "no H deficit"}

    # ---- Branch 2: joint charge-constrained rounding (neutralise within faithfulness) ----
    fixed_comp = Counter(sc.get_chemical_symbols()[i] for i in range(len(sc)) if i not in _disidx)
    if h_plan["apply"]:
        fixed_comp['H'] = fixed_comp.get('H', 0) + h_plan["n_target"]   # planned protons (added per conf)
    # CONSTITUTIONAL vacancies (dummy-derived): their occupancy is the deliberate
    # truth -> match the corresponding orbit(s) by element-occupancy and FIX them, so charge-balancing does
    # not fill the vacancy (that would silently restore the full-occupancy stoichiometry). Residual charge is then honest.
    fixed_sigs = set()
    for _c in (strip_rep.get('constitutional_occ') or []):
        for sig in go:
            oo = {}
            for (role, el), o in state_probs[sig].items(): oo[el] = oo.get(el, 0) + o
            if set(oo) == set(_c) and all(abs(oo[e] - _c[e]) < 0.03 for e in _c):
                fixed_sigs.add(sig)
    go_tgt, charge_status = balance_charge(go, go_tgt, fixed_comp, ev['oxidation_states'], fixed_sigs=fixed_sigs)

    # cap the random-sample count at the number of DISTINCT decorations: a small combinatorial space
    # (few sites) doesn't benefit from NR=30 -- sampling more just relaxes duplicates. Astronomical for
    # solid solutions -> never caps. (Per-orbit multinomial ng!/(Π count! · vac!), product over orbits.)
    import math as _math
    def _ndist(cap):
        tot = 1
        for _sig, _gids in go.items():
            _ng = len(_gids); _t = go_tgt[_sig]; _vac = _ng - sum(_t.values())
            _den = _math.factorial(max(_vac, 0))
            for _c in _t.values(): _den *= _math.factorial(_c)
            tot *= _math.factorial(_ng) // _den
            if tot > cap: return cap + 1
        return tot
    NR = min(NR, _ndist(NR))

    # ---- Finding 2: detect rigid/orientational units; flag if per-site decoration is unreliable ----
    _ru = orient_units(s)
    rigid_flag = {"centers_detected": len(_ru),
                  "auto_handleable": sum(1 for r in _ru if r['status'] == 'clean'),
                  "ambiguous": sum(1 for r in _ru if r['status'].startswith('ambiguous')),
                  "note": ("ORIENTATIONAL rigid units present -> per-site decoration is UNRELIABLE; "
                           "route to a CUSTOM one-orientation build (the auto-enumerator does not "
                           "generalise -- combinatorial ambiguity)") if _ru else None}

    n_merged = sum(1 for g in g_members if len(g) > 1)

    # COUPLE_CUT data-derived: the former-anion BOND is measured -> the evidence reports the cross-orbit
    # contacts. Use the shortest former-anion contact x1.15 (captures the real bond, excludes longer
    # averaging-artifact contacts). Override with --couple-cut; fallback to the module default. NOT rigid:
    # if the shortest contact isn't the bonding one, the LLM should set it from chemistry.
    # WHICH ELEMENTS PLAY WHICH ROLE in the coupling path. `COORD_FORMERS`/`COORD_ANIONS` are fixed
    # tables, and D5 already reports when a structure's cross-orbit contacts match neither -- naming
    # a Co intermetallic and a Ga chalcogenide. What was missing is a way to ACT on that report:
    # `--couple-cut` retargets
    # the cutoff, it cannot add a role, so the gap was unfixable by any value (D37).
    # These REPLACE the table for this structure, exactly as `former_coordination(formers=...)` does.
    # NOT auto-derived, and that is measured, not caution: asking which elements BEHAVE like formers
    # (prevalence + uniform CN) fired on La, Mg, Fe, Ti, Zr, Ca, Cs, Er, Ni, Zn, Gd, Cu, Ba and a K
    # with CN 27 across 15 structures -- see `unexplained_dangling`'s "WHAT NOT TO DO". Whether Ga
    # centres a unit here is chemistry, so the engine asks with the numbers attached and the analyst
    # answers with a flag.
    cplF = set(couple_formers) if couple_formers else COORD_FORMERS
    cplA = set(couple_anions) if couple_anions else COORD_ANIONS
    _all_xc = ev['disorder'].get('cross_orbit_contacts_below_2.6A') or []
    _xc = [c['d_A'] for c in _all_xc
           if (set((c.get('orbit_a') or {})) & cplF and set((c.get('orbit_b') or {})) & cplA)
           or (set((c.get('orbit_b') or {})) & cplF and set((c.get('orbit_a') or {})) & cplA)]
    _els = sorted({e for c in _all_xc for k in ('orbit_a', 'orbit_b') for e in (c.get(k) or {})})
    cpl_cut = (couple_cut if couple_cut is not None
               else (round(min(_xc) * 1.15, 2) if _xc else COUPLE_CUT))
    # DEFECT (5): this said "no contact in evidence" whenever the FILTERED list came back empty,
    # which is only true when there were no contacts at all. On set A it was false for 3 of 4
    # entries -- and it reads as "nothing to see", so the self-driving loop stopped on an 82.8
    # meV/at spread instead of chasing it. The two cases need different actions (accept the default
    # vs. ask whether the element table covers this chemistry), so they must not share a sentence.
    couple_cut_report = {"value_A": cpl_cut, "basis": (
        "override --couple-cut" if couple_cut is not None
        else f"shortest former-anion contact {min(_xc):.2f} A x1.15" if _xc
        else f"no cross-orbit contact in evidence -> default {COUPLE_CUT} A" if not _all_xc
        else (f"{len(_all_xc)} cross-orbit contact(s) present but none matched "
              f"COORD_FORMERS/COORD_ANIONS (elements: {', '.join(_els)}) "
              f"-> default {COUPLE_CUT} A"))}
    # DEFECT (5). One string covered three different situations, and on set A it was TRUE for one entry
    # and FALSE for three: one really had no cross-orbit contact, while a Li conductor (9), the Co
    #                  intermetallic (3) and
    # the Ga chalcogenide (1) all had contacts that simply matched no former/anion rule. Split them.
    #   `vacuous`   -- no cross-orbit contact at all; the default is right by construction.
    #   `unmatched` -- contacts exist, none matched. The engine CANNOT tell a correct non-match
    #                  (Li is not a network former) from a table gap (Co and Ga are
    #                  absent from COORD_FORMERS/ANIONS). That distinction is chemistry, which is why
    #                  the analyst exists -- so ASK rather than guess, and hand over the numbers.
    _cev = {"n_contacts": len(_all_xc), "n_former_anion": len(_xc),
            "closest_A": round(min(c['d_A'] for c in _all_xc), 2) if _all_xc else None,
            "elements": _els}
    _cev['roles'] = {'formers': sorted(cplF) if couple_formers else 'COORD_FORMERS',
                     'anions': sorted(cplA) if couple_anions else 'COORD_ANIONS'}
    if couple_formers or couple_anions:
        _p = MEV.provenance('overridden',
                            couple_cut_report["basis"] + ' (roles set by hand)', _cev,
                            source='--couple-formers/--couple-anions')
    elif couple_cut is not None:
        _p = MEV.provenance('overridden', couple_cut_report["basis"], _cev, source='--couple-cut')
    elif _xc:
        _p = MEV.provenance('derived', couple_cut_report["basis"], _cev)
    elif not _all_xc:
        _p = MEV.provenance('vacuous', f"no cross-orbit contact below {MEV.COUPLE_A} A", _cev)
    else:
        _p = MEV.provenance('unmatched',
                            f"{len(_all_xc)} cross-orbit contact(s) below {MEV.COUPLE_A} A, none former-anion "
                            f"under COORD_FORMERS/COORD_ANIONS (elements: {', '.join(_els)}). "
                            f"If one of those elements DOES play the role here, say so with "
                            f"--couple-formers / --couple-anions -- the cutoff cannot add a role.",
                            _cev, source=f'COUPLE_CUT module default {COUPLE_CUT} A')
    couple_cut_report["provenance"] = _p

    # What the supercell ACTUALLY reached, next to what was asked for (defect D13). `overridden`
    # when the atom budget won: the value came from somewhere other than the derivation, which is
    # what that state means. A cut multiplier that still clears MIN_A kept the contract -- the
    # shortfall is the fact worth flagging, not the back-off.
    _cell_achieved = round(float(min(np.linalg.norm(C, axis=1))), 2)
    _cell_ev = {"min_cell_A_requested": MIN_A, "min_cell_A_achieved": _cell_achieved,
                "supercell_requested": list(mult_req), "supercell": list(mult),
                "n_atoms_template": int(len(sc)), "atom_budget": budget,
                "atom_budget_source": ('override --max-atoms' if max_atoms
                                       else f'default MAXAT*1.6={int(MAXAT * 1.6)}')}
    supercell_prov = (
        MEV.provenance('derived', f'{list(mult)} from min_cell {MIN_A} A', evidence=_cell_ev)
        if _cell_achieved >= MIN_A else
        MEV.provenance('overridden',
                       f'atom budget cut the supercell {list(mult_req)} -> {list(mult)}; shortest '
                       f'axis {_cell_achieved} A is BELOW the requested {MIN_A} A',
                       evidence=_cell_ev, source=f'atom budget {budget}'
                              + ('' if max_atoms else
                                 f' (default MAXAT*1.6; raise it with --max-atoms if the cell rule '
                                 f'matters more than the cost)')))

    def _orbit_exclusion(gids):
        """Which mechanism keeps this orbit's excluded sites apart -- reported, not left to inference.

        clique-merged : the component was a clique of alternates, so it is ONE group and exactly one
                        atom is placed. The exclusion lives in the merge.
        pairwise      : the component was a chain/network, so it was split into singletons and its
                        conflict edges are enforced at decoration (independent sets). `merged` is
                        false here and exclusion IS enforced -- that is the pair that was misread.
        none          : no site pair of this orbit falls under the cutoff. Nothing to exclude.
        """
        mem = {m for g in gids for m in g_members[g]}
        n_pairs = sum(1 for a in mem for b in excl_adj.get(a, ()) if b in mem and a < b)
        if len(g_members[gids[0]]) > 1:
            mode, note = 'clique-merged', 'one atom per merged group; exclusion is in the merge'
        elif n_pairs:
            mode, note = 'pairwise', ('sites kept separate; excl_adj forbids the excluded pairs at '
                                      'decoration, so exclusion IS enforced despite merged=false')
        else:
            mode, note = 'none', 'no site pair of this orbit falls under the cutoff'
        return {'mode': mode, 'n_pairs': n_pairs, 'note': note}

    info = {"icsd_id": iid, "supercell": list(mult), "supercell_provenance": supercell_prov,
            "n_atoms_template": int(len(sc)),
            "charge_balancing": charge_status, "rigid_units": rigid_flag, "h_restore": h_restore,
            "minority_rescue": minority_rescue or None,
            "full_occ_clash": foc,
            "dummy_species": dummy if dummy.get('detected') else None,
            "exclusion_merge": exclusion_merge_report, "couple_cut": couple_cut_report,
            "axes_A": [round(float(x), 1) for x in np.linalg.norm(C, axis=1)],
            "exclusion_A": EXCL, "n_groups_merged": n_merged,
            "group_orbits": {go_ids[sig]: {
                "members_per_group": len(g_members[gids[0]]),
                "merged": len(g_members[gids[0]]) > 1, "n_groups": len(gids),
                # D35: `merged: false` was read as "no exclusion" and produced three consecutive
                # misreads on one entry. It only ever meant "not clique-merged" -- a NON-clique
                # component is split into singletons and its pairwise edges are enforced at
                # decoration through `excl_adj` (valid occupations are the independent sets), and
                # `excl_adj` never left the engine. So `generation_recipe` alone could not verify
                # exclusion handling, and the scorer had to re-derive it from the CIF operators by
                # hand on three entries. Name which of the three cases this orbit is.
                "exclusion": _orbit_exclusion(gids),
                "states": {f"{r}:{e}": round(p, 3) for (r, e), p in state_probs[sig].items()},
                "target": {f"{r}:{e}": c for (r, e), c in go_tgt[sig].items()}}
                for sig, gids in go.items()},
            "sampling": {"n_random": NR, "modes": ["random", "dispersed", "clustered"],
                         "seed": seed, "min_cell_A": MIN_A,
                         "min_cell_A_achieved": _cell_achieved, "d3": d3},
            "coupling_signal": ev['disorder']['cross_orbit_contacts_below_2.6A'],
            "ordered_sibling": ev['ordered_sibling']}
    # ONE place a gate can check, instead of walking the tree. Stage (2) (diagnostics must be consumed)
    # hangs off `n_unconfident`; today it is reporting only.
    info["provenance_summary"] = MEV.provenance_summary({
        "exclusion_merge": merge_prov, "couple_cut": couple_cut_report["provenance"],
        "supercell": supercell_prov,
        # The charge balancer belongs here, not only in `charge_balancing`: it is the step that can
        # silently DELETE the disorder an entry exists for, and a summary that cannot see it
        # reported n_unconfident = 0 while exactly that happened.
        "charge_balancing": charge_status.get('provenance'),
        "ordered_sibling": ev.get('ordered_sibling_provenance')})

    # ---- coupling precompute: constructive co-placement when the loop discovers a dangling/under-coord defect ----
    def orbit_els(sig):
        return {el for (_r, el) in state_probs[sig]}
    former_nbrs = {}                                       # disordered-anion global idx -> bonded disordered-former global idxs
    if couple:
        _fc = [dsites[m][0] for sg in go for gg in go[sg] if (orbit_els(sg) & cplF) for m in g_members[gg]]
        _ac = [dsites[m][0] for sg in go for gg in go[sg]
               if (orbit_els(sg) & cplA) and not (orbit_els(sg) & cplF) for m in g_members[gg]]
        if _fc and _ac:
            _Dfa = get_distances(P[_ac], P[_fc], cell=C, pbc=True)[1]
            for _k, _ag in enumerate(_ac):
                former_nbrs[_ag] = [_fc[_fi] for _fi in np.where(_Dfa[_k] < cpl_cut)[0]]

    def decorate(mode, rng):
        ss = list(sc.get_chemical_symbols()); drop = []; assign = {}; short = 0
        occ_local = set()                                  # conflict-graph: occupied dsites-local indices this config
        # COUPLE: co-select former-anion UNITS via bipartite matching (each anion group paired to a DISTINCT
        # former site through a bond), so every occupied anion sits on an occupied former -> no dangling.
        Arole, occ_Fgrp = {}, set()
        if couple and former_nbrs:
            F_groups = [g for sg in go if (orbit_els(sg) & cplF) for g in go[sg]]
            A_groups = [g for sg in go if (orbit_els(sg) & cplA) and not (orbit_els(sg) & cplF)
                        for g in go[sg]]
            glob2F = {dsites[g_members[g][0]][0]: g for g in F_groups}      # former site global idx -> F group
            Aadj = {}
            for ag in A_groups:
                opts = [(glob2F[fg], role) for role, m in enumerate(g_members[ag])
                        for fg in former_nbrs.get(dsites[m][0], ()) if fg in glob2F]
                rng.shuffle(opts); Aadj[ag] = opts
            matchF = {}
            def _aug(ag, seen):
                for fg, role in Aadj[ag]:
                    if fg in seen: continue
                    seen.add(fg)
                    if fg not in matchF or _aug(matchF[fg], seen):
                        matchF[fg] = ag; Arole[ag] = role; return True
                return False
            ag_order = list(A_groups); rng.shuffle(ag_order)
            for ag in ag_order: _aug(ag, set())
            occ_Fgrp = set(matchF)
        for sig, gids in go.items():
            ng = len(gids); els = orbit_els(sig)
            flat = []                                     # target states (minority first); first ntar are non-VAC
            for sk in sorted(go_tgt[sig], key=go_tgt[sig].get):
                flat += [sk] * go_tgt[sig][sk]
            ntar = len(flat); flat += ['VAC'] * (ng - ntar)
            if couple and former_nbrs and (els & COORD_FORMERS):
                # occupy MATCHED former groups first, top up to target with random others
                matched = [gl for gl in range(ng) if gids[gl] in occ_Fgrp]
                rest = [gl for gl in range(ng) if gids[gl] not in occ_Fgrp]; rng.shuffle(rest)
                order = matched + rest
            elif couple and former_nbrs and (els & COORD_ANIONS):
                # occupy MATCHED anion groups first (so they're the ones kept if target<ng), at their bonded role
                matched = [gl for gl in range(ng) if gids[gl] in Arole]
                rest = [gl for gl in range(ng) if gids[gl] not in Arole]; rng.shuffle(rest)
                order = matched + rest
            else:
                order = list(range(ng))
                if mode == 'random':
                    rng.shuffle(order)
                elif mode in ('dispersed', 'clustered'):
                    cent = np.array([g_centroid[g] for g in gids])
                    D = get_distances(cent, cent, cell=C, pbc=True)[1]
                    seq = [int(rng.integers(ng))]
                    while len(seq) < ng:
                        rest = [i for i in range(ng) if i not in seq]
                        key = (lambda i: min(D[i, j] for j in seq))
                        seq.append(max(rest, key=key) if mode == 'dispersed' else min(rest, key=key))
                    order = seq
            placed = 0                                    # conflict-graph: fill the target as an INDEPENDENT SET
            for gl in order:
                g = gids[gl]; mem = g_members[g]
                vacate = placed >= ntar                   # target already met -> vacate the rest
                if not vacate:
                    sk = flat[placed]; role, el = sk
                    if couple and former_nbrs and (els & COORD_ANIONS):
                        role = Arole.get(g, role)         # matched anion uses its bonded role
                    chosen = mem[role]
                    if excl_adj.get(chosen) and (excl_adj[chosen] & occ_local):
                        vacate = True                     # would sit on an occupied neighbour -> skip, keep pursuing target
                if vacate:
                    assign[g] = 'VAC'
                    for m in mem: drop.append(dsites[m][0])
                    continue
                assign[g] = sk; ss[dsites[chosen][0]] = el; occ_local.add(chosen)
                for m in mem:
                    if m != chosen: drop.append(dsites[m][0])
                placed += 1
            # The fill is a SINGLE greedy pass: a group skipped because a conflict neighbour is already
            # occupied is vacated for good, never retried. On a conflict cycle that can exhaust `order`
            # before the target is met -- e.g. a 6-cycle reaches 2 of 3 about a quarter of the time.
            # Such a config is a DIFFERENT COMPOSITION (cation-deficient), not a decoration of this one,
            # and per-atom energy would REWARD it (dropping an atom lowers E/atom). Report it; the caller
            # invalidates the sample rather than letting take-lowest select it.
            if placed < ntar:
                short += ntar - placed
        at = sc.copy(); at.set_chemical_symbols(ss)
        if drop: del at[sorted(set(drop))]
        if h_plan.get("mode") == "compensating":          # Fix C: charge-compensating Bronsted protons
            at, _ = _place_compensating_h(at, h_plan["substituent"], h_plan["n_target"])
        elif h_plan["apply"]:                             # restore X-ray-invisible H
            at, _ = _complete_h(at, h_plan["n_target"], ox=ox,
                                mode=h_plan.get("mode") or 'capacity')
        return at, assign, short

    def features(assign):
        """minority-state dispersal per group-orbit: mean NN distance among minority-occupied
        sites, normalised by the group-orbit NN distance. >1 dispersed, <1 clustered."""
        f = {}
        for sig, gids in go.items():
            sk = go_minor[sig]
            if sk is None: continue
            role, el = sk
            pts = [P[dsites[g_members[g][role]][0]] for g in gids if assign.get(g) == sk]
            key = f"go{go_ids[sig]}_minor_disp"
            if len(pts) < 2:
                f[key] = None; continue
            Dm = get_distances(np.array(pts), np.array(pts), cell=C, pbc=True)[1]
            np.fill_diagonal(Dm, 9e9)
            nn = go_nn.get(sig, 1.0)
            f[key] = round(float(Dm.min(axis=1).mean() / nn), 2) if nn > 1e-6 else None
        return f

    confs, samples, rng = [], [], np.random.default_rng(seed)
    if couple:
        # CONSTRUCTIVE round: each decoration co-places anions onto occupied formers (no oversample/filter
        # needed -- the coupling is enforced at generation, so every config is in the accessible manifold).
        for k in range(NR):
            at, asg, shf = decorate('random', rng); confs.append(at)
            samples.append({"label": f"cpl{k}", "mode": "coupled", "features": features(asg), "short_fill": shf})
        info["coupled_round"] = {"mode": "constructive co-placement (anion bonded to occupied former)",
                                 "couple_cut_A": COUPLE_CUT}
    elif select is None:
        for k in range(NR):
            at, asg, shf = decorate('random', rng); confs.append(at)
            samples.append({"label": f"rand{k}", "mode": "random", "features": features(asg), "short_fill": shf})
        for mode in ('dispersed', 'clustered'):
            at, asg, shf = decorate(mode, rng); confs.append(at)
            samples.append({"label": mode, "mode": mode, "features": features(asg), "short_fill": shf})
    else:
        # CONSTRAINED round: oversample cheap decorations, score by the discovered-defect proxy on the
        # UNRELAXED structure, relax only the NR lowest-defect -> confines the ensemble to the accessible
        # (low-energy) manifold instead of wasting relaxations on rule-violating configs.
        cand = []
        for k in range(oversample * NR):
            at, asg, shf = decorate('random', rng); cand.append((float(select(at)), at, asg, shf))
        cand.sort(key=lambda t: t[0])
        # `score`, not `sc`: `sc` is the SUPERCELL Atoms built at the top of this function. Rebinding
        # it to a float here is harmless only because nothing reads it afterwards -- a silent trap
        # for the next edit that does.
        for r, (score, at, asg, shf) in enumerate(cand[:NR]):
            confs.append(at)
            samples.append({"label": f"sel{r}", "mode": "constrained", "features": features(asg),
                            "select_score": round(score, 3), "short_fill": shf})
        info["constrained_round"] = {"oversampled": oversample * NR, "kept": len(confs),
                                     "select_score_range": [round(cand[0][0], 3), round(cand[min(NR, len(cand)) - 1][0], 3)]}

    if relax is None:
        S.use_model(model='7net-nano', d3=d3)
        E, rel, val = S.batched_fire_relax(confs)  # memory sized by torch-sim's autobatcher
    else:
        E, rel, val = relax(confs)                # injected relaxer -> pluggable MLIP backend (no spinner global)
    for i, smp in enumerate(samples):
        if smp.get("short_fill"):                                     # decoration never reached the occupancy
            smp.update(E_per_atom=None, valid=False,                  # target -> not a decoration of this
                       invalid_reason=f"short-filled by {smp['short_fill']} site(s)")   # composition
        elif val[i]:
            at = rel[i]; pos = at.get_positions(); cc = np.array(at.get_cell())
            D = get_distances(pos, pos, cell=cc, pbc=True)[1]; np.fill_diagonal(D, 9e9)
            syms = at.get_chemical_symbols()
            _ij = np.unravel_index(int(D.argmin()), D.shape)          # element pair of the closest contact
            mdpair = '-'.join(sorted([syms[_ij[0]], syms[_ij[1]]]))   # so a real bond (C-O 1.14) != a clash
            q = sum(ox.get(e, 0.0) for e in syms)
            smp.update(E_per_atom=round(float(E[i]) / len(at), 4), charge=round(float(q), 2),
                       min_dist_A=round(float(D.min()), 2), min_dist_pair=mdpair, n_atoms=len(at),
                       valid=True, composition=dict(Counter(syms)), descriptors=config_descriptors(at))
        else:
            smp.update(E_per_atom=None, valid=False)
    # NEVER rank samples of different composition by per-atom energy -- E/atom is not comparable across
    # them, and the deficient one wins. short_fill catches the decoration path; this catches whatever else
    # changes the atom count downstream (H restoration, compensating protons, minority rescue).
    _cmp = [tuple(sorted(s["composition"].items())) for s in samples if s.get("valid")]
    if _cmp:
        _modal = Counter(_cmp).most_common(1)[0][0]
        for s in samples:
            if s.get("valid") and tuple(sorted(s["composition"].items())) != _modal:
                s.update(valid=False, E_per_atom=None,
                         invalid_reason="composition differs from the modal decoration")
    if not any(s.get("valid") for s in samples):
        # Every sample was rejected. The usual cause is an exclusion cut so wide that the occupancy
        # target is unreachable as an independent set -- report it instead of dying on min() of an
        # empty list downstream, and say which reason dominated so the cut can be re-chosen.
        _why = Counter(s.get("invalid_reason") or "relaxation did not converge" for s in samples)
        info["error"] = ("no valid decoration: " +
                         "; ".join(f"{n}x {r}" for r, n in _why.most_common()))
        info["short_fill_per_sample"] = [s.get("short_fill") for s in samples]
        return ev, None, info
    return ev, (confs, rel, val, samples, info), info

def run_self_driving(iid=None, NR=30, MIN_A=15.0, EXCL=1.1, d3=False, max_rounds=3, same_excl=None, couple_cut=None,
                     structure=None, evidence=None, relax=None, max_atoms=None,
                     couple_formers=None, couple_anions=None):
    """SELF-DRIVING loop: enumerate -> if the energy spread is broad, DIAGNOSE the spectrum for the
    defect descriptor that drives the high-energy configs, then RE-ENUMERATE constrained to suppress it;
    repeat. Stops when the spread is within the normal range (manifold tight) OR no descriptor tracks
    energy (residual spread is genuine SRO, not a removable defect). The constraint is DISCOVERED from
    the energy-structure correlation, not pre-specified. Returns (ev, built_final, info, trace)."""
    trace = []
    ev, built, info = build(iid, NR=NR, MIN_A=MIN_A, EXCL=EXCL, d3=d3, seed=0, max_atoms=max_atoms, same_excl=same_excl, couple_cut=couple_cut,
                            structure=structure, evidence=evidence, relax=relax,
                            couple_formers=couple_formers, couple_anions=couple_anions)
    if built is None:
        return ev, None, info, trace
    for rnd in range(max_rounds):
        confs, rel, val, samples, info = built
        ok = [s for s in samples if s.get('valid')]
        Es = [s['E_per_atom'] for s in ok]
        spread = 1000.0 * (max(Es) - min(Es)) if len(Es) >= 2 else 0.0
        lo = min(ok, key=lambda s: s['E_per_atom']); lod = lo.get('descriptors', {})
        # D12: count only the dangling anions that re-decorating could actually remove. One entry
        # spent all three rounds trying to fix 100 anions whose centre was simply missing from
        # COORD_FORMERS -- an unreachable target, the same failure as D6's unreachable CN.
        _dangling = lod.get('dangling_unexplained', lod.get('dangling_anions', 0))
        rep_coupling_defects = _dangling + lod.get('under_coord_formers', 0)
        entry = {"round": rnd, "mode": ("unconstrained" if rnd == 0 else "constrained"),
                 "n_valid": len(ok), "spread_meV": round(spread, 1),
                 "rep_coupling_defects": rep_coupling_defects}
        if lod.get('dangling_explained_by_unlisted_former'):
            entry["unlisted_former_candidates"] = lod['dangling_explained_by_unlisted_former']
            entry["note"] = (f"{lod.get('dangling_anions')} anion(s) counted as dangling, "
                             f"{_dangling} of them unexplained; the rest bond an element absent from "
                             f"COORD_FORMERS -- confirm whether it centres a unit here")
        # TRIGGER: broad spread OR the representative itself carries coupling defects (dangling / under-
        # coordinated formers). The latter catches the case where merging the rings already collapsed the
        # spread to the SRO scale but the anion ORIENTATION still isn't matched to its former (residual
        # under-coordination) -- spread alone would miss it; coordination-integrity is the reliable trigger.
        if spread <= BROAD_SPREAD_MEV and rep_coupling_defects == 0:
            entry["action"] = "stop: spread normal AND representative coordination-clean (accessible manifold)"
            trace.append(entry); break
        diag = diagnose_spectrum(Es, [s.get('descriptors', {}) for s in ok])
        entry["diagnosis"] = {"top": diag["top"], "ranked": diag["ranked"][:5]}
        if diag["top"] is None and rep_coupling_defects == 0:
            entry["action"] = "stop: no defect descriptor tracks energy -> residual spread is genuine SRO"
            trace.append(entry); break
        # pick what to enforce: a coupling defect (energy-correlated OR present in the representative) ->
        # CONSTRUCTIVE co-placement; otherwise the diagnosed generic defect -> oversample+filter.
        dstar = (diag["top"]["descriptor"] if diag["top"] else "under_coord_formers")
        coupling_defect = dstar in ("dangling_anions", "under_coord_formers", "bridging_anions") or rep_coupling_defects > 0
        if coupling_defect and dstar not in ("dangling_anions", "under_coord_formers", "bridging_anions"):
            dstar = "under_coord_formers"   # representative-defect-driven couple
        # D6: the loop used to chase a target it had no way of reaching. When `diag["top"]` is None
        # nothing correlated with energy, but a representative carrying defects still forced a round
        # -- on the entry that found this, the descriptor it went after had rho = -0.682, i.e. FEWER
        # of those "defects" went with HIGHER energy. Removing them was the wrong direction, and the
        # loop spent its whole budget on it. Two guards, both from evidence already in hand.
        _rho = {k: r for k, r, _lo, _hi in diag["ranked"]}
        entry["enforced"] = dstar
        if diag["top"] is None and _rho.get(dstar, 0.0) <= -diag["min_rho"]:
            entry["action"] = (
                f"stop: '{dstar}' tracks energy NEGATIVELY (rho {_rho[dstar]}) -- suppressing it "
                f"would RAISE the energy, so the count is not a removable defect here (it is more "
                f"likely stoichiometric: the target coordination cannot be reached by re-decorating)")
            trace.append(entry); break
        _prev = trace[-1] if trace else None
        if (_prev and _prev.get("enforced") == dstar
                and _prev.get("rep_coupling_defects", -1) <= rep_coupling_defects):
            entry["action"] = (
                f"stop: the previous round already enforced '{dstar}' and the representative did not "
                f"improve ({_prev['rep_coupling_defects']} -> {rep_coupling_defects}) -- repeating it "
                f"spends the budget on an unreachable target")
            trace.append(entry); break
        _why = (f"rho {diag['top']['rho']}, low-E {diag['top']['low_E_value']} -> high-E {diag['top']['high_E_value']}"
                if diag["top"] else f"representative has {rep_coupling_defects} coupling defects")
        entry["action"] = (f"enforce: re-enumerate "
                           f"{'CONSTRUCTIVELY co-placing (couple)' if coupling_defect else 'oversample+filter'} "
                           f"to minimise '{dstar}' ({_why})")
        trace.append(entry)
        if rnd == max_rounds - 1: break                      # round budget spent; keep current ensemble
        if coupling_defect:
            ev, nb, info = build(iid, NR=NR, MIN_A=MIN_A, EXCL=EXCL, d3=d3, seed=rnd + 1, couple=True, max_atoms=max_atoms, same_excl=same_excl, couple_cut=couple_cut,
                                 structure=structure, evidence=evidence, relax=relax,
                                 couple_formers=couple_formers, couple_anions=couple_anions)
        else:
            sel = (lambda key: (lambda at: config_descriptors(at).get(key, 0)))(dstar)
            ev, nb, info = build(iid, NR=NR, MIN_A=MIN_A, EXCL=EXCL, d3=d3, seed=rnd + 1, select=sel, max_atoms=max_atoms, same_excl=same_excl, couple_cut=couple_cut,
                                 structure=structure, evidence=evidence, relax=relax,
                                 couple_formers=couple_formers, couple_anions=couple_anions)
        if nb is None: break
        built = nb
    info["self_driving"] = trace
    return ev, built, info, trace

def relax_single(atoms, ox, d3, final):
    """Relax ONE structure (a model-repaired/custom build, an ordered sibling for the ΔE COMPARISON,
    or an entry with no disorder) through the same nano(+D3) -> omni-mpa(+D3) tiers. The uniform final
    endpoint. NOTE: a sibling relaxed here is only ever COMPARED (ΔE); it is never adopted as the MAR."""
    S.use_model(model='7net-nano', d3=d3)
    E, rel, val = S.batched_fire_relax([atoms])
    if not val[0]: return {"valid": False}
    at = rel[0]; pos = at.get_positions(); cc = np.array(at.get_cell())
    D = get_distances(pos, pos, cell=cc, pbc=True)[1]; np.fill_diagonal(D, 9e9)
    q = sum(ox.get(e, 0.0) for e in at.get_chemical_symbols())
    out = {"valid": True, "E_per_atom": round(float(E[0]) / len(at), 4),
           "charge": round(float(q), 2), "min_dist_A": round(float(D.min()), 2), "n_atoms": len(at)}
    if final:
        S.use_model(model='7net-omni', modal='mpa', d3=d3)
        Ef, relf, vf = S.batched_fire_relax([at])
        if vf[0]: out["E_final_per_atom"] = round(float(Ef[0]) / len(relf[0]), 4)
    return out

def _ox_of(ev):
    return {el: (sum(v) / len(v) if isinstance(v, list) else v)
            for el, v in ev['oxidation_states'].items()}

def main():
    args = sys.argv[1:]
    iid = int([a for a in args if not a.startswith('-')][0])
    NR = int(args[args.index('--nr') + 1]) if '--nr' in args else 30
    MIN_A = float(args[args.index('--min-nm') + 1]) * 10 if '--min-nm' in args else 15.0
    EXCL = float(args[args.index('--excl') + 1]) if '--excl' in args else 1.1
    max_atoms = int(args[args.index('--max-atoms') + 1]) if '--max-atoms' in args else None  # raise the cell-size budget
    same_excl = float(args[args.index('--same-excl') + 1]) if '--same-excl' in args else None  # override the data-derived
    couple_cut = float(args[args.index('--couple-cut') + 1]) if '--couple-cut' in args else None  # same-element/former-anion cutoffs
    d3 = '--d3' in args                               # DeMARS uses NO dispersion by default; --d3 opt-in only
    _real = sys.stdout; sys.stdout = sys.stderr

    if '--from' in args:                              # relax a provided structure (custom build, or sibling for ΔE comparison)
        from ase.io import read as aseread
        src = args[args.index('--from') + 1]; at = aseread(src)
        r = relax_single(at, _ox_of(MEV.evidence(iid)), d3, '--final' in args)
        sys.stdout = _real
        print(json.dumps({"icsd_id": iid, "mode": "from", "source": src, "single_MAR": r},
                         indent=1, default=str)); return 0

    max_rounds = int(args[args.index('--rounds') + 1]) if '--rounds' in args else (1 if '--no-auto' in args else 3)
    ev, built, info, sd_trace = run_self_driving(iid, NR=NR, MIN_A=MIN_A, EXCL=EXCL, d3=d3, max_rounds=max_rounds,
                                                 same_excl=same_excl, couple_cut=couple_cut,
                                                 max_atoms=max_atoms)
    if built is None:
        if info.get("error") == "no disordered sites":   # nothing to enumerate -> relax as-is
            r = relax_single(load(iid), _ox_of(ev), d3, '--final' in args)
            sys.stdout = _real
            print(json.dumps({"icsd_id": iid, "mode": "no-disorder", "single_MAR": r,
                              "ordered_sibling": ev['ordered_sibling']}, indent=1, default=str)); return 0
        sys.stdout = _real
        print(json.dumps({"icsd_id": iid, "error": info.get("error")})); return 1
    confs, rel, val, samples, info = built
    ok = [smp for smp in samples if smp.get("valid")]
    Es = sorted(smp["E_per_atom"] for smp in ok)
    # the DISTRIBUTION still describes every valid sample (probes included -- bounding it is their
    # job); only the representative is chosen from the ship candidates.
    ship = ship_candidates(samples)
    lo = min(ship, key=lambda x: x["E_per_atom"]); hi = max(ok, key=lambda x: x["E_per_atom"])
    dist = {"n_relaxed": len(ok), "n_total": len(samples), "ground_E": Es[0], "highest_E": Es[-1],
            "spread_meV": round(1000 * (Es[-1] - Es[0]), 2),
            "std_meV": round(1000 * float(np.std([s["E_per_atom"] for s in ok])), 2),
            "energies_sorted": Es, "lowest": lo, "highest": hi,
            "n_ship_candidates": len(ship), "bracket_probes": bracket_margins(samples)}
    from .mar_record import dist_stats as _ds                # #4: indicative distribution descriptors (small-n robust)
    dist.update({k: v for k, v in _ds(Es).items() if k in ('mean_meV_per_atom', 'shape', 'sro_read_indicative')})
    # broad-spread flag: TRIGGER (not a verdict) for the investigate-and-re-plan loop. A spread this
    # large means the per-atom decoration produced wildly different energies -> a coupled/rigid unit it
    # shattered (a planar oxo-anion ring shattered by per-atom decoration reaches ~0.5 eV/at) OR genuine
    # strong ordering. Distinguish by examining coordination integrity / hypothesising a unit and
    # re-enumerating. Complementary to exclusion-merge, which catches split-site coupling that relaxation
    # heals to a SMALL spread (tens of meV).
    dist["broad_spread_flag"] = {
        "spread_meV_per_atom": dist["spread_meV"], "threshold_meV_per_atom": BROAD_SPREAD_MEV,
        "triggered": dist["spread_meV"] > BROAD_SPREAD_MEV,
        "action": ("INVESTIGATE: missed coupled/rigid unit (re-enumerate by unit) OR strong ordering"
                   if dist["spread_meV"] > BROAD_SPREAD_MEV else None)}
    info["broad_spread_flag"] = dist["broad_spread_flag"]   # surface alongside the other engine flags
    # coordination-integrity of the representative (lowest) -- a deterministic check that catches a
    # shattered/defective covalent unit even when the spread is small (the reliable trigger + the gate
    # that verifies any re-plan). Computed on the relaxed lowest decoration.
    _loi = samples.index(lo)
    if val[_loi]:
        ci = coordination_integrity(rel[_loi])
        dist["coordination_integrity"] = ci; info["coordination_integrity_flag"] = ci["integrity_flag"]
    out = {"icsd_id": iid, "engine": info, "distribution": dist, "samples": samples}

    final_struct = None
    if '--final' in args:
        widx = samples.index(lo)
        # Same two rules as api._final_omni, which is the OTHER way into this tier:
        #  (a) hand the sampling tier's GPU memory back first -- otherwise nano and omni sit on the
        #      card together, on top of the allocator pool the sampling run grew;
        #  (b) a tier that was ASKED FOR and could not run must say so. Leaving the key unset makes
        #      it indistinguishable from "no --final", because tools/demars_engine.py coerces a
        #      missing value to {} -- which is exactly the empty final_MAR of D10.
        S.release_models()
        S.use_model(model='7net-omni', modal='mpa', d3=d3)
        n_at = len(rel[widx])
        try:
            Ef, relf, vf = S.batched_fire_relax([rel[widx]])
        except Exception as exc:
            vf = [False]
            out["final_MAR"] = {"available": False, "requested": True, "n_atoms": n_at,
                                "modal": "mpa" + ("+D3" if d3 else ""),
                                "reason": f"{type(exc).__name__}: {exc}"}
        if vf[0]:
            final_struct = relf[0]
            out["final_MAR"] = {"engine_label": lo["label"],
                                "E_final_per_atom": round(float(Ef[0]) / len(relf[0]), 4),
                                "n_atoms": len(relf[0]), "modal": "mpa" + ("+D3" if d3 else "")}
        elif "final_MAR" not in out:
            out["final_MAR"] = {"available": False, "requested": True, "n_atoms": n_at,
                                "modal": "mpa" + ("+D3" if d3 else ""),
                                "reason": "the omni-mpa relaxation did not converge"}

    if '--out' in args:                               # persist the ENSEMBLE (deliverable)
        from ase.io import write as asewrite
        import os as _os
        od = args[args.index('--out') + 1]; _os.makedirs(od, exist_ok=True)
        frames = []
        for i, smp in enumerate(samples):
            if smp.get('valid'):
                at = rel[i].copy()
                at.info.update(label=smp['label'], mode=smp['mode'],
                               nano_E_per_atom=smp['E_per_atom'], charge=smp.get('charge'))
                frames.append((smp['E_per_atom'], at))
        frames.sort(key=lambda x: x[0])
        asewrite(f'{od}/ensemble.xyz', [a for _, a in frames], format='extxyz')   # energy-marked, OVITO-native
        # the written representative must be the SAME sample `distribution.lowest` names and the same
        # one the final tier relaxed -- all three go through `ship_candidates`, so a bracket probe
        # cannot become the deliverable by winning take-lowest.
        _ship_labels = {s.get('label') for s in ship_candidates(samples)}
        _rep = next((a for _, a in frames if a.info.get('label') in _ship_labels), frames[0][1])
        asewrite(f'{od}/representative.vasp', _rep, format='vasp', sort=True)
        asewrite(f'{od}/representative.cif', _rep, format='cif')
        ef = {'ensemble': f'{od}/ensemble.xyz', 'n_frames': len(frames),
              'representative': f'{od}/representative.vasp'}
        if final_struct is not None:
            asewrite(f'{od}/representative_final.vasp', final_struct, format='vasp', sort=True)
            ef['representative_final'] = f'{od}/representative_final.vasp'
        out['ensemble_files'] = ef

    if '--summary' in args:
        e = out['engine']; d = out['distribution']
        L = [f"ICSD {iid}  supercell {e['supercell']} -> {e['n_atoms_template']} atoms (template), axes {e['axes_A']} A",
             f"  exclusion-merge <{e['exclusion_A']}A cross / same-element data-derived "
             f"{e.get('exclusion_merge',{}).get('same_element_cut_per_element')}: {e['n_groups_merged']} groups merged",
             f"    basis: {e.get('exclusion_merge',{}).get('basis')}"]
        for sm in e.get('exclusion_merge', {}).get('suspicious_merges', []):
            L.append(f"    ** SUSPICIOUS MERGE: {sm} **")
        L.append(f"  couple_cut (former-anion bond): {e.get('couple_cut',{}).get('value_A')} A "
                 f"({e.get('couple_cut',{}).get('basis')})")
        for gid, g in e['group_orbits'].items():
            L.append(f"  group-orbit {gid}: {g['n_groups']} groups x {g['members_per_group']} site"
                     f"{'s (MERGED)' if g['merged'] else ''}  states {g['states']}  target {g['target']}")
        L += [f"  coupling signal: {len(e['coupling_signal'])} cross-orbit <2.6A   sibling: {e['ordered_sibling']}",
              f"  distribution: {d['n_relaxed']}/{d['n_total']} relaxed  GROUND {d['ground_E']}  "
              f"SPREAD {d['spread_meV']} meV/at  std {d['std_meV']} meV/at",
              f"  lowest = {d['lowest']['label']} ({d['lowest']['mode']})  feat {d['lowest']['features']}  "
              f"q={d['lowest'].get('charge')}  min_dist={d['lowest'].get('min_dist_A')}",
              f"  highest = {d['highest']['label']} ({d['highest']['mode']})  q={d['highest'].get('charge')}  "
              f"min_dist={d['highest'].get('min_dist_A')}"]
        if d['broad_spread_flag']['triggered']:
            L.append(f"  ** BROAD-SPREAD FLAG: {d['spread_meV']} > {BROAD_SPREAD_MEV} meV/at -> "
                     f"investigate missed coupled/rigid unit (re-enumerate by unit) OR strong ordering **")
        ci = d.get('coordination_integrity')
        if ci and ci['integrity_flag']:
            L.append(f"  ** COORDINATION-INTEGRITY FLAG: heteroCN={ci['heterogeneous_CN']} "
                     f"stretched={ci['n_stretched']} clashes={ci['n_clashes']} bridging={ci['bridging_anions']} "
                     f"-> shattered/defective unit (verify / re-plan) **")
        for t in e.get('self_driving', []):
            L.append(f"  self-driving round {t['round']} ({t['mode']}): spread {t['spread_meV']} meV/at "
                     f"-> {t['action']}"
                     + (f"  [ranked: {[(r[0], r[1]) for r in t['diagnosis']['ranked'][:3]]}]"
                        if t.get('diagnosis') else ''))
        if 'final_MAR' in out:
            L.append(f"  FINAL MAR (omni mpa{'+D3' if e.get('sampling',{}).get('d3',True) else ''}): "
                     f"E={out['final_MAR']['E_final_per_atom']} eV/at")
        print('\n'.join(L), file=sys.stderr)

    sys.stdout = _real
    print(json.dumps(out, indent=1, ensure_ascii=False, default=str))
    return 0

if __name__ == '__main__':
    sys.exit(main())
