"""MAR pipeline stage ⑥ — RECORD assembler (schema mar-1.0).

Merges the DETERMINISTIC engine + evidence output with the analyst's JUDGMENT into the canonical
MAR record. The LLM (stage ⑤) supplies a small `judgment.json` (its verdict); this assembler
guarantees the deterministic parts (provenance, energy distribution, generation recipe, charge &
fidelity gates) are correct and the schema is complete — the LLM never hand-assembles the whole
record. "Code proves": charge & fidelity gates are recomputed here, not transcribed.

Usage:
  python mar_record.py <icsd_id> --engine <engine.json> --judgment <judgment.json> \
                       [--hull <hull.json>] --out <dir>
  stdout = the finalized record.json (also written to <dir>/record.json)
  --hull: self-consistent omni-mpa(+D3) convex-hull result from mar_hull.py (fills E_above_hull).

judgment.json (what the analyst produces at ⑤) — all optional but expected:
  { "class": "A",                       # soft mechanism class; hybrids ok e.g. "D+E"
    "class_label": "configurational antisite (HT)",
    "interpretation": "<plain-physics, with CIF evidence (+literature for ordered/disordered pairs)>",
    "disorder_pattern": "dispersed Li/Mn antisite SRO",
    "ordering_read": "SRO / dispersed-favoured",     # the LLM's read of the distribution SHAPE
    "confidence": "high",                            # high | medium | [screening]
    "principles_applied": ["representability -> maintain", "HT pair -> explain"],
    "quoted_corrections": ["~14% antisite is a T-dependent HT population"],
    "sibling_comparison": {"sibling": "<sibling id>", "dE_meV_per_atom": -10.4, "artifact_note": null},
    # RESELECT a frame of the engine's own ensemble (the clustered-lowest escape). Name the LABEL;
    # the code finds the frame, reads its composition and energy, and keeps the enumeration gate
    # basis. NOT representative_override -- that means you BUILT a structure (D19).
    "representative_pick": {"config_label": "rand19", "reason": "the lowest is the clustered draw"},
    "representative_override": {"file": "...", "E_final_eV_per_atom": -9.57, "note": "custom build (our construction)"},
    "build_recipe": {                     # ONLY for custom builds -- the reproducible construction logic.
        "parent": "fluorite CaF2 (Fm-3m)",            # parent structure / structure-type the model derives from
        "supercell": "2x2x2 conventional (32 cation FCC + 64 tetrahedral F)",
        "steps": ["substitute 6/32 Ca->Th (x=0.1875)",       # ordered integer edits, in order applied
                  "add 12 interstitial F at octahedral holes (2 per Th)",
                  "cluster Th+interstitials near origin (one defect cluster)"],
        "charge_balance": "26*2 + 6*4 = 76+ = 76 F- -> neutral",
        "relaxation": "ideal interstitials placed, then nano->omni-mpa (--from) relax heals geometry",
        "rationale": "engine merge over-collapsed the percolating F network; built from the named model instead"},
    "polymorph_ladder": {                 # OPTIONAL: same-tier ΔE of the MAR vs ordered polymorphs (consistent comparison)
        "tier": "omni-r2scan (matpes_r2scan), no D3, relaxed",
        "members": [{"label": "Cc", "source": "mp-38970", "n_atoms": 10,
                     "E_per_atom_eV": -20.2251, "dE_meV_per_atom": 0.0, "role": "ordered ground state"}]},
    "ce_mc": {"warranted": true, "reason": "60 meV SRO spread"},
    "escalate": null,
    "decision_trace": ["setup: no repair", "engine: proceed (no rerun)", "sibling: report ΔE (never adopt)"] }
"""
import sys, json, os

def _arg(args, flag, default=None):
    return args[args.index(flag) + 1] if flag in args else default

_PMG = None
def _ox_range(el):
    """physical oxidation-state range of an element (ICSD states, then common)."""
    global _PMG
    if _PMG is None:
        from pymatgen.core import Element as _E; _PMG = _E
    try:
        e = _PMG(el)
        st = list(e.icsd_oxidation_states) or list(e.common_oxidation_states) or list(e.oxidation_states)
        return (min(st), max(st)) if st else None
    except Exception:
        return None

def _ox_override(judg):
    """-> {element: state} the analyst declared, or None.

    judgment.json:  "oxidation_override": {"states": {"Mo": 6}, "reason": "...", "evidence": "..."}

    Only `states` reaches the gate; `reason` and `evidence` are there so the claim cannot be made
    without saying why, and they travel with the judgment into the record. A bare mapping is also
    accepted, but the record then carries an override with no stated basis, which is the shape this
    is meant to discourage.
    """
    ov = (judg or {}).get('oxidation_override')
    if not ov:
        return None
    st = ov.get('states') if isinstance(ov, dict) and 'states' in ov else ov
    return {k: v for k, v in st.items()} if isinstance(st, dict) else None


def valence_aware_charge(comp, oxidation_states, tol=0.3, ox_override=None):
    """Branch-1 charge gate. If the fixed-mean charge != 0, a CIF-flagged MULTIVALENT element
    (the CIF gives it a non-integer mean, or a list of oxidation labels) may FLOAT its valence
    within its physical range to neutralize the cell -- keeping the composition faithful (no
    rebuild). Returns the gate verdict + the implied-valence fact for the LLM. If no multivalent
    element can absorb it, the cell is NOT Branch 1 -> route to Branch 2 (count adjust) / escalate.

    `ox_override` {element: state} replaces the CIF's declared value for that element BEFORE the
    gate runs, and is recorded in the verdict so the substitution is never silent.

    It exists because the float-a-multivalent-element machinery below can only move an element the
    CIF ITSELF declared with a non-integer value. A deposited loop that is simply wrong about an
    INTEGER-declared species is therefore untestable. A deposited garnet loop declaring `La3+ 3.4`
    -- not a real species, and outside La's own range in `_ox_range` -- together with `Mo4+ 4`, when
    the real chemistry is Mo6+, is the shape of it: under the CIF's numbers a physically correct
    rebuild reads a large positive q and FAILS; under Mo6+ the same structure is exactly neutral. Without an override the analyst can only
    argue in prose next to a gate that says the opposite, and the record ships a failing charge
    gate for a composition that is right.

    The override is a claim, not a fix: it must carry provenance in the judgment that supplies it,
    and the verdict below repeats it so a reader sees which states were used and which came from
    the CIF."""
    def mean(v): return sum(v) / len(v) if isinstance(v, list) else v
    if ox_override:
        oxidation_states = dict(oxidation_states or {})
        oxidation_states.update(ox_override)
    oxm = {el: mean(o) for el, o in oxidation_states.items()}
    q = sum(comp.get(el, 0) * oxm.get(el, 0.0) for el in comp)
    g = {"q_per_cell": round(q, 3), "mode": "neutral(mean)", "pass": abs(q) < tol}
    if ox_override:
        g["oxidation_override"] = dict(ox_override)      # never silent: say which states were used
        g["mode"] = "neutral(mean, analyst oxidation override)"
    if g["pass"]:
        return g
    cands = [el for el in comp if comp.get(el, 0) > 0 and (
        isinstance(oxidation_states.get(el), list) or
        (isinstance(oxidation_states.get(el), (int, float)) and abs(oxm[el] - round(oxm[el])) > 1e-6))]
    best = None
    for el in cands:
        rng = _ox_range(el)
        if not rng:
            continue
        n = comp[el]; implied = -(q - n * oxm[el]) / n
        inrange = rng[0] - 1e-9 <= implied <= rng[1] + 1e-9
        r = {"multivalent": el, "implied_valence": round(implied, 3),
             "allowed_range": list(rng), "neutralizable": bool(inrange)}
        if inrange:
            best = r; break
        best = best or r
    if best:
        g.update(best); g["pass"] = best["neutralizable"]
        g["mode"] = "neutral(valence-float)" if g["pass"] else "charge-unresolved -> Branch2/escalate"
    else:
        g["mode"] = "charge-unresolved -> Branch2/escalate"
    return g

def fidelity_from_recipe(go):
    """max |realized fraction - intended occupancy| over all group-orbit states (deterministic)."""
    worst = 0.0
    for gid, g in (go or {}).items():
        n = g.get('n_groups', 0) or 1
        states = g.get('states', {}); target = g.get('target', {})
        for sk, prob in states.items():
            realized = target.get(sk, 0) / n
            worst = max(worst, abs(realized - prob))
    return round(worst, 4)

def _comp_counts_from_file(path):
    """element -> integer atom count, read from a structure file (POSCAR/.vasp, CIF) via ASE."""
    from ase.io import read
    from collections import Counter
    return dict(Counter(read(path).get_chemical_symbols()))

def _count_xyz_frames(path):
    """Number of frames in an (ext)xyz file: each frame is `natoms`-line + comment + natoms rows.
    Distinguishes a single de-averaged frame (custom build) from a real multi-config ensemble."""
    try:
        lines = open(path).read().splitlines()
        n = i = 0
        while i < len(lines):
            if not lines[i].strip():
                i += 1; continue
            nat = int(lines[i].strip()); n += 1; i += nat + 2
        return n
    except Exception:
        return None

def _ensemble_energies(path):
    """Per-frame E_per_atom from an energy-marked (ext)xyz ensemble, sorted. Lets a custom-build record
    report ITS OWN distribution (not the discarded engine enumeration). None if energies are absent."""
    try:
        from ase.io import read
        es = [a.info.get('E_per_atom') for a in read(path, index=':')]
        es = sorted(float(e) for e in es if e is not None)
        return es or None
    except Exception:
        return None

def cell_scatter(path):
    """Geometric 'framework rigidity' metric over a multi-frame ensemble: how much the relaxed CELL
    varies across configs. A rigid fully-occupied framework PINS the cell (all configs relax to ~the
    same cell -> low scatter); when every site is partial (no backbone) the cell is SOFT and follows
    each vacancy arrangement -> high scatter (the geometric signature of 'framework unclear'; e.g.
    it ranks at the very top of a census of disordered entries). Returns {n_frames, len_cv_pct (mean coeff-of-variation of a,b,c),
    angle_std_deg (mean std of alpha,beta,gamma), vol_cv_pct} or None for <3 frames (single-frame
    custom builds have no ensemble to scatter)."""
    try:
        import numpy as np
        from ase.io import read
        frames = read(path, index=':')
        if len(frames) < 3:
            return None
        P = np.array([a.cell.cellpar() for a in frames])          # (n,6): a,b,c,al,be,ga
        V = np.array([a.get_volume() / len(a) for a in frames])
        abc, ang = P[:, :3], P[:, 3:]
        return {"n_frames": len(frames),
                "len_cv_pct": round(float(np.mean(np.std(abc, axis=0) / np.mean(abc, axis=0))) * 100, 2),
                "angle_std_deg": round(float(np.mean(np.std(ang, axis=0))), 2),
                "vol_cv_pct": round(float(np.std(V) / np.mean(V)) * 100, 2)}
    except Exception:
        return None

def fidelity_from_composition(adopted_comp, nominal_formula):
    """Fidelity for an ADOPTED (ordered) structure: it has no partial occupancies, so faithfulness
    is whether its stoichiometry matches the disordered CIF's nominal composition. Returns the max
    |atomic-fraction(adopted) - atomic-fraction(CIF nominal)| over all elements (0.0 == exact match)."""
    from pymatgen.core import Composition
    a = Composition(adopted_comp); b = Composition(nominal_formula)
    els = set(a.elements) | set(b.elements)
    worst = max((abs(a.get_atomic_fraction(e) - b.get_atomic_fraction(e)) for e in els), default=0.0)
    return round(worst, 4)

def dist_stats(energies_sorted):
    """First-class energy-distribution descriptors -- SCOPED to what ~10-24 samples can support:
    n, mean, std, spread (meV/atom above ground) + a COARSE small-n-robust shape tag. Deliberately NO
    skewness number and NO bimodality claim (need ~50+ / many more samples -> false precision). The
    ensemble + per-config energies are the ground truth; this is an INDICATIVE screen, and a quantitative
    distribution is the CE+MC escalation, not these few relaxations."""
    import math
    es = [e for e in (energies_sorted or []) if e is not None]
    n = len(es)
    if n == 0:
        return {}
    if n == 1:
        # A spread of 0.0 over ONE point is not a narrow distribution, and the tags below would call
        # it 'narrow' / 'near-degenerate -> solid-solution-like'. Two records shipped that positive
        # finding next to their own prose retracting it (two entries): the analyst knew there was
        # no distribution, and a downstream consumer parsing sro_read_indicative read the opposite.
        # Same vocabulary as disorder_descriptor -- say the read is unavailable and why.
        return {'n_configs': 1, 'mean_meV_per_atom': 0.0, 'std_meV_per_atom': 0.0,
                'spread_meV_per_atom': 0.0, 'shape': None,
                'sro_read_indicative': None,
                'sro_read_available': False,
                'sro_read_unavailable_reason':
                    'one valid configuration: no distribution to read an ordering signal from. '
                    'The zeros above are the arithmetic of a single point, not a narrow spread.'}
    g = min(es); rel = [1000 * (e - g) for e in sorted(es)]   # meV/atom above ground
    mean = sum(rel) / n
    std = math.sqrt(sum((x - mean) ** 2 for x in rel) / n)
    spread = rel[-1]
    # coarse tag from signals that stay stable at small n (magnitude + an obvious lone outlier)
    if spread < 10:
        shape, sro = 'narrow', 'near-degenerate -> solid-solution-like (any typical config ~equivalent)'
    elif n >= 4 and (rel[-1] - rel[-2]) > 0.5 * spread and rel[-2] < 15:
        shape, sro = 'tight+outlier', 'pack near-degenerate + one unfavourable config -> solid-solution-like'
    elif spread > 60:
        shape, sro = 'broad', 'wide spread -> ordering / SRO tendency (single config under-represents; CE+MC if quantitative)'
    else:
        shape, sro = 'moderate', 'moderate spread -> mild SRO'
    return {'n_configs': n, 'mean_meV_per_atom': round(mean, 2), 'std_meV_per_atom': round(std, 2),
            'spread_meV_per_atom': round(spread, 2), 'shape': shape, 'sro_read_indicative': sro,
            'note': f'indicative from {n} relaxed configs; ensemble.xyz carries the per-config energies'}

def review_ledger(review):
    """Normalise the adversarial reviewer's output into the record's review LEDGER -- one entry per
    analyst->reviewer round, oldest first.

    `None` means NOT REVIEWED, which is NOT the same as reviewed-and-clean -- the same distinction
    the ordered-sibling 'none' carries. A record whose `review` is null must never be reported as
    confirmed; it was never checked.

    `final_verdict` and `unresolved_blocking` are DERIVED here from the rounds, not copied from the
    reviewer's prose: a round that still carries a blocking objection cannot be recorded as a clean
    confirm, and the contradiction is surfaced rather than smoothed over. Same rule as the gates --
    code proves, the LLM's transcription is not trusted."""
    if review is None:
        return None
    rounds = review if isinstance(review, list) else [review]
    out = []
    for i, r in enumerate(rounds, 1):
        if not isinstance(r, dict):
            continue
        objs = [o for o in (r.get('objections') or []) if isinstance(o, dict)]
        out.append({"round": r.get('round', i), "verdict": r.get('verdict'),
                    "n_objections": len(objs),
                    "n_blocking": sum(1 for o in objs if o.get('severity') == 'blocking'),
                    "objections": objs, "checks": r.get('checks'),
                    "confidence_in_review": r.get('confidence_in_review')})
    if not out:
        return None
    last = out[-1]
    unresolved = [o.get('issue') for o in last['objections'] if o.get('severity') == 'blocking']
    block = {"n_rounds": len(out), "final_verdict": last.get('verdict'),
             "unresolved_blocking": unresolved, "rounds": out}
    if last.get('verdict') == 'confirm' and unresolved:
        block["verdict_warning"] = ("confirm recorded with unresolved blocking objection(s) -- "
                                    "not a clean confirm")
    return block

def portable_path(path, out_dir):
    """-> (value to STORE in the record, absolute path that RESOLVES, or None).

    A record used to keep whatever path the engine emitted, which was relative to the working
    directory of the run. Move the tree and every one of them dangles -- measured on an archived
    50-record campaign, 49 of 50 no longer resolved, and with them the connectivity gate's only way
    to find the shipped frame. Two rules fix that:

      * STORE relative to the record's own directory when the file lives under it, so the whole entry
        directory can be moved or renamed and stays internally consistent. A file OUTSIDE that
        directory (an input CIF from a shared corpus) is stored absolute instead, because a relative
        path to it would only survive a move of everything at once.
      * RESOLVE with a fallback for a tree that already moved: if the recorded path does not exist,
        try its TAILS under `out_dir`, longest first, and take the first that does.
        `<old>/<entry>/_work/r0/rep.cif` is found again at `<new>/<whatever>/_work/r0/rep.cif` --
        the entry directory may even have been renamed, since only the tail below it has to match.

    `out_dir=None` (no output directory given) leaves the value untouched and only resolves.
    """
    if not path or not isinstance(path, str):
        return path, None
    if os.path.isabs(path) and os.path.exists(path):
        found = path
    elif os.path.exists(path):
        found = os.path.abspath(path)
    elif out_dir:
        # the tree moved: re-root the longest tail of the recorded path that exists under out_dir
        parts = [q for q in path.replace('\\', '/').split('/') if q not in ('', '.')]
        found = None
        for k in range(len(parts)):
            cand = os.path.join(out_dir, *parts[k:])
            if os.path.exists(cand):
                found = os.path.abspath(cand)
                break
    else:
        found = None
    if not out_dir:
        return path, found
    base = os.path.abspath(out_dir)
    if found and os.path.commonpath([found, base]) == base:
        return os.path.relpath(found, base), found
    return (found or path), found


def connectivity_gate(conn, source=None):
    """Normalise the stage-⑤ connectivity audit (`tools/demars_connectivity.py --json`) into the
    third gate block.

    Why this exists: `gates` carried charge / fidelity / sibling_comparison only, so the one gate
    that catches STRUCTURAL error had no place in the schema. Analysts put the result in
    `decision_trace` prose, where no program can read it, and re-running this stage to stamp the
    review -- which rebuilds `gates` from a literal -- DELETED any value put there by hand (D21).
    The 08-17 scoring had to re-run the audit by hand on 14 of 15 frozen-run entries for that reason.

    States are `mar_evidence.PROV_STATES`, not a second vocabulary:
      not_run    no audit supplied; the gate did not execute. NEVER report this as clean.
      vacuous    it ran and nothing in this structure centres a unit (a chloride has no ClO4).
      ambiguous  a multi-frame audit carries no per-frame coverage, so "checked and clean" cannot be
                 told from "nothing examined".
      derived    centres were examined and the verdict means what it says.

    `pass` is None for not_run / vacuous / ambiguous -- deliberately NOT True. 6 of the 15
    frozen-run "clean" verdicts had `examined_centres == []`, and counting those as passes is the
    laundering the scoring caught: a consumer that sums `pass is True` must count only frames where
    a centre was actually examined. A defect found is `pass: False` whatever the coverage -- a
    broken unit is broken.

    This does NOT run the audit itself. Making the gate self-running is the separate
    🔴 "게이트 자동화" item, and it has a stated prerequisite (`COORD_FORMERS` role split --
    one entry produced 160 false positives by grouping cyanide C with methyl C). Auto-failing a record
    on a gate that over-flags would be worse than the hole this closes.
    """
    from .mar_evidence import PROV_CONFIDENT

    if not conn:
        return {"state": "not_run", "confident": False, "pass": None,
                "basis": source or "no stage-⑤ audit supplied and none could be run here",
                "source": source, "frame_index": None,
                "file": None, "n_frames": None, "tol": None, "n_defects": None,
                "examined_centres": None, "not_examined": None, "defects": None}

    summ = conn.get('per_frame_summary') or {}
    cov = summ.get('_coverage') or {}
    examined = cov.get('examined_centres')
    if examined is None and summ:
        # a summary without _coverage predates D25; the centre keys are the examined set
        examined = sorted(k for k in summ if not str(k).startswith('_'))
    n_def = conn.get('n_defects')
    if n_def is None:
        n_def = len(conn.get('defects') or [])

    if n_def:
        state, ok = 'derived', False
    elif examined:
        state, ok = 'derived', True
    elif examined == []:
        state, ok = 'vacuous', None
    else:
        state, ok = 'ambiguous', None

    basis = {'derived': 'stage-⑤ audit of the shipped frame(s)',
             'vacuous': 'stage-⑤ audit ran; no element here centres a former-ligand unit',
             'ambiguous': 'stage-⑤ audit of multiple frames; per-frame coverage not reported, so a '
                          'clean result cannot be distinguished from an empty one'}[state]
    return {"state": state, "confident": state in PROV_CONFIDENT, "pass": ok, "basis": basis,
            # WHICH frame was audited. Without it a reader cannot tell a repick's gate from one that
            # audited frame 0 of the same ensemble file.
            "source": source, "file": conn.get('file'), "frame_index": conn.get('frame_index'), "n_frames": conn.get('n_frames'), "tol": conn.get('tol'),
            "n_defects": n_def, "examined_centres": examined,
            # The per-role coordination DISTRIBUTION, next to the verdict it produced. `n_defects`
            # counts deviations from the expectation but cannot say what shape they are, and the
            # shape is what decides whether a FAIL is a broken unit or one element sitting in two
            # environments: {4: 47, 3: 1} is a unit that lost a ligand, {4: 128, 0: 64} is a
            # square net beside isolated anions. On a polytelluride the analyst asserted the second
            # reading from outside the record and the reviewer refuted it from a corrupted copy of
            # this histogram, because the record carried no version of its own. It adds no
            # threshold and changes no verdict -- the reader judges. None for a multi-frame audit,
            # where the CLI reports no per-frame summary to read it from.
            "coord_hist": {k: {"expected": v.get('expected'), "hist": v.get('coord_hist')}
                           for k, v in summ.items()
                           if not str(k).startswith('_') and isinstance(v, dict)
                           and v.get('coord_hist')} or None,
            "not_examined": cov.get('not_examined'),
            # a sample, not the lot: one entry produced 160 and the record is not a log
            "defects": (conn.get('defects') or [])[:20]}


def audit_representative(path, tol=1.25, expect=None, index=None):
    """Run the stage-⑤ connectivity audit HERE, on the shipped frame, and return it in exactly the
    shape `tools/demars_connectivity.py --json` emits.

    Why stage ⑥ runs it rather than waiting to be handed the artifact: charge and fidelity are
    recomputed automatically and see composition only; connectivity is the one gate that catches
    STRUCTURAL error, and leaving it to a manual call inverts the priority. The 08-17 scoring
    measured what that costs -- `gates.connectivity` was absent on 15 of 15 entries even with every
    entry reviewed, and the scorer had to re-run the audit by hand on 14 of them.

    One shape for both sources means `connectivity_gate` does not care where the audit came from;
    only its `source` field says. An artifact still WINS when one is supplied, because it can carry
    what this cannot: `--expect` (an enforced coordination, for a build that relaxed into a mixture
    and would otherwise hide behind its own majority) and a multi-frame audit of `ensemble.xyz`.

    -> (audit_dict, None) or (None, reason). NEVER raises: a probe that kills the run is not a probe
    (D24), and 9 of the 15 frozen-run records already carry a `representative.file` that no longer
    resolves -- an archived or moved tree is the normal case, not the exception. The reason string
    lands in the gate as `not_run`, which is UNCHECKED and never clean.
    """
    if not path:
        return None, 'the record carries no representative file to audit'
    if not os.path.exists(path):
        return None, f'representative file does not resolve: {path}'
    try:
        from ase.io import read
        from ..connectivity import audit_frame
        frames = read(path, index=':' if index is None else index)
        if not isinstance(frames, list):
            frames = [frames]
        per_frame = []
        all_def = []
        for fi, at in enumerate(frames):
            summ, defs = audit_frame(at, tol=tol, expect=expect)
            per_frame.append(summ)
            for d in defs:
                d2 = dict(d); d2['frame'] = fi; all_def.append(d2)
    except Exception as e:                                  # unreadable / unsupported / no cell
        return None, f'{type(e).__name__}: {e}'
    return {'file': path, 'frame_index': index, 'n_frames': len(frames),
            'clean': not all_def, 'tol': tol,
            'expect': expect, 'n_defects': len(all_def), 'defects': all_def,
            'per_frame_summary': per_frame[0] if len(frames) == 1 else None}, None


def pick_final_tier(pick, pick_info):
    """The final-tier relaxation of a REPICKED frame, from the `--from` run that produced it.

    The research pipeline had an invariant PACKAGE lost: the shipped structure always carried a
    final-tier energy, because both of its two paths produced one -- take-lowest was what the engine
    itself relaxed at the final tier, and a `representative_override` was realised through
    `--from --final`, which relaxes. A repick is a third path and it has neither: the final tier ran
    on the enumeration's lowest, not on the frame that shipped. Measured on a 50-entry campaign, 17
    records shipped a repick and every one of them carried `file_final: null`.

    That matters now that `gates.hull` exists, because a hull is only a stability number when it is
    computed at the tier the structure it describes was finalized with.

    So a pick may name the `--from` run that finalized it -- `representative_pick.final_from`, the
    path to that run's `engine.json` -- and this reads it back. **The ensemble and the gate basis
    still come from the ONE `--engine` given to this stage**: `final_from` contributes an energy and
    a relaxed file, never the ensemble, which is what kept D19(b) from recurring.

    -> ({'E_final_eV_per_atom', 'file_final', 'final_from'}, None) or (None, reason it was rejected).
    """
    src = (pick or {}).get('final_from')
    if not src:
        return None, None
    try:
        with open(src, encoding='utf-8') as fh:
            fr = json.load(fh)
    except Exception as exc:
        return None, f'could not read representative_pick.final_from ({src}): {exc}'
    if fr.get('mode') != 'from':
        return None, (f'{src} is not a --from run (mode={fr.get("mode")!r}); a picked frame is '
                      'finalized by relaxing THAT frame, not by another enumeration')
    sm = fr.get('single_MAR') or {}
    ef = sm.get('E_final_per_atom')
    if ef is None:
        return None, (f'{src} ran but its final tier did not: '
                      f'{(sm.get("final_MAR") or {}).get("reason", "no E_final_per_atom")}')
    # the frame that was finalized must be the frame that shipped -- atom count is the cheap check
    # available from both sides, and a mismatch means a different structure was relaxed.
    n_pick = (pick_info or {}).get('n_atoms')
    if n_pick and sm.get('n_atoms') and int(sm['n_atoms']) != int(n_pick):
        return None, (f'{src} finalized a {sm["n_atoms"]}-atom structure but the picked frame has '
                      f'{n_pick}; that is not the same frame')
    return {'E_final_eV_per_atom': round(float(ef), 5),
            'file_final': (fr.get('files') or {}).get('representative_final'),
            'final_from': src, 'final_input': fr.get('input')}, None


def resolve_pick(engine, pick, out_dir=None):
    """Resolve `judgment.representative_pick` to a concrete frame of the engine's OWN ensemble.

    Choosing a different frame from the same ensemble is normal and `strategy.md` asks for it -- a
    `lowest.mode == 'clustered'` draw must not ship as-is. But the only field that could express it
    was `representative_override`, which means "the analyst built this structure by hand", and two
    things came out wrong (D19):

      (a) a RESELECTION was recorded as a custom build. Measured on a real record: `source` =
          `custom-build`, gate basis = `custom-build structure (composition match; ordered, occ=1)`
          -- while the shipped frame was one of the engine's own 30 random seeds. `custom_reason`
          then demanded a reason for a build that never happened, so a SELECTION reason sat in a
          BUILD reason's slot and a reader saw construction that did not occur.
      (b) the `ensemble` block and the representative came from different places. `build_record`
          reads the ensemble from the ONE `--engine` it is given; the representative came from the
          override. If the override pointed at a `--from` run, whichever engine.json you passed, one
          side was wrong -- so an analyst copied `r0/ensemble.xyz` next to the override and the run
          dir ended up with TWO byte-identical ensembles (507294 B each). That is D7(e): a program
          cannot tell which one is the entry's ensemble.

    A pick names the LABEL, and this resolves it. The engine already writes `label`, `mode`,
    `E_per_atom` and `charge` into each extxyz frame comment, so nothing is transcribed by hand: the
    analyst says `rand19` and the code finds the frame, reads its composition and its energy, and
    the ensemble block keeps coming from the same engine.json. There is no second copy to make.

    `frame_index` is accepted as an escape for an ensemble written before the labels existed.

    -> (info, None) or (None, reason). NEVER raises, and it NEVER falls back to the lowest: shipping
    the frame the analyst rejected, quietly, is worse than an unresolved pick.
    """
    if not pick:
        return None, None
    ef = engine.get('ensemble_files') or {}
    path = ef.get('ensemble')
    if not path:
        return None, 'the engine output names no ensemble file'
    # heal a path recorded before the tree moved -- the same rule `portable_path` applies to what the
    # record stores. Without it a moved archive cannot resolve its own picks, and the record reports
    # the pick unresolved and ships the frame the analyst rejected.
    _stored, _abs = portable_path(path, out_dir)
    if not _abs:
        return None, f'ensemble file does not resolve: {_stored}'
    path = _abs
    want = pick.get('config_label')
    idx = pick.get('frame_index')
    if want is None and idx is None:
        return None, 'representative_pick needs a config_label (or a frame_index)'
    try:
        from ase.io import read
        frames = read(path, index=':')
        if not isinstance(frames, list):
            frames = [frames]
    except Exception as e:
        return None, f'could not read the ensemble ({type(e).__name__}: {e})'
    hit = None
    if want is not None:
        for i, at in enumerate(frames):
            if str(at.info.get('label')) == str(want):
                hit = (i, at); break
        if hit is None:
            have = [str(a.info.get('label')) for a in frames]
            return None, f'no frame labelled {want!r} in {path} (labels: {have})'
    else:
        if not (0 <= int(idx) < len(frames)):
            return None, f'frame_index {idx} outside the ensemble (n_frames={len(frames)})'
        hit = (int(idx), frames[int(idx)])
    i, at = hit
    comp = {}
    for s in at.get_chemical_symbols():
        comp[s] = comp.get(s, 0) + 1
    return {'file': path, 'frame_index': i, 'config_label': at.info.get('label', want),
            'mode': at.info.get('mode'), 'E_per_atom': at.info.get('E_per_atom'),
            'charge': at.info.get('charge'), 'composition': comp, 'n_atoms': len(at)}, None


BRACKET_MODES = ('clustered', 'dispersed')   # the two extremes the engine always appends as probes


def sqs_gate(lowest, representative):
    """Did this record ship one of the engine's BRACKET PROBES as the representative?

    `strategy.md` samples random-only and treats the appended `dispersed`/`clustered` extremes as
    BRACKET PROBES: they bound the distribution, they are not candidates to ship. A probe winning
    take-lowest is usually an artifact rather than the phase. That instruction was enforced by
    nobody -- it needed the analyst to notice -- the same inversion `gates.connectivity` had.

    BOTH extremes count, not just `clustered`. The 08-18 scoring measured three records that shipped
    a probe (one clustered, two dispersed) with `representative.source` reading
    `engine-lowest` and no disclosure anywhere in the record. Their margins over the best random
    frame were 2.8 / 4.1 / 0.8 meV/at -- at or under the nano resolution floor, so the structures are
    not wrong; the label hid what they were. A gate that covered `clustered` only would still miss
    two of those three.

    States are `mar_evidence.PROV_STATES`:
      not_run   there is no distribution to look at (should not happen with an enumeration).
      vacuous   no enumeration ran at all (a `--from` build), so there was no draw to ship wrongly.
                `pass` is None, not True -- nothing was examined, the same rule the connectivity
                gate follows.
      derived   an enumeration ran and the shipped frame is known.

    `pass` is False only for the one thing this checks: the enumeration's own lowest IS the clustered
    draw AND that draw is what shipped. A record that set the draw aside -- a custom build, or a
    repick inside the same ensemble -- passes, and the basis says which.

    **Now a WATCHDOG.** The engine used to offer bracket probes to take-lowest, so this gate fired
    often and the analyst escaped it by hand with `representative_pick`. `mar_engine.ship_candidates`
    removed the probes from the candidate pool (2026-08-24), so a probe reaches the representative
    only through the fallback -- no random sample survived relaxation -- or if some future call site
    bypasses that helper. Both are exactly what this should still catch, so it stays.
    """
    from .mar_evidence import PROV_CONFIDENT

    lowest = lowest or {}
    rep = representative or {}
    mode, label = lowest.get('mode'), lowest.get('label')
    if not lowest:
        state, ok, basis = 'not_run', None, 'no distribution in the engine output'
    elif not mode:
        state, ok, basis = ('vacuous', None,
                            'no enumeration ran (a --from build), so there was no draw to ship')
    elif rep.get('source') != 'engine-lowest':
        state, ok, basis = ('derived', True,
                            f"the engine draw ({mode}) was set aside; shipped {rep.get('source')}")
    else:
        ok = mode not in BRACKET_MODES
        state, basis = 'derived', (
            f"shipped the enumeration's own lowest, sampling mode {mode!r}"
            + ('' if ok else f' -- {mode} is a BRACKET PROBE, not a candidate to ship '
                             '(strategy.md is random-only); repick a random frame inside the '
                             'ensemble with representative_pick, or disclose why this one ships'))
    return {"state": state, "confident": state in PROV_CONFIDENT, "pass": ok, "basis": basis,
            "lowest_mode": mode, "lowest_label": label,
            "shipped": rep.get('config_label') or rep.get('source')}


# Stability threshold for `gates.hull`, in eV/atom. **None by default, and that is deliberate.**
#
# The research pipeline (`mar_hull.py`) computed E_above_hull and REPORTED it; its record's gates
# were {charge, fidelity, sibling_comparison} and the hull was a block beside them, never a
# pass/fail. There is no inherited threshold to port, and a threshold is not derivable from the
# structure either -- the two terms that set it are both system-dependent:
#
#   * CONFIGURATIONAL ENTROPY, which the 0 K hull does not contain. A MAR is an ORDERED approximant
#     of a phase the entropy stabilises, so it sits ABOVE the hull legitimately. One mixed site at
#     x=0.5 is kB*T*ln2 = 60 meV per mixed site at 1000 K -- but diluted by that sublattice's share
#     of the atoms, which is 5% in one compound and 50% in the next, and T is the CIF's refinement
#     temperature, not a constant.
#   * THE ENERGY SCALE, whose error enters the subtraction differently in each mode: `mp-direct`
#     compares an MLIP target against MP's DFT references, `self-consistent` compares MLIP against
#     MLIP.
#
# So the line is a POLICY INPUT, set by whoever knows the campaign, not a constant of nature. Unset,
# this gate reports the number and refuses to judge it -- `state: derived`, `pass: null` -- which is
# what the research pipeline did, and is honest in a way a made-up number would not be. Set it (a
# float here, or `--hull-tol` on stage 6) and the gate judges against it and records which value.
HULL_TOL_EV_PER_ATOM = None


def hull_gate(hull, tol=HULL_TOL_EV_PER_ATOM):
    """Is the MAR thermodynamically reachable? The one gate charge / fidelity / connectivity / sqs
    cannot speak to: all four ask whether the cell is a faithful realisation of the CIF, and a cell
    can be perfectly faithful and still be a structure that would decompose.

    **Unlike `connectivity` and `sqs` this gate does NOT run itself.** It needs a Materials Project
    call and a relaxation, so it is `not_run` until stage 4b's artifact is handed in with `--hull`.
    That is a real cost of promoting it: a pipeline that skips 4b now has an UNCHECKED gate, and by
    this record's own rules an unchecked gate is not a pass.

    States are `mar_evidence.PROV_STATES`:
      not_run   no hull artifact, or one that refused (no MP key, fetch failed, the MAR would not
                relax, a disordered structure was passed). The artifact's own reason is carried.
      ambiguous the hull ran but cannot be read as one: an incomplete simplex.
      derived   a hull was computed. `E_above_hull_eV_per_atom` means what it says.

    `pass` is True/False only when `tol` is set; with no threshold configured it is None even in the
    `derived` state -- the number is REPORTED, not judged. See HULL_TOL_EV_PER_ATOM for why there is
    no default line to port. `below_reference_hull` carries the research pipeline's `below_mp_hull`
    flag: a MAR under the hull is either a genuinely better ordering or a reference set missing a
    competitor, and which of those it is was never something code decided here.
    """
    from .mar_evidence import PROV_CONFIDENT

    def _g(state, ok, basis, **extra):
        return {"state": state, "confident": state in PROV_CONFIDENT, "pass": ok, "basis": basis,
                "tol_eV_per_atom": tol, **extra}

    if not hull:
        return _g('not_run', None,
                  'no hull artifact supplied -- run tools/demars_hull.py on the shipped '
                  'representative and pass it with --hull. UNCHECKED, not clean')
    hstate = hull.get('state') or ('not_run' if hull.get('error') else 'derived')
    reason = hull.get('reason') or hull.get('error')
    chemsys = hull.get('chemsys')
    if hstate != 'derived':
        return _g('not_run' if hstate == 'not_run' else 'ambiguous', None,
                  f'the hull stage returned {hstate!r}: {reason}', chemsys=chemsys)
    e = (hull.get('mar') or {}).get('E_above_hull_eV_per_atom')
    if e is None:
        return _g('not_run', None, 'the hull reports `derived` but carries no E_above_hull',
                  chemsys=chemsys)
    e = float(e)
    # The SCALE belongs in the gate, not only in the artifact: on a mixed-valence oxide the two
    # scales can differ by ~0.1 eV/atom AND name different decomposition products, so a number
    # without the scale beside it is not a reproducible statement.
    corr = hull.get('corrections')
    common = {"chemsys": chemsys, "E_above_hull_eV_per_atom": e, "tier": hull.get('tier'),
              "mode": hull.get('mode'), "corrections": corr,
              "n_refs_used": hull.get('n_refs_used'),
              "below_reference_hull": bool(e < -1e-3),
              "decomposition": (hull.get('mar') or {}).get('decomposition')}
    where = (f'tier {hull.get("tier")}, mode {hull.get("mode")}, corrections {corr}')
    if tol is None:
        return _g('derived', None,
                  f'E_above_hull = {e:.3f} eV/atom ({where}); decomposes to '
                  f'{common["decomposition"]}. NO stability threshold is configured, so this gate '
                  'reports and does not judge -- read the number, do not read `pass`', **common)
    ok = e <= tol
    return _g('derived', ok,
              (f'E_above_hull = {e:.3f} eV/atom <= {tol} ({where})') if ok else
              (f'E_above_hull = {e:.3f} eV/atom EXCEEDS the configured {tol} ({where}); it would '
               f'decompose to {common["decomposition"]}'), **common)


def build_record(iid, engine, judg, hull=None, evidence=None, review=None,
                 connectivity=None, hull_tol=None, out_dir=None):
    """`evidence` is the ①a bundle. Pass it to stay ICSD-free (a file-sourced CIF
    has no id); omitted, it falls back to the ICSD lookup as before. `review` is the
    mar-reviewer output -- one round dict or the list of rounds (see `review_ledger`);
    omitted, the record records itself as UNREVIEWED. `connectivity` is the stage-⑤ audit
    JSON; omitted, `gates.connectivity` says `not_run` -- which is not "clean". `hull_tol` is the
    stability threshold `gates.hull` judges against; omitted, that gate reports its number and
    leaves `pass` null (see HULL_TOL_EV_PER_ATOM -- there is no inherited line to default to)."""
    ev = evidence
    if ev is None:
        from . import mar_evidence as MEV
        ev = MEV.evidence(iid)
    p = ev['provenance']; einfo = engine.get('engine', {}); dist = engine.get('distribution', {})
    # `mar_engine` names the shipped frame `lowest`; `record.write_custom_ensemble` names it
    # `representative` (it is the lowest SHIP candidate, not the pool's). Read both, or a custom
    # build loses its composition/charge to the override branch and `sqs_gate` reads `not_run`.
    lowest = dist.get('lowest') or dist.get('representative') or {}
    ef = engine.get('ensemble_files', {}); fin = engine.get('final_MAR', {})

    # ---- deterministic gates (recomputed here, not trusted from the LLM) ----
    # When the analyst supplies its OWN custom/repaired build (representative_override -- NEVER an
    # adopted sibling; sibling adoption was removed), the gates must describe THAT deliverable, not the
    # discarded blind enumeration cell. Read its composition and score charge on it + fidelity as a
    # composition match to the CIF's nominal stoichiometry.
    ovr = judg.get('representative_override')
    # A repick is a frame of the engine's OWN enumeration, so it keeps the enumeration's gate basis
    # and the enumeration's ensemble block -- that is the whole point of D19. An override is a
    # structure the analyst BUILT; the two are not interchangeable and cannot both apply.
    pick = judg.get('representative_pick')
    pick_info = pick_err = _pf = _pf_why = None
    if pick and ovr and ovr.get('file'):
        pick_err = ('both representative_override and representative_pick were given; the override '
                    'wins -- a hand-built structure is not a frame of the enumeration')
    elif pick:
        pick_info, pick_err = resolve_pick(engine, pick, out_dir=out_dir)
        _pf, _pf_why = pick_final_tier(pick, pick_info)
    gate_basis = None; ovr_comp = None; ovr_nat = None; fid_basis = None
    if ovr and ovr.get('file'):
        try:
            acomp = _comp_counts_from_file(ovr['file'])
            ovr_comp = acomp; ovr_nat = sum(acomp.values())   # the ACTUAL deliverable's composition
            charge_gate = valence_aware_charge(acomp, ev['oxidation_states'],
                                               ox_override=_ox_override(judg))
            fdev = fidelity_from_composition(acomp, p.get('formula_sum'))
            fidelity_gate = {"max_occ_deviation": fdev, "pass": fdev <= 0.08}
            gate_basis = "custom-build structure (composition match; ordered, occ=1)"
        except Exception as e:
            ovr_fail = f"could not read custom-build structure ({e}); gates fell back to blind recipe"
            ovr = dict(ovr); ovr['gate_warning'] = ovr_fail   # surface in the representative note path
            gate_basis = None
    if gate_basis is None:                       # genuine enumeration (or custom-build file unreadable)
        # the PICKED frame when there is one -- the gates have to describe what shipped, and a repick
        # ships a different frame of the same enumeration (its composition is read from the frame,
        # not transcribed). Fidelity stays recipe-based: the orbit targets apply to every frame.
        comp = (pick_info or {}).get('composition') or lowest.get('composition')
        if comp:
            charge_gate = valence_aware_charge(comp, ev['oxidation_states'],   # Branch-1 aware
                                               ox_override=_ox_override(judg))
        else:
            q = lowest.get('charge')
            # A missing charge is an ABSENCE, not a measurement -- the same distinction the
            # fidelity gate makes just below. `q is not None and ...` folded it to False, so an
            # entry with no build at all (engine status=error, no distribution) came back as a
            # FAILING charge gate on a structure that was never made. `pass` is None there.
            charge_gate = {"q_per_cell": q, "mode": "legacy",
                           "pass": (abs(q) < 0.3) if q is not None else None}
            if q is None:
                charge_gate["state"] = "vacuous"
        dev = fidelity_from_recipe(einfo.get('group_orbits'))
        fidelity_gate = {"max_occ_deviation": dev, "pass": dev <= 0.08}
        gate_basis = "engine-enumeration (occupancy realization vs CIF)"
        if not (einfo.get('supercell') and einfo.get('group_orbits')):
            # No enumeration recipe (a --from build). fidelity_from_recipe walks an empty
            # group_orbits and returns 0.0 -- an ABSENCE, not a measurement, and the basis above
            # would claim an enumeration that never ran. `pass` is None for vacuous, exactly as
            # the sqs gate does for the same build; it must never read True off nothing.
            fidelity_gate = {"max_occ_deviation": None, "pass": None, "state": "vacuous"}
            fid_basis = ("no enumeration recipe (a --from build), so occupancy realization "
                         "against the CIF was never measured")
    charge_gate = {**charge_gate, "basis": gate_basis}
    fidelity_gate = {**fidelity_gate, "basis": fid_basis or gate_basis}
    sib = judg.get('sibling_comparison') or judg.get('sibling_sufficiency')   # ΔE comparison only (never an adopt decision); legacy key tolerated

    # ---- convex hull, filled from `demars_core.hull` if the stage was run ----
    # `hull: null` means the stage was NOT RUN, and nothing else may mean that. A hull that ran and
    # came back refused (no API key, MP cannot cover the chemsys, the MAR would not relax) used to
    # be nulled out here, which made "we did not ask" and "we asked and could not know" the same
    # record -- the exact confusion the gates' four states exist to prevent. Keep the block and
    # carry its state; `E_above_hull` is present only when that state is `derived`.
    hm = (hull or {}).get('mar', {}) if hull else {}
    hull_state = (hull or {}).get('state')
    if hull and hull_state is None:                  # a pre-state artifact: infer, do not guess silently
        hull_state = 'not_run' if hull.get('error') else 'derived'
    e_hull = hm.get('E_above_hull_eV_per_atom') if hull_state == 'derived' else None
    hull_block = None
    if hull:
        hull_block = {"state": hull_state,
                      "reason": hull.get('reason') or hull.get('error'),
                      "tier": hull.get('tier'), "mode": hull.get('mode'),
                      "chemsys": hull.get('chemsys'),
                      "E_above_hull_eV_per_atom": e_hull,
                      "formation_E_eV_per_atom": hm.get('formation_E_eV_per_atom'),
                      "on_hull": hm.get('on_hull'), "below_mp_hull": hm.get('below_mp_hull'),
                      "decomposition": hm.get('decomposition'),
                      "n_refs_used": hull.get('n_refs_used')}

    # ---- representative: engine lowest, unless the analyst supplied its OWN custom/repaired build ----
    # (ovr was resolved above so the gates could be computed on that built structure; NEVER a sibling)
    if ovr:
        note = ovr.get('note')
        if ovr.get('gate_warning'):
            note = (note or '') + f" [GATE WARNING: {ovr['gate_warning']}]"
        # WHY custom-built (the trigger -- what the engine default got wrong), distinct from build_recipe (the how).
        # Prefer an explicit judgment field; else the build_recipe rationale; else an excerpt of the override note.
        custom_reason = (judg.get('custom_reason') or (judg.get('build_recipe') or {}).get('rationale')
                         or ((ovr.get('note') or '').strip()[:220] or None))
        representative = {"source": "custom-build", "file": ovr.get('file'),
                          "composition": ovr_comp, "n_atoms": ovr_nat,
                          "E_final_eV_per_atom": ovr.get('E_final_eV_per_atom'),
                          "custom_reason": custom_reason,
                          "note": note, "E_above_hull_eV_per_atom": e_hull}
    elif pick_info:
        # `file` + `frame_index` IS the identifier -- no frame is extracted to a new file. Writing one
        # would create a second answer next to the ensemble (D7(e)), which is the very thing the
        # ensemble-copy workaround did.
        representative = {"source": "engine-repick", "file": pick_info['file'],
                          "frame_index": pick_info['frame_index'],
                          "file_final": _pf.get('file_final') if _pf else None,
                          "composition": pick_info['composition'],
                          "config_label": pick_info['config_label'],
                          "sampling_mode": pick_info.get('mode'),
                          "n_atoms": pick_info['n_atoms'],
                          "supercell": einfo.get('supercell'),
                          "E_nano_eV_per_atom": pick_info.get('E_per_atom'),
                          "E_final_eV_per_atom": _pf.get('E_final_eV_per_atom') if _pf else None,
                          # A repick is the one shipped structure the engine did NOT relax at the
                          # final tier -- it relaxed the enumeration's lowest. `final_from` names the
                          # `--from` run that finalized THIS frame and restores the research
                          # pipeline's invariant (the shipped structure always has a final tier).
                          # Absent, say so: a bare null reads as "never requested" (D23).
                          "final_from": _pf.get('final_from') if _pf else None,
                          "E_final_unavailable_reason": (
                              None if _pf else
                              (f'representative_pick.final_from was rejected -- {_pf_why}'
                               if _pf_why else
                               "this frame was repicked from the ensemble; the final tier ran on the "
                               "enumeration's lowest, not on it. Relax this frame with "
                               "`demars_engine.py --from <frame> --final` into its own _work tag and "
                               "name that run in representative_pick.final_from")),
                          # the SELECTION reason, in its own slot -- not in `custom_reason`, which
                          # asks why a structure was BUILT
                          "pick_reason": pick.get('reason'),
                          "E_above_hull_eV_per_atom": e_hull}
    else:
        representative = {"source": "engine-lowest", "file": ef.get('representative'),
                          "file_final": ef.get('representative_final'),
                          # the ACTUAL composition, as the custom-build branch above does. This held
                          # `label` ('rand26') until 2026-08-12 -- a config id where downstream reads
                          # a composition. The label is provenance (WHICH config won), so it stays,
                          # under its own key.
                          "composition": lowest.get('composition'),
                          "config_label": lowest.get('label'),
                          "n_atoms": fin.get('n_atoms') or lowest.get('n_atoms'),
                          "supercell": einfo.get('supercell'),
                          "E_nano_eV_per_atom": lowest.get('E_per_atom'),
                          "E_final_eV_per_atom": fin.get('E_final_per_atom'),
                          # A null above is ambiguous on its own -- never requested, or requested
                          # and refused? This carries the answer when it is the second.
                          "E_final_unavailable_reason": (fin.get('reason')
                                                         if fin.get('available') is False else None),
                          "E_above_hull_eV_per_atom": e_hull}   # from demars_core.hull; None unless state=='derived'

    # ---- ensemble block: default = the engine enumeration. For a custom build, describe the ACTUAL
    # deliverable next to the override file: `representative.xyz` = a single de-averaged frame (no
    # configurational ensemble — the engine enumeration was set aside); `ensemble.xyz` = a genuine
    # multi-config distribution (e.g. 6319). File presence + frame count drive it. ----
    # D8: the record used to carry only how many configs SURVIVED. One set-A entry ranked and
    # shipped a representative chosen from 4 of 30 and recorded `status: de-averaged` -- the record
    # had no field in which the other 26 could be missed. `n_enumerated` is what the engine built;
    # the shortfall is a RISK WARNING (a 4-sample ensemble does not represent a typical arrangement,
    # so `confidence` should feel it), not a value, so it is reported rather than gated. WHERE to
    # stop -- 50 %? 80 %? -- is the open question of 규약 2단계 and is deliberately not invented here.
    _n_cfg = ef.get('n_frames') or dist.get('n_relaxed')
    _n_enum = dist.get('n_total')
    ens_block = {
        "file": ef.get('ensemble'), "n_configs": _n_cfg,
        "n_enumerated": _n_enum,
        "sampling_shortfall": (None if not _n_enum or _n_cfg is None or _n_cfg >= _n_enum else
                               f"{_n_cfg}/{_n_enum} enumerated configs survived relaxation; the "
                               f"representative was chosen from the survivors"),
        "engine_model": "7net-nano" + ("+D3" if einfo.get('sampling', {}).get('d3', False) else ""),
        "energy_distribution": {
            "ground_eV_per_atom": dist.get('ground_E'),
            "spread_meV_per_atom": dist.get('spread_meV'),
            "std_meV_per_atom": dist.get('std_meV'),
            "energies_sorted": dist.get('energies_sorted'),
            # the probes are excluded from the REPRESENTATIVE (ship_candidates) but their margins are
            # the ordering FLAG strategy.md asks for -- dropping them here would erase the signal the
            # exclusion was careful to preserve, and leave `ordering_read`'s claims unverifiable.
            "n_ship_candidates": dist.get('n_ship_candidates'),
            "bracket_probes": dist.get('bracket_probes'),
            **dist_stats(dist.get('energies_sorted')),   # n/mean/std/spread + coarse INDICATIVE shape tag
            "ordering_read": judg.get('ordering_read')}}
    if ovr and ovr.get('file'):
        # a custom build's frame count is not a survival rate of the engine enumeration -- comparing
        # the two would manufacture a shortfall that never happened
        ens_block['sampling_shortfall'] = None
        _rd = os.path.dirname(ovr['file'])
        _rx = os.path.join(_rd, 'representative.xyz'); _ex = os.path.join(_rd, 'ensemble.xyz')
        _df = _rx if os.path.exists(_rx) else (_ex if os.path.exists(_ex) else None)
        _nf = _count_xyz_frames(_df) if _df else None
        if _df:
            ens_block['file'] = _df
        if _nf == 1:                                  # single de-averaged frame -> no ensemble to summarise
            _e = ovr.get('E_final_eV_per_atom')
            ens_block['n_configs'] = 1
            ens_block['energy_distribution'] = {
                "ground_eV_per_atom": _e, "spread_meV_per_atom": 0.0, "std_meV_per_atom": 0.0,
                "energies_sorted": ([_e] if _e is not None else None),
                "n_configs": 1, "mean_meV_per_atom": 0.0, "shape": "single",
                "sro_read_indicative": "single de-averaged frame — no configurational ensemble",
                "ordering_read": judg.get('ordering_read'),
                "note": "single de-averaged frame (custom build); engine enumeration set aside, no configurational ensemble"}
        elif _nf:
            ens_block['n_configs'] = _nf
            _en = _ensemble_energies(_df)          # custom multi-config build: report ITS OWN distribution
            if _en:
                ens_block['energy_distribution'] = {
                    "ground_eV_per_atom": round(min(_en), 4),
                    "energies_sorted": [round(e, 4) for e in _en],
                    **dist_stats(_en),
                    "ordering_read": judg.get('ordering_read')}
    # geometric framework-rigidity signal over the actual ensemble (None for single-frame builds)
    ens_block['cell_scatter'] = cell_scatter(ens_block.get('file'))

    # disorder descriptor (Antypov O/S/V/P set) + the no_full_backbone triage flag (every orbit
    # vacancy-bearing -> no rigid scaffold -> "framework unclear"; pairs with ensemble cell_scatter).
    #
    # Read from the CIF the evidence bundle came from -- `ev['source']`, the FILE, is the key here;
    # there is no id lookup. A bundle without one (build_record handed a hand-made evidence dict)
    # leaves the field UNAVAILABLE-WITH-A-REASON rather than `null`, which in this schema is what an
    # unrun optional step leaves -- a reader has to be able to tell "computed, came back empty" from
    # "this build could not compute it". Same distinction AGENTS.md demands of the sibling search.
    _src = ev.get('source')
    if not _src:
        disorder_descriptor = {"available": False,
                               "reason": "no CIF source in the evidence bundle (ev['source']); the "
                                         "descriptor is read from the input CIF, not from an id"}
    else:
        try:
            from ..disorder_class import classify as _classify
            _dc = _classify(_src, oxidation_states=ev.get('oxidation_states'), iid=iid)
            disorder_descriptor = {"disorder_set": _dc.get("disorder_set"), "multiset": _dc.get("multiset"),
                                   "no_full_backbone": _dc.get("no_full_backbone"),
                                   "available": True}
        except Exception as _e:
            disorder_descriptor = {"available": False,
                                   "reason": f"could not classify {_src}: {_e}"}

    if pick_err:
        # NOT a fallback to the lowest: the analyst rejected that frame. The record says so, the CLI
        # warns, and if the lowest is the clustered draw `gates.sqs` fails on top -- which is exactly
        # the situation a repick exists to escape.
        representative = {**representative, "pick_unresolved": pick_err}

    # ---- the two gates that used to wait for a manual call (Phase 5 「게이트 자동화」) ----------
    # A supplied artifact wins: it can carry --expect and a multi-frame ensemble audit, which the
    # auto-run cannot. Otherwise audit the shipped frame here -- preferring the FINAL relaxed file,
    # since that is the deliverable, not the pre-final draw.
    # Store every structure path the way `portable_path` describes -- relative to this record when
    # the file lives beside it -- and hand the gate below the RESOLVED absolute path, so a tree that
    # has already been moved is audited rather than reported unreadable.
    _resolved = {}
    for _blk, _key in ((representative, 'file'), (representative, 'file_final'),
                       (representative, 'final_from'), (ens_block, 'file')):
        if _blk and _blk.get(_key):
            _stored, _abs = portable_path(_blk[_key], out_dir)
            _blk[_key] = _stored
            _resolved[(id(_blk), _key)] = _abs

    conn_source = 'stage-⑤ artifact (--connectivity)'
    if not connectivity:
        _shipped = (_resolved.get((id(representative), 'file_final'))
                    or _resolved.get((id(representative), 'file'))
                    or representative.get('file_final') or representative.get('file'))
        connectivity, _why = audit_representative(_shipped,
                                                 index=representative.get('frame_index'))
        conn_source = ('auto: stage ⑥ audited the shipped frame' if connectivity else
                       f'auto-run could not read the shipped frame -- {_why}')

    return {
        "schema_version": "mar-1.0",
        "icsd_id": iid,
        "provenance": {
            "formula_sum": p.get('formula_sum'), "chemical_name": p.get('chemical_name_common'),
            "structure_type": p.get('structure_type'),
            "spacegroup": ev['cell'].get('spacegroup'),
            "spacegroup_number": ev['cell'].get('spacegroup_number'), "Z": ev['cell'].get('Z'),
            "citation_title": p.get('citation_title'), "journal": p.get('journal'),
            "year": p.get('year'), "authors": p.get('authors'),
            "measurement_temperature_K": p.get('measurement_temperature_K'),
            "measurement_pressure_kPa": p.get('measurement_pressure_kPa'),
            "R_factor": p.get('R_factor')},
        "mechanism": {
            "class": judg.get('class'), "class_label": judg.get('class_label'),
            "interpretation": judg.get('interpretation'),
            "disorder_pattern": judg.get('disorder_pattern'),
            "confidence": judg.get('confidence'),
            "principles_applied": judg.get('principles_applied', []),
            "quoted_corrections": judg.get('quoted_corrections', [])},
        "disorder_descriptor": disorder_descriptor,
        "ensemble": ens_block,
        "representative": representative,
        "hull": hull_block,
        "polymorph_ladder": judg.get('polymorph_ladder'),   # optional: same-tier ΔE vs ordered polymorphs
        "gates": {"charge": charge_gate, "fidelity": fidelity_gate,
                  "connectivity": connectivity_gate(connectivity, source=conn_source),
                  "sqs": sqs_gate(lowest, representative),
                  # The one gate that does NOT run itself -- it needs stage 4b's artifact. Absent,
                  # it is `not_run`, which is UNCHECKED and never a pass.
                  "hull": hull_gate(hull, tol=hull_tol),
                  "sibling_comparison": sib},
        "review": review_ledger(review),   # null = UNREVIEWED (not reviewed-and-clean)
        "ordered_sibling": einfo.get('ordered_sibling'),
        "generation_recipe": {
            "engine": "mar_engine.py",
            # For an ENGINE build these fields ARE the recipe (the CE+MC seed). For a CUSTOM build they
            # describe the engine enumeration that was ATTEMPTED and set aside; `build_recipe` (from the
            # analyst) is the real construction, and `engine_enumeration_used` flags which is authoritative.
            # Did the engine enumeration RUN -- not "is the shipped frame the engine's own pick",
            # which `representative.source` already answers. Keying this off the override erased a
            # real enumeration: a run that swaps the representative to a random-pool draw
            # WITHIN its own enumeration (the SQS principle) then read
            # "engine_enumeration_used: false", i.e. cell / n_configs / spread not comparable at all.
            # A genuine custom build carries neither field; an enumeration carries both.
            "engine_enumeration_used": bool(einfo.get('supercell') and einfo.get('group_orbits')),
            "supercell": einfo.get('supercell'), "exclusion_merge_A": einfo.get('exclusion_A'),
            # D29: `exclusion_merge_A` is the CROSS-element cut only. The SAME-element cut is what
            # `--same-excl` overrides, and it lived in `exclusion_merge.same_element_cut_per_element`,
            # which this recipe did not copy -- so a run corrected with `--same-excl 3.0` came out
            # BYTE-IDENTICAL to a default run (measured on one entry, round 2). Scoring 100 entries by
            # `generation_recipe` could not tell a corrected build from an uncorrected one. The report
            # is small (0.4-1.5 kB across the bundled fixtures), so copy it whole rather than project
            # it and have to guess later which field the scorer needed.
            "exclusion_merge": einfo.get('exclusion_merge'),
            "n_groups_merged": einfo.get('n_groups_merged'),
            "group_orbits": einfo.get('group_orbits'),
            "sampling": einfo.get('sampling'),
            # compute provenance: which weights produced these energies, their hash when pinned,
            # what resolved them, and the versions underneath. Without it the record cannot be
            # checked against the run that claims to have produced it.
            "mlip": engine.get('mlip'),
            "model_repair": judg.get('model_repair'),
            "build_recipe": judg.get('build_recipe')},   # custom-build construction logic (None for engine builds)
        "ce_mc": judg.get('ce_mc', {"warranted": None, "reason": None}),
        "escalate": judg.get('escalate'),
        "decision_trace": judg.get('decision_trace', []),
    }

def main():
    args = sys.argv[1:]
    iid = int([a for a in args if not a.startswith('-')][0])
    engine = json.load(open(_arg(args, '--engine')))
    judg = json.load(open(_arg(args, '--judgment'))) if _arg(args, '--judgment') else {}
    hull = json.load(open(_arg(args, '--hull'))) if _arg(args, '--hull') else None
    rev = json.load(open(_arg(args, '--review'))) if _arg(args, '--review') else None
    conn = json.load(open(_arg(args, '--connectivity'))) if _arg(args, '--connectivity') else None
    rec = build_record(iid, engine, judg, hull=hull, review=rev, connectivity=conn)
    out = _arg(args, '--out')
    if out:
        import os; os.makedirs(out, exist_ok=True)
        json.dump(rec, open(f'{out}/record.json', 'w'), indent=1, ensure_ascii=False)
    # sanity: warn (to stderr) on missing judgment / failing gates
    miss = [k for k in ('class', 'interpretation', 'disorder_pattern', 'confidence') if not judg.get(k)]
    if miss: print(f"WARN: judgment missing {miss}", file=sys.stderr)
    for g, v in rec['gates'].items():
        if isinstance(v, dict) and v.get('pass') is False:
            print(f"WARN: gate {g} FAILS: {v}", file=sys.stderr)
    print(json.dumps(rec, indent=1, ensure_ascii=False))
    return 0

if __name__ == '__main__':
    sys.exit(main())
