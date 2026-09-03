# MAR construction strategy

Once the mechanism is called, pick the construction strategy. The governing idea
(**optimal-representation principle**): truncate the representation at the **dominant degree of
freedom by energy scale**; keep that faithfully, quote the rest as a correction. The MAR is the
*minimal* structure that still carries the physics — not the largest faithful supercell.

The five strategies are how you BUILD. The sections after them — cell size, ensemble-vs-single,
random-only sampling, rigid-unit relaxation — apply to **every** strategy, custom builds included.

## The five strategies

1. **maintain** — keep the fractional occupancy, realized as integers in the smallest supercell
   that represents it. Default for configurational (A) and vacancy (B) when the occupancy is
   *representable* (see rule). Preserves the real stoichiometry, including dilute antisites and
   aliovalent minorities.

2. **integer-resolve** — round each orbit's occupancy to integer counts via **largest-remainder**
   targets in a supercell (composition AND per-site occupancies conserved; random *assignment*,
   not random counts). Use when maintain's exact ratio needs a supercell within budget.

3. **purify** — drop a minority species and adopt the clean host. **Only** when the minority is
   *dilute AND isovalent AND not representable* within the supercell budget. Never purify a
   concentrated or aliovalent minority (it changes the chemistry/charge). Representability, not a
   fixed 0.15 threshold, is the test.

4. **custom** — build the physically correct structure by hand when the generic constructor can't
   (e.g. restore an unresolved/missing light atom and build one ordered unit). Reserved for cases
   where the CIF model is an artifact of the experiment (scattering-degenerate identity, missing
   light atoms). **Also the handler for ORIENTATIONAL disorder of a rigid unit** — a bonded group whose
   internal geometry is fixed and whose ORIENTATION is the degree of freedom. Do not carry a list of
   which chemistries qualify: the detector is geometric and element-agnostic, so an anion, a molecular
   cation, a neutral cluster and a coordination shell all reach it the same way. **Read
   `rigid_unit_signal` (evidence) / `rigid_units` (engine output)** — it fires on a centre whose partial
   light-atom ligand shell has more sites than atoms (a complete shell smeared over orientations) with
   same-element sites too close to coexist, and it emits neutral facts for you to judge. The engine's
   per-site distribution is unreliable on those orbits, and the auto-enumerator does NOT generalise
   (combinatorial ambiguity — tested), so place the units by hand.

   **Hand-building settles HOW a frame is made, never HOW MANY.** The orientation choice is a
   configurational DOF like any other, so unless the alternatives are non-degenerate you owe an
   ENSEMBLE here exactly as you would for a solid solution: draw N random orientation assignments over
   the units in the cell — **the assignment across units, not just the choice within one** — relax them
   all, and let the energy distribution answer the question. Ship a single orientation only by passing
   and LOGGING the SINGLE-CONFIG GATE below; *"it is a rigid unit, therefore one frame"* is the
   inference that gate exists to stop. **Both answers occur here and the mechanism name predicts
   neither**: a unit smeared at occ 1/N over N symmetry-equivalent settings with a large ADP on the
   centre is a rotator CANDIDATE, not a verdict. Some rigid units sit hundreds of meV/atom apart and
   have a real ground state; others come back inside MLIP noise and are genuinely disordered. Relax the
   alternatives and read their ΔE — that number decides, and nothing about the formula does.
   Realise a hand-built structure with `tools/demars_engine.py <cif> --from <mybuild.vasp>`.

   A custom build is held to the SAME cell-size, ensemble and sampling rules as an engine build —
   the sections below. Where it differs: **generate that ensemble by your OWN design logic** for this
   case (a build/sample script that respects the repair) — do NOT route a flagged custom case back
   through the generic engine's enumeration mode: the engine already failed on it, and a generic
   enumerator would re-apply the blind spot that flagged it
   (e.g. charge-balance an itinerant metal and revert the repair). Use `--from` to *relax* your build, not
   to re-enumerate it. Codifying a pattern into the engine is the exception, earned only when it is
   specific + general + validated.

   **Whatever relaxes your build, WRITE THE ENSEMBLE THROUGH THE CONTRACT.** A hand-written script
   inherits none of the engine's guarantees, and **printing energies is not persisting them** — a
   script that reports its numbers to stdout and saves nothing leaves a completed relaxation with no
   ensemble, no representative and nothing to run the gates against, so it has to be run again.
   Close it with one call, which writes the same files under the same names as the stock engine:

   ```python
   from demars_core.api import write_custom_ensemble, _make_relax, _final_omni
   relax, mode, prov = _make_relax('sevennet', d3=False, rigid_units=True)
   E, rel, val = relax(confs)
   write_custom_ensemble(OUTDIR, cif, 'sevennet', rel, E, val,
                         oxidation_states=ox, mlip=prov,           # else the record cannot say
                         final_struct=final_atoms, final_MAR=fm)   # which weights produced it
   ```

   It returns a `MARRecord`, so the gates, stage ⑥ and the gallery cannot tell your build from an
   engine build — which is the point: a custom build is a different way to REACH the ensemble, not
   a different kind of result.

5. **enumerate-then-pick** — generate diverse occupation-preserving decorations, relax (nano),
   take the ground/representative. Use to (a) distinguish ordered vs random (the energy spread)
   and (b) pick a representative snapshot for mobile-ion (C) or frustrated (A3) cases.
   **GUARDED:** only valid when there is **no slaving/coupling** — if the bundle shows any
   `cross_orbit_contacts_below_2.6A`, independent enumeration is INVALID; handle the coupling at
   setup (exclusion-merge / `--couple-cut`) or custom-build.

## The representability rule (maintain vs purify)
A species count `v` (atoms of that species the cell would contain) is **representable** if
`v / n_sites * MAXAT >= 0.5` — i.e. it survives as ≥1 atom somewhere within the supercell budget.
(`MAXAT = 500` is the nominal budget; the hard ceiling the engine actually enforces is
`MAXAT * 1.6 = 800` atoms. Check `sampling.min_cell_A_achieved` in the record — if the budget bit,
the supercell was cut and the provenance says `overridden`.)

- **Per-species AND per-orbit.** Check globally *and* per orbit: a species present elsewhere
  globally can still be dropped from a *specific* antisite orbit — escalate that orbit too
  (per-orbit maintain).
- If representable → **maintain** it. If not representable AND dilute AND isovalent → **purify**.
- Aliovalent or concentrated minority that isn't representable → escalate the supercell, don't
  purify (it would shift charge/chemistry).

## Supercell budget and the cell-length cutoff
- `MAXAT = 500` atoms (`mar_engine.py:36`). The engine takes `mult = ceil(MIN_A / |a_i|)` per axis
  and backs off the largest multiplier while `n_atoms > MAXAT * 1.6`.
- Escalate the cell to capture a faithful ratio (a larger cell can hit a more faithful, more neutral
  ratio than rounding in the primitive cell) **when** the cell fits.
- If accommodating the fractional occupancy needs a bigger cell and it fits the budget, **do it**
  — prefer the faithful ratio over a rounded one (the dilute-dopant / escalation directive).

**CELL-LENGTH CUTOFF — applies to CUSTOM builds TOO, not just engine builds.** A hand-built MAR must
satisfy the SAME minimum cell size as the engine: **every lattice-vector length ≥ 1.5 nm (15 Å)** —
the engine's exact measure, `L = np.linalg.norm(base.cell, axis=1)` then `ceil(MIN_A / L)`
(`demars-core/demars_core/_engine/mar_engine.py:610`). Judge a non-orthogonal / primitive cell by its
ACTUAL vector norms — NOT by a conventional-cubic span or a minimum-image hand-wave (a primitive cell
with short edges at a non-90° angle is exactly as short as those edges, even when its atoms sit on a much
larger conventional lattice — the periodic image IS at the primitive edge length). If the repaired
parent's natural cell is smaller, EXPAND it (multiplicity-aware) until all three axes clear 15 Å,
choosing the supercell shape that PRESERVES exact stoichiometry: a **conventional** setting will often
clear the cutoff at the same multiplier while keeping the integer ratio, where bumping the primitive
multiplier by one instead would break the integer count. Verify and RECORD the final cell lengths in
`build_recipe`.

## Ensemble or single config?

**Emit an ENSEMBLE when the residual disorder is CONFIGURATIONAL — custom builds included, not just
engine builds.** For a custom case the hand-build step is the *repair* (fix the cell/charge/identity,
restore atoms, de-double a superstructure). After repairing, look at what DOF remains: if it is an
*arrangement* with many near-degenerate realizations — a substitutional solid solution (A), a vacancy
arrangement (B), a mobile-ion snapshot (C/A3), or a choice among rigid-unit orientations — then SAMPLE
it: generate N realizations by RANDOM sampling ONLY (no dispersed/clustered extremes), relax all
(nano → omni-mpa), write `ensemble.xyz` + the energy distribution, and take the representative from
those RANDOM configs. Do NOT emit one snapshot and merely *quote* the spread — the recurring anti-pattern
is a cation-vacancy solid solution shipped as a single dispersed config, where structurally equivalent
vacancy / channel cases were correctly shipped as multi-config ensembles. The faithful
per-config distribution belongs in the record, exactly like maintain/enumerate. **Emit a SINGLE config
ONLY when the disorder resolves to a genuinely UNIQUE structure:** an ordered superstructure /
twin-averaged aristotype (D+F), a single displacive distortion (D), an isotopic/label no-op, or a
model-repair whose result is uniquely ordered. Rule of thumb: if you would call `enumerate-then-pick`
on the *repaired* parent, you owe an ensemble.

**SINGLE-CONFIG GATE — PROVE uniqueness, do NOT assert it.** A single ordered frame makes the STRONGEST
possible claim: "no residual configurational DOF exists." Earn that claim with a logged test; never infer it
from the mechanism type ("it's displacive / orientational / a rigid unit → therefore one frame" is exactly how
a WRONG single frame gets shipped). Before emitting one config: (1) ENUMERATE every candidate residual DOF,
**including inter-unit ones** — the relative phase/orientation BETWEEN the several chains / columns / layers /
rigid units in the cell, not just the choice within one; (2) build a cell big enough to HOST them —
**multiplicity-aware: ≥2 (ideally ≥3) independent copies of each disordered unit along each lattice direction,
with special attention PERPENDICULAR to low-D units** (a 1×1×n cell along a chain does NOT expose inter-column
disorder — it holds a single column; expand in-plane); (3) generate a handful of alternative orderings /
orientations and relax them; (4) ship one frame ONLY if the alternatives are symmetry-equivalent or the chosen
one is clearly, non-degenerately lowest — and RECORD the test (cell used, alternatives tried, ΔE) in the
judgment. If the alternatives come back near-degenerate, the disorder is REAL → emit an ENSEMBLE. Two
otherwise near-twin structures can differ in VERDICT by the supercell alone: a cell holding a single copy of
the disordered unit reports no inter-unit disorder because it cannot host any. (Engine default is ≥1.5 nm /
`MAXAT = 500`, ceiling `MAXAT * 1.6 = 800`, which forces such in-plane expansion in most cases — but
**verify it actually happened for YOUR cell**: `sampling.min_cell_A_achieved` vs `min_cell_A` in the record.)

## Random-only sampling — the SQS principle

**RANDOM-ONLY, and why (the SQS principle):** a disordered solid solution's real state is the *random*
arrangement (entropy-stabilized / kinetically frozen at synthesis T), so the MAR must be a typical random
snapshot — NOT the 0 K energy minimum. With any clustering tendency the minimum is the PHASE-SEPARATED
(clustered) config, which does not represent the disordered phase; including a clustered/dispersed extreme
in the pool makes take-lowest ship that artifact. So sample random only and pick the representative from
the random configs. A clustered config sitting *far* below the random ones is a FLAG that the material may
genuinely ORDER (reconsider the class → D+F / averaged-superstructure), not a config to ship.

**The ENGINE now does this for you.** It used to offer the `dispersed`/`clustered` brackets to
take-lowest, so `distribution.lowest` could be a probe and shipping it needed a manual escape. Since
2026-08-24 `distribution.lowest` is chosen from the RANDOM samples only (`ship_candidates`), and the
probes stay in the ensemble where bounding the distribution is their job. So read
**`distribution.bracket_probes`** instead: it gives each probe's `margin_meV_vs_lowest_ship`.
A few meV is noise. **Tens of meV negative on `clustered` is the FLAG above** — the material may
genuinely order, and that is a mechanism-class question (D+F / averaged superstructure), not a frame
to pick. `distribution.n_ship_candidates` says how many samples were eligible.
**`representative_pick` is still there** {config_label, reason, final_from} for a reselection you make
on other grounds — the engine no longer forces one for a bracket probe. It keeps the record honest
(`source: engine-repick`, enumeration gate basis, one ensemble); do **not** reach for
`representative_override`, which says you BUILT the structure (that misuse was D19). One cost to know:
the engine relaxes the enumeration's *lowest* at the final tier, so a picked frame has only the nano
energy until you relax it yourself — `--from <frame> --final` into its own `_work` tag, then name that
run in `final_from`, or the record carries `E_final: null` and `gates.hull` has no tier to match.
`gates.sqs` remains as the watchdog for a probe reaching the representative anyway.

## Relaxing a build that contains a discrete unit — two stages, not one

Correct rigid PLACEMENT does not survive a naive relax: the ligands start at their right offsets but the
surrounding cations are still at averaged CIF positions, sometimes inside bonding range, and a free relax lets
one of them pull a ligand off its centre. Hold the unit's internal bond lengths at FIXED CELL
first, then release and run the normal variable-cell relax — pass **`--rigid-units`** to
`tools/demars_engine.py`, in `--from` mode as well. The units are derived from the structure itself
(covalent-radii bonds + the coordination actually observed), so it applies to oxo-anions, molecular
ions and octahedral frameworks alike and is a no-op where there is no unit. Then run the
connectivity gate on the representative AND the ensemble. NOTE: the flag forces the serial ASE path
(torch-sim has no bond constraint), so it is much slower and cannot be combined with `--d3`.

## Charge & valence discipline
- Use the **CIF oxidation states** (`oxidation_states` in the bundle) — read, don't infer.
- Build neutral: strict gate `|q| < 0.3` per cell. Charge-aware integer adjustment is **bounded**
  to each species' rounding window `[floor(raw), ceil(raw)]` — never fabricate atoms beyond it
  (the over-fill failure mode). Buffer at most ~1 oxidation unit/atom.
- For unlocated light atoms (H, sometimes N): if the formula expects them but they're not
  deposited (`composition_vs_formula` shows a deficit), the engine's automatic **H-restore** runs
  by bond-valence saturation. The standalone allocators (`mar_h_alloc` / `mar_h_topup`) are **not
  ported here** — if a case needs one, `escalate`. For a missing non-H light atom whose identity
  the experiment couldn't resolve, that's a **custom** build.

## Fidelity gate
- Round-trip fidelity tolerance `fid_tol = 0.08` (occupancy deviation). A maintained fractional
  occupancy in a bounded cell may not match exactly — report the deviation honestly; antisite
  cases are the hardest. `tools/demars_record.py` recomputes this; don't transcribe it.

## Tool catalog — the levers you wield (DEFAULTS WITH A REASON, not rigid rules)

DeMARS is a **toolbox + your judgment**. Each knob below has a physical MEANING and a data-derived
default; the engine REPORTS how it set each one (`engine.json` → `exclusion_merge`, `couple_cut`,
`h_restore`, `self_driving`, `broad_spread_flag`, `coordination_integrity_flag`, `full_occ_clash`,
`rigid_units`, `minority_rescue`, `dummy_species`). **Your job: understand the meaning, AUDIT the
engine's choice against the chemistry, and OVERRIDE when this system is an exception.**

**Those same flag names are the trigger keys in `episodes/episodes.json`.** When one is set in your
`engine.json`, look it up there and read the `lesson` — it is what an earlier entry taught us about
exactly that situation, and it may say the engine's default is wrong here. A few episodes key on a
situation you reach by judgment rather than a flag (`situation:custom_build`,
`situation:single_config`, `situation:hull`, …); consult those when you are in one. The lesson is
what applies. The `provenance` beside it records which entries it was measured on, for auditing the
de-identified wording — **you do not need it to apply the lesson, and reasoning from a past entry's
answer instead of your own evidence is the contamination this project exists to avoid.**

**`episodes.json` is READ-ONLY to you.** It is shared campaign state: an episode added mid-run
changes the doctrine the *remaining* entries are judged under, so the batch stops being comparable
with itself, and a lesson that has not passed a reviewer is doctrine before anyone has checked it.
When this entry teaches something general, say so in `escalate` — name the chemistry and the
failure mode, propose the wording — and stop there. Someone outside the run adds it.

Two guardrails keep tuning honest, never fitting: **(a) set the value FROM THE DATA** (the structure's
own geometry / the evidence), never to match a hoped-for answer; **(b) VERIFY with the gates**
(coordination-integrity, charge, energy/spread). A value-from-data that passes the gates is right; one
that doesn't, isn't.

| Tool (flag) | Meaning / when it bites | How the engine derives it | When YOU override (exceptions) |
|---|---|---|---|
| `--excl` (cross-element merge, 1.1 Å) | catastrophic overlap of *different*-element disordered sites | fixed small (a 1.2–1.5 Å cross-element contact may be a real bond) | rarely; raise only if two different elements are genuinely a shared/averaged site |
| `--same-excl` (same-element merge) | same-element disordered sites too close to coexist = **alternates** of one site (the collapsed-ring failure mode: a planar oxo-anion's ligand half-sites merged into one) | **read from the distance GAP**: alternates cluster below a gap to real neighbours → cut in the gap; fallback `max(EXCL,1.3)`+`CHECK` flag if no gap | a close same-element pair that is a **real bond** — peroxide O–O ~1.49, metal–metal cluster (Mo/Re/W…), short M–M intermetallic → do **not** merge (lower `--same-excl`). The engine flags `suspicious_merges` (group occ-sum >1.2) for exactly this |
| `--couple-cut` (former–anion co-placement) | the former–anion BOND defining a unit (BO₃, PO₄, NH₄…); decoupled placement → dangling anions | **from the evidence cross-orbit contact** (shortest former–anion contact ×1.15); fallback 2.0 | if the shortest contact isn't the bonding one, or the unit's bond is unusual → set from chemistry |
| `--min-nm` / supercell | faithful ≥1.5 nm cell | largest-remainder occupancy in the smallest cell ≥ MIN_A under MAXAT | escalate the cell for a more faithful ratio when it fits |
| H-restore (automatic) | formula declares H but no coordinates (water/OH/amine/hydroxide layers/zeolite Brønsted) | per-acceptor valence saturation: an acceptor bonded to a substitutional **cation** (hydroxide/LDH) is completed; greedy partial-tier fill; **aliovalent-former → charge-compensating Brønsted H** (B³⁺-for-Si⁴⁺); only **orientational/rigid** disordered acceptors flagged | organic/Zundel/ammonium/ice (orientational proton order) needing the orientation builder → custom (these fail the charge gate **loud**) |
| broad-spread + coordination-integrity triggers | a missed coupled/rigid unit (loud in energy OR a coordination defect) | spread >0.2 eV/at OR representative carries dangling/under-coord | a broad spread that is genuine strong ordering / SRO (not a defect) → don't force-suppress |
| `full_occ_clash` (TRIAGE, element-normalized) | impossible contact on a **fully-occupied** orbit — no occupancy decoration can relieve it | ρ = d/(r_cov sum) on full-occ single-species sites: **`flag`** (ρ<0.5, distinct atoms) = **SUPERPOSITION** = average over a modulated/twinned/doubled-orbit structure (fingerprint: a sub-Ångström contact between two FULL-occupancy heavy-atom orbits — geometrically impossible in any single cell); **`dup_flag`** (d<0.2 Å, coincident) = pymatgen symmetry-expansion duplicates, **auto-deduped** before building (`n_coincident_removed`) | a `flag` (superposition) is **yours to judge from the evidence** — incommensurate modulation / wrong SG / overlaid phases → it is *not* decoratable disorder: build a commensurate de-doubled approximant + **quote the modulation** and **escalate** (don't pretend a finite cell is the truth). `dup_flag` is already fixed — just note it |
| `no_full_backbone` + ensemble `cell_scatter` (TRIAGE, "framework unclear") | EVERY orbit is vacancy-bearing (no fully-occupied O/full-S/P scaffold) → no rigid framework to template the disorder | structural: disorder set has no full backbone (record `disorder_descriptor.no_full_backbone`); geometric: relaxed-ensemble cell-shape scatter (`ensemble.cell_scatter` — edge-length CV%, angle std°) is high (⚠ ≥4.5%/≥4° vs ≲1%/≲1° for rigid frameworks). The two signals converge | **do NOT random-decorate.** Two resolutions, told apart by formula-vs-occupancy: (a) formula MANDATES the vacancies (the stoichiometry itself forces a fixed cation-vacancy fraction) → real **correlated/ordered-defect compound** → CE+MC or a reasoned ordered-vacancy build; (b) formula implies the site is FULL but the CIF lists it partial (e.g. manganite Mn₁O₃ at occ 0.5) → **cell-doubling/averaging artifact** masking a full backbone → un-double + restore the framework. Both are silent-failure traps where literal decoration gives dangling atoms / non-neutral cells |
| `dummy_species` (TRIAGE; engine flags, you decide) | a CIF site labelled with a NON-ELEMENT symbol (e.g. `L1+` sharing an alkali site) — a **vacancy placeholder OR a mislabeled/truncated element** (L→Li). Structure+charge CANNOT decide (charge may even favour the ELEMENT reading: filling the site with the candidate element can balance the cell exactly where the vacancy reading leaves a residual) | engine `dummy_species` report: `element_candidates` (prefix, e.g. L→Li/La/Lu), `charge_under_readings` (per reading), `citation_hint`, `recommendation`, `escalate`. Engine default-strips to vacancy so it can proceed, but **escalate=True means UNRESOLVED** | **resolve CIF-FIRST then WEB (public reference only)**: (1) read the CIF `citation_title`/`chemical_name` — "vacancy/deficient" → vacancy; "doped/substituted" → the element; (2) if still ambiguous, **web-search the public reference** (title/DOI/authors — NEVER paste the structure) to see what the paper reports; (3) decide **vacancy** (keep engine default) / **remap to the element** (rebuild with that element) / **escalate** (present both). Do NOT trust the charge alone — it can favour the wrong reading |
| `--rounds` (self-driving) | how many diagnose→re-enumerate passes | default 3; stop on clean+normal or SRO-only | raise for stubborn multi-defect systems |
| `--nr` (sample count) | how many occupancy-preserving decorations to relax | **default 30**, auto-**capped at the number of distinct decorations** so tiny combinatorial spaces don't relax duplicates | raise for a stubborn/borderline SRO read; the cap handles small spaces automatically. Lower it only under a wall-clock constraint you record |
| spread descriptor (how you READ the distribution) | degenerate (solid solution) vs ordering/SRO | report **std as the PRIMARY width** (n-robust); the raw max−min **range inflates with sample count**, so range is NOT comparable across entries run at different NR | use **std** for the genuine-disorder-vs-orderable call and any cross-entry comparison; range is a secondary within-entry cue only |
| custom build (`--from`) | occupancy decoration can't represent it (refinement artifact / rigid molecular unit) | flagged by the detectors | build it OURSELVES (custom/repaired/fix-the-reach) and realise it with `--from`; **never** adopt someone else's ordered structure as the MAR |

**The discipline (what makes this general, not a pile of fudges):** when DeMARS hits a NEW problem, the
move is *not* "guess a number" and *not* "give up" — it is: read what the engine derived and why →
recognise which tool the diagnosis points to → set that tool **from this structure's data** → re-run →
let the gates judge. A threshold is legitimate when it is a measured property of the structure verified
by the outcome; it is fitting when it is a constant chosen to reproduce a known answer. Apply with
judgment; the engine's reported `basis`/`CHECK`/`suspicious` fields are there for you to exercise it.
