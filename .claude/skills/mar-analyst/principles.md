# Field-tested principles & failure modes

Distilled from case-by-case work into **general rules — not memorized cases.** Match the
*situation* (the signals), not a remembered entry. These are operationally precise without naming
any compound; a case you half-remember should never override what the evidence in front of you says.

## Reading the situation → the move

The signal → class → default-move table is [taxonomy.md](taxonomy.md) §1. Below are only the
readings that table does NOT give you.

- **A scattering-degenerate pair averaged ~50/50** (neighbouring-Z atoms: C/N, Al/Si, Fe/Mn,
  Co/Ni…), or a light element present in `formula_sum` but **missing from the structure**: often an
  X-ray artifact (the experiment can't distinguish them), NOT real chemical disorder. Check for an
  ordered sibling and/or restore the identity → **model-repair** (restore identity); compare the
  ordered sibling by ΔE — **never adopt it as the representative**.
- **A recurring local motif shared by the low-energy samples**: a bound, charge-compensated complex
  (defect cluster) → keep the host + quote the complex.
- **Near-zero enumeration spread on a structure that "should" have configurational freedom, or a
  lowest sample ≫ a known ordered sibling**: the
  disordered model may be an *average* of an ordered structure, or your enumeration cannot *reach*
  the ordered superstructure (wrong cell shape/size). **Fix the reach** first — resize/reshape the
  cell, enforce the coupling, or custom-build the ordered unit yourself. Report the sibling ΔE
  (omni-mpa, no D3) as a comparison, and add an **artifact note** (with evidence) if
  provenance/chemistry shows the disordered CIF is a refinement artifact. If the sibling search came
  back `none`, that weakens an F call but does not settle it (the ordered parent may sit at another
  nominal stoichiometry, or not be in ICSD) — cap confidence accordingly.

## Strategy selection (general)
- **Representability decides maintain vs purify.** Keep a minority that survives as ≥1 atom within
  the supercell budget; **purify only a dilute, isovalent minority that cannot be represented.**
  Never purify a concentrated or aliovalent minority (it changes the chemistry / charge).
- **Escalate the cell for a faithful ratio.** When a larger (within-budget) supercell captures the
  occupancy ratio more faithfully — and more neutrally — than rounding in the primitive cell, use it.
- **Pick the surrogate by physics.** Simple short-range pair energetics → the deposited cell / a
  classical CE may already be the leading order; multi-species multi-sublattice → GNN-CE/MC;
  relaxation-dominated coupled systems → go direct (enumerate + relax).
- **Never adopt a sibling; the sibling is a COMPARISON, not a replacement.** The
  representative is always our own de-averaged MAR. A lower-energy ordered sibling does **not**
  license swapping it in — the same energy gap arises
  in two physically distinct cases, and the mechanism call changes only the *note*, never the
  deliverable: **(a) refinement artifact** (the disorder isn't real — an averaged subcell of a
  missed superstructure, twin/intergrowth, or unresolved modulation) → present our MAR, add an
  **artifact note** with the evidence, and if our enumeration can't reach the ordered
  superstructure, **fix the reach** (cell/coupling/custom-build); **(b) genuine disorder** (a real,
  distinct material — quenched/HT polymorph, kinetically trapped, or entropy-stabilized solid
  solution) → the sibling is a **different material**; record it as a *separate* MAR and just report
  the ΔE. Decide which case from the **mechanism + measurement context** (CIF citation TITLE,
  measurement T, synthesis route, ADP/occupancy signatures, Z-subcell-vs-superlattice), NOT from ΔE
  — the gap only ranks stability, it can't tell artifact from metastable.
- **Compute any ΔE on the final (omni-mpa) tier, not nano.** A near-noise gap (≲ ~15 meV/atom) can
  **flip sign** between nano and omni-mpa, so a *reported* ΔE must come from the final tier. For
  near-degenerate gaps tag **medium** confidence — never "high" on a sign-ambiguous gap — and anchor
  the artifact-vs-genuine *note* on the chemistry (entropy / an HT tag), not the gap.

## Failure modes (don'ts — hard-won)
- Don't **infer** oxidation states when the CIF has the oxidation-number loop — **read** it.
- Don't purify a representable / aliovalent / concentrated minority.
- Don't enumerate sites whose occupancies are coupled (bonded <2.6 Å) — exclusion-merge first.
- Don't ship a polyanion build without the **connectivity gate** — charge & fidelity are
  composition-only and will pass a malformed unit (an SO₃ with the right O count). When
  custom-building a polyanion, place each unit RIGIDLY (`centre + min_image(ligand − centre)`),
  never at absolute CIF ligand coords — interleaved half-occupied ligand half-sites otherwise
  mis-credit ligands between neighbouring centres, and relaxation only partly heals it. Run
  `tools/demars_connectivity.py`.
- **Rigid PLACEMENT is not enough — a discrete unit also needs a rigid first RELAX** (`--rigid-units`;
  the two-stage relax and why a free relax breaks the unit are in [strategy.md](strategy.md)).
- Don't over-fill a partial orbit with unbounded charge-aware adjustment — bound it to the rounding
  window `[floor(raw), ceil(raw)]`.
- Don't read ordering off a too-small (primitive) cell — expand to ≥1.5 nm; small cells force
  finite-size artifacts.
- Don't trust that relaxation will *reveal* coupling — it **heals** split-site clashes; catch
  coupling geometrically at setup.
- Don't force a single class label — hybrids are normal; name both, say which DOF dominates by
  energy scale.
- Don't match an ordered sibling by the reduced-formula **string** — the ①a search already does the
  right thing (nominal-reduced-formula equality on a fully-ordered site table); don't second-guess
  it with a looser string comparison of your own.
- Don't quote a number no tool here produced. **E_above_hull comes from stage ④b or not at all** —
  if that stage reports `state: "not_run"` (normally: no MP key), record it as UNCHECKED
  rather than as a passing hull, and never fill the number in from memory or from MP's own page. And
  don't report a `none` sibling as proof no ordered form exists; it means none of *this nominal
  composition* is in ICSD.
- Don't silently trade rigour for wall-clock.
