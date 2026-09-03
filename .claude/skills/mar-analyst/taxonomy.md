# Disorder mechanisms — signals, classes, and the MAR move

## How to use this file
**Reason from the evidence SIGNALS to a STRATEGY (what to build).** The A–F class is a **soft,
post-hoc SUMMARY** of the signals — useful as a checklist (so you don't overlook a mechanism), as
a prior for the default MAR move, and as a DB metadata label. It is **not** a decision you make
first and then act on. Real entries are often **hybrids** (a split site that is also coupled; a
vacancy that is really an averaged ordered superstructure) — name both, add a caveat, tag
confidence. **The signals decide; the class describes; the strategy is the deliverable.**

One thing is a **hard gate**, applied from the signal regardless of any label: **coupling** (below).

---

## 1. Signals → reading (mine these from the ①a bundle, then reason)
| signal in the bundle | what it suggests | points toward strategy |
|---|---|---|
| `mixed` orbit (2+ elements share a site, occ ≈1), oxidation clean | substitution on a shared sublattice | maintain / integer-resolve |
| single-element `partial` orbit (occ<1) | vacancy **or** a positional split — check stoichiometry next | (decide below) |
| cell **exactly stoichiometric** (Σ mult×occ = ideal formula) | partials are **splits**, NOT vacancies | collapse split → custom / one position |
| cell **sub-stoichiometric** on the partial species | genuine **vacancy** | integer-resolve vacancies in a supercell |
| two same-element alternates <0.8 Å apart, or large/anisotropic ADP | **displacive** double-well / off-centering | pick ONE well |
| `cross_orbit_contacts_below_2.6A` non-empty | **coupled** occupancies (HARD GATE) | never enumerate independently → exclusion-merge / `--couple-cut` at setup |
| `rigid_unit_signal` non-empty | a rigid molecular/polyanion unit (BO₃, CO, MFₓ) the enumerator does NOT generalise over | **custom** build, one orientation placed rigidly |
| heavily `partial` light-cation sublattice spread over many sites | **mobile-ion** (averaged trajectory) | one representative ordered snapshot |
| element in `formula_sum` but `missing_from_structure` | unresolved / unlocated atom | H: the engine's auto H-restore; identity unknown → **custom** |
| ordered sibling exists **+ large energy gap** (≫ tens meV) | averaged model of an ordered structure | present our MAR + **report ΔE** + artifact note; fix the reach — **never adopt** |
| ordered sibling exists **+ small gap** (few meV) | quasi-degenerate; real/entropic at measurement T | maintain our MAR; **report ΔE** — **explain the pair** |
| `oxidation_states` all 0 | intermetallic; no charge story → partials are structural | (use the split/vacancy/coupling signals) |
| **HT** in name/title + clean mix + ordered sibling | entropy-stabilized configurational | maintain — **explain the pair** |
| ordered sibling in the **same SG** / in a **different SG** | same-SG = a re-refinement that resolved the disorder ⇒ suspect **F** or HT-**A**; other-SG = ordered polymorph/superstructure | either way **explain the pair** (the one place literature is unavoidable) |
| `nominal_cell_charge` ≠ 0 with a clean formula | a species is missing/over-counted in the deposited model | check `composition_vs_formula` |

The enumeration **energy spread** (when you run it on a slaving-free case) reads ordering:
~0.5 meV/atom = ideal random solid solution · tens of meV = dispersed/ordered preference ·
>100 meV = strongly ordered (an averaged model of an ordered structure). Read **std**, not the
raw range — the range inflates with `--nr` and is not comparable across entries.

**Reading a `none` sibling result.** The search is strict by design (nominal-reduced-formula
equality + a fully-ordered site table — see [tools.md](tools.md)). `"none — no fully-ordered ICSD
entry of this composition"` is a **checked** answer, but it does not mean the phase has no ordered
form: the ordered parent may sit at a different nominal stoichiometry, or simply not be in ICSD.
So a `none` weakens an F call without settling it — when the *other* evidence points at an averaged
ordered model (near-zero enumeration spread, a suspicious subcell, a scattering-degenerate pair),
say so and cap confidence. Distinguish it from `"none (no sibling DB configured)"`, which means
**unchecked** — never report that as "no sibling exists."

## 2. The hard gate — coupling (act on the signal, not the label)
If `cross_orbit_contacts_below_2.6A` is non-empty, disordered sites of **different** orbits sit
within bonding range — their occupancies are **not independent** (a follower H/O on a partial
parent; a vacancy bonded to a split cation; two splits that share a well choice). You may **never**
decorate/enumerate them independently. Handle it at setup: the engine's **exclusion-merge**
(`--excl`) collapses coupled sites into one-atom groups, and `--couple-cut` co-places a
former–anion bond. Apply this **before** (and regardless of) naming a class. If the merge cannot
express the coupling (a rigid unit, an orientation that must move as a whole), that is a **custom**
build — see strategy.md #4.

**When `couple_cut` comes back `unmatched` (or `confident: false`), read the raw contact list, not
the verdict.** That state means the *role filter* matched nothing, and the filter asks whether an
element is a covalent network former — so a coupling carried by cation **size and charge** instead
of by a bond can never match, and the report reads like a clean non-match. The gate still holds if
**either side of a contact is a mixed-occupancy orbit** (two or more species sharing one site),
whatever the element tables say: which species takes the shared site decides which neighbouring
partial sites can be occupied, so the two orbits are one degree of freedom. Name the roles with
`--couple-formers` / `--couple-anions` — the cutoff alone cannot add one. Miss it and it fails
**silently**: the same tables drive the defect descriptors and the connectivity gate, so the run
reports zero defects, a gate that examined nothing, and a narrow ensemble spread — three
clean-looking signals produced by the miss itself, two of which stop the self-driving loop.

---

## 3. The classes (soft summary + default MAR move)
A named pattern of the signals above. Use to orient and to recall the default move; allow hybrids.

- **A — Configurational.** `mixed` orbit(s), clean oxidation. Substitution on a shared sublattice
  (isovalent or aliovalent). Subtypes: **A-dilute** (minority dopant) · **A3-frustrated** (no
  orderable ground state). *Move:* maintain (representability rule) or integer-resolve; purify only
  a dilute **isovalent** non-representable minority.
- **B — Vacancy.** single-element `partial`, cell sub-stoichiometric, charge balanced aliovalently
  or by a motif. *Move:* integer-resolve vacancies in a supercell; place by chemistry/bond-valence.
- **C — Mobile-ion.** a diffusing light-cation sublattice averaged over many sites; framework
  ordered. *Move:* one low-energy ordered snapshot at the right stoichiometry; mobility quoted.
- **D — Displacive.** same-element split (<0.8 Å) / large ADP = double-well or orientational
  average; not a chemical mixture. *Move:* pick ONE well/orientation (may be a **custom** build);
  averaging quoted.
- **E — Slaved.** the coupling gate above — bonded cross-orbit coupling. *Move (hard):*
  exclusion-merge / co-place at setup; never enumerate independently.
- **F — Twin / not-disorder.** pseudo-symmetric averaging / artifact / averaged ordered
  superstructure; ordered sibling within a few meV, or a large gap with zero enumeration spread.
  *Move:* present our de-averaged MAR + an **artifact note** + the sibling ΔE (**never** adopt the
  sibling); fix the reach (resize/reshape cell, enforce coupling, custom-build) if the cell can't
  capture the order.

**Hybrids are normal and often correct.** A split site that is also coupled = D+E (pick a well, but
merge at setup). A vacancy that orders into a superstructure = B+F (artifact note; build the
superstructure ourselves if the cell can't reach it — never adopt). Name both, say which degree of
freedom dominates by energy scale, and tag confidence.
