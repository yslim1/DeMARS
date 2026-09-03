# Instrument shelf

You **orchestrate** these tools; you do **not** reimplement their physics. They live in
`tools/` in the repo root and are thin wrappers over the installed `demars_core`
package — run them from the repo root through `tools/py`:

```bash
cd "${DEMARS_ROOT:-$PWD}" && tools/py tools/<tool>.py …
#                            ^ the demars env's interpreter, from paths.python in demars.yaml
#                              (demars_core is installed editable from ./demars-core)
```

**Prefix every Bash call with that whole chain.** A subagent's `cd` does not persist between tool
calls, so a one-time setup followed by a bare `tools/…` runs in the wrong directory.

**Never run a bare `python`, and never activate an env.** Your shell is re-initialized from the
profile on every call, which restores whatever env that profile sets up: a bare `python` dies with
`ModuleNotFoundError: demars_core` however the session was launched, and an activation of your own
is undone by the very next call. `tools/py` resolves the interpreter per call instead, and nothing
can undo it.

Command lines below are written as bare `python tools/…` for readability — substitute `tools/py`
for that `python` and prepend the `cd`. `icsd-query` needs no prefix beyond the `cd`: it is on
`PATH` and independent of the env.

**The primary key here is a FILE PATH, not an ICSD id.** Every tool takes the structure file
(CIF preferred — partial occupancies survive only in CIF). Pass the *same* file to every stage.
If the CIF you are analysing *is* an ICSD entry, also pass `--icsd-id N` to stage ①a so it
self-excludes from the sibling search.

**The local ICSD DB is wired up when `paths.icsd_db` is set in `demars.yaml`** (`$ICSD_DB_DIR` overrides it). When it is configured, the
ordered-sibling search works — see [`icsd-query`](#icsd-query--the-icsd-database) below.
**No Materials Project key is configured**, so there is no convex hull; read
[Missing capabilities](#missing-capabilities) before you quote an `E_above_hull`.

---

## `tools/demars_evidence.py <struct.cif> [--summary] [--out <rundir>] [--icsd-id N] [--no-siblings]` — stage ①a, the evidence bundle
**Clean contract.** stdout = JSON; `--summary` also prints a human digest to stderr.
**`--out <rundir>`** also writes `<rundir>/evidence.json`. Always pass it — and then **let stdout
go**. Saving both leaves an `evidence_stdout.json` that is byte-for-byte the same file.
This is the CIF *mined* (principle 8). Reason from it first. Schema:

```
icsd_id                      # null unless you passed --icsd-id
source                       # absolute path of the CIF you passed
provenance:
  chemical_name_common, chemical_name_mineral, structure_type,
  formula_structural, formula_sum,
  citation_title,            # often states the finding outright
  journal, year, authors[],
  measurement_temperature_K, measurement_pressure_kPa, R_factor, density_diffrn
cell: a,b,c,alpha,beta,gamma,volume, Z, spacegroup, spacegroup_number, n_atoms_cell
oxidation_states_in_cif      # true if the CIF has the oxidation-number loop (read, don't infer)
oxidation_states: {el: ox or [ox,...]}
mixed_valence_elements: []
nominal_cell_charge          # using the CIF oxidation states; ~0 if balanced; null if no states
composition_vs_formula: [    # formula_sum x Z vs what is actually deposited
  {element, formula_expected, located, deficit, missing_from_structure} ]   # catches missing N, unlocated H
sites: [ {frac, members:[{label,element,type,oxidation,occ,wyckoff,multiplicity,adp,adp_type}],
          occ_sum, wyckoff, multiplicity, is_mixed, is_partial, signature} ]
disorder:
  n_partial_positions, n_mixed_positions, n_disordered_orbits,
  orbits: [ {signature, n_positions, mixed, partial, wyckoff} ],
  cross_orbit_contacts_below_2.6A: [ {d_A, count, orbit_a, orbit_b} ],   # NON-EMPTY => coupling/slaving; do NOT enumerate independently
  rigid_unit_signal: []      # BO3/CO/MFx-type units the auto-enumerator does NOT generalise over
  slaved_units: [ {follower_signature, parent_signature, trigger_element,     # class E, COMPUTED
                   n_parent_positions, followers_per_parent, follower_element} ]
  orientation_units: [ {signature, element, occupancy, n_anchors,
                        positions_per_anchor, alternatives_per_anchor} ]
  hydrogen: {located, formula_expected, deficit} | null
ordered_sibling: "ORDERED same-SG: ICSD <id> (+N more)" | "ORDERED other-SG <sg>: ICSD <id> ..."
                 | "none — no fully-ordered ICSD entry of this composition"
ordered_sibling_ids: [id, ...]   # FULL ranked candidate list: same-SG first, then other-SG
scope: {n_elements, n_atoms_cell, excluded, reason}    # gt6-elements / high-pressure / partial-structure
```

**`slaved_units` and `orientation_units` are the class-E signal, computed rather than eyeballed.**
`taxonomy.md` defines **E — Slaved** and until now nothing measured it: `cross_orbit_contacts` gave
you distances and you drew the pairing yourself. These two name the pairing.

- **`slaved_units`** — a partial orbit whose atoms exist only where a partial PARENT orbit is
  occupied (a hydroxide H following its O, the O completing a half-occupied former's coordination).
  `followers_per_parent` is how many the parent carries: 1 for an OH, 3 for a carbonate. It is
  reported only when EVERY parent position finds a follower, and never with a full parent — a full
  parent encodes no choice.
- **`orientation_units`** — a partial single-element orbit (occupancy 0.25–0.75) clustered into
  discrete alternatives around full-occupancy anchors of the SAME element. The choice is which
  orientation, not which atom. Distinct from `rigid_unit_signal`, which keys on a former CENTRE and
  its partial light ligand shell; neither sees the other's case, so read both.

**Non-empty means per-site decoration is invalid on those orbits** — place the unit, not the atoms.
The research pipeline's enumeration diagnostic simply refused to run in that situation. Here it is
your call: set `--couple-cut` from the bonding contact, use `--rigid-units`, or build the unit
yourself. An **empty** list is not evidence the entry is simple — both criteria are strict, and
`orientation_units` in particular needs a full-occupancy anchor of the same element.

**How the sibling match works** (so you can judge a null result): the search takes the CIF's
chemsys, and keeps a candidate only if its **nominal reduced formula** — every element's
partial-occupancy count rounded to its full-occupancy integer — EQUALS the target's, and its CIF
site table is fully ordered (no mixed site, no partial occupancy). That strictness is deliberate:
it separates "an ordered version of the SAME phase" from a different nearby compound that a flat
composition tolerance would conflate. A `none` is therefore a real, checked answer — but it means
"no ordered ICSD entry of *this nominal composition*", not "this phase has no ordered form".

Note the CIF-provenance fields are only as rich as the file. A COD/CSD/user CIF often has no
`_citation_title` and no oxidation loop — say so in your judgment rather than inventing one.

### `tools/py -m demars_core.disorder_class <struct.cif>` — the O/S/V/P descriptor (①b, optional)

The published orbit descriptor (Antypov, Collins, Dyer, Claridge & Rosseinsky, *J. Appl. Cryst.*
**58**, 659–677 (2025), Table 1): every crystallographic orbit labelled **O** (ordered) /
**S** (substituted, full) / **V** (vacancy-bearing) / **P** (intersecting = split sites) and their
combinations (SV, SP, VP, SVP), plus the **`no_full_backbone`** flag. Pure CIF, no relaxation, ~1 s.
stdout = JSON `{orbit_labels, multiset, disorder_set, n_orbits, no_full_backbone}`.

Stage ⑥ runs it itself off `evidence['source']` and files it under `record.disorder_descriptor`, so
you never pass it in. Run it **by hand at triage** anyway, because `no_full_backbone` is a
*pre-build* signal: see the `no_full_backbone` row in `strategy.md` — every orbit vacancy-bearing
means no fully occupied scaffold anywhere to template the disorder, and literal random decoration is
then the known silent failure. Corroborate it geometrically with the ensemble `cell_scatter` after
the build.

This is the **published descriptor, not the DeMARS mechanism call (A–F)**. Quote it as a structural
signal; the class is still yours to judge.

## `tools/demars_engine.py <struct.cif> --out <rundir>/_work/<tag> [flags]` — stage ③, THE CORE ENGINE
**This is the uniform enumeration engine — the main build tool.** Builds a faithful ≥1.5 nm
supercell; **exclusion-merges** split/coupled sites (<`--excl`, default 1.1 Å) into one-atom groups
(the primary coupling handling); samples diverse occupancy-preserving decorations; relaxes with
7net-nano (no D3); emits the FULL per-sample distribution; and (`--final`) recomputes the lowest
with the omni `mpa` modal (no D3).

Flags: `--nr N` (samples, default 30) · `--min-nm 1.5` (**nanometres**) · `--excl 1.1` ·
`--same-excl X` · `--couple-cut X` · `--couple-formers EL[,EL]` · `--couple-anions EL[,EL]` ·
`--rounds 3` (self-driving) · `--final` · `--d3` (opt-in;
OFF by default) · `--rigid-units` · `--calculator sevennet|<ase-calc>` · `--summary`.

**`--rigid-units`** — hold each discrete former–ligand unit's bond lengths rigid through a
fixed-cell pre-relax, then release for the normal variable-cell relax. **Turn it on whenever the
evidence shows a rigid unit** (`rigid_unit_signal` non-empty, or the composition contains an
oxo-anion / molecular ion / octahedral framework). Without it a free relax lets a neighbouring
cation tear a ligand off its centre — the unit breaks, and charge & fidelity are composition-only
and wave it through. The units are detected from the structure (covalent-radii bonds + the
coordination actually observed), so it needs no per-compound input and is a no-op where there is no
unit. It forces the SERIAL ASE path (torch-sim has no bond constraint), so it is **much slower** —
budget for it, and it is incompatible with `--d3`. Works in `--from` mode too, which is where a
hand-built polyanion most needs it.

Writes **to its `--out` dir**: `ensemble.xyz` (energy-marked, OVITO-native), `representative.{cif,vasp}`,
`representative_final.{cif,vasp}` (with `--final`), and **`engine.json`** — the stage-③ contract
that `tools/demars_record.py` consumes. **That is why `--out` is `_work/<tag>/`, never the run dir**:
a rerun or a sibling relaxation would otherwise drop a rival `representative*` set beside the record.
You adopt a run simply by passing its `engine.json` to the record assembler.
```
distribution: {n_relaxed, n_total, ground_E, highest_E, spread_meV, std_meV, energies_sorted, lowest, highest}
samples are inside ensemble.xyz frame info: {label, mode, E_per_atom, charge}
engine: {supercell, n_atoms_template, axes_A, exclusion_merge, exclusion_A, n_groups_merged,
         couple_cut, coupling_signal,
         group_orbits:{id:{members_per_group, merged, n_groups, states, target,
                           exclusion:{mode, n_pairs, note}}},
         sampling, broad_spread_flag, coordination_integrity_flag, ordered_sibling}
self_driving: [ ...the diagnose->re-enumerate trace... ]
provenance_summary: {n_unconfident, unconfident:[names], states:{name: state}}   # READ THIS FIRST
final_MAR: {E_final_per_atom, n_atoms, modal}    # with --final (modal = "mpa")
ensemble_files: {ensemble, n_frames, representative, representative_final}
```

### `provenance_summary` — what the engine could NOT decide

**Read this before you read the distribution.** Every parameter the engine *derives* now says how it
got its value. `n_unconfident > 0` means the engine acted on, or defaulted, a value it could not
justify — and those are exactly the places a build goes plausibly wrong without tripping any gate.

| `state` | meaning | your move |
|---|---|---|
| `derived` | the data determined it | nothing |
| `overridden` | you set it (`--same-excl` / `--couple-cut` / `--couple-formers` / `--couple-anions`) | nothing |
| `vacuous` | the derivation ran, **zero candidates, none expected** — the default is right | nothing |
| `ambiguous` | it ran but the evidence is weak (no clean distance gap) | judge it |
| `out_of_window` | **data exists but sits outside the range the derivation inspects** | judge it |
| `unmatched` | **candidates exist but no element/type rule matched them** | judge it |
| `not_run` | the derivation never executed (capability absent, or it raised) | say so; never read as absence |

**`charge_balancing` is one of the names in here (since 2026-08-18) and it is the sharpest one.**
The engine may move site counts away from the CIF's refined occupancies to reach neutrality — that
is designed, since a refined occupancy can itself be charged. But when a move takes a partially
occupied orbit to **FULL or EMPTY**, the disorder the entry exists to represent is gone, and the
state is `ambiguous`. Read `evidence.orbits_filled_to_full` / `orbits_emptied`.
**Do not just accept it.** A refined alkali occupancy of ~0.95(1) — a ~5σ vacancy, i.e. the disorder
that entry existed for — was filled to 100 % because the cell offered exactly one lever and the gate
is an ABSOLUTE `|q| < 0.3`. The right move was to give the charge model a lever instead (a repaired
CIF floating the counter-cation's oxidation state off its integer), not to accept the filled
vacancy. Before this flag existed you had to read `engine.json`'s `charge_balancing` by hand to
notice, and the summary reported `n_unconfident = 0` while it happened.

The first three are `confident: true`. For the rest, the block carries an `evidence` dict with the
numbers you need — **use those instead of re-mining `evidence.json`.**

Two you will meet often, and what they actually ask you:

- **`exclusion_merge.<El>` = `out_of_window`** — same-element pairs exist but all lie above the
  1.3 Å window, so **no exclusion cut was set for that element**. Distance alone cannot say whether
  such a pair is a split image (mutually exclusive) or a real bond / face-sharing pair (both occupied).
  That is chemistry. A face-sharing Mg–Mg at 2.87 Å landed here and needed `--same-excl 3.0`.
- **`couple_cut` = `unmatched`** — cross-orbit contacts exist but none is a former–anion pair under
  the engine's element tables, so the cut fell back to the module default. **The engine cannot tell a
  CORRECT non-match from a table gap**, and both happen: Li–O is correctly not a former–anion bond,
  while Co in a Co–As intermetallic and Ga in Ga–Te are simply absent from the tables. Decide which.
  ⚠️ **`--couple-cut` cannot fix a table gap** — it retargets the cutoff, and the contact is not even
  in the filtered list, so no value reaches it (that was D37). If an element DOES play the role here,
  say so: **`--couple-formers <EL>`** / **`--couple-anions <EL>`**. They REPLACE the tables for this
  structure, and then the cutoff derives itself from the measured bond. Measured on a chalcogenide
  whose former element is absent from the tables: default → `unmatched` and the cut falls back to the
  module default while the real bond sits well above it; with the role named → `overridden` and the
  cut derived from that bond.
  The engine will never promote an element itself: asking which elements *behave* like formers was
  measured to fire on La, Mg, Fe, Ti, Zr, Ca, Cs, … and a K with CN 27. This is your call.

### `group_orbits[*].exclusion` — how exclusion is actually enforced

**`merged: false` does NOT mean "no exclusion".** It means "not clique-merged", and that was
misread three times in a row on one entry. Read `exclusion.mode`:

| `mode` | what it means |
|---|---|
| `clique-merged` | the sites were a clique of alternates → ONE group, exactly one atom placed. Exclusion is in the merge (`merged: true`) |
| `pairwise` | a chain/network, so the sites were split into **singletons** and the conflict edges are enforced at decoration (valid occupations = independent sets). **`merged: false` and exclusion IS enforced** — `n_pairs` counts the excluded pairs |
| `none` | no site pair of this orbit falls under the cutoff. Genuinely nothing to exclude |

One structure can carry both `pairwise` and `none` orbits, so you cannot read it off `merged`
alone — and before this field existed the only way to know was to re-derive the self-symmetry
pairs from the CIF operators by hand. `generation_recipe` in `record.json` carries it too, so a
reader of the record can check exclusion handling without the engine output.

Record your call in `judgment.json` `decision_trace` for every unconfident entry — "checked, default
is correct" is a valid answer, silence is not.
- `distribution.lowest` is the representative sample. Its features + the spread SHAPE are what
  you classify from. Report **std** as the primary width (n-robust); the raw range inflates with
  `--nr` and is not comparable across entries.
- **`--from <struct>` mode:** `tools/demars_engine.py <cif> --from <other.vasp|cif> --out <dir>`
  relaxes ANY provided structure through the same nano→omni-mpa tiers → `single_MAR`. Two uses:
  (a) realise a **custom/repaired MAR of your own construction**, and (b) compute an **ordered
  sibling's energy on the SAME tier** for the ΔE comparison. The sibling is **NEVER adopted** as
  the representative — see [principles.md](principles.md).
- No-disorder entries fall through to a single relax automatically.

A production run (`--nr 30`, ≥1.5 nm) takes a long time — routinely more than the 600 s ceiling on a
foreground Bash call, so you have to start it with the Bash tool's `run_in_background: true` —
never a shell `nohup … &` or a bare `&` (see below). Scope your `--nr`/`--min-nm`
deliberately, and never quietly shrink the cell below the 15 Å cutoff to save wall-clock — record
what you ran.

### Then WAIT FOR IT IN THE FOREGROUND. Do not end your turn.

**You are a subagent. You cannot wait.** Ending a turn without a tool call does not pause you — it
*terminates* you, and whatever you wrote becomes your return value. There is no notification coming
to wake you up; that mechanism is for the main session, not for you. So a turn that ends with

> *"I've launched the relaxation and will continue once it completes."*
>
> *"I'll hold here until the run finishes and then pick it back up."*
>
> *"I'll pause and resume once something tells me the relaxation is done."*

Sentences of this shape ended four entries across two validation campaigns — every one of those
analysts had launched the run correctly, every one expected to be woken, and every one was
terminated mid-entry. **Nothing will call you back.** You have no monitor, no watcher and no
notification: naming one in your closing sentence does not create it.

reads like patience and is actually death mid-entry: the relaxation keeps running, nobody reads its
result, and your caller receives that sentence as your finished answer. The work is not saved by
having completed — a finished relaxation nobody collected has to be run again.

Poll it instead, with a **foreground** call, so every turn ends in a tool call and you stay alive:

```bash
python tools/demars_engine.py ... --out <dir>/_work/r0     # run_in_background: true

# then, foreground, timeout: 600000 -- repeat this call until it prints "done"
for _ in $(seq 60); do [ -f <dir>/_work/r0/engine.json ] && { echo done; break; }; sleep 10; done
```

One such call covers the 600 s ceiling; a longer job just needs several of them, which is cheap and
is the whole fix. **Wait on the artifact the job writes, never on a process id.** `run_in_background`
hands you a task id, not a PID, so `kill -0` has nothing to take — and reaching for `$!` instead
means `nohup … &`, which is the trap: it hides the job from the harness (so no completion
notification can ever fire) without protecting it from the session ending, so the job dies
uncollected and you park forever waiting on it. If the loop stops printing `done`, read the job's
log rather than inferring anything from a process being absent — that is equally true the instant
before it starts.

**Also make the job leave something behind.** A relaxation whose result exists only in the stdout of
a process nobody is reading is a relaxation you will have to run twice — write the ensemble through
`write_custom_ensemble()` (see [strategy.md](strategy.md) §4) or `--out`, then read the file.

## `icsd-query` — the ICSD database
The local ICSD metadata DB + CIF zip (`paths.icsd_db` in `demars.yaml`). Stage ①a
already uses it for the sibling search; you use `icsd-query` directly to **fetch a sibling's CIF**
so you can run the ΔE comparison, or to explore the chemsys.

```bash
icsd-query show <id>                       # full DB row + raw CIF
icsd-query extract <id> [<id>...] -o <dir> # write CIFs -> <dir>/icsd_<0-padded id>.cif
icsd-query chemsys La-Mn-O-Sr --ordered    # exact chemsys, fully-ordered only
icsd-query chemsys Ag-Ge-Te --disordered   # partial-occupancy only
icsd-query subsystem Li,La,Zr,O            # contains AT LEAST these elements
icsd-query formula LiFePO4                 # by reduced formula
icsd-query sg 14 --chemsys Cr-O-Te         # spacegroup + chemsys
icsd-query stats
```

**The sibling ΔE workflow** (when `ordered_sibling_ids` is non-empty):
```bash
icsd-query extract <sibling_id> -o <rundir>/_work/sibling
python tools/demars_engine.py <struct.cif> --from <rundir>/_work/sibling/icsd_<id>.cif \
       --out <rundir>/_work/sibling --final
```
It stays in `_work/sibling/` and is **never promoted** — the ΔE is the deliverable, the structure
is not. (one entry left a relaxed sibling next to the MAR: 12× smaller, and indistinguishable
from the real `representative_final.cif` to anything that globs the run dir.)
Compare that `single_MAR.E_final_per_atom` to your own MAR's `final_MAR.E_final_per_atom` — **both
on the omni-mpa tier, never nano** (a ≲15 meV/atom gap can flip sign between tiers). Report the ΔE
in `judgment.json.sibling_comparison`. **Never adopt the sibling as the representative.**

## Custom builds — case-by-case scripts
The path when the generic constructor can't (restore an unresolved/missing atom, build the
physically correct ordered unit, then relax). Write a small self-contained script that emits the
structure(s), then realise them with `demars_engine.py --from`. Write build artifacts to a scratch
dir, never over an existing result dir.

## Verification gates
- **charge** — from the engine's per-sample `charge` (it sums the CIF oxidation states); `|q| < 0.3`.
  Recomputed authoritatively by `demars_record.py` — don't transcribe it.
- **fidelity** — the engine's realized integer targets vs the CIF occupancies
  (`engine.group_orbits.target`); deviation ≤ ~0.08. Also recomputed by `demars_record.py`.
- **connectivity** — `python tools/demars_connectivity.py <struct> [--json] [--tol 1.25] [--expect S=4,P=4]`
  (also run it on `ensemble.xyz`, which audits ALL frames). It finds every discrete former–ligand
  unit and checks each centre carries a consistent ligand count; exit 0 = clean, 1 = defect (prints
  the offending atoms). **No per-compound table**: bonds are detected at `covalent_radii × --tol`
  and the target coordination is the MODE observed in *this* structure, so BO₃→3, SO₄/PO₄/SiO₄→4,
  MoO₆→6 and NH₄→4 are handled by the same code, and a structure with no discrete unit is simply
  reported as having nothing to check. **The mode is taken per ROLE, not per element** — a centre
  is labelled `C[N]` / `C[H+N]` by the SET of ligand elements it carries when one element centres
  more than one unit type (single-role elements keep the bare `C`). And an atom of a dual-membership
  element (N, S, Se, Te, halogens) that is a *ligand* of another centre is reported under
  `_coverage.ligand_role` instead of as a 0-coordination stripped centre. Both were false-positive
  factories: one cyanide / methylated-cation entry reported 160 defects on a chemically fine cell
  without them. A centre that lost
  **every** ligand is still a defect — it is judged against its element's dominant role. Use `--expect` when the intended coordination is known and
  must be **enforced** rather than inferred — the mode follows the majority, so a build that relaxed
  into a *mixture* of coordinations can otherwise hide behind its own majority.
  Charge & fidelity are composition-only and **cannot see any of this**. MUST be clean before
  shipping any structure with an oxo-anion / molecular-ion / octahedral unit. Two distinct causes
  when it flags a custom build: you placed a ligand absolutely instead of rigidly (rebuild as
  `centre + min_image(ligand − centre)`), **or** you free-relaxed and a cation tore the unit apart
  (use the two-stage constrained relax — principles.md).

## `tools/demars_hull.py <representative_final.vasp> [--calculator omni] [--out <rundir>]` — stage ④b, the CONVEX HULL

Is this MAR thermodynamically **reachable**, and what would it decompose into? Every gate asks
whether the cell is a faithful realisation of the CIF; this is the only check that asks whether the
phase is stable at all — and where the composition has **no ordered sibling**, it is the only
stability reference you have. Writes `<out>/hull.json` (stage ⑥ takes it with `--hull`) **and the
tier-relaxed MAR as `<out>/representative_final.{vasp,cif}`** — one hull run also upgrades the
deliverable to the final tier. **So point `--out` at `<rundir>/_work/<tag>/`, never the run dir**, or
you leave a second `representative_final` beside the engine's own.

Run it on **the structure that shipped**, at **the tier that structure was finalized with**
(`--calculator omni` = the DeMARS final tier, the default). An `E_above_hull` computed at a different
level than the structure it describes is not a stability number. Handing it the input CIF is refused
outright: ASE cannot hold partial occupancies, so a disordered input would silently hull a
composition nobody built.

Two modes, stamped in `mode`. **`self-consistent` is the default**: MP gives the competing phases'
*structures* and every one is re-relaxed at your tier, so the MLIP-vs-DFT offset cancels out of the
subtraction instead of landing on the MAR alone — MARs that sit **on** the hull once the references
are recomputed can read tens of meV/atom above it the other way. `--mode mp-direct` takes MP's own energies as
vertices and relaxes only the MAR: much cheaper, **omni-mpa only** (asking for it at another tier is
refused, not quietly swapped), and it falls back to self-consistent when MP's GGA thermo has a gap
(it is empty for some f-elements) — `mode_note` says when that happened.

**The energy SCALE is stamped in `corrections`, and it is not cosmetic.** Default `mp2020` =
MP's own hull scale (`MaterialsProject2020Compatibility`, applied to references **and** the MAR) —
the scale MP's published `energy_above_hull` is on, so it is the only one comparable to a number
quoted from materialsproject.org. `--corrections none` = raw PBE(+U); internally consistent, but not
MP's hull. MP applies a correction only where an element sits **as an anion**, and GGA+U only for
oxides and fluorides of V/Cr/Mn/Fe/Co/Ni/W/Mo, so the offsets **cancel** in a hull decomposition
unless it crosses that role boundary — 0.0 meV/atom measured on a sulfide and a cobaltate. A
mixed-valence oxide does cross it, and the two scales can then disagree by ~0.1 eV/atom **and name
different decomposition products** — the raw scale putting a metallic phase among them where the
corrected one gives the ternary oxide. Read `corrections` before you read the number. If you switch
to `none`, say why in `decision_trace`.

**It needs a key**: `$MP_API_KEY`, or `materials_project.api_key` in `demars.yaml` (the `mp-api`
client itself is a hard dependency now). Read the `state` before you read any number:

| `state` | means |
|---|---|
| `derived` | computed. `E_above_hull_eV_per_atom` is real; `decomposition` names the competing phases |
| `ambiguous` | ran and could not conclude — an element with no reference, too few refs for the simplex |
| `not_run` | **UNCHECKED** — no key, no `mp-api`, MP fetch failed, the MAR would not relax, or you passed a disordered structure |

`not_run` is never "on the hull", and a negative `E_above_hull` is not automatically a triumph — it
is either a genuinely new stable ordering or a reference set missing a competitor. Say which you
think it is. **Never quote a hull number you did not compute.**

**This artifact IS `gates.hull` (the fifth gate).** Pass it to stage ⑥ with `--hull` every time —
it is the one gate that cannot re-run itself there, so skipping ④b leaves it `not_run`, and by this
record's rules an unchecked gate is not a pass.

**By default this gate REPORTS and does not judge.** With no `hull.tol_eV_per_atom` configured you
get `state: derived` and **`pass: null`** — read `E_above_hull_eV_per_atom`, and say in your judgment
what you make of it. That is deliberate: the research pipeline computed this number and left the call
to a person, there is no inherited threshold, and the terms that would set one are system-dependent
(the configurational entropy the 0 K hull omits — a MAR is an *ordered approximant* of an
entropy-stabilised phase, so sitting tens of meV/atom above the hull is normal, scaled by how much of
the cell the mixed sublattice is and by the CIF's own refinement temperature). `below_reference_hull`
is a flag, not a verdict: under the hull means either a better ordering or a reference set missing a
competitor, and deciding which is yours. If a threshold IS configured the gate judges against it and
records the value.

## `tools/demars_record.py <struct.cif> --engine <engine.json> --judgment <judgment.json> [--hull <hull.json>] [--connectivity <connectivity.json>] --out <dir>` — stage ⑥
The RECORD assembler. You write `judgment.json` (your ⑤ verdict, fields below); it merges that with
the engine + evidence into the canonical **mar-1.0** record and **recomputes the charge & fidelity
gates itself** (it does not trust your transcription).

`distribution` also carries **`bracket_probes`** (each bracket's `margin_meV_vs_lowest_ship`) and
**`n_ship_candidates`**. The engine chooses `lowest` from the random samples only, so a
`dispersed`/`clustered` extreme can no longer win take-lowest; read the margins instead — tens of
meV negative on `clustered` is the "this material may genuinely order" flag, which is a
mechanism-class question, not a frame to repick.

**Two of the five gates run themselves.** `gates.connectivity` audits the shipped frame (preferring
`representative_final`) and `gates.sqs` checks whether the enumeration's **clustered** lowest is what
shipped — `strategy.md` forbids shipping that as-is. Both use the `PROV_STATES` vocabulary:
`derived` (examined; the verdict means what it says) · `vacuous` (ran, nothing here to check —
no unit centre / no enumeration) · `ambiguous` (multi-frame audit with no coverage reported) ·
`not_run` (could not run — for connectivity, the shipped frame did not resolve; `basis` says why).
**`pass` is `null` for the last three — only `derived` yields true/false**, so a `vacuous` gate is
never a pass. 6 of the frozen run's 15 "clean" connectivity verdicts were vacuous.

`--connectivity` is therefore **optional**: supply the stage-⑤ artifact only when it carries what the
auto-run cannot — `--expect` (an enforced coordination) or a multi-frame `ensemble.xyz` audit. A
supplied artifact wins. **Never hand-edit a gate into the record**: this stage rebuilds `gates` from
scratch every run, which is what used to make stamping a review erase what the review asked for. Writes `<dir>/record.json`; stdout = the
record; **warns on stderr** for missing judgment fields and failing gates — read those warnings.
Pass the same structure file you gave the engine.

**`judgment.json` (you produce):** `class` (A–F + subtype; hybrids ok, e.g. "D+E") · `class_label` ·
`interpretation` · `disorder_pattern` · `ordering_read` (your read of the distribution shape) ·
`confidence` (high/medium/`[screening]`) · `principles_applied[]` · `quoted_corrections[]` ·
`representative_pick` {config_label, reason} — **reselect a frame of the engine's OWN ensemble**
(the clustered-lowest escape). Name the LABEL as it appears in `distribution.lowest.label` /
`ensemble.xyz`; stage ⑥ finds the frame, reads its composition and engine-tier energy from it, keeps
the **enumeration** gate basis, audits *that* frame for connectivity, and leaves the `ensemble`
block on the same `engine.json`. Nothing is copied and no file is written. `E_final` comes back
null **with a reason** (the final tier ran on the enumeration's lowest, not on your frame). An
unresolvable label is NOT a silent fallback: the record stamps `pick_unresolved` and the tool warns ·
`representative_override` {file, E_final_eV_per_atom, note} (only for your OWN custom/repaired
build — **not** for choosing a different frame; that was D19) · `build_recipe` {parent, supercell, steps[], charge_balance, relaxation, rationale}
`representative_pick` {config_label, reason, **final_from**} — the reselection. `final_from` is the
`engine.json` of a `--from --final` run on THAT frame, and it is how the shipped structure gets a
final-tier energy: the engine relaxed the enumeration's **lowest** at the final tier, not your pick.
Every other path has one (take-lowest by construction, a custom build through `--from`), so without
it the record carries `E_final: null` and `gates.hull` has no tier to match. Extract the frame,
`demars_engine.py <cif> --from <frame> --final --out <rundir>/_work/<tag>_final`, and name that
`engine.json`. The ensemble and the gate basis still come from the ONE `--engine` you adopt.
(**custom builds ONLY** — the reproducible construction logic; threads into the record's
`generation_recipe`) · `ce_mc` {warranted, reason} · `escalate` · `decision_trace[]` ·
`model_repair` · `sibling_comparison` {sibling, dE_meV_per_atom, artifact_note} — fill it in when
`ordered_sibling_ids` is non-empty (the local ICSD sibling search **is** wired up; see the ΔE
workflow above). **Never an adopt decision in any case** — the ΔE is the deliverable, the sibling
structure is not.

**mar-1.0 record (assembled):** `schema_version` · `icsd_id` (null) · `source` · `provenance` ·
`mechanism` (your judgment) · `ensemble` {file, n_configs, engine_model, energy_distribution} ·
`representative` {file(+final), composition, supercell, E_nano, E_final, E_above_hull} · `hull`
(when `--hull` given) · `gates` {charge, fidelity (both recomputed), sibling_comparison} ·
`ordered_sibling` · `generation_recipe` (the engine spec — the CE+MC seed) · `ce_mc` · `escalate` ·
`decision_trace`.

## Deliverable per entry
`record.json` (mar-1.0) at the run dir top level, plus the `ensemble.xyz` +
`representative.{vasp,cif}` (+ `representative_final.vasp`) that the engine wrote in the
`_work/<tag>/` of the run you adopted.

**You adopt a run by passing its `engine.json` to `demars_record.py` — there is no separate step,
and nothing needs copying.** `record.json`'s `representative.file` / `file_final` come straight from
the engine's own `ensemble_files`, so **the record is the only authority on which structure is the
MAR** and that answer is code-proven. Never identify the MAR by globbing a run dir (one dir can hold
several `representative_final.cif`, and one of them may be the forbidden ordered sibling), and never
copy one up to the top level to mark it — that creates a second answer that can contradict the record.
**Custom single-frame builds:** when the de-averaging yields ONE ordered unit (no configurational
ensemble to sample — rigid-unit/twin/orientational cases run via `--from`), write the xyz as
`representative.xyz` (a single energy-marked frame), NOT `ensemble.xyz` — there is no ensemble.

---

## Missing capabilities

Do not fake a tool's output; say plainly in your judgment when a check was unavailable.

- **Hull / E_above_hull.** Ported (stage ④b above); its client `mp-api` is installed with the
  package, so what it still needs is a Materials Project key. Where that is absent the artifact
  says `state: "not_run"` — that is UNCHECKED, so record it as unchecked and `escalate` if the
  call genuinely turns on stability.
  **Never quote a hull number you did not compute.** Omitting `--hull` at stage ⑥ leaves
  `record.hull` null, which means the stage was never run; a hull that ran and refused keeps its
  block and its reason, and those two must not be reported alike.
- **Unlocated-H reconstruction.** The automatic H-restore keeps located H, adds the formula's
  deficit, places protons clash-checked (retrying, and re-offering a blocked host's allotment), and
  accepts O/N/F/Cl/S/Se. Read `h_restore.method` — it says which acceptor model decided:
  `valence-driven completion` (capacity = valence − neighbours), `charge-compensating Bronsted`
  (aliovalent former), or **`bond-valence deficit (fallback)`** — the last is reached only where the
  capacity model found nothing placeable, and it is what lets a hydrous silicate or phosphate build
  at all (a bridging O with an ionic Ca–O contact reads capacity 0 and bond-strength deficit 0.87).
  It needs oxidation states in the CIF; without them it declines. `applied: false` still happens —
  an orientational acceptor risk (Zundel/ice/amine) or a deficit neither model can seat — and then
  the protons are **yours** to build (custom, then `--from`) or to `escalate`. Never report an
  unrestored deficit as restored.

The **Antypov O/S/V/P descriptor IS available** (ported 2026-08-24 as
`demars_core.disorder_class`; earlier versions of this skill and of the record reported
it as absent). `record.disorder_descriptor` now carries real labels — and it still says
`{"available": false, "reason": ...}` instead of `null` when a record was built from an evidence
bundle with no `source` to read. Read which one you got.

The **ordered-sibling lookup IS available** (it was missing in an earlier version of this skill —
it is wired up now). If the DB ever goes missing the bundle degrades gracefully to
`"none (no sibling DB configured)"` — that string, unlike `"none — no fully-ordered ICSD entry of
this composition"`, means **unchecked**, not **checked and absent**. Read which one you got.
