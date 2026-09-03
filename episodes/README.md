# Episodes

What earlier entries taught this program, kept in the form that can be applied to a new one.

`episodes.json` holds them. Each episode is keyed by **the trigger that should call it up** — almost
always a flag the engine already emits into `engine.json` (`dummy_species`, `full_occ_clash`,
`h_restore`, `rigid_units`, …), sometimes a situation the analyst reaches by judgment
(`situation:custom_build`, `situation:single_config`). `triggers` maps each key to the episodes it
fires.

```
episodes["E-cell-length-cutoff"] = {
  "lesson":      "…",                         # what applies. This is the whole point.
  "artifact":    "doctrine",                  # doctrine | gate | deterministic code | diagnostic | open
  "shipped_as":  [{"where": …, "text": …}],   # where the de-identified wording lives in the tree
  "provenance":  {"entries": […], "detail": …}
}
```

**`lesson` is what you apply. `provenance` is not.** It records which entries the lesson was measured
on, so the de-identified wording in the skills and engine comments can be audited against what it
actually stood for. Reasoning from a past entry's answer instead of the evidence in front of you is
the contamination this project exists to avoid — see `CLAUDE.md`, *name the chemistry, not the entry*.

Two flags on a provenance block are worth reading:

- `residual_hint: true` — de-identification removed the identifier but the chemistry alone still
  points at the entry, so the wording is as far as de-identification could go on its own.
- `open_defect` — the episode is not closed; `demars-core/docs/DEFECTS.md` has the defect.

## Adding one

**Not from inside a run.** The `mar-analyst` reads this file and never writes it: adding an episode
mid-campaign re-bases the doctrine partway through, so the earlier and later entries of one batch are
no longer answering the same question — that has happened on three runs. An analyst that has a lesson
puts it in the record's `escalate`; it is added here afterwards, once the reviewer has been past it.

Record the episode here **first**, then write the de-identified wording into the skill or comment —
in the other order the next corpus run is contaminated before anyone notices, which has happened.
The lesson has to hold without the number; if it doesn't, it isn't a lesson yet.
