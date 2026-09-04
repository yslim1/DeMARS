# DeMARS version workflow

`state.json` records the current version and whether protected production code is mutable or frozen.
A release tag identifies the protected snapshot for each frozen version.

## Modes

- `mutable`: protected code may change. Production Analyst and Reviewer runs are blocked; an explicit
  rehearsal may run when its prompt contains the exact line `VERSION_MODE: rehearsal`.
- `frozen`: the current release tag must exist, protected tracked files must match that tag, and no
  untracked protected file may be present. Production runs are allowed.

The hooks in `.codex/hooks.json` verify the state at session start, before campaign-agent spawns,
and when a session stops. They report inconsistencies but never modify or repair files.

## Open a frozen version

Opening a version is a human decision. This layer does not require a particular maturation loop,
memory size, or event sequence before that decision.

1. After a human authorizes changes, set only `mode` from `frozen` to `mutable`. Keep the current
   `release_tag`; it remains the informational baseline for the work being opened.
2. Commit that state-only change with the reason for opening the version.
3. Make the approved changes. The hooks allow protected files to change while mutable, but block
   production Analyst and Reviewer runs until a new version is frozen.

Do not create the next version entry or clear the previous tag merely to open the working period.
Those changes belong to promotion, after the release scope is known.

## Change protected code

1. If the current version is frozen, open it using the procedure above.
2. Change and test the protected code in ordinary commits.
3. Once the release scope is known, add the new version at the front of `versions`, with its planned
   tag, previous tag in `based_on`, change summaries, and known limits.
4. Commit the completed protected code and create the release tag on that commit.
5. Set `mode` to `frozen`, commit the state-only activation, and run
   `python3 version/check_version.py verify`.

Never move an existing tag. If a candidate is rejected, create a new candidate tag after fixing and
retesting it. Python and package versions are run provenance rather than freeze constraints.
