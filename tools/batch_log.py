#!/usr/bin/env python3
"""Append-only ledger for a batch de-averaging run.

One row per structure, written the moment that structure finishes (or fails), so a
batch that dies halfway still leaves a truthful record of what was done. Writes
both a machine-readable `batch.jsonl` and a human-readable `batch.log`.

Usage:
  # after each structure -- ALWAYS call this, success or failure
  python tools/batch_log.py <logdir> --file A.cif --status ok \
         --rundir runs/A --class A+B --confidence high \
         --gates "charge=pass,fidelity=pass,connectivity=clean,sqs=pass,hull=derived" --verdict confirm

  python tools/batch_log.py <logdir> --file B.cif --status error \
         --stage engine --error "relaxation failed: <msg>"

  python tools/batch_log.py <logdir> --file C.cif --status declined \
         --stage engine --error "no reliable MAR: <what the evidence cannot decide>"

  python tools/batch_log.py <logdir> --summary        # table + counts + rerun list
  python tools/batch_log.py <logdir> --done           # newline-separated FINISHED files -- last row
                                                      # `ok`, `declined` or `skipped`. An `error`
                                                      # entry is left out so a resume runs it again.

`--status`: ok | declined | error | skipped
  ok        the pipeline RAN and produced a MAR. A failing gate is still `ok` -- record it in --gates.
  declined  it ran to a CONCLUSION and the conclusion is that no reliable MAR can be produced from
            this evidence. Give --stage and, in --error, what a human or the literature has to
            supply. This is the second of the two ways an entry legitimately ends, and it is FINISHED:
            re-running reproduces the same refusal, so a resume must not offer it again. Distinct
            from `error` (something broke -- retry) and from `skipped` (never processed). One entry
            found this hole: the engine ran three times, the analyst proved geometrically that no
            decoration exists at any cutoff, and the row went in as `error` -- so the driver's retry
            loop re-ran it to the same refusal, for hours, and could not terminate.
  error     a stage could not complete. Give --stage and the real --error text.
  skipped   deliberately not processed. Finished; a resume does not come back to it.
`--stage` : evidence | engine | connectivity | record | review
`--round` : which analyst->reviewer pass this verdict came from. **There is ONE round** (`AGENTS.md`
  §3): a `revise` verdict is logged as `revise` and the entry is DONE -- the analyst is not re-run to
  answer the reviewer, so `--round 1` is the normal value. That keeps the revise rate measurable
  across a campaign, which is the number that says whether the analyst layer or the review layer
  needs work. The field stays in the schema so that a re-run pass the USER asked for can still be
  told apart from the first.
"""
import argparse
import json
import os
import sys
from datetime import datetime

FIELDS = ('ts', 'file', 'status', 'stage', 'rundir', 'klass', 'confidence',
          'gates', 'verdict', 'round', 'escalate', 'error')


def _paths(logdir):
    os.makedirs(logdir, exist_ok=True)
    return os.path.join(logdir, 'batch.jsonl'), os.path.join(logdir, 'batch.log')


def read_rows(logdir):
    jsonl, _ = _paths(logdir)
    if not os.path.exists(jsonl):
        return []
    rows = []
    with open(jsonl, encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if line:
                try: rows.append(json.loads(line))
                except json.JSONDecodeError: pass      # a torn write must not kill the summary
    return rows


def append(logdir, row):
    jsonl, log = _paths(logdir)
    with open(jsonl, 'a', encoding='utf-8') as fh:
        fh.write(json.dumps(row, ensure_ascii=False) + '\n')
    mark = {'ok': 'OK   ', 'declined': 'DECL ', 'error': 'ERROR',
            'skipped': 'SKIP '}.get(row['status'], '?    ')
    bits = [f"{row['ts']}  {mark}  {row['file']}"]
    if row.get('klass'):      bits.append(f"class={row['klass']}")
    if row.get('confidence'): bits.append(f"conf={row['confidence']}")
    if row.get('gates'):      bits.append(f"gates={row['gates']}")
    if row.get('verdict'):    bits.append(f"review={row['verdict']}"
                                          + (f"(r{row['round']})" if row.get('round') else ''))
    if row.get('escalate'):   bits.append(f"ESCALATE={row['escalate']}")
    if row.get('stage'):      bits.append(f"stage={row['stage']}")
    if row.get('error'):      bits.append(f"error={row['error']}")
    with open(log, 'a', encoding='utf-8') as fh:
        fh.write('  '.join(bits) + '\n')


def summary(logdir):
    rows = read_rows(logdir)
    if not rows:
        print(f'(no batch rows in {logdir})')
        return 0
    # A file can hold SEVERAL rows (an `error` row and a later retry, or a re-run pass the user
    # asked for), so count STRUCTURES, not rows: its last row is its outcome. Counting rows would
    # inflate `ok` and misreport the batch.
    last = {}
    for r in rows:
        last[r.get('file', '')] = r
    counts = {}
    for r in last.values():
        counts[r.get('status', '?')] = counts.get(r.get('status', '?'), 0) + 1
    w = max(len(r.get('file', '')) for r in rows)
    print(f"{'file'.ljust(w)}  {'status':8} {'class':14} {'conf':9} {'review':9} note")
    for r in rows:
        note = r.get('error') or r.get('escalate') or r.get('gates') or ''
        print(f"{r.get('file','').ljust(w)}  {r.get('status',''):8} "
              f"{(r.get('klass') or '-'):14} {(r.get('confidence') or '-'):9} "
              f"{(r.get('verdict') or '-'):9} {note}")
    print('\n' + '  '.join(f'{k}={v}' for k, v in sorted(counts.items()))
          + f'  total={len(last)} structures ({len(rows)} rows)')
    errs = [r for r in last.values() if r.get('status') == 'error']
    if errs:
        print(f'\n{len(errs)} FAILED — rerun these:')
        for r in errs:
            print(f"  {r['file']}  (failed at {r.get('stage') or '?'}: {r.get('error')})")
    # Kept OUT of the rerun list on purpose: a declined entry re-runs to the same refusal. It is
    # listed separately because it still needs a person -- just not another pass of the pipeline.
    decl = [r for r in last.values() if r.get('status') == 'declined']
    if decl:
        print(f'\n{len(decl)} DECLINED — not a rerun; these need a human or the literature:')
        for r in decl:
            print(f"  {r['file']}  ({r.get('error')})")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logdir', help='batch log dir (e.g. runs/_batch)')
    ap.add_argument('--summary', action='store_true', help='print the table and exit')
    ap.add_argument('--done', action='store_true',
                    help='print files that are FINISHED (last row `ok`, `declined` or `skipped`) -- '
                         'for resuming. An `error` entry is NOT printed: a resume should run it again')
    ap.add_argument('--file', help='the structure this row is about')
    ap.add_argument('--status', choices=('ok', 'declined', 'error', 'skipped'))
    ap.add_argument('--stage', default=None, help='evidence|engine|connectivity|record|review')
    ap.add_argument('--rundir', default=None)
    ap.add_argument('--class', dest='klass', default=None)
    ap.add_argument('--confidence', default=None)
    ap.add_argument('--gates', default=None)
    ap.add_argument('--verdict', default=None)
    ap.add_argument('--round', type=int, default=None,
                    help='analyst->reviewer pass this verdict came from (1-based; there is ONE round '
                         '-- see AGENTS.md 3 -- so this is 1 unless the user asked for a re-run)')
    ap.add_argument('--escalate', default=None)
    ap.add_argument('--error', default=None)
    args = ap.parse_args()

    if args.summary:
        return summary(args.logdir)
    if args.done:
        # A file can hold SEVERAL rows (a retry, or a re-run pass), so its LAST row is its outcome --
        # first-row-wins would keep re-offering an entry that errored once and then succeeded.
        last = {}
        for r in read_rows(args.logdir):
            f = r.get('file', '')
            if f:
                last[f] = r.get('status')
        # `error` is deliberately NOT finished: a stage could not complete, no record exists, and the
        # whole point of a resume is to run it again. `skipped` IS finished -- that is the status that
        # says "do not come back to this one". Treating `error` as done drops an entry from the
        # campaign silently, and the final count then reports it as a success.
        # `declined` is finished for the opposite reason: the pipeline reached a conclusion and the
        # conclusion is a refusal. Re-running reproduces it, so offering it to a resume is a loop
        # that cannot end -- which is exactly what it did before this status existed.
        for f, status in last.items():
            if status in ('ok', 'declined', 'skipped'):
                print(f)
        return 0
    if not args.file or not args.status:
        ap.error('--file and --status are required unless --summary/--done')

    row = {'ts': datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
    for k in FIELDS:
        if k != 'ts':
            row[k] = getattr(args, k)
    append(args.logdir, row)
    print(f"logged {args.status}: {args.file}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
