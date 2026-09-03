"""Build a self-contained, growing web gallery from de-averaging results.

TWO record schemas are rendered, and they answer different questions:

  * `deaverage_output.json` (a `MARRecord`) -- the deterministic engine's own result: the energy
    distribution, the representative, the tiers. Written by `deaverage()` / `demars run`.
  * `record.json` with `schema_version: "mar-1.0"` -- the stage-6 record of a judged entry: the
    mechanism class, the five gates with their states, the review verdict, the hull. This is what a
    CAMPAIGN produces, and the index page over a directory of them is the audit view.

Point `build_gallery` at a directory of such result folders (one per structure) and it emits a
static site:

  index.html            card grid (one card per result)
  r-<slug>.html         per-result DETAIL page: energy histogram of the ensemble,
                        the full record, and a "View 3-D ensemble" button
  view3d.html + js/     the vendored 3-D viewer (offline; bundles 3Dmol.js)
  files/<staged>.xyz    each result's ensemble, staged for the viewer

Detail pages live at the site ROOT (not a subdir) so their links to `files/…`
and `view3d.html` carry no `..` — which the viewer's `?f=` guard rejects.

Re-run any time to refresh — new result folders just show up. Fully offline;
works under a plain `python -m http.server`.
"""
import glob
import hashlib
import html
import json
import os
import re
import shutil
import warnings
from urllib.parse import quote

__all__ = ['build_gallery', 'serve']

_VIEWER_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'viewer')
_STRUCT_PREF = ('ensemble.xyz', 'representative.cif', 'representative.vasp')


# ---------------------------------------------------------------- discovery ----
def _find_results(results_dir):
    """Yield (name, result_dir, record) for every folder holding a de-average output —
    the results_dir itself and/or its immediate subdirectories.

    `deaverage_output.json` is the current name; `record.json` is the pre-2026-08-13 name for the
    SAME schema, and is also the name of the stage-6 mar-1.0 record. Both are rendered now, by
    different builders, and the record itself says which: `schema_version: "mar-1.0"`. Anything that
    is neither is skipped loudly rather than drawn as a page of empty fields.
    """
    def _load(d):
        for base in ('deaverage_output.json', 'record.json'):
            p = os.path.join(d, base)
            if not os.path.isfile(p):
                continue
            try:
                with open(p, encoding='utf-8') as fh:
                    rec = json.load(fh)
            except Exception:
                continue
            if not isinstance(rec, dict):
                continue
            if rec.get('schema_version') == 'mar-1.0' or 'gates' in rec:
                return rec
            if 'distribution' in rec or 'status' in rec:
                return rec
            warnings.warn(f'{p} is neither a de-average output nor a mar-1.0 record (keys: '
                          f'{", ".join(sorted(rec)[:6])}); skipping.')
        return None

    root_rec = _load(results_dir)
    if root_rec is not None:
        yield os.path.basename(os.path.abspath(results_dir)), results_dir, root_rec
    for name in sorted(os.listdir(results_dir)):
        sub = os.path.join(results_dir, name)
        if os.path.isdir(sub):
            rec = _load(sub)
            if rec is not None:
                yield name, sub, rec


def _structure_file(result_dir):
    for fn in _STRUCT_PREF:
        p = os.path.join(result_dir, fn)
        if os.path.isfile(p):
            return p
    return None


def _mar10_structure(result_dir, rec):
    """A mar-1.0 record does NOT keep its structures beside itself -- they live in the engine's own
    `_work/<tag>/`, and `record.json` is the single authority on which one is the MAR. So resolve the
    path the record gives (relative to the record, as `portable_path` stores it) instead of guessing
    by filename, which is what `never resolve that by globbing` is about."""
    ens = (rec.get('ensemble') or {}).get('file')
    rep = rec.get('representative') or {}
    for cand in (ens, rep.get('file_final'), rep.get('file')):
        if not cand:
            continue
        for base in (result_dir, os.getcwd()):
            p = cand if os.path.isabs(cand) else os.path.join(base, cand)
            if os.path.isfile(p):
                return p
    return None


def _slug(name, seen):
    s = re.sub(r'[^A-Za-z0-9._-]+', '-', name).strip('-') or 'result'
    base, i = s, 1
    while s in seen:
        s = f'{base}-{i}'
        i += 1
    seen.add(s)
    return s


# ---------------------------------------------------------------- rendering ----
def _stat_chip(label, value):
    if value is None:
        return ''
    return f'<span class="chip"><span class="k">{html.escape(label)}</span>{html.escape(str(value))}</span>'


def _is_mar10(rec):
    return rec.get('schema_version') == 'mar-1.0' or 'gates' in rec


# A gate's `pass` is None for `vacuous` / `ambiguous` / `not_run`, and that is NOT a pass -- it is
# "nothing was examined" or "could not run". Collapsing the five gates to green/red would erase the
# only distinction the state vocabulary exists to make, so unchecked gets its own colour and its own
# word. `derived` + `pass: null` is the hull gate reporting a number without a configured threshold.
_GATE_CLS = {True: 'ok', False: 'err', None: 'unchecked'}


def _gate_chip(key, g):
    # A gate the record does not carry at all -- an older record written before that gate existed --
    # must not render as nothing, which a reader cannot tell from a pass. Say `absent`.
    if not isinstance(g, dict):
        return (f'<span class="gate unchecked" title="this record carries no {html.escape(key)} '
                f'gate — it predates it, or the stage that fills it did not run">'
                f'<span class="k">{html.escape(key)}</span>absent</span>')
    state, ok = g.get('state'), g.get('pass')
    cls = _GATE_CLS[ok if ok in (True, False) else None]
    if ok is True:
        word = 'pass'
    elif ok is False:
        word = 'FAIL'
    elif state == 'derived':
        word = 'reported'                  # computed, no threshold to judge against
    else:
        word = state or 'not_run'
    tip = html.escape(str(g.get('basis') or state or ''), quote=True)
    return (f'<span class="gate {cls}" title="{tip}">'
            f'<span class="k">{html.escape(key)}</span>{html.escape(word)}</span>')


def _gate_chips(rec):
    gates = rec.get('gates') or {}
    return ''.join(_gate_chip(k, gates.get(k)) for k in
                   ('charge', 'fidelity', 'connectivity', 'sqs', 'hull'))


def _mar10_flags(rec):
    """-> (list of audit flags, worst severity). What an index page must surface without a click."""
    gates = rec.get('gates') or {}
    flags, worst = [], 'ok'
    for k in ('charge', 'fidelity', 'connectivity', 'sqs', 'hull'):
        g = gates.get(k)
        if not isinstance(g, dict):
            flags.append(f'{k} absent')
            worst = 'err' if worst == 'err' else 'unchecked'
            continue
        if g.get('pass') is False:
            flags.append(f'{k} FAILS'); worst = 'err'
        elif g.get('pass') is None and g.get('state') != 'derived':
            flags.append(f'{k} {g.get("state") or "not_run"}')
            worst = 'err' if worst == 'err' else 'unchecked'
    if rec.get('escalate'):
        flags.append('ESCALATED'); worst = 'err'
    rv = rec.get('review')
    if rv is None:
        flags.append('UNREVIEWED')
        worst = 'err' if worst == 'err' else 'unchecked'
    elif rv.get('unresolved_blocking'):
        flags.append(f'blocking objection ({rv.get("final_verdict")})'); worst = 'err'
    return flags, worst


def _mar10_card(name, rec, detail_href, has_view):
    j = rec.get('mechanism') or {}          # mar-1.0 keeps the judgment here, not under `judgment`
    rep = rec.get('representative') or {}
    prov = rec.get('provenance') or {}
    title = prov.get('formula_sum') or rep.get('composition') or name
    # A class is meant to be `A` / `B+C` / `D+A [screening]`, but an analyst can write a sentence
    # into it. The badge truncates and keeps the whole string in the tooltip; the detail page has it
    # in full next to `class_label`.
    cls_full = str(j.get('class') or '?')
    cls = cls_full if len(cls_full) <= 16 else cls_full[:15] + '…'
    flags, worst = _mar10_flags(rec)
    rv = rec.get('review') or {}
    verdict = rv.get('final_verdict') or ('unreviewed' if rec.get('review') is None else '?')

    hull = rec.get('hull') or {}
    e_hull = hull.get('E_above_hull_eV_per_atom')
    dd = rec.get('disorder_descriptor') or {}
    chips = ''.join([
        _stat_chip('atoms', rep.get('n_atoms')),
        _stat_chip('E_hull', f'{e_hull} eV/at') if e_hull is not None else '',
        _stat_chip('O/S/V/P', '/'.join(dd.get('disorder_set') or [])) if dd.get('available') else '',
        _stat_chip('conf', j.get('confidence')),
        _stat_chip('shipped', rep.get('source')),
    ])
    search = html.escape(' '.join(str(x) for x in (title, name, cls_full, verdict, *flags)).lower(),
                         quote=True)
    flag_html = (f'<div class="flags {worst}">{html.escape(" · ".join(flags))}</div>'
                 if flags else '<div class="flags ok">all gates derived · reviewed</div>')
    foot = 'record &amp; 3-D →' if has_view else 'record →'
    return (f'<a class="card" data-status="{html.escape(worst)}" data-search="{search}" '
            f'href="{detail_href}">'
            f'<h3>{html.escape(str(title))}</h3>'
            f'<div class="sub">{html.escape(name)}</div>'
            f'<div class="badges"><span class="badge cls" title="{html.escape(cls_full, quote=True)}">'
            f'{html.escape(cls)}</span>'
            f'<span class="badge {"ok" if verdict == "confirm" else "muted"}">'
            f'{html.escape(str(verdict))}</span></div>'
            f'<div class="gates">{_gate_chips(rec)}</div>'
            f'{flag_html}'
            f'<div class="chips">{chips}</div>'
            f'<div class="view">{foot}</div></a>')


def _mar10_detail(name, rec, viewer_href, title):
    j = rec.get('mechanism') or {}          # see _mar10_card
    rep = rec.get('representative') or {}
    prov = rec.get('provenance') or {}
    ens = rec.get('ensemble') or {}
    formula = prov.get('formula_sum') or rep.get('composition') or name
    rows = []

    def row(k, v):
        if v not in (None, '', {}, []):
            rows.append(f'<tr><td class="k">{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>')

    row('mechanism class', ' — '.join(str(x) for x in (j.get('class'), j.get('class_label')) if x))
    row('interpretation', j.get('interpretation'))
    row('disorder pattern', j.get('disorder_pattern'))
    row('confidence', j.get('confidence'))
    row('escalated', rec.get('escalate'))
    row('CE+MC warranted', (rec.get('ce_mc') or {}).get('warranted'))
    row('decision trace', ' → '.join(rec.get('decision_trace') or []))
    row('formula (CIF)', prov.get('formula_sum'))
    row('name / structure type', ' · '.join(
        str(x) for x in (prov.get('chemical_name'), prov.get('structure_type')) if x))
    row('representative', f"{rep.get('source')} · {rep.get('config_label') or ''} · "
                          f"{rep.get('composition') or ''} · n_atoms={rep.get('n_atoms')}")
    row('representative file', rep.get('file_final') or rep.get('file'))
    row('E (nano / final)', f"{rep.get('E_nano_eV_per_atom')} / {rep.get('E_final_eV_per_atom')} eV/at")
    if rep.get('E_final_unavailable_reason'):
        row('no final tier', rep['E_final_unavailable_reason'])
    row('ensemble', f"{ens.get('n_frames')} frames · spread {ens.get('spread_meV')} meV/at · "
                    f"std {ens.get('std_meV')} meV/at")
    row('cell scatter', ens.get('cell_scatter'))
    dd = rec.get('disorder_descriptor') or {}
    row('disorder descriptor', ('/'.join(dd.get('disorder_set') or []) +
                                (' · no_full_backbone' if dd.get('no_full_backbone') else ''))
        if dd.get('available') else f"unavailable — {dd.get('reason')}")
    h = rec.get('hull') or {}
    if h:
        row('hull', (f"{h.get('state')} · E_above_hull={h.get('E_above_hull_eV_per_atom')} eV/at · "
                     f"{h.get('mode')} · corrections={h.get('corrections')} · "
                     f"decomposes to {h.get('decomposition')}")
            if h.get('state') == 'derived' else f"{h.get('state')} — {h.get('reason')}")
    row('ordered sibling', rec.get('ordered_sibling'))
    rv = rec.get('review')
    row('review', 'UNREVIEWED — a null review is not reviewed-and-clean' if rv is None else
                  f"{rv.get('final_verdict')} after {rv.get('n_rounds')} round(s)"
                  + (f" · UNRESOLVED: {rv.get('unresolved_blocking')}"
                     if rv.get('unresolved_blocking') else ''))

    grows = []
    for k in ('charge', 'fidelity', 'connectivity', 'sqs', 'hull'):
        g = (rec.get('gates') or {}).get(k)
        if not isinstance(g, dict):
            grows.append(f'<tr><td class="k">{html.escape(k)}</td><td class="unchecked">absent</td>'
                         f'<td>this record carries no such gate</td></tr>')
            continue
        ok = g.get('pass')
        word = {True: 'pass', False: 'FAIL'}.get(ok, f"{g.get('state')} (pass: null)")
        grows.append(f'<tr><td class="k">{html.escape(k)}</td>'
                     f'<td class="{_GATE_CLS[ok if ok in (True, False) else None]}">'
                     f'{html.escape(word)}</td>'
                     f'<td>{html.escape(str(g.get("basis") or ""))}</td></tr>')

    raw = html.escape(json.dumps(rec, indent=1))
    btn = (f'<a class="btn" href="{viewer_href}" target="_blank" rel="noopener">View 3-D ensemble →</a>'
           if viewer_href else '<span class="muted">no structure file for this result</span>')
    return (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(str(formula))} — {html.escape(title)}</title>'
            f'<style>{_CSS}</style></head><body><div class="detail">'
            f'<a class="back" href="index.html">← all results</a>'
            f'<h1>{html.escape(str(formula))}</h1><div class="sub">{html.escape(name)} · mar-1.0</div>'
            f'<div style="margin:16px 0">{btn}</div>'
            f'<div class="sec">gates</div><table class="gt">{"".join(grows)}</table>'
            f'<div class="sec">record</div><table>{"".join(rows)}</table>'
            f'<details><summary>raw record.json</summary><pre class="raw">{raw}</pre></details>'
            f'</div></body></html>')


def _card(name, rec, detail_href, has_view):
    d = rec.get('distribution') or {}
    rep = d.get('representative') or {}
    title = rec.get('formula') or name
    status = rec.get('status', '?')
    calc = rec.get('calculator', '')
    chips = ''.join([
        _stat_chip('E₀', f"{d['ground_E_per_atom']} eV/at") if 'ground_E_per_atom' in d else '',
        _stat_chip('spread', f"{d['spread_meV']} meV/at") if 'spread_meV' in d else '',
        _stat_chip('configs', f"{d.get('n_relaxed')}/{d.get('n_total')}") if 'n_relaxed' in d else '',
        _stat_chip('atoms', rep.get('n_atoms') or rec.get('n_atoms_template')),
    ])
    badge_cls = {'de-averaged': 'ok', 'no-disorder': 'muted', 'error': 'err'}.get(status, 'muted')
    els = ' '.join((rec.get('oxidation_states') or {}).keys())
    search = html.escape(' '.join(str(x) for x in (title, name, els) if x).lower(), quote=True)
    foot = 'details &amp; 3-D →' if has_view else 'details →'
    return (f'<a class="card" data-status="{html.escape(str(status))}" data-search="{search}" href="{detail_href}">'
            f'<h3>{html.escape(str(title))}</h3>'
            f'<div class="sub">{html.escape(name)}</div>'
            f'<div class="badges"><span class="badge {badge_cls}">{html.escape(status)}</span>'
            f'{f"<span class=badge_calc>{html.escape(calc)}</span>" if calc else ""}</div>'
            f'<div class="chips">{chips}</div>'
            f'<div class="view">{foot}</div></a>')


def _hist_svg(energies):
    """Inline SVG histogram of ensemble config energies (Δ above the ground
    state, meV/atom). Self-contained; styled via CSS so it is theme-aware."""
    if not energies or len(energies) < 2:
        return '<div class="muted">single config — no energy distribution to plot.</div>'
    emin = min(energies)
    deltas = sorted((e - emin) * 1000.0 for e in energies)      # meV/atom
    real_dmax = deltas[-1]
    dmax = real_dmax or 1.0
    n = len(deltas)
    nbins = min(12, max(4, n // 2))
    counts = [0] * nbins
    for dlt in deltas:
        b = min(nbins - 1, int(dlt / dmax * nbins)) if real_dmax > 0 else 0
        counts[b] += 1
    cmax = max(counts) or 1
    W, H, PL, PB, PT, PR = 540, 190, 36, 30, 12, 10
    iw, ih = W - PL - PR, H - PT - PB
    bw = iw / nbins
    bars = []
    for i, c in enumerate(counts):
        bh = (c / cmax) * ih
        bars.append(f'<rect class="bar" x="{PL + i * bw:.1f}" y="{PT + ih - bh:.1f}" '
                    f'width="{max(1, bw - 2):.1f}" height="{bh:.1f}"><title>{c} config(s)</title></rect>')
    axis = (f'<line class="axl" x1="{PL}" y1="{PT}" x2="{PL}" y2="{PT + ih}"/>'
            f'<line class="axl" x1="{PL}" y1="{PT + ih}" x2="{PL + iw}" y2="{PT + ih}"/>')
    labels = (f'<text class="axt" x="{PL}" y="{H - 10}" text-anchor="start">0</text>'
              f'<text class="axt" x="{PL + iw / 2:.0f}" y="{H - 10}" text-anchor="middle">Δ above ground state (meV/atom)</text>'
              f'<text class="axt" x="{PL + iw}" y="{H - 10}" text-anchor="end">{real_dmax:.0f}</text>'
              f'<text class="axt" x="{PL - 6}" y="{PT + 9}" text-anchor="end">{cmax}</text>'
              f'<text class="axt" x="{PL - 6}" y="{PT + ih}" text-anchor="end">0</text>')
    return (f'<svg viewBox="0 0 {W} {H}" class="hist" role="img" '
            f'aria-label="ensemble energy histogram">{axis}{"".join(bars)}{labels}</svg>')


def _detail_page(name, rec, viewer_href, title):
    d = rec.get('distribution') or {}
    rep = d.get('representative') or {}
    formula = rec.get('formula') or name
    rows = []

    def row(k, v):
        if v not in (None, '', {}, []):
            rows.append(f'<tr><td class="k">{html.escape(k)}</td><td>{html.escape(str(v))}</td></tr>')

    row('source', rec.get('source'))
    row('status', rec.get('status'))
    row('calculator', rec.get('calculator'))
    row('formula (representative)', rec.get('formula'))
    row('supercell', rec.get('supercell'))
    row('template atoms', rec.get('n_atoms_template'))
    row('oxidation states', rec.get('oxidation_states'))
    if d:
        row('ground-state E', f"{d.get('ground_E_per_atom')} eV/atom")
        row('ensemble spread', f"{d.get('spread_meV')} meV/atom")
        row('ensemble std', f"{d.get('std_meV')} meV/atom")
        row('configs relaxed', f"{d.get('n_relaxed')}/{d.get('n_total')}")
    if rep:
        row('representative', f"{rep.get('label')} · E={rep.get('E_per_atom')} eV/at · "
                              f"q={rep.get('charge')} · min_dist={rep.get('min_dist_A')} Å · "
                              f"n_atoms={rep.get('n_atoms')}")
    if rec.get('final_MAR'):
        fm = rec['final_MAR']
        # "requested but could not run" must not render as `E=None`, which reads like a missing
        # number rather than a tier that was asked for and refused (api._final_unavailable).
        row('final (omni-mpa)',
            f"NOT AVAILABLE — {fm.get('reason')}" if fm.get('available') is False
            else f"E={fm.get('E_final_per_atom')} eV/at · modal={fm.get('modal')}")

    hist = _hist_svg(d.get('energies_sorted') or [])
    raw = html.escape(json.dumps(rec, indent=1))
    btn = (f'<a class="btn" href="{viewer_href}" target="_blank" rel="noopener">View 3-D ensemble →</a>'
           if viewer_href else '<span class="muted">no structure file for this result</span>')
    return (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(str(formula))} — {html.escape(title)}</title>'
            f'<style>{_CSS}</style></head><body><div class="detail">'
            f'<a class="back" href="index.html">← all results</a>'
            f'<h1>{html.escape(str(formula))}</h1><div class="sub">{html.escape(name)}</div>'
            f'<div style="margin:16px 0">{btn}</div>'
            f'<div class="sec">ensemble energy distribution</div>{hist}'
            f'<div class="sec">record</div><table>{"".join(rows)}</table>'
            f'<details><summary>raw record.json</summary><pre class="raw">{raw}</pre></details>'
            f'</div></body></html>')


_CSS = """
:root{--bg:#f6f7f9;--card:#fff;--fg:#1c2024;--sub:#6b7280;--line:#e5e7eb;--accent:#2f6feb;
 --okc:#1f7a4d;--okbg:#e6f4ec;--errc:#a3341f;--errbg:#fbe9e5;--unkc:#8a6a1f;--unkbg:#f7efdc}
@media (prefers-color-scheme:dark){:root{
 --okc:#6dd39b;--okbg:#14291f;--errc:#f0a08c;--errbg:#2e1a16;--unkc:#dcb765;--unkbg:#2b2415}}
.gates{display:flex;flex-wrap:wrap;gap:4px;margin:8px 0 6px}
.gate{font:11px/1.4 ui-monospace,SFMono-Regular,Menlo,monospace;padding:2px 6px;border-radius:4px;
 white-space:nowrap}
.gate .k{opacity:.7;margin-right:5px}
.gate.ok{background:var(--okbg);color:var(--okc)}
.gate.err{background:var(--errbg);color:var(--errc);font-weight:600}
.gate.unchecked{background:var(--unkbg);color:var(--unkc)}
.flags{font-size:11.5px;margin:2px 0 6px}
.flags.ok{color:var(--okc)}
.flags.err{color:var(--errc);font-weight:600}
.flags.unchecked{color:var(--unkc)}
.badge.cls{background:var(--accent);color:#fff}
table.gt td.ok{color:var(--okc)}
table.gt td.err{color:var(--errc);font-weight:600}
table.gt td.unchecked{color:var(--unkc)}
@media (prefers-color-scheme:dark){:root{--bg:#0f1115;--card:#171a20;--fg:#e6e8eb;--sub:#9aa2ad;--line:#262b33;--accent:#5b8cff}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--fg);
 font:15px/1.5 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif}
header{padding:28px 24px 8px;max-width:1200px;margin:0 auto}
header h1{margin:0 0 4px;font-size:22px}header .meta{color:var(--sub);font-size:13px}
.grid{max-width:1200px;margin:0 auto;padding:16px 24px 48px;display:grid;gap:16px;
 grid-template-columns:repeat(auto-fill,minmax(260px,1fr))}
.card{display:block;background:var(--card);border:1px solid var(--line);border-radius:12px;
 padding:16px 16px 12px;text-decoration:none;color:inherit;transition:.15s}
.card:hover{border-color:var(--accent);transform:translateY(-2px);box-shadow:0 6px 20px rgba(0,0,0,.08)}
.card h3{margin:0 0 2px;font-size:18px}
.sub{color:var(--sub);font-size:12px;word-break:break-all;margin-bottom:10px}
.badges{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:10px}
.badge{font-size:11px;padding:2px 8px;border-radius:999px;border:1px solid var(--line)}
.badge.ok{background:rgba(47,111,235,.12);border-color:transparent;color:var(--accent)}
.badge.err{background:rgba(220,60,60,.14);border-color:transparent;color:#e05555}
.badge_calc{font-size:11px;padding:2px 8px;border-radius:999px;border:1px solid var(--line);color:var(--sub)}
.chips{display:flex;gap:6px;flex-wrap:wrap}
.chip{font-size:12px;color:var(--fg);background:var(--bg);border:1px solid var(--line);border-radius:8px;padding:3px 8px}
.chip .k{color:var(--sub);margin-right:5px}
.view{margin-top:12px;font-size:12px;color:var(--accent)}
.empty{max-width:1200px;margin:40px auto;padding:0 24px;color:var(--sub)}
.muted{color:var(--sub)}
.toolbar{max-width:1200px;margin:0 auto;padding:12px 24px 0;display:flex;gap:10px;align-items:center;flex-wrap:wrap}
.toolbar input[type=search]{flex:1;min-width:220px;padding:9px 12px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);font-size:14px}
.toolbar select{padding:9px 10px;border:1px solid var(--line);border-radius:8px;background:var(--card);color:var(--fg);font-size:14px}
.tcount{color:var(--sub);font-size:13px;white-space:nowrap}
.detail{max-width:900px;margin:0 auto;padding:22px 24px 56px}
.detail h1{margin:6px 0 0;font-size:24px}
.sec{margin:26px 0 4px;font-weight:600;font-size:14px}
.detail table{border-collapse:collapse;width:100%;font-size:13px;margin:6px 0}
.detail td{border-bottom:1px solid var(--line);padding:6px 8px;vertical-align:top}
.detail td.k{color:var(--sub);width:210px;white-space:nowrap}
.btn{display:inline-block;background:var(--accent);color:#fff;padding:9px 15px;border-radius:8px;
 text-decoration:none;font-size:14px;font-weight:500}
.back{color:var(--sub);text-decoration:none;font-size:13px}
.hist{width:100%;max-width:560px;height:auto;margin-top:6px}
.hist .bar{fill:var(--accent)}
.hist .axl{stroke:var(--line);stroke-width:1}
.hist .axt{fill:var(--sub);font-size:11px}
details{margin-top:20px}summary{cursor:pointer;color:var(--sub);font-size:13px}
pre.raw{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;
 overflow:auto;font-size:12px;max-height:420px}
"""

_FILTER_JS = """
<script>
(function(){
  var q=document.getElementById('q'), st=document.getElementById('st'),
      count=document.getElementById('count'), empty=document.getElementById('nomatch'),
      cards=[].slice.call(document.querySelectorAll('.card'));
  function apply(){
    var term=(q.value||'').trim().toLowerCase(), s=st.value, shown=0;
    for(var i=0;i<cards.length;i++){
      var c=cards[i],
          okText=!term||c.getAttribute('data-search').indexOf(term)>=0,
          okStat=(s==='all')||(c.getAttribute('data-status')===s),
          vis=okText&&okStat;
      c.style.display=vis?'':'none';
      if(vis)shown++;
    }
    count.textContent=shown;
    empty.style.display=shown?'none':'';
  }
  q.addEventListener('input',apply);
  st.addEventListener('change',apply);
  apply();
})();
</script>
"""


# ---------------------------------------------------------------- build ----
def build_gallery(results_dir, out_dir=None, title='DeMARS results', generated=None):
    """Scan `results_dir` for de-averaging result folders and (re)build a static
    gallery at `out_dir` (default `<results_dir>/_gallery`). Returns out_dir."""
    results_dir = os.path.abspath(results_dir)
    out_dir = os.path.abspath(out_dir or os.path.join(results_dir, '_gallery'))
    os.makedirs(os.path.join(out_dir, 'files'), exist_ok=True)

    # vendored viewer (copy if missing/stale)
    shutil.copy2(os.path.join(_VIEWER_SRC, 'view3d.html'), os.path.join(out_dir, 'view3d.html'))
    os.makedirs(os.path.join(out_dir, 'js'), exist_ok=True)
    shutil.copy2(os.path.join(_VIEWER_SRC, 'js', '3Dmol-min.js'), os.path.join(out_dir, 'js', '3Dmol-min.js'))
    # drop stale detail pages from a previous build so removed results don't linger
    for old in glob.glob(os.path.join(out_dir, 'r-*.html')):
        os.remove(old)

    cards, seen, present, n = [], set(), set(), 0
    for name, rdir, rec in _find_results(results_dir):
        n += 1
        mar10 = _is_mar10(rec)
        # the index filter groups by what a reader is scanning FOR: a de-average output by run
        # status, a judged record by whether anything about it is unchecked or failing.
        present.add(_mar10_flags(rec)[1] if mar10 else (rec.get('status') or '?'))
        src = _structure_file(rdir) or (_mar10_structure(rdir, rec) if mar10 else None)
        viewer_href = None
        if src:
            staged = hashlib.md5(name.encode()).hexdigest()[:8] + '_' + os.path.basename(src)
            dst = os.path.join(out_dir, 'files', staged)
            if (not os.path.exists(dst) or os.path.getsize(dst) != os.path.getsize(src)
                    or os.path.getmtime(dst) < os.path.getmtime(src)):
                shutil.copy2(src, dst)
            label = ((rec.get('provenance') or {}).get('formula_sum') if mar10
                     else rec.get('formula')) or name
            viewer_href = f'view3d.html?f=files/{quote(staged)}&label={quote(str(label))}'
        detail_file = f'r-{_slug(name, seen)}.html'
        with open(os.path.join(out_dir, detail_file), 'w', encoding='utf-8') as fh:
            fh.write((_mar10_detail if mar10 else _detail_page)(name, rec, viewer_href, title))
        cards.append((_mar10_card if mar10 else _card)(name, rec, detail_file, bool(viewer_href)))

    body = (f'<div class="grid">{"".join(cards)}</div>' if cards
            else '<div class="empty">No result folders found. Run '
                 '<code>demars run struct.cif -o results/&lt;name&gt;</code> first.</div>')
    gen = f' · generated {generated}' if generated else ''
    head = (f'<!doctype html><html><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1">'
            f'<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>')
    if cards:
        opts = '<option value="all">all statuses</option>' + ''.join(
            f'<option value="{html.escape(s)}">{html.escape(s)}</option>' for s in sorted(present))
        toolbar = (f'<header><h1>{html.escape(title)}</h1>'
                   f'<div class="meta">{n} result{"s" if n != 1 else ""}{gen} · '
                   f'click a card for details + the 3-D ensemble</div>'
                   f'<div class="toolbar"><input id="q" type="search" autocomplete="off" '
                   f'placeholder="filter by formula, element, or name…">'
                   f'<select id="st">{opts}</select>'
                   f'<span class="tcount"><b id="count">{n}</b> / {n} shown</span></div></header>')
        nomatch = '<div id="nomatch" class="empty" style="display:none">No results match the filter.</div>'
        page = head + toolbar + body + nomatch + _FILTER_JS + '</body></html>'
    else:
        page = head + f'<header><h1>{html.escape(title)}</h1></header>' + body + '</body></html>'
    with open(os.path.join(out_dir, 'index.html'), 'w', encoding='utf-8') as fh:
        fh.write(page)
    return out_dir


def serve(out_dir, port=8000, bind='127.0.0.1'):
    """Serve the gallery over HTTP (needed so the viewer can fetch structure files
    — a file:// page can't). Blocks until Ctrl-C."""
    import functools
    from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
    handler = functools.partial(SimpleHTTPRequestHandler, directory=os.path.abspath(out_dir))
    httpd = ThreadingHTTPServer((bind, port), handler)
    print(f'serving gallery at http://{bind}:{port}/  (Ctrl-C to stop)', flush=True)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        print('\nstopped.')
