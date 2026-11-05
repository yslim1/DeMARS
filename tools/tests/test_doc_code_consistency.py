"""Claims the docs make about the code must still be true.

Every defect this file exists for was a sentence that used to be true: `tools.md` said
"no sibling DB" after the search was wired up, `strategy.md` quoted the atom budget as 500 when the
enforced ceiling is 800, and the record kept a `disorder_descriptor` key whose module was never
ported. None of them break anything visibly -- they mislead the analyst, which at 1,000 entries is
worse, because the analyst is the layer we are trusting to judge.

Only load-bearing, checkable claims are pinned here: a numeric constant quoted in prose, a
capability the docs promise, and a capability they deny. A test that tried to parse all prose would
be noise.

No MLIP and no GPU.
"""
import os
import re

import pytest

import _icsd_env                                       # noqa: F401  MUST precede demars_core

from demars_core._engine import mar_engine as ME       # noqa: E402


ROOT = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
SKILLS = os.path.join(ROOT, '.agents', 'skills', 'mar-analyst')


def _read(*parts):
    with open(os.path.join(ROOT, *parts), encoding='utf-8') as fh:
        return fh.read()


def test_the_atom_budget_quoted_in_strategy_is_the_one_the_engine_enforces():
    """`strategy.md` drives the analyst's cell decisions, so a stale budget there produces cells
    the engine then silently cuts (defect D13, from the other end)."""
    txt = _read('.agents', 'skills', 'mar-analyst', 'strategy.md')

    quoted = {int(m) for m in re.findall(r'`?MAXAT\s*=\s*(\d+)`?', txt)}
    assert quoted == {ME.MAXAT}, f'strategy.md quotes MAXAT={quoted}, code has {ME.MAXAT}'
    assert 'MAXAT * 1.6' in txt, (
        'the doc quotes the nominal budget but not the ceiling the engine actually enforces; '
        'a reader plans a cell against the wrong number')
    assert str(int(ME.MAXAT * 1.6)) in txt, 'the enforced ceiling is not stated anywhere'


def test_the_docs_do_not_deny_a_capability_the_package_has():
    """`tools.md` told the analyst to leave `sibling_comparison` null "— no sibling DB" while the
    same file documented the sibling ΔE workflow twenty lines earlier. AGENTS.md's whole point is
    that *unchecked* and *checked-and-absent* are different answers; a doc that says the search
    cannot run makes every analyst report the first when the truth is the second."""
    txt = _read('.agents', 'skills', 'mar-analyst', 'tools.md')
    assert 'ordered_sibling_ids' in txt, 'fixture check: this file should document the search'

    # "no sibling DB" is ALSO the sentinel the evidence bundle emits when the search really was not
    # run, and tools.md has to document that string -- AGENTS.md's unchecked-vs-absent distinction
    # lives in it. So flag the phrase only where it is the doc speaking, not quoting.
    for i, line in enumerate(txt.splitlines(), 1):
        if 'sibling db' not in line.lower():
            continue
        assert '"none (' in line or '`none (' in line.lower(), (
            f'tools.md:{i} tells the analyst the sibling DB is absent while the search is wired '
            f'up -- every entry would then be reported as UNCHECKED: {line.strip()!r}')


def test_a_capability_the_docs_deny_really_is_absent():
    """The mirror of the test above, so "just delete the sentence" is not a way to pass it. The
    descriptor and then the hull were pinned here in turn; both are ported, so the standalone
    unlocated-H allocator -- the last genuinely absent capability -- carries the mirror."""
    assert 'mar_h_alloc' in _read('tools', 'README.md'), \
        'README no longer lists the H allocator as not-ported -- did it get ported?'
    for name in ('mar_h_alloc', 'mar_h_topup'):
        with pytest.raises(ImportError):
            __import__(name)


def test_the_docs_do_not_deny_the_hull_now_that_it_is_ported():
    """The hull is here; what it needs is an extra and a key. A doc that still says "not ported"
    sends the analyst to `escalate` on a question this package can now answer -- while a doc that
    forgets the key requirement is worse, because an unconfigured install reports `not_run` and the
    analyst has to know that is UNCHECKED rather than a passing hull."""
    for parts in (('tools', 'README.md'), ('README.md',), (SKILLS, 'tools.md')):
        for i, line in enumerate(_read(*parts).splitlines(), 1):
            low = line.lower()
            if 'hull' not in low:
                continue
            assert not ('not ported' in low or 'no `mp_api_key` is set' in low), (
                f'{"/".join(parts)}:{i} still denies the hull: {line.strip()!r}')

    from demars_core.hull import compute_hull, mp_available
    ok, why = mp_available()
    assert (why is None) == ok
    if not ok:
        # the refusal has to be a REPORTED state, not an exception and not a null
        res = compute_hull(os.path.join(ROOT, 'demars-core', 'tests', 'fixtures',
                                        'cod_9003141.cif'))
        assert res['state'] in ('not_run', 'ambiguous') and res['reason'], res
        assert 'mar' not in res, 'a refused hull must not carry a stability number'


def test_the_docs_do_not_deny_the_descriptor_now_that_it_is_ported():
    """`demars_core.disorder_class` IS part of demars-core, so no doc may still tell the analyst to leave
    `disorder_descriptor` null -- `strategy.md` reads `no_full_backbone` out of that field as a
    triage signal, and an analyst told the field cannot exist will not look."""
    docs = (('tools', 'README.md'), ('README.md',),
            (SKILLS, 'tools.md'), (SKILLS, 'principles.md'))
    for parts in docs:
        for i, line in enumerate(_read(*parts).splitlines(), 1):
            low = line.lower()
            if 'disorder_descriptor' not in low and 'disorder_class' not in low:
                continue
            assert not ('not ported' in low or 'unavailable in this build' in low
                        or 'always report' in low), (
                f'{"/".join(parts)}:{i} still denies the descriptor: {line.strip()!r}')

    from demars_core._engine.mar_record import build_record
    from demars_evidence import evidence_for

    fixture = os.path.join(ROOT, 'demars-core', 'tests', 'fixtures', 'cod_1544358.cif')
    ev = evidence_for(fixture, siblings=False)
    rd = build_record(None, {'engine': {}, 'distribution': {}, 'ensemble_files': {}}, {}, evidence=ev)

    dd = rd['disorder_descriptor']
    assert dd.get('available') is True, dd
    assert dd['disorder_set'] and set(dd['disorder_set']) <= {'O', 'S', 'V', 'P', 'SV', 'SP',
                                                              'VP', 'SVP'}, dd
    assert isinstance(dd['no_full_backbone'], bool), dd


def test_an_unavailable_record_field_says_so_instead_of_going_null():
    """`null` in this schema is what an unrun optional step leaves. A field that could NOT be
    computed for this record has to be distinguishable from one that came back empty. The descriptor
    is read from the input CIF, so an evidence bundle with no `source` cannot produce one -- and that
    is the case that must not silently look like a clean empty answer."""
    from demars_core._engine import mar_evidence as MEV
    from demars_core._engine.mar_record import build_record

    fixture = os.path.join(ROOT, 'demars-core', 'tests', 'fixtures', 'cod_1544358.cif')
    with open(fixture, encoding='utf-8') as fh:
        ev = MEV.evidence_from_text(fh.read(), iid=None, search_siblings=False)
    assert 'source' not in ev, 'fixture check: evidence_from_text does not stamp a source'
    rd = build_record(None, {'engine': {}, 'distribution': {}, 'ensemble_files': {}}, {}, evidence=ev)

    dd = rd['disorder_descriptor']
    assert isinstance(dd, dict) and dd.get('available') is False, dd
    assert 'reason' in dd and dd['reason'], 'an unavailable field must say why'
