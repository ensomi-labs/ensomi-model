"""Synthetic metadata checks: no corpus files, audio, charts or model weights are needed."""
import hashlib
import json
from types import SimpleNamespace

import pytest

from n.experiments.n1_redesign.gpu.audit_inputs import check_rows, probe_bucket
from n.experiments.n1_redesign.gpu.sealed import build_exclusion
from n.experiments.n1_redesign.gpu.decision import Decision


def empty_exclusion():
    return {k: set() for k in ('audio_sha256', 'recordings', 'set_ids', 'keys')}


def row(**kw):
    return dict(dict(audio_rel='dataset/std-taiko/100/a.mp3', audio_sha256='a',
                     rec='recording-a', set_id=100, census_key='chart-a'), **kw)


def test_transitive_folder_seal_catches_another_audio(tmp_path):
    """An unlisted audio shares a folder with a dedup member linked to a held-out group."""
    (tmp_path / 'h1.csv').write_text('audio_sha256\nh1\n')
    (tmp_path / 'h2.csv').write_text('audio_sha256\nh2\n')
    membership = dict(members=[dict(recording='hard-rec', sets=[dict(
        set_id='900', audio_sha256='hard-audio', key='std-taiko/900@hard')])])
    hard = tmp_path / 'hard'
    hard.mkdir()
    data = json.dumps(membership).encode()
    (hard / 'membership.json').write_bytes(data)
    (tmp_path / 'groups.json').write_text(json.dumps([dict(members=[
        dict(sha256='b', path='dataset/std-taiko/100/b.mp3'),
        dict(sha256='c', path='dataset/0/200/c.mp3', corpus_group_ids=['linked']),
    ])]))
    (tmp_path / 'inventory.jsonl').write_text('\n'.join(json.dumps(r) for r in [
        dict(path='dataset/std-taiko/100/a.mp3', sha256='a'),
        dict(path='dataset/std-taiko/100/b.mp3', sha256='b'),
    ]) + '\n')
    args = SimpleNamespace(holdout1='h1.csv', holdout2='h2.csv', hard_holdout='hard',
                           hard_holdout_sha256=hashlib.sha256(data).hexdigest(),
                           groups='groups.json', inventory='inventory.jsonl')
    corpus = [dict(group_id='linked', set_dir='dataset/0/200', audio_sha256='c', eval_split='fit'),
              dict(group_id='linked', set_dir='dataset/0/200', audio_sha256='h2', eval_split='fit'),
              dict(group_id='r2', set_dir='dataset/0/300', audio_sha256='r2', eval_split='heldout')]
    sealed = build_exclusion(args, tmp_path, corpus, {})
    exclusion = empty_exclusion()
    assert check_rows([row()], exclusion, set()) == []
    assert check_rows([row()], exclusion, sealed)[0]['reasons'] == ['sealed_folder']
    assert {'dataset/0/200', 'dataset/0/300', 'dataset/std-taiko/900'} <= sealed
    assert check_rows([row(audio_rel='dataset/std-taiko/101/a.mp3')], exclusion, sealed) == []
    args.hard_holdout_sha256 = '0' * 64
    with pytest.raises(SystemExit, match='does not match the seal'):
        build_exclusion(args, tmp_path, corpus, {})


@pytest.mark.parametrize('key,value,reason', [
    ('audio_sha256', 'a', 'sealed_audio'),
    ('recordings', 'recording-a', 'sealed_recording'),
    ('set_ids', '100', 'sealed_set'),
    ('keys', 'chart-a', 'sealed_key'),
])
def test_direct_exclusion_identities(key, value, reason):
    exclusion = empty_exclusion()
    exclusion[key].add(value)
    assert check_rows([row()], exclusion, set())[0]['reasons'] == [reason]


@pytest.mark.parametrize('path', [
    '/dataset/0/100/a.mp3', 'dataset/0/../a.mp3', 'dataset/netease/100/a.mp3',
    'dataset/0/a.mp3', 'dataset/0/100/nested/a.mp3',
])
def test_foreign_and_traversing_paths_are_rejected(path):
    assert 'outside_allowed_sources' in check_rows(
        [row(audio_rel=path)], empty_exclusion(), set())[0]['reasons']


def test_probe_split_uses_recording_sha_and_prefix():
    sha = 'f' * 64
    expected = int.from_bytes(hashlib.sha256(('n1r-probe:' + sha).encode()).digest()[:4], 'big') % 8
    assert probe_bucket(sha) == expected


def test_decision_excludes_folder_group_and_aliases_but_keeps_new_seal_visible():
    d = Decision(dict(folders=['dataset/std-taiko/100'], corpus_groups=['excluded']),
                 [dict(group_id='excluded', set_dir='dataset/0/200', audio_filename='B.mp3', audio_sha256='b')],
                 [dict(path='dataset/std-taiko/100/a.mp3', sha256='a')])
    candidates = [row(), row(audio_rel='dataset/0/200/b.mp3', audio_sha256='b'),
                  row(audio_rel='dataset/0/300/alias.mp3', audio_sha256='a'),
                  row(audio_rel='dataset/0/400/other.mp3', audio_sha256='c', groups=['excluded']),
                  row(audio_rel='dataset/0/500/new.mp3', audio_sha256='new')]
    kept, removed = d.filter(candidates)
    assert len(removed) == 4 and kept == candidates[-1:]
    assert check_rows(kept, empty_exclusion(), {'dataset/0/500'})[0]['reasons'] == ['sealed_folder']
