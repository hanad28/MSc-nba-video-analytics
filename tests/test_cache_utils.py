"""Unit tests for pickle-based pipeline caching and fingerprint sidecar validation."""

from __future__ import annotations

import hashlib
import json
import pickle

import pytest

from basketball.cache.cache_utils import (
    data_digest,
    file_digest,
    load_cache,
    load_valid_cache,
    save_cache,
    save_cache_with_meta,
)

FINGERPRINT = {
    'video_digest': 'a' * 64,
    'model_digest': 'b' * 64,
    'conf_threshold': 0.5,
    'n_frames': 3,
}


def test_save_cache_round_trip(tmp_path):
    path = str(tmp_path / 'cache.pkl')
    data = {'tracks': [1, 2, 3]}

    save_cache(data, path)

    assert load_cache(path) == data


def test_save_cache_creates_missing_parent_directories(tmp_path):
    path = str(tmp_path / 'a' / 'b' / 'cache.pkl')

    save_cache([1, 2], path)

    assert load_cache(path) == [1, 2]


def test_save_cache_overwrites_an_existing_file(tmp_path):
    path = str(tmp_path / 'cache.pkl')

    save_cache('first', path)
    save_cache('second', path)

    assert load_cache(path) == 'second'


def test_save_cache_writes_a_real_pickle(tmp_path):
    path = tmp_path / 'cache.pkl'

    save_cache({'a': 1}, str(path))

    with open(path, 'rb') as f:
        assert pickle.load(f) == {'a': 1}


def test_load_cache_raises_for_a_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_cache(str(tmp_path / 'absent.pkl'))


def test_save_cache_with_meta_writes_the_cache_and_the_sidecar(tmp_path):
    path = tmp_path / 'cache.pkl'

    save_cache_with_meta({'a': 1}, str(path), FINGERPRINT)

    assert path.exists()
    sidecar = tmp_path / 'cache.pkl.meta.json'
    assert sidecar.exists()
    with open(sidecar) as f:
        assert json.load(f) == FINGERPRINT


def test_load_valid_cache_returns_the_data_on_a_full_fingerprint_match(tmp_path):
    path = str(tmp_path / 'cache.pkl')
    save_cache_with_meta([1, 2, 3], path, FINGERPRINT)

    assert load_valid_cache(path, dict(FINGERPRINT)) == [1, 2, 3]


def test_load_valid_cache_returns_none_for_a_missing_cache_file(tmp_path):
    assert load_valid_cache(str(tmp_path / 'absent.pkl'), FINGERPRINT) is None


def test_load_valid_cache_treats_a_missing_sidecar_as_unverifiable(tmp_path, capsys):
    path = str(tmp_path / 'cache.pkl')
    save_cache([1, 2, 3], path)

    assert load_valid_cache(path, FINGERPRINT) is None
    assert 'unverifiable' in capsys.readouterr().out


def test_load_valid_cache_raises_for_a_corrupt_cache_with_no_sidecar(tmp_path):
    # The unverifiable-sidecar path must not bypass load_cache's corrupt-cache
    # guard: every pre-sidecar cache takes this branch on its first run, and a
    # None here would let the regenerating stage overwrite the evidence.
    path = tmp_path / 'cache.pkl'
    path.write_bytes(b'not a pickle')

    with pytest.raises(IOError, match='unreadable'):
        load_valid_cache(str(path), FINGERPRINT)

    assert path.read_bytes() == b'not a pickle'


@pytest.mark.parametrize('key', sorted(FINGERPRINT))
def test_load_valid_cache_rejects_a_cache_when_one_key_mismatches(tmp_path, key):
    path = str(tmp_path / 'cache.pkl')
    save_cache_with_meta([1, 2, 3], path, FINGERPRINT)
    current = dict(FINGERPRINT)
    current[key] = 'different' if isinstance(current[key], str) else current[key] + 1

    assert load_valid_cache(path, current) is None


def test_load_valid_cache_names_the_differing_key_with_old_and_new_values(tmp_path, capsys):
    path = str(tmp_path / 'cache.pkl')
    save_cache_with_meta([1, 2, 3], path, FINGERPRINT)

    assert load_valid_cache(path, dict(FINGERPRINT, n_frames=4)) is None
    assert 'n_frames 3 -> 4' in capsys.readouterr().out


def test_load_valid_cache_abbreviates_digests_in_the_mismatch_message(tmp_path, capsys):
    path = str(tmp_path / 'cache.pkl')
    save_cache_with_meta([1], path, FINGERPRINT)

    assert load_valid_cache(path, dict(FINGERPRINT, video_digest='c' * 64)) is None
    assert 'video_digest aaaaaaaa… -> cccccccc…' in capsys.readouterr().out


def test_data_digest_is_stable_for_equal_data():
    assert data_digest({'a': [1, 2]}) == data_digest({'a': [1, 2]})


def test_data_digest_differs_for_unequal_data():
    assert data_digest([1, 2, 3]) != data_digest([1, 2, 4])


def test_file_digest_matches_a_directly_computed_sha256(tmp_path):
    path = tmp_path / 'weights.pt'
    path.write_bytes(b'checkpoint bytes')

    assert file_digest(str(path)) == hashlib.sha256(b'checkpoint bytes').hexdigest()


def test_file_digest_is_memoised_per_path_for_the_process_lifetime(tmp_path):
    path = tmp_path / 'weights.pt'
    path.write_bytes(b'first')
    first = file_digest(str(path))
    path.write_bytes(b'second')

    # Documented behaviour: one hash per path per process, so an in-process
    # rewrite at the same path is not observed until a new process rehashes it.
    assert file_digest(str(path)) == first
