import numpy as np

from ensomi_model.system.measures import (arrangement_measures, audio_recurrence, beat_mel, dependence, jack_rate,
                                          jack_sound, repeats, tracking)

from .helpers import grid


def test_jack_rate_repeats_and_dependence():
    masks = np.array([1, 1, 2, 4, 4, 4, 8, 1])
    assert np.isclose(jack_rate(masks), 3 / 8)
    b = np.arange(64) * 0.5
    periodic = np.tile([1, 2, 4, 8, 1, 2, 4, 8], 8)          # repeats every 4 beats
    r = repeats(b, periodic)
    assert r['R_4'] == 1.0 and r['R_8'] == 1.0 and r['R_1'] == 0.0
    rng = np.random.default_rng(0)
    alternating = np.tile([3, 12], 200)
    assert dependence(alternating) > 0.9 > abs(dependence(rng.integers(1, 16, 400)))


def song_with_sections(rng, beats=256, beat_ms=500.0):
    """Two sounds alternating every 8 beats; 10 ms frames."""
    F = int(beats * beat_ms / 10) + 100
    a, c = rng.normal(size=128), rng.normal(size=128)
    mel = np.zeros((F, 128), np.float32)
    section = (np.arange(F) * 10 // (8 * beat_ms)).astype(int) % 2
    mel[:] = np.where(section[:, None] == 0, a, c) + 0.3 * rng.normal(size=(F, 128))
    return mel, section


def test_audio_recurrence_sees_arrangements_that_follow_the_audio():
    rng = np.random.default_rng(1)
    g = grid()
    mel, _ = song_with_sections(rng)
    b = np.arange(4, 250, 0.5)
    t = g.time_of_beat(b)
    follow = np.where(((b // 8) % 2 == 0)[:, None], [True, False, False, False], [False, False, True, True])
    noise = np.eye(4, dtype=bool)[rng.integers(0, 4, len(b))]
    bm = beat_mel(mel, g, 4, 249)
    assert audio_recurrence(b, follow | noise, bm, 4, 249)['audio_recurrence'] > 0.1
    assert abs(audio_recurrence(b, noise, bm, 4, 249)['audio_recurrence']) < 0.05
    assert len(t) == len(b)


def test_jack_sound_sees_jacks_on_repeated_sound():
    rng = np.random.default_rng(2)
    g = grid()
    K = 600
    t = 1000.0 + 250.0 * np.arange(K)
    mel = np.repeat(rng.normal(size=(K + 20, 128)), 25, axis=0).astype(np.float32)   # a new sound every 250 ms
    same = rng.random(K) < 0.3
    for k in np.flatnonzero(same):                     # some heads repeat the previous head's sound
        if k:
            f, p = int(t[k] / 10), int(t[k - 1] / 10)
            mel[f - 2:f + 3] = mel[p]
    heads = np.eye(4, dtype=bool)[rng.integers(0, 4, K)]
    for k in range(1, K):
        heads[k] = heads[k - 1] if same[k] else np.roll(heads[k - 1], 1)
    assert jack_sound(t, heads, g, t[-1] + 1000, mel)['jack_sound'] > 0.3


def test_tracking_and_the_whole_record():
    K = 200
    actions = np.zeros((K + 1, 4), np.int64)
    actions[:K, 0] = 1                                  # one tap per row on lane 0
    track = tracking(actions, K, [0.0, 1.0, 0.0])
    assert track == dict(track_rmse_held=0.0, track_rmse_heads=0.0, track_rmse_busy=0.0)
    rng = np.random.default_rng(3)
    mel, _ = song_with_sections(rng, beats=120)
    t = 1000.0 + 250.0 * np.arange(K)
    m = arrangement_measures(t, t[-1] + 1000, grid(), actions, mel, target=[0, 1, 0])
    assert m['J'] == (K - 1) / K and m['nh'] == 1.0 and m['track_rmse_heads'] == 0.0
    for key in ('D', 'S3long', 'audio_recurrence', 'jack_sound'):
        assert key in m
