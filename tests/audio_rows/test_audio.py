import numpy as np

from ensomi_model.audio_rows.audio import FEATURE_DIM, MEL_BINS, features


def test_beat_probabilities_are_sampled_at_the_mel_frame_centres():
    beat = np.zeros(50, dtype=np.float32)
    beat[10] = 1.0                                                      # BeatThis frame 10 is centred at 200 ms
    f = features(np.zeros((100, MEL_BINS), dtype=np.float32), beat, np.zeros(50, dtype=np.float32))
    assert f.shape == (100, FEATURE_DIM)
    assert np.argmax(f[:, MEL_BINS]) == 18 and f[18, MEL_BINS] == 1.0    # Mel frame 18 spans [180, 220) ms


def test_listener_takes_the_bar_phase_from_the_downbeats(monkeypatch):
    import torch
    from ensomi_model.audio_rows import audio
    from ensomi_model.audio_rows.fitter.types import TimingSegment
    from ensomi_model.features import audio as feature_audio

    t = np.arange(3000) * 20.0                                          # 60 s of BeatThis frames
    def logits(times):
        p = np.clip(0.002 + 0.997 * np.exp(-0.5 * ((t[:, None] - times[None]) / 12.0) ** 2).max(1), 1e-6, 1 - 1e-6)
        return torch.tensor(np.log(p / (1 - p)), dtype=torch.float32)
    beats = np.arange(1000.0, 59_000.0, 500.0)
    listener = audio.Listener.__new__(audio.Listener)
    listener.load = lambda path: (np.zeros(22050, dtype=np.float32), 22050)
    listener.frames = lambda wave, sr: (logits(beats), logits(beats[2::4]))  # bars start on the third beat
    monkeypatch.setattr(feature_audio, 'load_audio_file', lambda path, sr: np.zeros(sr, dtype=np.float32))
    monkeypatch.setattr(audio, 'lead_seconds', lambda path, decoder: 0.0)
    monkeypatch.setattr(audio, 'compute_log_mel_10ms', lambda wave, sample_rate, config: np.zeros((100, MEL_BINS)))
    monkeypatch.setattr(audio, 'fit_segments', lambda beat, downbeat, fps: (TimingSegment(1500.0, 500.0),))
    song = audio.Listener.listen(listener, 'song.mp3')
    assert np.allclose(song.segments, [[2000.0, 500.0]], atol=0.01)    # refit on the peaks, then the downbeats' bar


def test_listener_moves_the_grid_to_the_mel_onsets(monkeypatch):
    import torch
    from ensomi_model.audio_rows import audio
    from ensomi_model.audio_rows.fitter.mel_phase import MAPPER_OFFSET_MS
    from ensomi_model.audio_rows.fitter.types import TimingSegment
    from ensomi_model.audio_rows.grid import LAG_MS
    from ensomi_model.features import audio as feature_audio

    t = np.arange(3000) * 20.0
    def logits(times):
        p = np.clip(0.002 + 0.997 * np.exp(-0.5 * ((t[:, None] - times[None]) / 12.0) ** 2).max(1), 1e-6, 1 - 1e-6)
        return torch.tensor(np.log(p / (1 - p)), dtype=torch.float32)
    beats = np.arange(1000.0, 59_000.0, 500.0)
    onsets = beats - LAG_MS + 4.0 - MAPPER_OFFSET_MS                   # the mappers' beats sit 4 ms after BeatThis's grid
    env = np.exp(-0.5 * (((np.arange(6000) * 10.0 + 15.0)[:, None] - onsets[None]) / 8.0) ** 2).max(1)
    log_mel = np.repeat(np.cumsum(env)[:, None], MEL_BINS, axis=1)
    listener = audio.Listener.__new__(audio.Listener)
    listener.load = lambda path: (np.zeros(22050 * 60, dtype=np.float32), 22050)
    listener.frames = lambda wave, sr: (logits(beats), logits(beats[2::4]))
    monkeypatch.setattr(feature_audio, 'load_audio_file', lambda path, sr: np.zeros(sr, dtype=np.float32))
    monkeypatch.setattr(audio, 'lead_seconds', lambda path, decoder: 0.0)
    monkeypatch.setattr(audio, 'compute_log_mel_10ms', lambda wave, sample_rate, config: log_mel)
    monkeypatch.setattr(audio, 'fit_segments', lambda beat, downbeat, fps: (TimingSegment(1500.0, 500.0),))
    song = audio.Listener.listen(listener, 'song.mp3')
    assert np.allclose(song.segments, [[2004.0, 500.0]], atol=1.0)     # the bar phase, then 4 ms to the onsets
