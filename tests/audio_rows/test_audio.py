import numpy as np

from ensomi_model.audio_rows.audio import FEATURE_DIM, MEL_BINS, features


def test_beat_probabilities_are_sampled_at_the_mel_frame_centres():
    beat = np.zeros(50, dtype=np.float32)
    beat[10] = 1.0                                                      # BeatThis frame 10 is centred at 200 ms
    f = features(np.zeros((100, MEL_BINS), dtype=np.float32), beat, np.zeros(50, dtype=np.float32))
    assert f.shape == (100, FEATURE_DIM)
    assert np.argmax(f[:, MEL_BINS]) == 18 and f[18, MEL_BINS] == 1.0    # Mel frame 18 spans [180, 220) ms
