import torch

from ensomi_model.audio_rows.data import Dataset
from ensomi_model.audio_rows.evaluate import label_shuffle
from ensomi_model.audio_rows.model import LABEL, HeadModel

from .helpers import tiny_build


def test_shuffle_delta_nll_is_exactly_zero_only_when_the_model_ignores_the_label(tmp_path):
    data = Dataset(tiny_build(tmp_path), 'chart')
    record = dict(label_mean=data.label_mean, label_std=data.label_std)
    torch.manual_seed(0)
    model = HeadModel().eval()
    reads = label_shuffle(model, record, data)
    with torch.no_grad():
        model.gru.weight_ih_l0[:, model.cfg.enc + model.cfg.token + LABEL] = 0.0
    blind = label_shuffle(model, record, data)
    assert reads['same_audio_chart_shuffle']['charts'] == 2
    for shuffle in ('section_shuffle', 'same_audio_chart_shuffle'):
        assert reads[shuffle]['mean'] != 0.0 and blind[shuffle]['mean'] == 0.0
