"""CPU NumPy inference for the frozen, calibrated band readouts."""
from common import *
from scipy.special import softmax


def load_head(path):
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def head_logits(head, features):
    out = []
    for lo in range(0, len(features), 8192):
        x = (features[lo:lo+8192].astype(np.float32)-head['mean']) / head['sd']
        x = x @ head['w0'].T + head['b0']
        if 'w1' in head:
            x = np.maximum(x, 0) @ head['w1'].T + head['b1']
        out.append(x)
    return np.concatenate(out)


def probabilities(head, features):
    return softmax(head_logits(head, features).astype(np.float64)/float(head['temperature']), axis=1)
