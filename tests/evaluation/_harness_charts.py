"""Synthetic charts for the harness tests."""
from ensomi_model.evaluation.case import Chart
from ensomi_model.evaluation.redlines import RedLine
from ensomi_model.osu_core.hitobjects import ManiaHitObject, ManiaHitObjectKind

TAP, HOLD = ManiaHitObjectKind.TAP, ManiaHitObjectKind.HOLD


def tap(t, lane):
    return ManiaHitObject(float(t), float(t), lane, TAP)


def hold(t, end, lane):
    return ManiaHitObject(float(t), float(end), lane, HOLD)


def chart(objects, lines=((0.0, 500.0, 4),)):
    return Chart(tuple(sorted(objects, key=lambda o: (o.start_time_ms, o.lane))),
                 tuple(RedLine(o, b, m) for o, b, m in lines))


def busy_chart(bars=32, beat=500.0):
    """120 BPM 4/4: eighth-note rows cycling over lanes, a chord on every beat, a short hold every bar.

    No hold crosses a bar line; every head is on a canonical 1/2 or 1/4.
    """
    objects = []
    step = beat / 2
    for b in range(bars):
        t0 = b * 4 * beat
        for k in range(8):
            t = t0 + k * step
            lane = (k + b) % 4
            if k == 0:
                objects.append(hold(t, t + beat, lane))
                objects.append(tap(t, (lane + 2) % 4))
            elif k % 2 == 0:
                objects.append(tap(t, lane))
                objects.append(tap(t, (lane + 1) % 4))
            else:
                objects.append(tap(t, lane))
        objects.append(tap(t0 + 3.5 * beat + beat / 4, (b + 1) % 4))
    return chart(objects)


def ln_chart(bars=24, beat=500.0):
    """120 BPM 4/4 with long holds, two of which cross into the next bar, and taps between them.

    In odd bars lane k holds from beat k + 0.5 for 2.5 beats (lanes 2 and 3 into the next bar); even bars
    have eighth-note taps on lanes 0 and 1 while those holds last, then on every lane.
    """
    objects = []
    for b in range(bars):
        t0 = b * 4 * beat
        if b % 2:
            for k in range(4):
                objects.append(hold(t0 + (k + 0.5) * beat, t0 + (k + 3.0) * beat, k))
        else:
            for k in range(8):
                objects.append(tap(t0 + k * beat / 2, k % 2 if k < 4 else k % 4))
    return chart(objects)


def osu_file(c, path, title='harness test'):
    """Write ``c`` as an ``.osu`` file at ``path``."""
    from ensomi_model.evaluation.osu_text import osu_text
    path.write_text(osu_text(c, title=title), encoding='utf-8')
    return str(path)
