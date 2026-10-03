"""Corpus-referenced evaluation of generated charts.

A generated chart is judged by how close it is to the ranked and loved corpus,
at several scales and views, under the condition it was generated with.

- ``case``: what one evaluation judges. A chart, a condition composed of a
  timing grid or a skeleton, a context and audio, and a scope whose given and
  scored parts are free span sets; ``.osu`` files and their sections load here.
- ``redlines``: a chart's red lines and which of them state the music's beat;
  BPM gimmicks and other expressive lines are kept out of the grid.
- ``beats``: canonical beat coordinates, each segment folded into 80 to 160 BPM.
- ``corpus``: the R2 corpus, one row per usable 4K chart of ``dataset/``, with
  song groups and the evaluation split; whole files are filtered here.

Rhythm, arrangement and load operators are not implemented yet; ``case``
defines the protocol they will follow.
"""
