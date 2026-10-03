"""Corpus-referenced evaluation of generated charts.

A generated chart is judged by how close it is to the ranked and loved corpus,
at several scales and views, under the condition it was generated with. A
condition composes any of: a timing grid or a skeleton (grid plus head and
release times), given context (fixed parts of the chart), and full audio. Both
the given part and the scored part of a chart are free spans.

Built so far: ``corpus`` (one-row-per-file inventory of ``dataset/`` with song
groups and the evaluation split) and ``beats`` (canonical beat coordinates).
Rhythm, arrangement and load operators are not implemented yet.
"""
