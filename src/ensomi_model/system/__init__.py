"""The whole system: audio to a playable osu!mania 4K chart, with the R2 demo's controls and no source chart.

``make`` is the command: the Listener grid and the head rows (``audio_rows``), the corpus prior that replaces
what the R2 demo read from a source chart (``prior``), frozen R2 arranging the rows under the demo's controls
(``arrange``, ``controls``) with the copy path underneath when on (``copy``, trained by ``copytrain``), then the
export with a receipt.
"""
