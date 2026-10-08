"""Head rows from audio for R2: a whole-song beat grid, then the heads one canonical beat at a time.

A path beside R2, outside its plan. BeatThis and the legacy grid fitter give the song's beat grid
(``audio``, ``fitter``, ``grid``). Per canonical beat of that grid the generator (``model``)
decides a lattice (duple 1/16 or triple 1/12), a head count and the slots that carry a head, from
the log-Mel and the BeatThis activations, its own earlier beats and a section difficulty. Heads
sit on lattice slots with zero offsets. R2 then arranges them unchanged
(``ensomi_model.r2.sampling.continue_chart``) and the chart is exported as ``.osu``
(``generate``).

The difficulty is log(W_H / s) per section: the ras-v1 reference workload of the section's head
rows per second (``ensomi_model.r2.strain``), the rows-only part of R2's r = sqrt(W / W_H).

Modules: ``data`` builds training data from the R2 cache and reads it; ``train`` trains;
``generate`` runs audio to ``.osu``; ``evaluate`` holds the development measures; ``lattice``
converts head times to per-beat targets and back.
"""
