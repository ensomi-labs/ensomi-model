# Decoding stage: style preference and hard constraints

<a id="h-decoding-style"></a>**Suggestion (2026-10-06):** Lele Liu, an MIR researcher at the University of Würzburg (CAIDAS), suggested in conversation with the human that style preference be trained at the decoding stage, and mentioned an HMM. Her exact meaning was relayed second hand ([private, local](private/human-inputs/dc997baa-3539-4368-a0b2-c72bd4cea5e8.md#prompt-6)), so the agent reading below needs her confirmation.

**Agent reading.** In transcription and beat tracking, a network gives local probabilities and a separate decoder imposes structure: an HMM or DBN with Viterbi, as in tempo-continuity beat trackers. The decoder's few transition parameters can be fitted or tuned without retraining the network. Carried over to R2, the generator proposes; a decoder searches over whole continuations with this score:

  log p_R2(decision) + λ · log p_style(s_t | s_{t−1}) + λ · log e(decision | s_t)

Here s_t is a small latent style state, for example the Lens concepts or the pattern mode of a passage. Exact Viterbi is impossible because R2's state is not a finite Markov chain. Beam search or sequential Monte Carlo over pairs (chart prefix, s_t) is the practical form; it is the same as shallow fusion in speech recognition.

This bears on three map nodes:
- `selection`: choosing among proposed futures.
- `control`: steering without retraining; strength could be a decoding weight.
- `realtime`: the search budget sits inside the latency budget.

It cannot reach styles to which the generator gives almost no probability.

<a id="s-decoding-now"></a>**R2 decoding today [code].** Ancestral sampling: Gumbel-max at temperature 1, a fair coin for the orientation, then the pointer lane by lane. There is no truncation, search or reranking (`operating_point.py:17-18`, `sampling.py:84-105`). Every decision has an exact distribution over a finite legal set, and charts replay exactly. Masks, logit penalties, temperature, beam or SMC search, reranking of sampled sections, and guidance between conditioned and unconditioned logits therefore all apply. Decoding is part of the operating-point hash, so each change is a new operating point.

<a id="p-min-hold-mask"></a>**Proposal: the 40 ms rules at decoding (agent, not decided).**
- **Minimum hold** ([d-r2-spec](r2-ln-design.md#d-r2-spec): none for now, 40 ms the candidate). Real charts have no holds under 60 ms. A hard mask is exact and costs nothing. It must cover two levels:
  - release candidates closer than the minimum to the hold's start;
  - row codes that would release in the gap when no candidate survives.
  Otherwise a row action can leave the pointer with no candidate. Putting the same mask in the action contract keeps training and decoding consistent.
- **Release 1-40 ms before another lane's head.** This rises with star, from 0.05% at 2-3★ to 5.5% at 5-6★ ([s-ln-census](r2-ln-design.md#s-ln-census)), so a hard ban would cut real high-star style. At most it gets a soft, star-aware penalty, and only where the generated rate exceeds the source rate at the same star. Selection guard (iv) already compares against the source rate.
