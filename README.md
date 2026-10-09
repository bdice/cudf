# R8 LRB string-sort benchmark artifacts

These artifacts support the `bdice:radix-lrb-string-sort` draft PR to NVIDIA/cuDF.
The measured implementation is `6e05c89801c083ef403d755cdb239b2e1b455fc9`;
`74dd068d7b819abc307eed2bbe607856d68b2b8e` adds the comparison report without changing code.
The segmented reference is PR #24498 at `a257d75a7b5a4bb5b4ee392861d8d466e950d297`.

Measurements use an NVIDIA H100 SXM 80GB and CUDA 13.3. Primary rankings use the
median of three independently warmed process means, with 20 samples per process.
Inputs and binaries are matched, and Nsight profile durations are excluded from rankings.
Synthetic states receive equal weight; results do not prescribe production dispatch.

- [Comparison report](pr24498-comparison.md)
- [Ranking plot](pr24498-ranking.png), [SVG](pr24498-ranking.svg)
- [Controlled run distributions](pr24498-segments.png), [SVG](pr24498-segments.svg)
- [Per-state medians, noise, and peak memory](pr24498-timings.csv)
- [All primary measured means](confirmation/raw.csv)
- [Pool summaries](pr24498-summary.csv)
- [Stable and descending supplement](pr24498-stable-descending.csv)
- [Nsight kernel records](profiles/kernels.csv)
- [Repeated noisy cases and PR-default wins](noise-recheck/summary.csv)
- [Public provenance](provenance.json)

PR-default timeouts are whole-process limits and are censored, not per-sort timings.
Follow-up probes and repeat checks are kept separate from the primary ranking.
The public provenance omits internal machine reservation records.
