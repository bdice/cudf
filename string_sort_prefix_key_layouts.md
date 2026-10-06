# Carrying the cached prefix with the row index in string sorts

Context: https://github.com/NVIDIA/cudf/pull/24267 (cached 8-byte prefixes for
single-column string `sorted_order`). Status: idea only, not built or measured.

## The problem

In the PR, the sort runs `cub::DeviceMergeSort::{Stable,}SortKeysCopy` over the
row indices `0..N-1`, with a comparator that does `prefixes[lhs]` and
`prefixes[rhs]`.

After the first few merge passes the indices being compared are scattered, so
each `prefixes[idx]` is a random 8-byte read that costs a 32-byte sector.
Coalescing tricks such as the funnel-shift loads in
`cpp/include/cudf/strings/detail/gather.cuh` do not help, because the access
pattern depends on the data.

The fix is to make the prefix travel with the index, so the comparator reads
its operands from the sequentially staged sort keys instead of gathering them
from `prefixes[]`.

## The idea

Sort a key type that holds both the prefix and the row index:

```cpp
struct prefix_row {
  uint64_t prefix;
  cudf::size_type row;
};
```

`sizeof(prefix_row)` is 16 bytes (4 bytes of padding). CUB's merge sort stages
keys through shared memory in coalesced tiles, and the comparator sees both
fields without any global gather. Only the tie path (equal prefixes) touches
the string data, via `row`.

Comparator sketch:

```cpp
template <bool has_nulls>
struct prefix_row_comparator {
  __device__ bool operator()(prefix_row const& l, prefix_row const& r) const
  {
    if (l.prefix != r.prefix) { return ascending ? l.prefix < r.prefix : l.prefix > r.prefix; }
    return tie_break(l.row, r.row);
  }
  column_device_view const d_column;
  bool ascending;
  null_order null_precedence;
};
```

`tie_break` is the existing post-prefix logic from `string_prefix_comparator`
(length check, then suffix compare starting at byte 8).

## Implementation plan

Work in `cpp/src/sort/sort_column_impl.cuh`, on top of the PR branch.

1. Add `prefix_row` and `prefix_row_comparator` next to
   `string_prefix_comparator`.
2. Change `string_prefix_extractor` to output `prefix_row{prefix, row}` into a
   `rmm::device_uvector<prefix_row>` (replacing the `uint64_t` array).
3. Call `cub::DeviceMergeSort::SortKeysCopy` (or `StableSortKeysCopy` when
   `method == sort_method::STABLE`) with `prefix_row` input and output
   buffers. The existing `merge_sort` helper takes the key iterators, so it
   needs generalizing from `size_type` keys to a key type template.
4. Write the sorted rows back to `indices` with a `thrust::transform` that
   projects `.row`.
5. Nulls: the comparator still needs the null check before the prefix compare,
   so keep the `has_nulls` specialization and check `d_column.is_null(row)` on
   both operands. This reads the null mask via `row`, which is a gather, but a
   cheap one because the mask is 1 bit per row. An alternative is to partition
   nulls out first and sort only the valid rows, which removes the check from
   the comparator. That is a bigger change, so try it second.
6. Keep the PR's early exits (size < 2, all null, all empty) unchanged.

Memory cost: the key array and the sort output are each 16 B/row, against the
PR's 8 B prefix plus 4 B indices. Expect roughly 32 B/row for keys plus CUB
temporary storage, versus about 16 B/row now. Report this alongside any speedup.

### Alternative to evaluate afterwards: radix sort on the prefix

`cub::DeviceRadixSort::SortPairs` (`SortPairsDescending` for descending) with
`uint64_t` keys and `size_type` values is stable and avoids comparator calls
entirely. It then needs a second step that finds runs of equal prefixes and
sorts those runs with the full string comparator, for example with
`cub::DeviceSegmentedSort`. Nulls and the length rule for zero-padded prefixes
complicate this, so it is the larger experiment. Do it only if the merge-sort
variant above shows the gather was the bottleneck.

## Validation

Correctness comes before timing.

- Run the PR's `cpp/tests/sort/string_sort_prefix_test.cpp` and the existing
  sort tests unchanged against the new implementation. The PR's 342-line test
  file covers shared prefixes, embedded zero bytes, nulls, and both sort
  orders.
- Add cases that stress ties: many rows with identical 8-byte prefixes and
  differing suffixes, and strings that differ only by trailing zero bytes.
- Compare `sorted_order` output of the new path against the PR's path on the
  same random inputs. Stable sorts must match exactly; unstable sorts must
  match as orderings of the string values.

## Benchmark plan

Benchmarks live in `cpp/benchmarks/sort/sort_strings.cpp`:
`sort_strings` (end to end), `sorted_order_strings`, and
`sorted_order_strings_multi`. The PR adds 20 more `sorted_order` cases
(low cardinality, shared prefixes, variable length, 0/50/100% nulls).

1. Build the benchmarks on three commits: `main`, the PR head, and the PR
   head plus this change. Use the repo's normal build flow (for example
   `./build.sh libcudf benchmarks`, or your existing build directory) and run
   the same GPU for all three.
2. Run each with the same NVBench settings the PR used so numbers are
   comparable, for example:

   ```
   SORT_NVBENCH -b sorted_order_strings --json base.json
   ```

   The executable is the `SORT_NVBENCH` target, found under the build
   directory's `benchmarks/`.
3. Compare with NVBench's `nvbench_compare.py` (shipped with NVBench, not in
   this repo): `nvbench_compare.py base.json new.json`.
4. Report geometric mean over the original cases and over the new cases
   separately, as the PR does. The PR notes the new cases are biased toward
   large gains.
5. Report peak memory for at least the largest case.

### Profile before investing

Run Nsight Compute on the PR head for one large case to decide whether this is
worth doing:

```
ncu --set full -k regex:"DeviceMergeSort|prefix" SORT_NVBENCH -b sorted_order_strings --run-once
```

Look at:
- time share of the extractor kernel versus the merge-sort kernels;
- L2 sector efficiency (`l1tex__t_sectors_pipe_lsu_mem_global_op_ld` versus
  requests) on the merge-sort kernels, which shows how scattered the
  `prefixes[]` reads are;
- fraction of comparator calls that reach the string tie path (count via a
  temporary device counter in a scratch build, not committed).

If the merge-sort kernels are not limited by those prefix loads, this change
will not pay for its extra memory.

## Open questions

- CUB version in the build may differ in which `DeviceMergeSort` entry points
  exist (for example whether there is a stable pairs-with-copy variant). Check
  the headers before choosing between a struct key and a keys/values pair.
- The 4 bytes of padding in `prefix_row` could carry something useful, such as a
  null flag or the string length clamped to 8, which would let the comparator
  skip the null-mask read and the length lookup on ties.

## Variants under evaluation

All variants keep the PR's early exits and its tie semantics (zero-padded
prefix, length rule, suffix resume after the cached bytes).

| Id | Key layout (16 B) | Cached bytes | Null handling in comparator | Tie threshold |
|----|-------------------|--------------|-----------------------------|---------------|
| P  | PR head: `uint64_t prefixes[]` gathered by row id | 8 | `d_column.is_null(row)` | `size < 8` |
| A  | `{uint64 prefix; int32 row; 4 B pad}` | 8 | `d_column.is_null(row)` | `size < 8` |
| B  | `{uint64 hi; uint32 lo; int32 row}` | 12 | `d_column.is_null(row)` | `size < 12` |
| C  | `{uint64 prefix; int32 row; uint32 is_null}` | 8 | `is_null` field, no mask read | `size < 8` |

P is the reference, not a candidate. Comparing A to P isolates the effect of
carrying the prefix with the row. B and C are each compared against A, so each
isolates one change.

### Details that differ per variant

- B: `hi` is bytes 0-7 and `lo` is bytes 8-11, both big-endian. The comparator
  checks `hi`, then `lo` only when `hi` is equal. The suffix compare resumes at
  byte 12. Strings shorter than 12 bytes are zero-padded and resolved by length.
- C: the extractor writes `is_null = 1` and `prefix = 0` for null rows. The
  comparator checks both operands' `is_null` fields before comparing prefixes, so
  the null mask is never read in the sort. The `has_nulls == false`
  specialization still skips the check entirely. Rows with `is_null = 0` follow
  the A path.
- All keys are 16 B with 8 B alignment. Verify with `static_assert(sizeof(...) == 16)`
  in each variant so none silently grows.

### Implementation approach

Make the key type a template parameter of the extractor, the comparator and
`prefix_sorted_order_impl`, with one struct and one comparator per variant (A, B,
C). Select the variant with a compile-time constant while evaluating, for
example a macro or a template argument set in
`column_sorted_order_fn::operator()`. Do not add a runtime switch. The goal is to
pick one variant, so the losing variants and the selection mechanism are deleted
before the PR.

### Evaluation matrix

Run each variant on each of these axes. P is run on the same matrix.

- Existing `sorted_order_strings` cases from `sort_strings.cpp` (typical data).
- The PR's 20 added cases: low cardinality, shared prefixes, variable length,
  and 0%, 50%, 100% nulls.
- Add two targeted cases if the PR's set lacks them:
  - strings sharing exactly 8-11 leading bytes and diverging at byte 12 or
    later, which is where B should win and A and C should lose;
  - nullable columns at 50% nulls with short distinct strings, which is where C
    should win by avoiding mask reads.
- Both `sorted_order` (unstable) and `stable_sorted_order`, and both ascending
  and descending, since the comparator paths differ.

Report for each variant versus A: geometric mean over original cases, geometric
mean over added cases, the single worst regression, and peak memory on the
largest case. Expected outcomes to confirm or refute, not assumptions:

- B improves only on the 8-11 byte shared-prefix cases and is flat or slightly
  slower elsewhere.
- C improves only on nullable cases and is flat elsewhere.
- A versus P shows whether the struct key helps at all. If A is not faster than P,
  B and C are not worth pursuing in this form.

### Decision rule

Keep a variant only if it improves its target cases by a margin outside
run-to-run noise (use NVBench's reported noise; rerun if the noise exceeds 1%)
without regressing the original cases. If B and C each help their own cases,
a combined layout is not possible in 16 B with 12 B of prefix; the combination
would need a 24 B key or a null flag packed into the length-clamped field, which
is a separate experiment.

## Results

Measured 2026-10-06 on an RTX 3070 Ti (8 GB) under WSL. Code: PR head
`338b29de0d` merged with upstream/main, plus the variants in
`sort_column_impl.cuh`, selected at runtime with `CUDF_STRING_SORT_VARIANT`
(0 = upstream comparator, 1 = P, 2 = A, 3 = B, 4 = C). This runtime switch is
for evaluation only.

The A, B and C keys are sorted in place with `DeviceMergeSort::{Stable,}SortKeys`,
then `.row` is projected into `indices`. They use the same `memcpy` + `byteswap`
extraction and `size <= cached_bytes` length rule as the PR head. All five
configurations pass `SORT_TEST` (1627 tests). That count includes an added
randomized differential test around the 8- and 12-byte boundaries, with
embedded NULs and nulls, in both orders, stable and unstable.

Protocol:
- 20 samples per case:
  `--min-samples 20 --stopping-criterion stdrel --max-noise 100 --min-time 0`.
- 4 rounds with the variants interleaved. Each case uses the median across
  rounds of NVBench's mean GPU time.
- Percentages are `1 - geomean(t / t_ref)`, the same formula as the PR description.
- The 40 cases are `sorted_order_strings` (20, "original") plus
  `sorted_order_strings_distribution` and `sorted_order_strings_nulls`
  (20, "new"), matching the PR's comparison.
- Median per-case relative stdev is 6-17%, so differences under about 10% on a
  single case are not meaningful.

### Against upstream

| Variant | 40-case | Original | New |
|---------|--------:|---------:|----:|
| P       | 62.12%  | 27.36%   | 80.24% |
| A       | 69.97%  | 54.75%   | 80.07% |
| B       | 72.00%  | 57.91%   | 81.37% |
| C       | 72.74%  | 56.47%   | 82.92% |

P reproduces the PR's figures (H100 57.64 / 20.42 / 77.45, V100
62.86 / 27.13 / 81.08).

### Against P

| Variant | 40-case | Original | New | New, excluding fast-path cases |
|---------|--------:|---------:|----:|-------------------------------:|
| A       | 20.72%  | 37.71%   | -0.90% | 10.54% |
| B       | 26.08%  | 42.06%   | 5.71%  | 12.24% |
| C       | 28.03%  | 40.07%   | 13.56% | 11.26% |

Five of the new cases take the early exit (all 100%-null cases, plus
`cardinality_1_width_32` at 262144 rows). They run in tens of microseconds and
execute the same code in every variant, so their differences are noise. The
last column leaves them out.

### Per-variant findings

- A versus P: the struct key helps, so the gather was a bottleneck.
  - About 2.4x faster on `max_width=8` at 2M and 16M rows.
  - About 1.9x faster on `max_width=32` at 2M and 16M rows.
  - 3.3x faster on every `num_rows=32768` case with width 32 or more.
- B versus A: wins its target, the PR's `sorted_order_strings_prefixes` with
  `prefix_width=8, prefix_cardinality=1` (all rows share exactly 8 bytes):
  5.97 ms against 49.4 ms at 2M rows. Contrary to the expectation, it is also
  about 7% faster than A on the original cases instead of flat.
- C versus A: wins its target. It is 10-12% faster at 50% nulls with 2M rows and
  flat at 262144 rows. It is also about 4% faster on the original cases.

### Regressions

All struct-key variants are slower than P on heavy-tie inputs:

| Case | P | A | B | C |
|------|--:|--:|--:|--:|
| `shared_prefix_64`, 262144 rows | 54.1 ms | 166.2 ms | 152.8 ms | 163.3 ms |
| `cardinality_64_width_128`, 262144 rows | 12.9 ms | 18.8 ms | 16.9 ms | 18.3 ms |
| `cardinality_64_width_128`, 2M rows | 101.0 ms | 127.9 ms | 116.2 ms | 124.8 ms |

Cause: CUB's merge-sort tuning scales items per thread by key size
(`nominal_4B_items_to_items(17, key_size)`).
- 16 B keys get 4 items per thread, against 17 for P's 4 B row ids. Tiles are
  about 4x smaller.
- The sort therefore needs more merge passes: 8 against 6.
- The merge-path search costs more comparator calls per item.
- When nearly every comparison falls through to a 64-byte string compare, that
  overhead dominates.

Nsight Systems on `shared_prefix_64` at 262144 rows shows the merge kernels at
111.6 ms over 8 launches for A, against 18.2 ms over 6 launches for P.

Tuning partly fixes this. A with 8 items per thread
(`cuda::execution::tune` with a custom `MergeSortPolicy`, selected by
`CUDF_STRING_SORT_IPT=8`, 2 rounds only):
- 72.31% / 55.58% / 82.74% faster than upstream (40-case / original / new);
- 26.90% faster than P overall, 33.15% with the fast-path cases excluded;
- `shared_prefix_64` at 262144 rows: 96 ms, still 1.8x P;
- `shared_prefix_64` at 2M rows: 971 ms against P's 1728 ms.

With 11 items per thread, the original cases are about 4% slower than A's
default tuning. That is likely occupancy: 256 threads x 11 items x 16 B is about
45 KB of shared memory per block. B and C were not tuned.

### Memory

Peak memory for `sorted_order_strings` at 16M rows, `max_width=256`:

| Upstream | P | A, B, C |
|---------:|--:|--------:|
| 128 MiB | 256 MiB | 576 MiB |

That is about 36 B/row for the struct-key variants: 16 for the keys, 16 for
CUB's temporary key buffer, and 4 for the output indices.

### Decision-rule outcome

B and C each improve their target cases by more than the noise, and A beats P
on the original cases. No variant meets "without regressing", because of the
heavy-tie cases above and the 2.25x memory relative to P.

Next steps:
- Tune items per thread for B and C, and check whether the heavy-tie gap closes.
- Test a combined B + C layout, which needs a 24 B key or a packed null flag.
- Weigh the memory cost against the gain on typical data.

## Variant D: packed 8-byte key

D packs everything into one `uint64_t`:
- bits 63-32: a 4-byte big-endian prefix;
- bit 31: the null flag;
- bits 30-0: the row index, masked with `0x7FFF'FFFF`.

Row indices fit in 31 bits because `size_type` is a signed 32-bit integer.

The comparator works in three steps:
1. For nullable inputs, check bit 31 on both keys, so the null mask is not read during the sort.
2. Compare the 4-byte prefixes.
3. On a tie, fall back to the shared tie-break, which resumes at byte 4 with a
   `size <= 4` length rule.

The extractor uses the same byte loop as main. For an 8-byte key, CUB's default
tuning gives 8 items per thread (`17 * 4 / 8`). That is the same tile shape as
"A ipt8", with half the shared memory.

Code: branch `string-sort-prefix-variants` (commit `a35037ddf2`, on upstream/main
`7032a7b2c1`, which includes the merged PR). Select D with
`CUDF_STRING_SORT_VARIANT=5`. All variants pass `SORT_TEST` (1628 tests), which
includes a new 4-byte boundary test with zero-padded ties.

All variants were remeasured on this base with the same protocol as above:
4 interleaved rounds, RTX 3070 Ti. The table above is from the PR head, so the
numbers here differ from it slightly.

### Against upstream

| Variant | 40-case | Original | New |
|---------|--------:|---------:|----:|
| P (main) | 61.20% | 27.04% | 79.37% |
| A        | 71.10% | 56.17% | 80.95% |
| B        | 71.91% | 56.30% | 81.94% |
| C        | 71.63% | 55.71% | 81.83% |
| D        | 74.83% | 51.99% | 86.81% |
| A ipt8   | 72.87% | 55.64% | 83.41% |

### Against P

| Variant | 40-case | Original | New | New, excluding fast-path cases |
|---------|--------:|---------:|----:|-------------------------------:|
| A       | 25.51% | 39.93% | 7.64%  | 2.65%  |
| B       | 27.59% | 40.10% | 12.47% | 7.96%  |
| C       | 26.88% | 39.30% | 11.92% | 2.92%  |
| D       | 35.12% | 34.20% | 36.03% | 40.60% |
| A ipt8  | 30.08% | 39.20% | 19.58% | 23.58% |

### Selected cases

Times in ms, median of 4 rounds.

| Case | P | A | C | D | A ipt8 |
|------|--:|--:|--:|--:|-------:|
| `nulls` 50%, fixed_8, 2M | 8.37 | 5.68 | 5.23 | 3.31 | 5.65 |
| `nulls` 50%, variable_128, 262144 | 0.83 | 0.82 | 0.72 | 0.44 | 0.88 |
| `nulls` 0%, fixed_8, 262144 | 0.73 | 0.88 | 1.02 | 0.46 | 0.69 |
| `variable_128`, 2M | 11.80 | 5.74 | 5.64 | 4.25 | 5.09 |
| `sorted_order_strings` w=8, 2M | 14.66 | 6.35 | 6.19 | 5.97 | 6.41 |
| `sorted_order_strings` w=32, 16M | 256.8 | 137.6 | 137.8 | 170.8 | 134.8 |
| `sorted_order_strings` w=256, 262144 | 22.04 | 23.31 | 23.15 | 16.31 | 17.44 |
| `cardinality_64_width_128`, 262144 | 12.30 | 18.24 | 18.12 | 12.26 | 12.09 |
| `cardinality_64_width_128`, 2M | 96.56 | 122.80 | 122.82 | 115.83 | 69.91 |
| `shared_prefix_64`, 262144 | 51.78 | 160.04 | 159.95 | 129.60 | 94.43 |
| `prefixes` width 8, 2M | 65.95 | 47.58 | 47.66 | 109.25 | 35.69 |

### Findings

- D has the best 40-case and new-case geomeans of any variant. It is 35% faster
  than P overall and 41% faster on the non-trivial new cases. Most of that comes
  from nullable and short or variable-length inputs, where D is 25-45% faster
  than C. C was the previous best on nullable data.
  - Bit 31 avoids null-mask reads, as C's `is_null` field does.
  - The smaller key moves half the data.
  - The key size also gets a larger default tile.
- D is weaker than A, B and C on the original cases (34% vs about 40% faster
  than P). Its losses are on wide strings with many rows: `max_width` 32-256 at
  2M and 16M rows is 10-25% slower than A. With only 4 cached bytes, more
  comparisons tie and fall through to the string data.
- D loses badly on its worst case: strings that share 4-8 leading bytes. On
  `prefixes` width 8 it is 1.66x slower than P and 2.3x slower than A at 2M rows.
- D does not regress `cardinality_64_width_128` at 262144 rows (it matches P).
  It is 20% slower than P at 2M rows.
- D still regresses `shared_prefix_64` at 262144 rows (2.5x P). That is better
  than A, B and C (about 3x), but worse than A ipt8 (1.8x).
- Memory: D peaks at 320 MiB at 16M rows x 256 B, about 20 B/row (8 keys,
  8 CUB temp, 4 indices). That is 1.25x P (256 MiB) and much less than the
  16-byte variants (576 MiB).

### Updated decision-rule outcome

No variant beats P on every case. The candidates trade off as follows:

- **D** gives the largest overall gain for the least extra memory. Its weak spots
  are strings sharing 4-8 leading bytes and the heavy-tie case.
- **A ipt8** has the most even profile. Its worst case is a 1.8x regression on
  `shared_prefix_64` at 262144 rows, but it needs 2.25x P's memory.

Worth trying next:
- D with tuned items per thread.
- An unpadded 12-byte key: three `uint32_t` holding an 8-byte prefix and a
  row/null word, with 4-byte alignment. It would keep 8 cached bytes at 12 B/key.
  Padded to 16 B, this is just C.
