# CubeCL TT-Metal Implementation Phases

This document is the strategic roadmap for finishing the TT-Metal backend. It
tracks what is verifiably green today, what remains manual-only, and the
smallest honest sequence of work needed to complete the goal.

## Current Status

What is true today:

- The TT backend builds through `cubecl` with `--features tt_metal`.
- The live hardware-gated TT suite is green end to end.
- `cargo test -p cubecl-tt-metal -- --list` currently reports `301` tests.
- The current hardware result is `297 passed`, `0 failed`, `4 ignored`.
- The current TT lane already covers a large single-device subset of
  `cubecl_std` and `cubecl_core::runtime_tests`.
- The broad stream stress shape is green both in the live TT lane and in the
  manual characterization hooks.
- The staged-input broad-stream regression is green in the live TT lane.
- The topology lane is green in the full crate run, including
  `test_axis_components_linearized_u32` and `_u64`.

What is not true yet:

- The TT backend is not yet an honest "full CubeCL parity" backend.
- Broader shared-memory, subgroup, multi-device collective, tensormap,
  cluster, and CMMA semantics are still outside the proven contract.
- Burn downstream validation has not happened yet.
- Multi-core and performance work are still follow-on tasks, not current
  acceptance criteria.

## Launch-Model Guardrails

The remaining work should keep following the upstream launch guidance:

- `CubeDim` is runtime concurrency, not a fake GPU grid.
- `CubeCount` is scheduled work that the runtime can inject into sequential
  loops.
- Generated TT kernel bodies should stay sequential.
- Vectorization should track real TT SIMD/tile behavior, not substitute for
  launch geometry.

These rules matter most for the remaining stream, topology, tensor,
shared-memory, subgroup, and future multi-core work.

## Verified Surface

### `cubecl_std`

The current TT lane has honest hardware-green coverage for:

- `event`
- `reinterpret_slice`
- upstream `tensor_identity` for the proven `f32`/`u32` TT slice
- `trigonometry`
- TT-local `quantized_view`

Still not claimed:

- broader 2D tensor/topology parity beyond the validated identity-backed slice
- broader `quantized_view` layout/output-type coverage

### `cubecl_core::runtime_tests`

The live TT lane is green for the current documented subset, including:

- `all_reduce` single-device identity semantics
- `assign`
- `atomic` through the current single-unit generated suite
- `barrier` through the proven shared-scratch subset
- `binary_untyped::mulhi`
- `branch`
- `comparison`
- `const_match`
- `constants`
- `debug` helper-call subset
- `different_rank`
- `enums`
- `file`
- `index`
- `launch`
- `launch_untyped` dynamic-addressing subset
- `metadata`
- `minifloat` TT-local subset
- `numeric`
- `plane` through the current upstream suite plus TT-local regression coverage
- `properties`
- `saturating`
- `sequence`
- `slice`
- `stream` reduced and medium wrappers
- `stream` upstream-sized broad workload
- `synchronization` current subset
- `tensor` current `tensor_coordinate` subset
- TT-local `topology`
- `to_client` opportunistic path
- `unary_int`
- `unary` scalar and vectorized BF16/F32 subset through `vec16`
- `binary` scalar and vectorized BF16/F32 subset through `vec16`
- `unroll`
- `vector` current subset, including the narrow shared-memory case

## Manual-Only Green Surface

These are intentionally not in the always-on lane yet:

- `tests::cubecl_core_wrappers::stream::test_stream_broad_single_round_manual`
- `tests::cubecl_core_wrappers::stream::test_stream_broad_full_chain_manual`
- `tests::cubecl_core_wrappers::stream::test_stream_broad_single_round_diagnostic_manual`
- `tests::tt_program_cache_populates_for_cubetask_pipeline`

Interpretation:

- These hooks are no longer standing in for missing coverage.
- They remain as characterization and regression probes around the now-green
  live broad stream path.

## Remaining Phases

### Phase A: Stream Hardening and Regression Safety

Goal:
Keep the now-promoted broad stream path stable while preserving the repaired
staged-input contract and the split between immediate low-level launches and
the queued runtime path.

Why this still matters:

- The broad case is the heaviest current generic runtime stress shape in the
  live TT lane.
- The staged-input CB contract was previously wrong and should not regress.
- The `kernel()` versus `kernel_cube()` path split is still important to keep
  low-level characterization safe.

Acceptance gate:

- The reduced, medium, and broad stream wrappers remain green in the full TT
  lane.
- The staged-input regression remains green in the live TT lane.
- No low-level immediate `kernel()` characterization path regresses.

### Phase B: Finish Truthful Single-Device Parity

Goal:
Decide what the truthful single-device TT backend contract actually is beyond
the current large green subset.

This phase includes:

- topology and tensor semantics beyond the current decomposition model
- shared-memory semantics beyond the current shared-scratch subset
- synchronization semantics beyond the current proven subset
- vector widths beyond `vec16`
- wider atomic op/type/vector families
- true upstream wrapper promotion for TT block-float formats
- `Bfp2_b` support or explicit continued exclusion

Acceptance gate:

- Every widened category has focused TT regressions.
- Every supported capability matches real hardware behavior.
- Every unsupported category stays explicitly documented as such.

### Phase C: Multi-Device and Subgroup Semantics

Goal:
Resolve the categories that cannot be honestly closed on a single-device,
single-core contract alone.

This phase includes:

- real multi-device `all_reduce`
- broader plane/subgroup synchronization and collective semantics
- truthful `to_client` expectations when more than one device is visible

Acceptance gate:

- Either the capability is hardware-validated on 2+ devices, or it stays out of
  the supported contract.

### Phase D: Burn Downstream Smoke

Goal:
Prove the current CubeCL subset is sufficient for real downstream execution.

This phase includes:

- tensor allocation and transfer
- simple forward execution
- tiny backward/autograd
- one optimizer step

Acceptance gate:

- A tiny Burn TT smoke passes on hardware.
- Remaining failures are categorized as backend gaps, Burn integration gaps, or
  environment issues.

### Phase E: Multi-Core and Performance

Goal:
Scale only after the backend is trustworthy in the single-device, single-core
model and after Burn smoke exists.

This phase includes:

- multi-core scheduling based on the CPU-style launch model
- sharded buffers and per-core runtime args where required
- performance work guided by real workloads rather than synthetic optimism

Acceptance gate:

- Multi-core execution has dedicated tests.
- Single-core behavior remains green.

## Parity-Lane Categories

These stay explicitly outside the live contract until their underlying runtime
or compiler capability exists:

- `tensormap`
- `cluster`
- `cmma`

## Practical Meaning of "Done"

For this backend, "done" does not mean "every upstream macro is uncommented."
It means:

- the broad stream path is in the normal lane
- the remaining supported categories are backed by truthful capability claims
- the unsupported categories are explicit rather than accidental
- the backend passes a downstream Burn smoke on real hardware
