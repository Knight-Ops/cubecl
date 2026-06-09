# CubeCL TT-Metal Checklist

This checklist is the live execution board for the remaining TT-Metal backend
work. It intentionally omits already-completed bring-up tasks so it stays
focused on what still needs to happen to claim the goal honestly.

## Verified Baseline

- The hardware-gated TT lane is green end to end.
- `cargo test -p cubecl-tt-metal -- --list` currently reports `301` tests.
- The current hardware result is `297 passed`, `0 failed`, `4 ignored`.
- The ignored/manual tests are:
  - `tests::cubecl_core_wrappers::stream::test_stream_broad_single_round_manual`
  - `tests::cubecl_core_wrappers::stream::test_stream_broad_full_chain_manual`
  - `tests::cubecl_core_wrappers::stream::test_stream_broad_single_round_diagnostic_manual`
  - `tests::tt_program_cache_populates_for_cubetask_pipeline`
- The live TT lane already covers a large `cubecl_std` and `cubecl_core`
  subset, including the current topology, tensor-coordinate, plane, barrier,
  atomic, minifloat, unary, binary, vector, slice, sequence, and launch
  surfaces documented in `PHASES.md`.
- The live TT lane now also includes the upstream-sized broad stream workload:
  - `tests::cubecl_core_wrappers::stream::test_stream_broad`
- The live TT lane now also includes the staged-input regression that caught
  the broad stream layout bug:
  - `tests::cubecl_core_wrappers::stream::test_stream_broad_input_page_probe`
- `cubecl_std::tensor_identity` now runs through the real upstream kernel for
  the proven `f32`/`u32` TT slice, while broader 2D tensor/topology parity
  still needs explicit validation beyond that identity-backed path.

## Goal Definition

Treat "full CubeCL test suite" as complete only when all of the following are
true:

- The remaining enabled TT-local wrapper surface is either promoted, kept
  TT-local with an explicit reason, or declared out of scope with a truthful
  capability contract.
- The backend has a tiny downstream Burn smoke so we know the current green
  CubeCL subset translates into real model execution.

## Immediate Remaining Work

### 1. Stream Hardening

- [ ] Keep the low-level immediate `kernel()` characterization path separate
  from the queued `kernel_cube()` runtime path.
- [ ] Decide whether any additional runtime optimization is still needed for
  downstream workloads now that the broad case is in the live lane.

### 2. Truthful Single-Device Parity Expansion

- [ ] Decide how far to widen topology and tensor semantics beyond the current
  verified decomposition model.
- [ ] Decide how far to widen shared-memory and synchronization semantics beyond
  the current proven shared-scratch subset.
- [ ] Resolve whether broader vector widths beyond `vec16` are real backend
  support or should remain outside the contract.
- [ ] Resolve whether broader atomic op/type/vector families are real backend
  support or should remain outside the contract.
- [ ] Resolve true upstream wrapper promotion for TT block-float formats instead
  of relying only on TT-local logical-`f32` wrappers.
- [ ] Resolve `Bfp2_b` host/device tilize support or keep it explicitly blocked.
- [ ] Keep every capability claim aligned with real hardware validation rather
  than inherited generic C++ feature registration.

### 3. Multi-Device and Subgroup Semantics

- [ ] Validate real multi-device `all_reduce` semantics on 2+ TT devices.
- [ ] Decide whether broader plane/subgroup synchronization and collective
  semantics are implementable in the current backend model.
- [ ] Keep `to_client` and any collective-facing surface truthful when only one
  device is available.

### 4. Capability-Lane Decisions

- [ ] Leave `tensormap`, `cluster`, and `cmma` disabled until their underlying
  runtime/compiler capability exists.
- [ ] Document each parity-lane category as either "future implementation",
  "requires hardware/runtime capability we do not yet have", or "out of scope".

### 5. Burn Downstream Validation

- [ ] Add a tiny Burn TT smoke outside this repo.
- [ ] Validate allocation, transfer, simple forward, tiny backward, and one
  optimizer step.
- [ ] Classify each Burn failure as a backend gap, Burn integration gap, or
  environment issue.
- [ ] Backfill a reduced TT regression in this repo for every backend gap when
  practical.

### 6. Multi-Core and Performance

- [ ] Do not start multi-core bring-up until the Burn smoke exists.
- [ ] Once the Burn smoke is green, design multi-core scheduling around the
  CPU-style launch model:
  - `CubeDim` is real runtime concurrency
  - `CubeCount` is scheduled work
  - generated TT kernels stay sequential
- [ ] Treat performance work as a follow-on to correctness, stream hardening,
  and downstream validation rather than as a substitute for them.

## Documentation and Verification Discipline

- [ ] Keep `PHASES.md`, `BLOCKER.md`, `TODO.md`, and the TT harness summary in
  `src/lib.rs` aligned with the real hardware-verified surface.
- [ ] Keep `cargo test -p cubecl-tt-metal -- --list` aligned with the docs after
  every wrapper promotion or retirement.
- [ ] Add a focused TT regression before promoting any new generated category or
  widening any capability contract.
- [ ] Re-run the full hardware lane after every substantial runtime/compiler
  change:

```bash
TT_METAL_RUN_HARDWARE_TESTS=1 \
LD_LIBRARY_PATH=/usr/local/lib \
cargo test -p cubecl-tt-metal -- --test-threads=1
```
