# TODO — Remaining Roadmap

This file is the short-form roadmap for the remaining TT-Metal backend work.
It tracks what still needs to happen after the current hardware-green baseline.

## Current Baseline

- `cubecl-tt-metal` is green on hardware in its current live lane.
- The current test inventory is `301` total tests.
- The current hardware result is `297 passed`, `0 failed`, `4 ignored`.
- The broad stream workload is now in the live TT lane, and the broad manual
  probes still pass as characterization hooks.
- `cubecl_std::tensor_identity` now uses the real upstream kernel for the
  proven `f32`/`u32` TT slice; broader 2D tensor/topology parity still remains
  to be widened explicitly beyond that validated path.

## Highest-Priority Next Steps

### 1. Finish the Truthful Single-Device Contract

Files most likely involved:

- `src/compute/context.rs`
- `src/compute/command.rs`
- `src/compute/stream.rs`
- `src/lib.rs`
- `../cubecl-cpp/src/tt_metal/dialect.rs`
- `../cubecl-cpp/src/tt_metal/reader.rs`
- `../cubecl-cpp/src/tt_metal/writer.rs`

Deliverable:

- broader topology and tensor semantics beyond the current decomposition model
- broader shared-memory and synchronization semantics
- wider BF16/F32 vector widths beyond `vec16`
- broader atomic families beyond the current generated suite
- true upstream wrapper promotion for the remaining TT-local standard wrappers,
  especially wider `quantized_view` coverage and TT block-float formats
- `Bfp2_b`

Deliverable:

- every supported category is backed by focused TT regressions and a truthful
  hardware-verified capability contract

### 2. Resolve Multi-Device and Subgroup Work

Areas:

- real multi-device `all_reduce`
- broader plane/subgroup synchronization and collective semantics
- `to_client` expectations with more than one visible device

Deliverable:

- either explicit hardware-validated support or explicit continued exclusion

### 3. Add Burn Downstream Validation

Deliverable:

- a tiny Burn TT smoke covering allocation, transfer, forward, backward, and
  one optimizer step

### 4. Only Then Start Multi-Core and Performance Work

Areas:

- multi-core scheduling around the CPU-style launch model
- per-core runtime args and sharded storage where needed
- performance work guided by real workloads instead of guesswork

## Still Out of Scope for the Live Contract

Until underlying capability exists, keep these explicitly outside the claimed
surface:

- `tensormap`
- `cluster`
- `cmma`

## Verification Commands

Full TT lane:

```bash
TT_METAL_RUN_HARDWARE_TESTS=1 \
LD_LIBRARY_PATH=/usr/local/lib \
cargo test -p cubecl-tt-metal -- --test-threads=1
```

List current inventory:

```bash
cargo test -p cubecl-tt-metal -- --list
```

Manual broad stream chain:

```bash
TT_METAL_RUN_HARDWARE_TESTS=1 \
LD_LIBRARY_PATH=/usr/local/lib \
cargo test -p cubecl-tt-metal \
tests::cubecl_core_wrappers::stream::test_stream_broad_full_chain_manual \
-- --ignored --exact --nocapture --test-threads=1
```
