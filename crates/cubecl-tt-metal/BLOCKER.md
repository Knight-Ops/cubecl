# TT-Metal Stream Promotion Notes

This document tracks the remaining stream-specific follow-up after broad stream
promotion. The broad stream issue is no longer a timeout blocker or a pending
promotion task.

## Current Status

What is green today:

- `tests::cubecl_core_wrappers::stream::test_stream_small`
- `tests::cubecl_core_wrappers::stream::test_stream_medium`
- `tests::cubecl_core_wrappers::stream::test_stream_broad`
- `tests::cubecl_core_wrappers::stream::test_stream_broad_input_page_probe`
- `tests::cubetask_compile_pipeline`
- `tests::cubecl_core_wrappers::stream::test_stream_broad_single_round_manual`
- `tests::cubecl_core_wrappers::stream::test_stream_broad_full_chain_manual`
- `tests::tt_program_cache_populates_for_cubetask_pipeline`

What that means:

- Cross-stream ownership and resource-lineage handling are working for the live
  TT stream wrappers.
- The staged-input circular-buffer contract that broke the broad case has been
  repaired and now has an always-on regression.
- The upstream-sized broad workload is now green in the live TT lane.
- The full 300-round broad chain still completes on hardware in manual mode.

What is still pending:

- hardening and regression safety for the promoted broad stream path
- any further runtime optimization only if downstream workloads still need it

## Why Stream Still Deserves Focus

The broad stream case is still a meaningful stress shape because it combines:

- two logical streams
- 300 dependent launches in one chain
- a large staged input
- heavy generic scalar work in every launch

What still needs to stay true:

- correctness stays green in the always-on lane
- the runtime path remains stable under the broad chained workload
- the broad case does not regress back into special-case manual handling

## Remaining Work

### 1. Keep the Promoted Broad Workload Stable

- Keep the broad upstream `stream::test_stream` workload green in the live TT
  lane.
- Keep the reduced and medium wrappers green alongside it.

### 2. Keep the Runtime Split Honest

The backend still needs two distinct paths:

- `kernel()` for direct immediate low-level characterization
- `kernel_cube()` for the normal queued CubeTask runtime path

The broad stream promotion must not blur those paths together again.

### 3. Harden the Repaired Staged-Input Contract

- Keep the staged-input page probe green in the live lane so the repaired
  contiguous staged-input CB contract cannot regress silently.

### 4. Decide Whether More Optimization Is Necessary

If the broad case is correct but still too costly for the live lane, only then
revisit:

- queued/submitted/completed stream batching thresholds
- launch/setup churn in `src/compute/context.rs`
- runtime command path overhead in `src/compute/command.rs`
- additional program-cache reuse

## Acceptance Gate

The stream work is in a good resting state when all of the following are true:

- `test_stream_small` is green
- `test_stream_medium` is green
- the broad upstream-sized stream workload is green in the normal TT lane
- the manual broad probes remain available for targeted characterization

## Useful Commands

Run the live TT lane:

```bash
TT_METAL_RUN_HARDWARE_TESTS=1 \
LD_LIBRARY_PATH=/usr/local/lib \
cargo test -p cubecl-tt-metal -- --test-threads=1
```

Run the manual broad full chain:

```bash
TT_METAL_RUN_HARDWARE_TESTS=1 \
LD_LIBRARY_PATH=/usr/local/lib \
cargo test -p cubecl-tt-metal \
tests::cubecl_core_wrappers::stream::test_stream_broad_full_chain_manual \
-- --ignored --exact --nocapture --test-threads=1
```
