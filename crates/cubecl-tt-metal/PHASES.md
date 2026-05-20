# CubeCL TT-Metal Implementation Phases

This document turns the current TT-Metal roadmap into a practical execution plan.
It is organized for correctness-first backend work, with Burn training as the downstream acceptance target.

## Current Baseline

What is true today:
- The `tt_metal` feature is wired through `cubecl` and builds.
- The TT backend can run a narrow single-core tiled subset.
- TT compile-time args and launch metadata now flow from real `MeshBuffer` metadata and logical resource sizes instead of hardcoded values.
- TT source generation supports a narrow generic single-core scalar path for bounds-checked elementwise kernels and global reinterpret-style kernels, in addition to the existing copy/add path.
- Unsupported Phase 1 kernels now fail during compilation with explicit `CompilationError::UnsupportedInstruction` errors.
- The hand-written TT backend tests, focused Phase 1 source-generation regressions, and new Phase 2 runtime-hardening regressions are green.
- The `cubecl_std` `trigonometry`, global `reinterpret_slice`, and current `event` coverage now run through TT.

What is not true yet:
- The TT dialect is not a general CubeCL backend.
- Shared-memory reinterpret and range loops remain unsupported.
- Dynamic tensor metadata now works for the current single-core generic TT path, but broader SIMT-style metadata patterns are still intentionally gated.
- Broader generated `cubecl_std` and `cubecl_core` coverage is still intentionally gated rather than broadly enabled.
- A targeted TT-local `cubecl_core` subset is enabled and green on hardware, including numeric define coverage, sliced-array scalar indexing, scalar-kernel-argument branching, the vector `assign` case needed by the current harness, launch-untyped dynamic addressing, TT helper-call debug coverage, and opportunistic `to_client` coverage.
- Burn training has not been validated downstream.
- Multi-core/sharding/performance work is still out of scope.

## Shared Support Matrix

Keep this matrix aligned with the TT harness comment in `crates/cubecl-tt-metal/src/lib.rs` and with `cargo test -p cubecl-tt-metal -- --list`.

### `cubecl_std`

| Category | Status | Blocker | Next action |
|---|---|---|---|
| `event` | enabled | none in current subset | keep green in isolation and in the full TT lane |
| `reinterpret_slice` | enabled | global path only; shared-memory reinterpret still unsupported | keep green and route new aliasing bugs into TT regressions |
| `tensor_identity` | bring-up | the upstream kernel launches with 2D cube dimensions while TT still only models a single execution axis | revisit after TT supports non-1D cube dimensions or a TT-safe identity kernel path exists |
| `trigonometry` | enabled | broader math surface is still queued | keep the current pair green and use failures to drive compiler math support |
| `quantized_view` | bring-up | quantized values, scales, and view metadata have not been validated end-to-end on TT | characterize the int path first, then the fp4 path, before adding TT wrappers |

### `cubecl_core::runtime_tests`

| Category | Status | Blocker | Next action |
|---|---|---|---|
| `all_reduce` | parity lane | plane/subgroup semantics are not modeled by the current single-core TT execution shape | defer until plane/sync semantics are real on TT |
| `assign` | enabled | loop-heavy variants still depend on broader control-flow lowering | keep current wrappers green and add any remaining fixed regressions |
| `atomic` | parity lane | TT atomic/runtime feature support is not established for the generated matrix | defer until hardware feature support is explicit and testable |
| `barrier` | parity lane | shared/local-memory and barrier semantics are outside the current subset | defer until shared-memory execution is supported |
| `binary` | partially enabled | only the untyped `mulhi` subset is exposed through CubeCL wrappers; TT-native direct `add`/`sub`/`mul` kernels are characterized but not yet safe for ordinary row-major wrappers | keep `mulhi` green, keep the native direct regressions green, and add a row-major/tile-layout bridge before promoting float wrapper coverage |
| `branch` | partially enabled | switch and loop cases still depend on broader control-flow lowering | keep `select` green, then add switch cases one by one |
| `cluster` | parity lane | cluster/distributed semantics are out of scope for single-core TT | revisit only after multi-core/distributed execution exists |
| `cmma` | parity lane | WMMA/CMMA support is intentionally out of scope | defer until a real TT matmul path is required |
| `comparison` | enabled | none in current scalar subset | keep green and expand only if new comparison shapes are needed |
| `const_match` | enabled in TT-local wrappers | needs isolated TT hardware confirmation before broader promotion | keep the wrapper green and route any comptime-match bugs into TT regressions |
| `constants` | enabled | none in current subset | keep green in isolation and in the full TT lane |
| `debug` | enabled in TT-local helper-call subset | `debug_print` output handling is still uncharacterized on TT | keep simple/nested helper-call tests green, then evaluate `debug_print` separately |
| `different_rank` | enabled | none in current metadata subset | keep green and route metadata regressions back into TT tests |
| `enums` | enabled in TT-local wrappers | needs isolated TT hardware confirmation before broader promotion | keep the wrappers green and narrow failures by enum shape if needed |
| `file` | enabled in TT-local wrappers | file-backed/runtime-side behavior still needs isolated TT hardware confirmation | keep the wrapper green and treat failures as runtime-integration bugs first |
| `index` | partially enabled | `test_kernel_shuffle` requires local-array/shuffle behavior | keep `test_assign_index` green and leave shuffle in the parity lane |
| `launch` | enabled | error-path cases stay upstream-ignored; TT only wraps green cases | keep the existing basic launch wrappers green |
| `launch_untyped` | enabled in TT-local dynamic-addressing subset | shared-memory and max-unit error-path cases are still upstream-ignored | keep dynamic-addressing wrappers green and revisit error-path coverage later |
| `metadata` | enabled | broader SIMT metadata patterns are still outside scope | keep current shape/stride/len coverage green |
| `minifloat` | enabled in TT-local feature-gated wrappers | meaningful coverage still depends on TT advertising the needed conversion features | keep the wrappers green on hardware when features are present and treat skips as capability reporting |
| `numeric` | enabled | none in the current define-oriented subset | keep green and backfill TT regressions for any define-specific bug |
| `plane` | parity lane | plane operations require subgroup semantics that TT does not expose yet | defer until plane support is real |
| `properties` | enabled | none in current subset | keep green as the runtime capability baseline |
| `saturating` | enabled in TT-local `i32`/`u32` wrappers | needs isolated TT hardware confirmation on integer lowering | keep the narrow wrappers green before widening integer coverage further |
| `sequence` | queued behind control-flow | range-loop lowering is still intentionally unsupported | revisit after loop support is implemented |
| `slice` | partially enabled | `slice_for` still depends on range-loop lowering | keep select/len/mut-assign/mut-len green and leave the loop case isolated until loops are supported |
| `stream` | parity lane | longer-lived stream semantics are not yet validated beyond focused TT regressions | revisit only after ordinary test-failure teardown is fully stable |
| `synchronization` | parity lane | 2D launch and plane sync semantics are outside the current execution model | defer until TT exposes those semantics explicitly |
| `tensor` | parity lane | `test_tensor_coordinate` requires 2D cube dimensions | revisit only after non-1D launch geometry is supported |
| `tensormap` | parity lane | tensor maps are explicitly outside the current TT subset | defer until tensor-map support is real |
| `to_client` | enabled opportunistically | meaningful coverage needs more than one visible device | keep the wrapper present; it no-ops when only one device is visible |
| `topology` | queued | multi-device/runtime-topology behavior is not yet characterized on TT | evaluate after `to_client` and other simple runtime-facing tests |
| `traits` | helper only | this is support code, not a backend category | no TT harness action needed |
| `unary` | partially enabled | only the integer abs/bit-operation subset is wired today; TT-native direct `sqrt` is characterized but ordinary float wrappers still lower through a row-major/scalar path that is not ready for auto-promotion | keep the unary-int subset green, keep the native direct `sqrt` regression green, and only widen float unary wrappers after the row-major/tile-layout bridge is real |
| `unroll` | queued behind control-flow | unrolled loop lowering needs dedicated validation | revisit after the simpler control-flow wave is green |
| `vector` | partially enabled | loop-unroll and shared-memory vector cases are still outside the current TT subset | keep index/index-assign/conditional/comparison wrappers green and leave the rest isolated |

## Phase 1: Broaden Single-Core Compute Coverage

Goal:
Replace copy/add-only source selection with a narrow but real single-core CubeCL subset that can compile bounds-checked elementwise and global reinterpret-style kernels.

Primary files:
- `crates/cubecl-cpp/src/tt_metal/dialect.rs`
- `crates/cubecl-cpp/src/tt_metal/compile.rs`
- `crates/cubecl-cpp/src/tt_metal/reader.rs`
- `crates/cubecl-cpp/src/tt_metal/writer.rs`
- `crates/cubecl-cpp/src/shared/*` as reference only

Work items:
- Analyze compiled kernels structurally in `tt_metal/compile.rs` instead of detecting only copy/add.
- Keep the Phase 1 supported subset explicitly narrow:
  - single-core only
  - no tensor maps
  - no shared-memory or local-array kernels
  - no dynamic metadata
  - no loops/switch/event-heavy IR
- Generate generic TT compute source for supported scalar/tile-safe kernels while keeping the proven copy/add fast paths.
- Current supported instruction families are:
  - global buffer reads/writes and index addressing
  - same-width reinterpret-friendly type handling and bitcasts
  - comparisons, boolean ops, and selects
  - arithmetic currently exercised by trigonometry-style kernels (`Assign`, `Add`, `Sub`, `Mul`)
  - simple `if` / `if-else` bounds checks
- Keep TT lowering explicitly single-core and tile-oriented.
- Reject unsupported IR with `CompilationError::UnsupportedInstruction` instead of falling through to bad codegen.
- Defer shared-memory reinterpret, range loops, dynamic metadata, and event/comptime-heavy patterns to later phases.

Acceptance criteria:
- TT source generation no longer special-cases only copy/add at the API level.
- Supported kernels no longer depend on weak kernel-kind heuristics.
- Unsupported kernels fail cleanly during compilation.
- Remaining follow-up is runtime and generated-suite expansion, not compiler fallthroughs.

Likely failure modes:
- Emitting C++ that looks valid but violates TT compute/dataflow API sequencing.
- Confusing scalar element semantics with TT tile-native semantics.
- Auto-routing ordinary row-major CubeCL wrappers into TT tile-native kernels before the buffer-layout bridge exists.
- Reinterpret kernels exposing layout or padded-buffer mismatches at runtime.
- Compile-time/event-expanded IR shapes slipping past the compile gate when they should still be rejected.

## Phase 2: Runtime Correctness for Real CubeCL Kernels

Goal:
Make launch, memory, metadata, and failure behavior robust enough for generic CubeCL kernels.

Primary files:
- `crates/cubecl-tt-metal/src/compute/context.rs`
- `crates/cubecl-tt-metal/src/compute/command.rs`
- `crates/cubecl-tt-metal/src/compute/server.rs`
- `crates/cubecl-tt-metal/src/compute/storage/gpu.rs`
- `crates/cubecl-tt-metal/src/runtime.rs`

Work items:
- Harden the launch path so every TT compile/build failure stays a Rust-side error.
- Make stream state recovery explicit after launch errors instead of leaving poisoned streams.
- Validate reader/writer/runtime arg ordering against compiled buffer metadata in every path.
- Expand support for metadata-driven addressing used by arrays, slices, and views.
- Revisit allocation/page-size assumptions for non-bf16 cases and small logical buffers.
- Confirm read/write behavior for padded allocations and partial logical tensor sizes.
- Add targeted regressions for data-format mapping, ordered launch preparation, recoverable launch failures, and bad-build conversion into Rust-visible errors.

Acceptance criteria:
- A bad TT kernel build never produces a segfault or TT teardown abort through normal test execution.
- Unsupported runtime patterns fail deterministically and descriptively.
- Multi-input and multi-output kernels use correct TT compile args and runtime args.
- Stream errors are surfaced once and become recoverable after `flush`/`sync` in the synchronous TT backend.

Likely failure modes:
- Stream remains in error state and causes cascading failures on later client operations.
- TT host/device tooling may still emit watcher or teardown noise even when Rust-side error handling recovers correctly.
- Shape/stride metadata does not match TT tile assumptions.
- Page-aligned backing buffers leak into logical tensor semantics.

## Phase 3: Re-enable `cubecl_std` Suites Incrementally

Goal:
Use the generated std suites as the next source of truth, one category at a time.

Primary files:
- `crates/cubecl-tt-metal/src/lib.rs`
- `crates/cubecl-std/src/tests/event.rs`
- `crates/cubecl-std/src/tests/reinterpret_slice.rs`
- `crates/cubecl-std/src/tests/tensor/identity.rs`
- `crates/cubecl-std/src/tests/trigonometry.rs`

Work items:
- Re-enable one ignored TT std suite at a time.
- Completed order so far:
  - `trigonometry`
  - `reinterpret_slice`
  - `event`
- Keep `tensor_identity` in isolated bring-up until TT supports the upstream 2D cube-dimension launch shape.
- Keep `quantized_view` in isolated bring-up until quantized values, scales, and view metadata are validated end-to-end on TT.
- Keep std coverage behind TT-local hardware-gated wrappers in `cubecl-tt-metal/src/lib.rs` instead of uncommenting the broad upstream `testgen!()` macro.
- Add TT-specific regression tests in `cubecl-tt-metal/src/lib.rs` for each bug found.
- Treat future std-suite expansion as category-by-category work, not a one-shot switch flip.
- Phase 3 is functionally complete; the remaining work from this phase was to keep that coverage in TT-local hardware-gated wrappers and sync the docs with the real harness shape.

Acceptance criteria:
- Each std suite is either fully green or explicitly ignored with a precise reason.
- No std suite aborts the process.
- Every fixed failure gets a backend regression test when practical.

Likely failure modes:
- Broader std re-enablement may pull in shared-memory reinterpret or other unsupported IR shapes that the manual TT suite list currently avoids.
- Future reinterpret work is now concentrated in shared-memory and broader aliasing cases rather than the global path.
- Future trigonometry and event work is now concentrated in broader math-surface and comptime/control-flow coverage rather than the currently enabled kernels.

## Phase 4: Enable Targeted `cubecl_core` Runtime Coverage

Goal:
Grow from std-level spot coverage into real backend compatibility coverage without pretending the current TT backend can run the full generated `cubecl_core` matrix.

Primary files:
- `crates/cubecl-tt-metal/src/lib.rs`
- `crates/cubecl-core/src/runtime_tests/*`

Work items:
- Keep `cubecl_core::testgen_all!` disabled and add TT-local hardware-gated wrappers instead.
- Start with a correctness-first scalar mix:
  - `f32` foundation
  - required untyped/runtime-wide coverage
  - `u32` companion coverage where the upstream tests are already comparison-oriented
  - defer dedicated `i32` generated coverage until the first green subset is stable
- The current green TT-local wrapper subset is:
  - launch basics (`with_generics`, `without_generics`, `with_comptime_tag`)
  - untyped launch dynamic addressing (`AddressType::U32`, `AddressType::U64`)
  - properties and constant-array coverage
  - metadata/addressing coverage for both `AddressType::U32` and `AddressType::U64`
  - different-rank tensor behavior
  - `assign` coverage (`assign_scalar`, `add_assign_array`, `add_assign_vector`)
  - `u32` comparisons
  - TT helper-call debug coverage (`debug::test_simple_call`, `debug::test_nested_call`)
  - scalar-kernel-argument branching (`branch::test_select_*`)
  - sliced-array scalar indexing (`index::test_assign_index`)
  - numeric define coverage (`numeric::*`)
  - opportunistic multi-device transfer coverage (`to_client::test_to_client`)
- Keep unsupported or still-failing categories explicitly unenabled in the TT harness with a reason comment instead of relying on broad ignored upstream macros:
  - `index::test_kernel_shuffle`: local array / shuffle behavior outside the current single-core subset
  - `tensor::test_tensor_coordinate`: requires 2D cube dimensions while TT still exposes a single-axis execution shape
- Do not enable broad unary/binary math, switch/loop branches beyond the scalar `select` subset, tensor-coordinate, sequence, plane/sync, atomics, tensor maps, or stream categories in this phase.
- Treat queued categories (`topology`) as isolated wrapper work, not broad macro re-enablement.
- Keep `binary` (`mulhi` only), `const_match`, `enums`, `file`, `minifloat`, the narrow `saturating` subset, the non-loop `slice` subset, the non-loop/non-shared-memory `vector` subset, and the unary-int subset in TT-local wrappers until they are hardware-validated in isolation.

Acceptance criteria:
- The enabled TT-local `cubecl_core` wrappers are green on TT hardware.
- The unsupported categories are intentional, documented, and still off by default.
- The backend can run a meaningful subset of generic CubeCL kernels without bespoke TT-only test bodies.

Likely failure modes:
- CubeCL runtime tests assume semantics closer to SIMT than current TT single-core mapping provides.
- Coarse upstream runtime-test modules bundle still-unsupported loops, switch lowering, local array behavior, or multi-dimensional launch semantics.
- Integer and metadata-heavy paths may diverge from the float-centric std coverage that drove the earlier phases.

## Phase 5: Burn Downstream Validation

Goal:
Prove the backend is useful for real model code, not just local runtime tests.

Primary repos/files:
- Burn repo, not this repo
- Any CubeCL-facing Burn backend glue used by Burn runtime selection

Work items:
- Add a minimal TT smoke workflow in Burn.
- Validate at least:
  - tensor allocation and reads/writes
  - elementwise forward ops
  - autograd/backward pass on a tiny graph
  - optimizer step
  - one tiny model training iteration
- Record every backend gap Burn hits that CubeCL tests did not expose.
- Backfill those gaps into CubeCL TT regressions where appropriate.

Acceptance criteria:
- A small Burn training example runs end-to-end on TT hardware.
- Failures found downstream are translated into actionable CubeCL backend tasks.

Likely failure modes:
- Burn depends on runtime contracts not covered by current CubeCL std/core tests.
- Training immediately exposes dtype/gradient/update paths not exercised by copy/add-centric kernels.
- Error handling that is acceptable in unit tests becomes unacceptable in longer-lived training loops.

## Phase 6: Expand Supported Surface Deliberately

Goal:
Close the gap between "minimal useful backend" and "native backend people can rely on".

Primary areas:
- more arithmetic and math ops
- more view/slice/tensor behavior
- matrix/matmul or WMMA-adjacent paths if needed
- broader dtype support

Work items:
- Use actual failing tests and Burn needs to drive scope.
- Add support category-by-category, not one-off per failing kernel.
- Only add WMMA/matmul-specific work when a real required path proves it is necessary.

Acceptance criteria:
- The supported feature set is documented and growing predictably.
- New features come with generated-test or downstream-test coverage.

Likely failure modes:
- Backend becomes a pile of one-off fixes without a coherent supported subset.
- Matmul-specific work crowds out more foundational runtime correctness work.

## Phase 7: Multi-Core and Performance

Goal:
Move beyond single-core correctness after the backend is already trustworthy.

Primary files:
- `crates/cubecl-tt-metal/src/compute/context.rs`
- `crates/cubecl-tt-metal/src/compute/server.rs`
- `crates/cubecl-tt-metal/src/compute/storage/gpu.rs`
- `crates/cubecl-cpp/src/tt_metal/dialect.rs`
- `crates/cubecl-cpp/src/tt_metal/reader.rs`
- `crates/cubecl-cpp/src/tt_metal/writer.rs`

Work items:
- Map cube/block concepts onto multiple cores deliberately.
- Add per-core tile partitioning.
- Replace replicated-buffer assumptions where needed with sharded layouts.
- Add distributed runtime args and workload partitioning.
- Revisit execution model, scheduling, and profiling once correctness is stable.

Acceptance criteria:
- Multi-core execution is introduced without regressing single-core correctness.
- Partitioning logic is covered by dedicated tests.

Likely failure modes:
- Trying multi-core before single-core semantics are stable.
- Reusing replicated-buffer assumptions in places that require sharded memory.
- Masking correctness bugs as performance bugs.

## Recommended Execution Order

1. Finish runtime hardening for the Phase 1 supported subset.
2. Keep the manually enabled `cubecl_std` suites green, use TT-local `tensor_identity` as the remaining std baseline, and characterize `quantized_view` before enabling it.
3. Enable targeted `cubecl_core` groups in Phase 4, starting with launch-untyped/value-semantics helpers before broader math/control-flow families.
4. Validate a tiny Burn training path in Phase 5.
5. Expand supported surface based on real needs in Phase 6.
6. Only then begin multi-core/perf work in Phase 7.

## Definition of Done for "Native Enough"

The backend should be considered natively integrated in a meaningful first version when all of the following are true:
- `cargo check -p cubecl --features tt_metal` is stable.
- The hand-written TT backend tests are green.
- The chosen `cubecl_std` suites are green, not ignored, and any remaining std add-ons have an explicit matrix status.
- A targeted but meaningful `cubecl_core` runtime subset is green, and the rest of the generated inventory is categorized as queued or capability-blocked.
- A tiny Burn training example runs on TT hardware.
- Unsupported functionality is explicit and documented rather than crashing.
