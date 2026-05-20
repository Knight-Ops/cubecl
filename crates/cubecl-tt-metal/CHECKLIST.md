# CubeCL TT-Metal Checklist

This checklist is the tactical companion to the TT-Metal phase plan.
Use it as the working execution board while implementing the backend.

## Ground Rules

- [x] Keep the backend single-core until correctness is stable.
- [x] Prefer explicit `UnsupportedInstruction` / ignored tests over flaky behavior.
- [x] Every fixed regression should add or strengthen a backend test.
- [x] Do not start multi-core or performance work before Burn smoke validation exists.

## Shared Support Matrix

Status goal:
Keep the TT docs and harness inventory aligned while coverage expands.

Tasks:
- [x] Keep the full `cubecl_std` + `cubecl_core` matrix in `PHASES.md`.
- [x] Mirror the matrix in the TT harness comment in `crates/cubecl-tt-metal/src/lib.rs`.
- [x] Use `cargo test -p cubecl-tt-metal -- --list` as the inventory check for wrapper changes.
- [ ] Update the matrix immediately when a category moves between enabled, queued, or parity-lane states.


## Phase 0: Maintain Baseline

Status goal:
Keep the current narrow TT backend healthy while expanding capability.

Tasks:
- [x] Keep `cargo check -p cubecl-tt-metal` green.
- [x] Keep `cargo check -p cubecl --features tt_metal` green.
- [x] Keep the current hand-written TT backend tests green.
- [x] Keep unsupported generated suites explicitly ignored instead of letting them crash.
- [x] Document newly discovered unsupported categories in code or docs.

Verification:
- [x] `cargo check -p cubecl-tt-metal`
- [x] `cargo check -p cubecl --features tt_metal`
- [x] `cargo test -p cubecl-tt-metal -- --test-threads=1`

## Phase 1: Broaden Single-Core Compute Coverage

Status goal:
Move beyond straight-line copy/add into a narrow generic single-core CubeCL subset at the compiler/source-generation layer.

Compiler/dialect tasks:
- [x] Audit `crates/cubecl-cpp/src/tt_metal/dialect.rs` and `tt_metal/compile.rs` for copy/add-only assumptions.
- [x] Implement load/store handling needed by the current bounds-checked elementwise and global reinterpret kernel shapes.
- [x] Implement index/addressing support needed by array/slice kernels.
- [x] Implement same-width reinterpret/bitcast handling for the dtypes used by the current std-facing kernel shapes.
- [x] Implement comparisons, boolean ops, and selects used by bounds checks and simple control flow.
- [x] Support the arithmetic currently needed by trigonometry-style kernels (`Assign`, `Add`, `Sub`, `Mul`).
- [x] Support simple single-path control flow that current std/core tests rely on.
- [x] Reject unsupported IR explicitly rather than silently emitting bad TT code.

Source generation tasks:
- [x] Stop relying on copy/add-only public semantics in `tt_metal/compile.rs`.
- [x] Validate compiled kernel shapes before TT source emission.
- [x] Keep copy/add fast paths and generic scalar source emission aligned with real buffer metadata and compile args.
- [x] Pass logical buffer sizes and static metadata into the compute-kernel contract instead of guessing from the first resource.

Verification:
- [x] Add focused regression tests for accepted and rejected Phase 1 kernel shapes.
- [x] Run isolated TT tests for the new source-generation categories before touching generated suites.
- [x] Confirm unsupported kernels return Rust compilation errors, not runtime crashes.

Remaining before Phase 3:
- [x] Re-enable `trigonometry` end-to-end.
- [x] Re-enable global `reinterpret_slice` end-to-end.
- [x] Keep shared-memory reinterpret, dynamic metadata, and loops explicitly unsupported until later phases.

Stop/go gate:
- [x] Unsupported compute paths now fail cleanly.

## Phase 2: Runtime Correctness and Failure Handling

Status goal:
Make TT launch, memory, and stream behavior safe for real CubeCL kernels.

Launch/runtime tasks:
- [x] Remove the `sources_from_repr(repr, 1)` plus later tile-count overwrite pattern from the normal launch path.
- [x] Use logical resource sizes instead of whole backing-buffer sizes when building compile/runtime metadata.
- [x] Audit `crates/cubecl-tt-metal/src/compute/context.rs` for remaining hardcoded assumptions.
- [x] Audit `crates/cubecl-tt-metal/src/compute/server.rs` for binding/order assumptions.
- [x] Audit `crates/cubecl-tt-metal/src/compute/command.rs` for partial-buffer and staging issues.
- [x] Audit `crates/cubecl-tt-metal/src/compute/storage/gpu.rs` for dtype/page-size assumptions beyond the current subset.
- [x] Ensure bad TT kernel builds always become Rust-visible `LaunchError`s.
- [x] Ensure stream state is recoverable or clearly terminal after a launch error.
- [x] Ensure read/write logic respects padded allocations while preserving logical tensor sizes in all paths.
- [x] Validate compile args and runtime args for multi-input and multi-output kernels.
- [x] Revisit F16/BF16/F32 tile sizing and data format mapping.
- [x] Confirm metadata-driven addressing works for the next re-enabled test categories.

Safety tasks:
- [x] Remove any remaining crash path caused by client panics after TT launch failure.
- [ ] Remove any remaining TT teardown abort caused by ordinary test failures.
- [x] Keep unsupported runtime patterns descriptive and deterministic.

Verification:
- [x] Add focused tests for partial logical tensor reads/writes.
- [x] Add focused tests for multi-input/multi-output compile-arg ordering.
- [x] Add focused tests for failure conversion from TT build errors into Rust errors.
- [x] Add focused tests for dtype/data-format mapping and recoverable stream errors.

Stop/go gate:
- [x] Do not re-enable generated suites until failure behavior is stable.

## Phase 3: Re-enable `cubecl_std` Suites One at a Time

Status goal:
Turn ignored std placeholders back into real backend coverage.

Recommended order:
- [x] `trigonometry`
- [x] `reinterpret_slice`
- [x] `event`
- [ ] `tensor_identity` after TT supports the upstream 2D cube-dimension launch shape
- [ ] `quantized_view` after end-to-end quantized-value, scale, and metadata validation

Per-suite workflow:
- [x] Unignore exactly one std suite.
- [x] Run only that suite.
- [x] Fix compiler/runtime/backend issues it exposes.
- [x] Add TT-specific regressions for the discovered bugs.
- [x] Re-run the single suite until green.
- [x] Keep the remaining unsupported suites ignored.
- [x] Keep std coverage behind TT-local hardware-gated wrappers instead of enabling the broad upstream macro.

Trigonometry checklist:
- [x] Support the basic compiler shape the suite uses: bounds checks, indexing, and multiply-by-constant arithmetic.
- [x] Validate output dtype/data format behavior for float kernels end-to-end.
- [x] Confirm the real suite runs green.

Reinterpret-slice checklist:
- [x] Support the compiler shape for global reinterpret read/write kernels.
- [x] Validate reinterpret behavior for vector widths used by the real suite end-to-end.
- [x] Validate read/write behavior for small logical tensors backed by padded buffers.
- [x] Validate metadata and indexing semantics for reinterpret helpers.

Event checklist:
- [x] Understand what compile-time event expansion becomes in the lowered IR.
- [x] Confirm the current std event kernels run end-to-end on TT.
- [x] Confirm failures do not poison the whole client lifecycle.

Verification:
- [x] Each re-enabled std suite runs green in isolation.
- [x] `cargo test -p cubecl-tt-metal -- --test-threads=1` stays stable after each suite is re-enabled.

Stop/go gate:
- [x] Do not enable all std suites at once.
- [x] Keep the Phase 3 std coverage in TT-local hardware-gated wrappers and keep the docs aligned with that harness.

## Phase 4: Enable Targeted `cubecl_core` Runtime Coverage

Status goal:
Expand from std spot checks into real backend compatibility coverage with TT-local, hardware-gated wrapper tests.

Scope tasks:
- [x] Choose a correctness-first first subset: `f32` foundation plus required untyped and `u32` companion coverage.
- [x] Keep dedicated `i32` generated coverage deferred until the first green subset is stable.
- [x] Keep `cubecl_core::testgen_all!` disabled in favor of TT-local wrappers in `crates/cubecl-tt-metal/src/lib.rs`.

Runtime test category order:
- [x] launch basics, properties, constants, metadata, comparisons, and different-rank tensor behavior
- [x] add the loop-free `assign` subset (`assign_scalar`, `add_assign_array`)
- [x] revisit numeric define coverage after the remaining TT correctness/environment issues are isolated
- [x] revisit sliced-array scalar indexing and scalar-kernel-argument branching after those backend gaps are closed
- [x] add TT-local `launch_untyped` dynamic-addressing coverage
- [x] add TT-local helper-call `debug` coverage and opportunistic `to_client` coverage
- [x] add TT-local wrappers for `const_match`, `enums`, and `file`
- [x] add a non-loop TT-local `slice` subset (`select`, `len`, `mut_assign`, `mut_len`)
- [x] add a narrow TT-local `saturating` subset (`i32`/`u32`)
- [x] add feature-gated TT-local `minifloat` wrappers
- [x] add TT-local `binary_untyped::mulhi` and the non-loop/non-shared-memory `vector` subset
- [x] add the TT-local unary-int subset (`abs`, `vector_sum`, and bit operations)
- [ ] then evaluate queued value-semantics families (`topology`) and the broader float `binary`/`unary` surfaces
  direct TT-native `sub`/`mul`/`sqrt` characterization is now in place, but wrapper promotion is blocked on the row-major/logical-buffer to tile-layout bridge
- [ ] only then evaluate broader switch/loop branches, `slice_for`, vector loop/unroll/shared-memory cases, and synchronization/plane categories

Execution tasks:
- [x] Enable the chosen `cubecl_core` wrappers in `crates/cubecl-tt-metal/src/lib.rs`.
- [x] Keep the green TT-local subset hardware-gated and documented in the harness.
- [x] Add the first extra Phase 4 category: loop-free `assign` coverage.
- [x] Keep unsupported and parity-lane categories explicitly unenabled and documented in the TT harness.
- [x] Track which failures are semantic mismatches versus missing instruction support.
- [x] Backfill focused TT regressions for TT-native `sub`/`mul`/`sqrt` direct-kernel coverage.
- [ ] Backfill focused TT regressions for every remaining category-level bug fixed.

Verification:
- [x] A documented subset of TT-local `cubecl_core` runtime tests is green for TT hardware.
- [x] Unsupported categories are intentional and named.

Stop/go gate:
- [x] Do not claim broad CubeCL compatibility until an explicit core subset is green on hardware.

## Phase 5: Burn Downstream Smoke Validation

Status goal:
Prove the backend is usable for actual model code.

Burn tasks:
- [ ] Select or create a minimal Burn TT smoke entrypoint.
- [ ] Validate tensor allocation and host/device transfer.
- [ ] Validate simple elementwise forward ops.
- [ ] Validate autograd/backward on a tiny graph.
- [ ] Validate one optimizer step.
- [ ] Validate one tiny model training iteration.

Feedback-loop tasks:
- [ ] Record each Burn failure as either a CubeCL backend gap or Burn-integration gap.
- [ ] Bring CubeCL backend gaps back into TT regressions in this repo.
- [ ] Keep Burn-specific glue issues tracked separately.

Verification:
- [ ] A tiny Burn training example completes on TT hardware.
- [ ] Any remaining failures are categorized and reproducible.

Stop/go gate:
- [ ] Do not begin multi-core work before this smoke path exists.

## Phase 6: Expand the Supported Surface Deliberately

Status goal:
Broaden the backend in a principled way after correctness and Burn smoke exist.

Tasks:
- [ ] Prioritize new features based on real failing tests or Burn needs.
- [ ] Group work by capability, not by one-off failing kernels.
- [ ] Add more arithmetic/math categories.
- [ ] Add more tensor/view/slice semantics.
- [ ] Add more dtype coverage.
- [ ] Evaluate whether matrix/matmul or WMMA-adjacent work is actually needed yet.

Verification:
- [ ] Every new supported category has test coverage.
- [ ] The supported subset is documented somewhere stable.

## Phase 7: Multi-Core and Performance

Status goal:
Scale the backend only after it is trustworthy.

Tasks:
- [ ] Design core partitioning for cube/block mapping.
- [ ] Add per-core tile range calculation.
- [ ] Replace replicated-buffer assumptions where sharded memory is required.
- [ ] Add per-core runtime args.
- [ ] Add distributed workload tests.
- [ ] Revisit execution model and profiling.
- [ ] Benchmark only after correctness is stable.

Verification:
- [ ] Multi-core execution has dedicated tests.
- [ ] Single-core behavior does not regress.

## Milestone Gates

Minimal useful TT backend:
- [x] TT crate builds cleanly.
- [x] Hand-written TT tests are green.
- [x] Unsupported generated suites are explicit and stable.

Native-enough CubeCL backend:
- [x] Chosen `cubecl_std` suites are green.
- [x] Chosen `cubecl_core` runtime subset is green.
- [x] Unsupported behavior is explicit and documented.

Burn-ready first version:
- [ ] Tiny Burn training example runs on TT hardware.
- [ ] Backend gaps exposed by Burn are tracked and reproducible.

## Recurring Commands

Build:
- [x] `cargo check -p cubecl-tt-metal`
- [x] `cargo check -p cubecl --features tt_metal`

TT backend tests:
- [x] `cargo test -p cubecl-tt-metal -- --test-threads=1`
- [x] `cargo test -p cubecl-tt-metal -- --list`

Focused TT test loops:
- [ ] `cargo test -p cubecl-tt-metal <test_name> -- --nocapture --test-threads=1`

When re-enabling generated suites:
- [ ] run only the single suite being worked on
- [ ] get it green in isolation before broadening the matrix
