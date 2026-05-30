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
- The full TT-local hardware-gated suite in `cubecl-tt-metal` is green end-to-end on real hardware (`280` tests in the current inventory).
- The `cubecl_std` `trigonometry`, global `reinterpret_slice`, current `event`, TT-local `tensor_identity`, and TT-local `quantized_view` coverage now run through TT.

What is not true yet:
- The TT dialect is not a general CubeCL backend.
- Shared-memory reinterpret and broader shared/local-memory execution semantics remain unsupported beyond the current single-unit scratch/shared-layout slice.
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
| `tensor_identity` | enabled in TT-local seeded wrapper | the current TT-local seeded wrapper is green, but upstream parity still needs 2D launch geometry or a modulo-safe identity lowering path | keep the TT-local wrapper green and revisit upstream parity after non-1D launch geometry or modulo lowering is supported |
| `trigonometry` | enabled | broader math surface is still queued | keep the current pair green and use failures to drive compiler math support |
| `quantized_view` | enabled in TT-local wrappers | the current per-tensor int/fp4 TT-local wrappers are green; broader quantized shapes and output-type coverage are still uncharacterized | keep the current wrappers green and route future quantized decode regressions into focused TT tests before widening the surface |

### `cubecl_core::runtime_tests`

| Category | Status | Blocker | Next action |
|---|---|---|---|
| `all_reduce` | single-device identity enabled; multi-device parity pending | the TT runtime now implements and hardware-validates single-device identity `all_reduce` semantics for both in-place and out-of-place calls, and the upstream wrapper stays clean in the current 1-device environment, but real collective semantics still need multi-device validation and plane/subgroup support | keep the single-device path green, keep the upstream wrapper isolated, and only claim parity after explicit 2+-device TT hardware validation |
| `assign` | enabled | loop-heavy variants still depend on broader control-flow lowering | keep current wrappers green and add any remaining fixed regressions |
| `atomic` | enabled in TT-local single-unit wrappers | the full current upstream generated atomic suite is green through a narrow single-unit backend slice (`Atomic<u32>` load/store/add, scalar `Atomic<i32>` min/max, and scalar/vectorized-2/vectorized-4 `Atomic<f32>` add/min/max), but broader multi-unit and extra-op atomic semantics are still outside the verified contract | keep the current generated suite green, keep capability reporting aligned with the proven single-unit contract, and only widen beyond the current op/type/vector matrix with explicit hardware validation |
| `barrier` | partially enabled in TT-local unit and narrow cube-shared wrappers | the full current upstream barrier runtime suite is now green on hardware through the generic writer's shared-scratch emulation path, but broader pipeline-backed synchronization and general shared-state semantics are still unsupported | keep the current barrier suite green, keep capability reporting aligned with the proven subset, and only widen beyond the current memcpy/barrier lifecycle shapes with explicit hardware validation |
| `binary` | partially enabled | the untyped `mulhi` subset is exposed, and scalar plus vectorized-2/vectorized-4/vectorized-8/vectorized-16 logical-BF16 and logical-F32 `add`/`sub`/`mul`/`div` now run through the TT-native tiled path | keep the scalar/vectorized-2/vectorized-4/vectorized-8/vectorized-16 BF16/F32 native subset green, then widen beyond vec16 and remaining math shapes only after explicit hardware validation |
| `branch` | enabled | none in the current upstream branch suite | keep the full branch suite green and route any break-lowering regressions into focused TT tests |
| `cluster` | parity lane | cluster/distributed semantics are out of scope for single-core TT | revisit only after multi-core/distributed execution exists |
| `cmma` | parity lane | WMMA/CMMA support is intentionally out of scope | defer until a real TT matmul path is required |
| `comparison` | enabled | none in current scalar subset | keep green and expand only if new comparison shapes are needed |
| `const_match` | enabled in TT-local wrappers | none in the current wrapper subset | keep the wrapper green and route any comptime-match bugs into TT regressions |
| `constants` | enabled | none in current subset | keep green in isolation and in the full TT lane |
| `debug` | enabled in TT-local helper-call subset | `debug_print` output handling is still uncharacterized on TT | keep simple/nested helper-call tests green, then evaluate `debug_print` separately |
| `different_rank` | enabled | none in current metadata subset | keep green and route metadata regressions back into TT tests |
| `enums` | enabled in TT-local wrappers | none in the current wrapper subset | keep the wrappers green and narrow failures by enum shape if needed |
| `file` | enabled in TT-local wrappers | none in the current wrapper subset | keep the wrapper green and treat failures as runtime-storage integration bugs first |
| `index` | enabled | the current upstream index suite is now green on hardware through `test_assign_index` plus the promoted local-array `test_kernel_shuffle` path in the TT generic writer | keep both index tests green and treat any local-array regressions as writer/codegen bugs first |
| `launch` | enabled | error-path cases stay upstream-ignored; TT only wraps green cases | keep the existing basic launch wrappers green |
| `launch_untyped` | enabled in TT-local dynamic-addressing subset | shared-memory and max-unit error-path cases are still upstream-ignored | keep dynamic-addressing wrappers green and revisit error-path coverage later |
| `metadata` | enabled | broader SIMT metadata patterns are still outside scope | keep current shape/stride/len coverage green |
| `minifloat` | enabled in TT-local feature-gated wrappers | the current CubeCL minifloat wrappers are green; TT-native `Bfp8_b`/`Bfp4_b` direct copy round-trips and storage suites are green, the full current direct TT-native unary/binary op surface (`add`/`sub`/`mul`/`div` and `abs`/`sqrt`/`rsqrt`/`sin`/`cos`/`tan`/`tanh`/`exp`/`log`) is green on hardware for both `Bfp8_b` and `Bfp4_b`, and TT-local logical-`f32` wrapper suites over that same native surface are now green; true upstream wrapper promotion and `Bfp2_b` remain blocked | keep the current wrappers green, keep the direct `Bfp8_b`/`Bfp4_b` copy/storage suites and the TT-local logical-`f32` wrapper suites green, and only promote wider TT-native block-float coverage once the upstream type-model gap is resolved |
| `numeric` | enabled | none in the current define-oriented subset | keep green and backfill TT regressions for any define-specific bug |
| `plane` | enabled for the full current upstream suite plus one TT-local regression | the TT generic upstream warp-lowering path is now green on hardware for the full current upstream plane suite: `vec1`/`vec2`/`vec4` `sum`/`prod`, inclusive/exclusive `sum`/`prod`, `max`/`min`, `broadcast`, `shuffle`/`shuffle_xor`/`shuffle_up`/`shuffle_down`, and `elect`, plus `vec1` `all`/`any`/`ballot`; the TT-local `vec1` `elect` regression remains as an extra focused check | keep both the full upstream plane suite and the focused TT-local `elect` regression green, then treat true collective validation as the remaining subgroup workstream |
| `properties` | enabled | none in current subset | keep green as the runtime capability baseline |
| `saturating` | enabled in TT-local `i32`/`u32` wrappers | none in the current integer subset | keep the narrow wrappers green before widening integer coverage further |
| `sequence` | enabled | none in the current upstream sequence suite | keep the full sequence suite green and route any sequence-lowering regressions into focused TT tests |
| `slice` | enabled | none in the current upstream slice suite | keep the full slice suite green and route any loop-lowering regressions into focused TT tests |
| `stream` | partially enabled in TT-local wrappers | the reduced and medium single-device cross-stream wrappers are now green on hardware after fixing cross-stream resource ownership, but the full upstream-sized stream workload still stalls and wider stream-facing runtime semantics remain under-validated | keep `test_stream_small` and `test_stream_medium` green in isolation and in the full TT lane, then revisit the full upstream stream workload with focused runtime/queue instrumentation |
| `synchronization` | partially enabled in TT-local wrappers | `sync_cube`, `finished_sync_cube`, `sync_cube_shared`, and the current `sync_plane` visibility subset are green on hardware through the generic writer's shared-scratch execution model; broader plane/subgroup synchronization semantics are still not modeled | keep the current synchronization subset green and defer broader subgroup-dependent synchronization until plane semantics are real |
| `tensor` | enabled in a TT-local wrapper | the current upstream `test_tensor_coordinate` case is green through the narrow 2D single-cube-count launch model; broader multidimensional cube-count semantics are still not claimed | keep `test_tensor_coordinate` green and only widen tensor geometry parity after broader non-1D launch semantics are hardware-validated |
| `tensormap` | parity lane | tensor maps are explicitly outside the current TT subset | defer until tensor-map support is real |
| `to_client` | enabled opportunistically | meaningful coverage needs more than one visible device | keep the wrapper present; it no-ops when only one device is visible |
| `topology` | enabled in TT-local wrappers | the full upstream absolute-position case is now green, and the current TT-local wrappers are green for linearized and 3D cube-count `CUBE_POS_X` / `CUBE_POS_Y` / `CUBE_POS_Z` decomposition plus 2D single-cube `UNIT_POS_X` / `UNIT_POS_Y` coverage, including tail handling; broader multidimensional tensor/shared-state semantics are still not fully modeled by the TT execution shape | keep the current topology wrappers green and revisit broader topology parity only after wider non-1D launch/topology semantics are real on TT |
| `traits` | helper only | this is support code, not a backend category | no TT harness action needed |
| `unary` | partially enabled | the integer abs/bit-operation subset is wired, and scalar plus vectorized-2/vectorized-4/vectorized-8/vectorized-16 logical-BF16 and logical-F32 `abs`/`sqrt`/`inverse_sqrt`/`sin`/`cos`/`tan`/`tanh`/`exp`/`log` now run through the TT-native tiled path | keep the unary-int subset green, keep the scalar/vectorized-2/vectorized-4/vectorized-8/vectorized-16 BF16/F32 native subset green, and only widen float unary wrappers beyond vec16 after explicit hardware validation |
| `unroll` | enabled | none in the current upstream unroll suite | keep the full unroll suite green and route any unroll-lowering regressions into focused TT tests |
| `vector` | partially enabled | index/index-assign/conditional/comparison plus `vector_loop_unroll` are green, and `test_shared_memory` is now green through a narrow single-unit scratch/shared-layout model; broader shared-memory vector semantics are still unsupported | keep the current subset green, keep the single-unit scratch/shared-layout case isolated, and only widen shared-memory vector behavior after the broader execution model is real on TT |

## Remaining Parity Workstreams

These are the concrete remaining implementation plans for full parity with the CubeCL test surface.
They are ordered by dependency, not by desirability: later workstreams assume the earlier runtime/compiler semantics are real.

### Workstream A: Broaden TT-Local Topology Toward Upstream Parity

Goal:
Move from the current single-axis TT-local topology subset toward the upstream topology surface without pretending TT already exposes GPU-like multi-axis launch semantics.

Why this is blocked today:
- `ABSOLUTE_POS`, `CUBE_POS_X`, and `UNIT_POS_X` are now green in the current single-axis TT-local wrappers, including a tail case.
- Upstream topology tests still assume broader multidimensional axis semantics than TT currently models.

Implementation plan:
- Keep the current single-axis topology wrappers as the baseline contract.
- Preserve focused TT regressions for `ABSOLUTE_POS`, `CUBE_POS_X`, `UNIT_POS_X`, and tail handling.
- Revisit topology in this order:
  - broader single-axis bookkeeping if new generated cases require it
  - non-1D launch/topology semantics
  - only then consider upstream multidimensional topology parity
- Do not enable the upstream generated topology macro until non-1D topology semantics are green on hardware.

Acceptance gate:
- TT-local axis-component topology wrappers are green on hardware for both full-tile and tail cases.
- Harness/docs still describe topology as single-axis only until non-1D launch semantics are real.

### Workstream B: Shared-Memory and Barrier Execution Model

Goal:
Expand from the current honest shared-memory execution slice on TT into the remaining vector shared-memory neighborhood and broader synchronization semantics without overstating TT's execution model.

Why this is blocked today:
- TT now supports a narrow shared-scratch generic-writer path that is sufficient for `vector::test_shared_memory` and the current upstream `barrier` runtime suite.
- Broader shared-memory visibility, local-array semantics, and pipeline-backed synchronization are still unsupported outside that proven subset.
- The current barrier success depends on serialized generic-writer execution over shared scratch, not on a general TT pipeline/barrier runtime model.
- TT-local synchronization (`sync_cube`, `finished_sync_cube`, `sync_cube_shared`, and the current `sync_plane` visibility subset) is now green through that same execution model, but broader subgroup/plane synchronization and true multi-device collectives are still unsupported.

Implementation plan:
- Keep the current single-unit scratch/shared-layout path green on hardware as the baseline contract.
- Preserve focused TT regressions for shared-memory declaration/codegen in the generic writer and for `vector::test_shared_memory` itself.
- Characterize the next honest step beyond the current slice before widening behavior:
  - live shared allocation lowering beyond a single unit
  - runtime/shared-buffer plumbing
  - cube-level visibility rules
  - local-array semantics
  - barrier lifecycle and synchronization semantics
- Only once those are real should broader synchronization categories and any barrier shapes beyond the current proven subset move out of parity lane.

Acceptance gate:
- `vector::test_shared_memory` stays green through the narrow honest TT shared-scratch model.
- The current upstream `barrier` runtime suite stays green through the proven generic-writer shared-scratch subset.
- Broader synchronization and barrier/pipeline semantics remain explicitly blocked until they are hardware-validated on TT.

### Workstream C: TT-Native Block-Float Promotion

Goal:
Turn the current direct TT-native block-float characterization into real end-to-end backend coverage without conflating CubeCL minifloat semantics with TT block-float storage semantics.

Why this is blocked today:
- `Bfp8_b` and `Bfp4_b` direct copy/storage regressions are green, the full current direct TT-native unary/binary op surface (`add`/`sub`/`mul`/`div` and `abs`/`sqrt`/`rsqrt`/`sin`/`cos`/`tan`/`tanh`/`exp`/`log`) is green for both formats, and TT-local logical-`f32` wrapper suites over that native surface are now green on hardware.
- The TT-local logical-`f32` wrapper suites are now green, but upstream wrapper promotion still does not map cleanly onto CubeCL `e4m3`/`e2m1x2` element semantics.
- CubeCL minifloat wrappers are not the same thing as TT block-float storage paths.
- `Bfp2_b` is still blocked by the host tensor conversion path.

Implementation plan:
- Keep direct TT-native `Bfp8_b`/`Bfp4_b` copy/storage characterization, the full current direct unary/binary op surface, and the TT-local logical-`f32` wrapper suites green.
- Keep the now-green direct `Bfp8_b`/`Bfp4_b` op surface stable in the hardware lane.
- Keep the now-green TT-local logical-`f32` wrapper suites stable in the hardware lane.
- Document the real TT-native arithmetic contract once characterized, instead of assuming host-side scalar re-quantization is identical.
- Treat true upstream wrapper promotion as a separate follow-on task after the CubeCL type-model gap is resolved.
- Keep `Bfp2_b` blocked until the host/device tilize path actually supports it.

Acceptance gate:
- TT-native block-float math has a hardware-validated semantic model.
- Wrapper promotion only starts after that model is encoded in focused TT regressions.

### Workstream D: Widen Native Float Vectorization Beyond `vec16`

Goal:
Push the proven TT-native BF16/F32 unary/binary path beyond `vec16` without regressing correctness.

Why this is blocked today:
- The current wrapper path is now green through `vec1`/`vec2`/`vec4`/`vec8`/`vec16` for the promoted BF16/F32 native unary and binary surface.
- Wider vectorization beyond `vec16` is still unproven and may need more bridge work or per-op qualification.

Implementation plan:
- Keep `vec16` as the highest verified wrapper width.
- Characterize any drift beyond `vec16` per op family instead of broadening all math at once.
- Start with the native unary/binary ops that already have the cleanest BF16/F32 behavior.
- Only promote a wider vector width when both representative isolated hardware tests and the full TT lane stay green.

Acceptance gate:
- Any width beyond `vec16` is only documented as supported after isolated and full-lane hardware validation on representative unary and binary kernels.

### Workstream E: Atomic Bring-Up

Goal:
Keep the now-green TT-local atomic suite honest while deciding whether to widen beyond the current generated single-unit contract.

Why this is blocked today:
- The full current upstream generated atomic suite is now green on hardware through a truthful narrow single-unit slice: `Atomic<u32>` load/store/add, scalar `Atomic<i32>` min/max, and scalar/vectorized-2/vectorized-4 `Atomic<f32>` add/min/max.
- TT runtime capability registration is now aligned with the proven single-unit contract instead of inheriting broad generic atomic claims.
- What remains is broader atomic semantics beyond the current generated suite: wider vectors, more op families, and any meaningful multi-unit behavior still lack explicit TT hardware validation and, where needed, real TT-native primitives.

Implementation plan:
- Keep TT capability reporting honest: only advertise atomic usages the backend is prepared to compile and run.
- Keep the full current generated backend slice green:
  - scalar `u32` load/store (`test_regression_issue_1218`)
  - scalar `u32` add (`test_atomic_add_int`)
  - scalar `i32` `min`/`max`
  - scalar/vectorized-2/vectorized-4 `f32` `add`/`min`/`max`
- If we widen beyond the current generated suite, do it one family at a time with explicit TT hardware validation.
- If TT lacks a real atomic primitive for any broader family, keep that family out of the supported contract and document it explicitly.

Acceptance gate:
- Any enabled atomic wrapper corresponds to both truthful capability reporting and hardware-green execution.
- The full current generated atomic suite stays green in isolation and in the full TT lane.
- No generated atomic category is enabled on the strength of inherited shared C++ feature registration alone.

### Workstream F: 2D Launch Geometry and Tensor Coordinate Semantics

Goal:
Add real non-1D launch semantics so topology parity and `tensor::test_tensor_coordinate` can become meaningful backend work instead of fake wrappers.

Why this is blocked today:
- TT now supports an honest non-1D generic-launch slice: `cube_dim.y > 1` with `z = 1`, real dispatched-unit counts from `CubeCount`, the full upstream absolute-position case, 3D cube-count decomposition for `CUBE_POS_X` / `CUBE_POS_Y` / `CUBE_POS_Z`, and `tensor_coordinate` are all green on hardware.
- What remains blocked is broader multidimensional tensor/shared-state semantics beyond this current decomposition model, not basic non-1D launch bookkeeping itself.

Implementation plan:
- Keep the current honest non-1D baseline green:
  - launched unit count comes from real `CubeCount` instead of inferred buffer length for the generic writer path
  - full upstream absolute-position coverage is green
  - 3D cube-count decomposition is green for `CUBE_POS_X` / `CUBE_POS_Y` / `CUBE_POS_Z`
  - `cube_dim.y > 1`, `z = 1` decomposition is green for 2D `UNIT_POS_X` / `UNIT_POS_Y` and `tensor_coordinate`
- Revisit non-1D semantics in this order:
  - wider tensor geometry cases that depend on broader shared-state or multidimensional semantics
  - broader shared-memory/barrier interactions with non-1D launch
  - only then any claims beyond the current decomposition model

Acceptance gate:
- The current non-1D decomposition model is backed by focused TT regressions and the full hardware lane.
- Broader tensor/shared-state semantics beyond that decomposition model are still blocked until they have the same level of focused hardware validation.

### Workstream G: Subgroup / Plane / Collective Semantics

Goal:
Resolve whether plane, synchronization, and collective categories are implementable on TT at all in the current backend architecture.

Why this is blocked today:
- `plane`, `all_reduce`, and parts of `synchronization` assume subgroup-like semantics that TT does not currently model in this backend.

Implementation plan:
- Treat these as a research workstream, not opportunistic bug-fixing.
- First identify which subgroup semantics TT kernel APIs can actually express.
- Only then decide whether these categories are backend work or permanent exclusions for this backend shape.

Acceptance gate:
- Each category is either given a concrete implementation path or explicitly declared out of scope for the current TT backend model.

### Workstream H: Tensor Maps, Stream, Cluster, and CMMA Parity Lane

Goal:
Keep the remaining parity-lane categories explicit and separate from ordinary bring-up.

Implementation plan:
- `tensormap`: wait for a real TT tensor-map story before enabling any wrapper.
- `stream`: only revisit when we deliberately select longer-lived stream/runtime suites.
- `cluster`: only revisit after multi-core/distributed execution exists.
- `cmma`: only revisit when a real TT matmul path is required by downstream workloads.

Acceptance gate:
- These categories move only when their underlying runtime/compiler capability exists, not because a generated test happens to be nearby.

### Workstream I: Burn Downstream Validation

Goal:
Translate the now-large TT-local CubeCL subset into a real downstream Burn smoke path, then use that to drive the remaining backend priorities.

Why this matters:
- The current TT-local CubeCL suite is meaningful and hardware-green, but it is still not proof that Burn training or even a real forward pass works.

Implementation plan:
- Add the smallest Burn TT smoke entrypoint first:
  - tensor allocation / transfer
  - simple elementwise forward ops
  - tiny autograd/backward check
  - tiny optimizer step
- Treat every Burn failure as one of two things:
  - a missing CubeCL backend/runtime semantic
  - a Burn integration/runtime-selection issue
- Feed CubeCL backend failures back into focused TT regressions here before broadening Burn scope.

Acceptance gate:
- A tiny Burn smoke runs on TT hardware.
- Training remains a later milestone until backward/update paths are proven.

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
  - no shared-memory or local-array execution semantics
  - no dynamic metadata beyond the current tested subset
  - no plane/sync/distributed IR; generic switch/range-loop codegen is only admitted where the current single-core writer path is characterized
- Generate generic TT compute source for supported scalar/tile-safe kernels while keeping the proven copy/add fast paths.
- Current supported instruction families are:
  - global buffer reads/writes and index addressing
  - same-width reinterpret-friendly type handling and bitcasts
  - comparisons, boolean ops, and selects
  - arithmetic currently exercised by trigonometry-style kernels (`Assign`, `Add`, `Sub`, `Mul`)
  - simple `if` / `if-else` bounds checks
- Keep TT lowering explicitly single-core and tile-oriented.
- Reject unsupported IR with `CompilationError::UnsupportedInstruction` instead of falling through to bad codegen.
- Defer shared-memory reinterpret, runtime promotion of loop-heavy kernels, broader dynamic metadata, and event/comptime-heavy patterns to later phases.

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
- Large logical buffers can still expose TT memory-pool sizing assumptions if the runtime-advertised max page size drifts below the real contiguous allocation sizes the backend uses.
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
  - TT-local seeded `tensor_identity`
  - TT-local `quantized_view`
- Keep upstream `tensor_identity` parity in isolated bring-up; the current TT-local seeded wrapper is green, but the upstream 2D launch/modulo path is still not enabled.
- Keep broader `quantized_view` parity isolated until additional quantized layouts/output types are validated beyond the current TT-local wrappers.
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
- Do not enable broad switch/loop branches beyond the scalar `select` subset, tensor-coordinate, sequence, plane/sync, atomics, tensor maps, or stream categories in this phase; wider native math/vectorization and TT-native block-float formats remain later work.
- Treat queued categories (`topology`) as isolated wrapper work, not broad macro re-enablement.
- Keep the TT-local `binary` subset (`mulhi` plus native BF16/F32 `add`/`sub`/`mul`/`div`), `const_match`, `enums`, `file`, `minifloat`, the narrow `saturating` subset, the full current `slice` subset, the current `vector` subset (including the single-unit scratch/shared-layout `test_shared_memory` case), and the unary subsets in TT-local wrappers until wider-vector coverage and TT-native block-float wrapper promotion are hardware-validated beyond the now-green direct `Bfp8_b`/`Bfp4_b` copy/storage suites and the full current direct TT-native block-float unary/binary op surface.

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
