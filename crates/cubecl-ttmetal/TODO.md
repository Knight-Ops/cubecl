# cubecl-ttmetal TODO

This file tracks the remaining work required to turn `cubecl-ttmetal` from a
scaffold into a backend that is well-formed by CubeCL standards and realistic to
upstream next to `cubecl-cpu`, `cubecl-cuda`, `cubecl-hip`, and `cubecl-wgpu`.

## Current State

Today the crate provides:

- a `MetaliumRuntime`, `MetaliumDevice`, and placeholder `MetaliumServer`
- a `MetaliumCompiler` that produces a `MetaliumSourceBundle`
- a `MetaliumBridge` seam for the eventual `tt_metal` FFI layer
- minimal tests proving compilation and controlled failure at launch time

Today it does **not** provide:

- real execution on Tenstorrent hardware
- real TT host-program creation
- real reader / compute / writer lowering from CubeCL IR
- runtime compilation or loading of TT kernels
- conformance with CubeCL runtime tests

## Acceptance Goal

The backend should be acceptable for the main codebase when it:

- executes real CubeCL kernels on TT hardware or supported TT simulation
- follows the same backend shape as the other runtimes
- compiles real IR rather than emitting TODO-only scaffolds
- supports enough core kernel functionality to run meaningful CubeCL / Burn code
- has tests, docs, feature gating, and failure behavior consistent with other backends

## Phase 0: Scope and Upstream Contract

- Decide the initial support target:
  - real device only
  - simulation only
  - both
- Decide the initial feature scope:
  - pointwise kernels only
  - reductions
  - matmul / MMA
  - tensor maps / TMA-like features
- Decide the supported data types for v1:
  - `bf16`
  - `f32`
  - integer types
  - atomics
- Document explicit non-goals for the first upstreamable version.
- Confirm how TT kernel compilation is expected to happen in CI and in user environments:
  - through installed `tt-metal`
  - through a vendored SDK
  - through simulator tooling

## Phase 1: Crate Shape and Public API Cleanup

- Audit `cubecl-ttmetal` public API against the style of the other backends.
- Remove placeholder-only public types if they are not intended to remain stable.
- Decide which public names should match existing backend conventions:
  - `MetaliumRuntime`
  - `MetaliumDevice`
  - bridge/config types
- Add crate-level docs explaining:
  - execution model mismatch
  - supported features
  - required system dependencies
  - known limitations
- Add a crate feature plan:
  - `std`
  - optional tracing
  - optional simulation support
  - optional experimental compute paths

## Phase 2: Build System and Dependency Strategy

- Choose the FFI approach:
  - `cxx` is the preferred path
  - document why if another route is chosen
- Add build integration for the TT SDK:
  - header discovery
  - library discovery
  - version checks
  - clear build errors
- Decide whether `cubecl-ttmetal` should compile without TT installed:
  - no, hard dependency
  - yes, behind a feature
  - yes, with stub bridge for docs/tests only
- Add `build.rs` if needed for:
  - include paths
  - link flags
  - generated bridge bindings
- Ensure the crate behaves correctly under:
  - `cargo test`
  - workspace builds
  - docs.rs-style builds or feature-disabled doc builds

## Phase 3: Real Device Model

- Replace the placeholder `MetaliumDevice` implementation with real device identity.
- Model TT device and mesh concepts cleanly:
  - single device
  - unit mesh
  - multi-device mesh if supported
- Implement real device enumeration.
- Populate runtime info with meaningful data:
  - bridge availability
  - SDK version
  - mesh shape
  - device architecture / SKU
- Decide how CubeCL `DeviceId.type_id` should map to TT device families.

## Phase 4: Runtime Properties and Capabilities

- Replace placeholder hardware properties in `runtime.rs` with real queried values.
- Define accurate `DeviceProperties` for TT:
  - memory alignment
  - page sizes
  - max bindings
  - shared/L1-style capacity
  - practical cube count mapping
  - supported vector sizes
- Define realistic `TargetProperties`.
- Decide how CubeCL concepts map to TT concepts:
  - cube count -> core range / workload decomposition
  - unit position / plane position -> TT execution semantics
  - shared memory -> circular buffers or L1 allocations
  - global buffers -> DRAM / interleaved / replicated buffers
- Document any semantic mismatches that must be enforced by validation.

## Phase 5: Compiler Architecture

- Replace the remaining scaffold semantics in `compiler.rs` with a real lowering pipeline.
- Keep moving toward `cubecl-cpp` rather than handwritten strings.

### 5.1 Typed TT Codegen Model

- Introduce typed TT-specific intermediate structures for:
  - host program
  - reader kernel
  - compute kernel
  - writer kernel
- Avoid representing these as raw strings until the final rendering step.
- Keep section rendering behind `Display` or equivalent typed formatting.

### 5.2 cubecl-cpp Integration

- Decide how much of compute lowering can reuse `cubecl_cpp::shared::CppCompiler`.
- Add a TT compute dialect or equivalent typed rendering layer.
- Reuse `cubecl-cpp` primitives for:
  - variable declarations
  - typed instructions
  - loops / branches
  - comments / debug lines
  - expression formatting
- Avoid direct string assembly in backend logic except at the very outer boundary.

### 5.3 IR Splitting Pass

- Implement a real analysis pass that splits one `KernelDefinition` into:
  - reader responsibilities
  - compute responsibilities
  - writer responsibilities
- Decide how to recognize:
  - global reads
  - global writes
  - temporaries
  - reductions
  - synchronization points
- Introduce explicit TT transfer operations in the typed model:
  - read DRAM to CB/L1
  - wait/reserve/push/pop circular buffers
  - write CB/L1 to DRAM
- Ensure the split preserves data dependencies and ordering.

### 5.4 Compute Lowering

- Lower actual `kernel.body` operations into TT compute operations.
- Start with a minimal supported subset:
  - assign
  - add
  - mul
  - fma
  - index / index assign where feasible
  - loops / ifs
- Validate unsupported operations explicitly and fail with actionable errors.
- Map CubeCL types to TT compute-compatible types.
- Decide how explicit CubeCL vectorization should map to TT hardware behavior.

### 5.5 Metadata and Binding Lowering

- Lower CubeCL scalar metadata into TT runtime arguments or compile-time arguments.
- Lower buffer bindings into TT buffer handles and address generators.
- Define the contract for:
  - compile-time args
  - runtime args
  - tensor metadata
  - shape/stride metadata
- Handle dynamic cube-count and metadata buffers correctly.

## Phase 6: Reader/Writer Kernel Semantics

- Replace comment-only reader/writer bodies with real TT dataflow kernels.
- Define how contiguous tensors are transferred first.
- Add explicit strategy for strided tensors:
  - reject
  - materialize contiguous copies
  - support selected stride patterns
- Map CubeCL reads/writes to:
  - `noc_async_read*`
  - `noc_async_write*`
  - `cb_wait_front`
  - `cb_push_back`
  - `cb_reserve_back`
  - `cb_pop_front`
- Validate reader/writer runtime arguments and page sizes.

## Phase 7: Host Program Construction

- Replace `UnavailableBridge` with a real bridge implementation.
- Through FFI, construct:
  - device / mesh handle
  - command queue
  - `Program`
  - circular buffers
  - reader kernel
  - compute kernel
  - writer kernel
  - runtime args
- Decide whether kernel sources are:
  - passed to TT JIT APIs directly
  - materialized to files and compiled externally
  - compiled ahead of time
- Implement source/binary caching with invalidation rules.
- Ensure launch failures preserve useful diagnostics from the TT toolchain.

## Phase 8: Real ComputeServer Behavior

- Replace host-memory-only execution assumptions with real TT resource management.
- Audit all `ComputeServer` methods for backend correctness:
  - `initialize_memory`
  - `read`
  - `write`
  - `launch`
  - `flush`
  - `sync`
  - `get_resource`
  - `memory_usage`
  - `memory_cleanup`
  - profiling hooks
- Ensure stream semantics are either:
  - implemented meaningfully
  - serialized intentionally with clear documentation
- Decide whether cross-stream hazards require extra synchronization.

## Phase 9: TT Memory Model Integration

- Stop treating TT buffers as plain CPU byte storage.
- Implement storage/resources that correspond to TT device allocations.
- Decide the ownership model for:
  - DRAM buffers
  - L1 scratch / CB backing allocations
  - replicated vs sharded buffers
- Integrate memory layout policy with TT constraints:
  - page size
  - interleaving
  - tile size
  - alignment
- Implement correct readback and host upload paths.
- Define behavior for zero-sized allocations and edge cases.

## Phase 10: Validation and Unsupported Features

- Add backend-specific validation before launch.
- Reject unsupported kernels with precise errors rather than vague bridge failures.
- Validate:
  - unsupported element types
  - unsupported vector sizes
  - unsupported atomics
  - unsupported stride/layout patterns
  - unsupported control flow or operations
  - cube dimensions that cannot map to TT execution
- Add compiler diagnostics that point to the CubeCL operation category where possible.

## Phase 11: Performance-Critical Design

- Establish a correctness-first path and a performance path.
- Correctness-first path:
  - small supported op set
  - simple contiguous dataflow
  - minimal optimizations
- Performance path:
  - tiled transfers
  - circular-buffer pipelining
  - TT math engine friendly layouts
  - reduced host overhead
- Decide whether to add TT-specific optimization passes in `cubecl-opt`.
- Measure launch overhead and compilation overhead.

## Phase 12: Feature Parity Expectations

To be comparable with the other backends, decide and document support status for:

- pointwise kernels
- reductions
- broadcasts
- tensor identity / view operations
- quantized view support
- atomics
- shared-memory dependent kernels
- plane/warp-like ops
- tensor-map-like features
- matmul / MMA
- profiling
- tracing
- autotuning

Each item should be one of:

- supported
- unsupported with explicit error
- planned

## Phase 13: Testing

- Add unit tests for the compiler split pass.
- Add tests for source bundle structure.
- Add tests for bridge behavior:
  - missing bridge
  - bridge compile failure
  - runtime launch failure
- Add backend conformance tests progressively using existing CubeCL test generators where realistic.
- Decide which runtime tests can run:
  - on CI with simulator
  - on hardware-only CI
  - locally only
- Add regression tests for:
  - metadata packing
  - cube count handling
  - buffer binding order
  - scalar ordering
  - error messages

## Phase 14: Documentation

- Expand the README into user-facing setup instructions.
- Document required TT environment variables and installation.
- Document supported hardware / simulator configurations.
- Add a high-level architecture doc:
  - CubeCL SIMT model
  - TT DAE model
  - split compiler architecture
  - bridge responsibilities
- Add examples showing how to select `MetaliumRuntime`.
- Document backend limitations honestly.

## Phase 15: Code Quality and Upstream Readiness

- Remove placeholder-only TODO comments that are no longer useful.
- Keep direct string-building out of backend logic except at the final render boundary.
- Keep public API small and intentional.
- Match naming, crate structure, and feature style used by the existing backends.
- Add comments only where the TT architecture mismatch would otherwise be hard to follow.
- Ensure error handling is precise and user-facing messages are actionable.
- Ensure the crate compiles cleanly under the workspace lint configuration.

## Phase 16: Final Upstream Checklist

- `cargo test -p cubecl-ttmetal` passes
- feature-gated `cubecl` build passes
- backend executes at least a minimal real kernel on supported TT environment
- unsupported features fail explicitly
- docs explain setup and limitations
- source generation uses typed codegen, not ad-hoc line assembly
- runtime/server semantics are no longer placeholders
- reviewers can compare the crate to existing backends and recognize a complete backend shape

## Suggested Implementation Order

1. Land the real FFI bridge and host program construction path.
2. Replace placeholder device/runtime properties with queried TT values.
3. Implement correctness-first contiguous reader/compute/writer lowering for a tiny op subset.
4. Execute one real pointwise kernel end to end.
5. Add validation for everything unsupported.
6. Expand operation coverage and runtime tests.
7. Optimize the dataflow and memory layout path.
8. Polish docs, tests, and feature gating for upstream review.
