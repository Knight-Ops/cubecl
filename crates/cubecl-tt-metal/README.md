# cubecl-tt-metal — CubeCL Tenstorrent Backend

A [CubeCL](https://github.com/tracel-ai/cubecl) runtime backend targeting
Tenstorrent hardware through the [TT-Metalium](https://github.com/tenstorrent/tt-metal)
programming model.

For detailed implementation status and roadmap, see **[TODO.md](TODO.md)**.

## Overview

Tenstorrent architectures (Wormhole, Blackhole) differ fundamentally from
CUDA/HIP GPUs in execution shape and memory behavior: there is no GPU-style
hardware scheduler, no cache-driven memory hierarchy, and no honest warp-based
execution model to lean on. The useful CubeCL mental model is closer to the CPU
runtime than to CUDA launch semantics: generated kernel code should stay
sequential, `CubeDim` should represent the actual concurrent workers/cores made
available by the runtime, and `CubeCount` should be injected as runtime
scheduling work rather than treated as a baked-in SIMT grid.

At the hardware level, each **Tensix core** is a small cluster with distinct
reader / decode / math / encode / writer roles, communicating through
circular buffers in software-managed L1 SRAM and the on-chip NoC. Native TT
compute is tile-oriented, with 32×32 tiles as the important execution unit.

This backend follows the existing `cubecl-hip` / `cubecl-cuda` pattern:
a C++ `Dialect` in `cubecl-cpp` generates kernel source; the TT-Metal host
compiler JIT-compiles it; execution goes via `Program` → `MeshWorkload` →
`MeshDevice::enqueue`.

## Architecture

```
cubecl-tt-metal/                  ← Runtime crate (THIS CRATE)
├── src/lib.rs                    ← Module root, test harness
├── src/device.rs                 ← TtDevice (impl Device)
├── src/runtime.rs                ← TtRuntime (impl Runtime), DeviceService
└── src/compute/
    ├── server.rs                 ← TtServer (impl ComputeServer + ServerCommunication)
    ├── context.rs                ← TtContext (mesh, compilation, kernels)
    ├── command.rs                ← Command (memory ops, kernel launch)
    ├── stream.rs                 ← TtStreamBackend (impl EventStreamBackend)
    ├── fence.rs                  ← Fence (synchronous no-op)
    └── storage/gpu.rs            ← TtStorage (impl ComputeStorage)

cubecl-cpp/src/tt_metal/          ← Compiler dialect (in cubecl-cpp crate)
├── arch.rs                       ← TtArchitecture
├── dialect.rs                    ← TtMetalDialect (Dialect* trait impls)
├── reader.rs                     ← Reader (dataflow) kernel codegen
├── writer.rs                     ← Writer kernel + compute kernel codegen
├── kernel.rs                     ← TtKernelSources struct
└── wmma.rs                       ← TtNoWmma (no-op DialectWmmaCompiler)
```

## Current State

- **Phase 0**: `libtt-metal-cxx` bindings extended with `MeshBuffer` I/O — ✅
- **Phase 1**: Crate scaffolding, dialect stubs, workspace wiring — ✅
- **Phase 2**: End-to-end copy kernel compiles and executes on hardware — ✅
- **Phase 3**: TT-local `cubecl_std` spot coverage (`trigonometry`, global `reinterpret_slice`, current `event`) — ✅
- **Phase 4**: Targeted TT-local `cubecl_core` hardware subset, including loop-free `assign` coverage — ✅
- **Phase 5**: Burn downstream smoke validation — 🟡 next

## Building & Testing

```bash
# Build
cargo build -p cubecl --features tt_metal

# Run hardware test (requires TT device)
TT_METAL_RUN_HARDWARE_TESTS=1 \
LD_LIBRARY_PATH=/usr/local/lib \
cargo test -p cubecl-tt-metal copy_tile_round_trip -- --nocapture

# Check compilation (no hardware needed)
cargo check -p cubecl --features tt_metal
```

## Design Decisions

- **CPU-style launch semantics**: the actionable upstream guidance is that TT should not invent a fake GPU grid model. `CubeDim` should stay small and match concurrent workers/cores, while `CubeCount` should be scheduled by the runtime and pushed into the inner loop, like the CPU runtime.
- **Sequential generated kernels**: the code emitted for a logical CubeCL kernel should stay sequential; concurrency belongs to the host/runtime launch contract rather than to GPU-like thread indexing baked into the generated source.
- **Vector width follows SIMD, not grid width**: vectorization should represent the real SIMD/tile capability of TT instructions, not be used as a substitute for launch geometry.
- **Synchronous execution today**: TT-Metal still does not behave like a CUDA-style async stream runtime. The current backend remains mostly synchronous, with ordered batching now used to reduce host submission overhead where possible.
- **ContiguousMemoryLayoutPolicy**: TT uses tile-aligned contiguous layouts, not row-pitched.
- **No-op WMMA compiler**: TT's FPU matrix unit uses `mm_init`/`matmul_tiles`, not WMMA intrinsics.
- **MeshDevice raw pointers**: `TtContext`/`TtStorage` access the `MeshDevice` via `*const` pointers, safe because `TtServer` owns and outlives them.

## Findings From `tt-lang` Review

A focused review of `../tt-lang` surfaced a few patterns that are worth
keeping as standing guidance for this backend:

- **Explicit layout metadata is mandatory, not optional.** `tt-lang` verifies
  that tensor operands carry a `ttl.layout` encoding and rejects lowering when
  layout metadata is missing. It also derives page size from the **tile type**
  carried by the layout, not from ad hoc dtype guesses alone. Relevant sources:
  `../tt-lang/lib/Dialect/TTL/IR/TTLOps.cpp`,
  `../tt-lang/lib/Dialect/TTL/Transforms/ConvertTTLToTTKernel.cpp`.
- **Tensor accessor materialization needs real memory-layout metadata.**
  `tt-lang`'s lowering explicitly distinguishes interleaved and sharded tensor
  accessor compile-time argument footprints and uses constexpr CTA-offset
  chaining to find the right runtime-arg slice for each tensor. That is a good
  reference for future TT tensor metadata, sharding, and tensormap work in
  CubeCL. Relevant source:
  `../tt-lang/lib/Dialect/TTL/Transforms/ConvertTTLToTTKernel.cpp`.
- **Tensor <-> dataflow-buffer movement is a first-class contract.**
  `tt-lang` treats tensor/CB copies as explicit operations and documents DFBs
  as producer-consumer storage with blocking `reserve`/`wait`, non-blocking
  `push`/`pop`, and `block_count=2` as the normal double-buffered default. This
  matches the TT hardware model better than implicit buffer aliasing. Relevant
  docs: `../tt-lang/docs/sphinx/tour/dataflow-buffers.md`.
- **Shape units depend on layout.** In `tt-lang`, tiled tensors use a tile as
  the shape unit, while row-major tensors use scalars. That is a useful mental
  model for future CubeCL layout-sensitive TT work, especially when mixing
  tiled kernels with row-major host-visible surfaces. Relevant docs:
  `../tt-lang/docs/sphinx/tour/dataflow-buffers.md`,
  `../tt-lang/examples/group_transfer_upsample.py`.
- **Launch grid and active work extent are separate concerns.** `tt-lang`'s
  PipeNet model explicitly separates the launch grid from the active coordinates
  that participate in communication, and requires guards for nodes outside the
  work extent. That is directly relevant to our future non-1D, collective, and
  cluster planning. Relevant docs: `../tt-lang/docs/development/PipeNets.md`.
- **Per-node block shape matters more than full tensor size for TT scheduling.**
  `tt-lang`'s DST planning, subblocking, and sync insertion are all described in
  terms of block shape per node, DST capacity, and tiles per sync region, not in
  terms of a giant logical tensor. That is a strong reference point for future
  performance work in this backend. Relevant docs:
  `../tt-lang/docs/development/DST_Utilization.md`,
  `../tt-lang/lib/Dialect/TTL/Transforms/TTLScheduleOperations.cpp`.
- **Linearized tile/CB indexing is intentional.** `tt-lang` lowers
  multi-dimensional tensor/CB movement by explicitly linearizing tile indices
  before `pack_tile`/NOC operations. We should keep avoiding hidden row-major or
  byte-offset assumptions when widening TT layout support. Relevant source:
  `../tt-lang/lib/Dialect/TTL/Transforms/ConvertTTLToTTKernel.cpp`.

## Actionable Upstream Guidance

The most useful maintainer guidance so far is:
- **SIMT as a frontend model is still acceptable**. The problem is not that CubeCL must abandon SIMT concepts entirely; it is that TT launch/runtime semantics should not be modeled as a GPU grid.
- **Use the CPU runtime as the reference for launch semantics**. `CubeDim` is host-side concurrency, `CubeCount` is scheduled work, and the generated kernel body should be sequential.
- **Map `CubeDim` to real TT concurrency**. For TT, that means thinking in terms of available Tensix workers/cores, not large GPU-style launch dimensions.
- **Keep vectorization tied to hardware SIMD/tile behavior**. On TT, native tiles and SIMD-style execution are a better fit for vector semantics than pretending those lanes are a launch grid.
- **Expect upstream tile abstractions to matter**. Planned CubeCL/Burn tile abstractions should reduce some of the impedance mismatch for TT’s native 32×32 tiled execution model.

## Related Documentation

- [TODO.md](TODO.md) — implementation roadmap and detailed status
- [TT Architecture and Metalium Guide](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md)
- [TT-Metal Programming Examples](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/programming_examples)
- [libtt-metal-cxx README](../../libtt-metal-cxx/README.md)
- [CubeCL HIP Backend](../cubecl-hip/) — reference implementation
