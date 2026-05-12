# cubecl-tt-metal — CubeCL Tenstorrent Backend

A [CubeCL](https://github.com/tracel-ai/cubecl) runtime backend targeting
Tenstorrent hardware through the [TT-Metalium](https://github.com/tenstorrent/tt-metal)
programming model.

## Overview

Tenstorrent architectures (Wormhole, Blackhole) differ fundamentally from
CUDA/HIP GPUs. There is no SIMT grid launch, no warp-based thread model, and
no traditional async streams. Instead:

- A **Tensix core** contains 5 Baby RISC-V CPUs and dedicated compute engines
  (FPU matrix unit, SFPU vector unit, pack/unpack).
- Each computation requires **three cooperating kernels** per core: a *reader*
  (NoC → L1 SRAM), a *compute* kernel (matrix/vector ops), and a *writer*
  (L1 SRAM → NoC).
- Kernels communicate through **circular buffers** in L1 SRAM.
- Data is natively **tile-based** (32×32 elements).
- Execution goes through `Program` → `MeshWorkload` → `MeshDevice::enqueue`.
- Input/output uses `MeshBuffer` (distributed buffer abstraction).

This backend follows the existing `cubecl-hip` and `cubecl-cuda` architecture:
a `Dialect` in `cubecl-cpp` generates C++ source, which the TT-Metal host-side
compiler JIT-compiles into device binaries.

## Architecture

```
cubecl-tt-metal/                  ← Runtime crate (THIS CRATE)
├── src/lib.rs                    ← Module root, TtWmmaCompiler alias, testgen
├── src/device.rs                 ← TtDevice (impl Device)
├── src/runtime.rs                ← TtRuntime (impl Runtime), DeviceService
└── src/compute/
    ├── mod.rs                    ← Module declarations
    ├── server.rs                 ← TtServer (impl ComputeServer + ServerCommunication)
    ├── context.rs                ← TtContext (mesh, compilation cache, compiled kernels)
    ├── command.rs                ← Command (memory ops, kernel launch)
    ├── stream.rs                 ← TtStreamBackend (impl EventStreamBackend)
    ├── fence.rs                  ← Fence (synchronous no-op)
    └── storage/
        ├── mod.rs
        └── gpu.rs                ← TtStorage (impl ComputeStorage)

cubecl-cpp/src/tt_metal/          ← Compiler dialect (in cubecl-cpp crate)
├── mod.rs                        ← Module root
├── arch.rs                       ← TtArchitecture (warp_size=32, tile constants)
├── dialect.rs                    ← TtMetalDialect (all Dialect* trait impls)
└── wmma.rs                       ← TtNoWmma (no-op DialectWmmaCompiler)

libtt-metal-cxx/                  ← CXX bridge to TT-Metal C++ host API
├── include/tt_metal_cxx/mesh_buffer.hpp   ← MeshBufferHandle C++ type
├── src/tt_metal_cxx/mesh_buffer.cc        ← Buffer I/O C++ impl
├── src/mesh_buffer.rs                      ← MeshBuffer Rust wrapper
└── src/distributed.rs                      ← MeshDevice read/write methods
```

## Current State — What's Implemented

### ✅ `libtt-metal-cxx` Bindings — Complete

The TT-Metal C++/Rust bridge has all APIs needed for the backend MVP:

| Feature | Status |
|---------|--------|
| Device management (open/close/query) | ✅ |
| Mesh device (unit mesh, multi-device) | ✅ |
| Program creation and kernel compilation | ✅ |
| Circular buffer configuration + program resources | ✅ |
| Runtime arguments (per-core, common) | ✅ |
| Kernel creation from file and inline source | ✅ |
| Buffer creation (interleaved, sharded) | ✅ |
| MeshWorkload enqueue | ✅ |
| **MeshBuffer creation (replicated, DRAM/L1)** | ✅ |
| **Blocking host↔device buffer I/O** | ✅ |

### ✅ `cubecl-cpp` `tt_metal` Dialect — Stub

All `Dialect*` trait implementations exist and compile. State:

| Trait | Status | Details |
|-------|--------|---------|
| `Dialect` | ✅ `TtMetalDialect<Wmma>` | Generic over WMMA compiler, follows HIP pattern |
| `DialectIncludes` | ✅ Stub | Emits `#include "compute_kernel_api.h"` + common/binary headers |
| `DialectTypes` | ✅ Stub | Maps CubeCL types → C++ types (float, int32_t, bfloat16, etc.) |
| `DialectBindings` | ✅ Stub | Emits `void kernel_main()` signature |
| `DialectCubeBuiltins` | ✅ Stub | Maps threadIdx/blockIdx → loop index `i` / compile-time constants |
| `DialectInstructions` | ⚠️ Stubs | Synchronisation ops are no-ops; warp ops, saturating ops `unimplemented!()` |
| `DialectWmmaCompiler` | ✅ `TtNoWmma` | No-op — no tensor core/WMMA support; returns zero supported combinations |
| `DialectWarpReduceCompiler` | ✅ Defaults | Uses default loop-based reductions (no warp intrinsics) |
| `DialectProcessors` | ✅ Empty | No IR post-processing yet |

### ✅ `cubecl-tt-metal` Runtime — Partial

All CubeCL traits are wired together and compile:

| Component | Status | Details |
|-----------|--------|---------|
| `TtDevice` | ✅ Complete | Wraps chip index, implements `Device` |
| `TtRuntime` | ✅ Complete | Implements `Runtime`, device enumeration, compilation options |
| `DeviceService` for `TtServer` | ✅ Complete | Opens mesh, queries properties, builds DeviceProperties |
| `TtServer` | ⚠️ Partial | Implements `ComputeServer` + `ServerCommunication`. `read`, `write`, `launch` are stubs. |
| `TtContext` | ⚠️ Partial | Manages mesh, compilation cache, compiled kernels. No `compile_kernel()` yet. |
| `Command` | ⚠️ Minimal | Memory ops and resource access. No kernel launch or read/write yet. |
| `TtStreamBackend` | ✅ Complete | Synchronous; creates per-stream MemoryManagement<TtStorage>. No async events. |
| `Fence` | ✅ Complete | No-op (all operations are blocking) |
| `TtStorage` | ⚠️ Stub | `ComputeStorage` trait impl; `alloc`/`get` are `todo!()` |
| Multi-core | ❌ Not started | Single-core only; documented expansion points below |

### ✅ Integration — Complete

- Feature flag `tt_metal` in the `cubecl` umbrella crate
- `testgen!()` and `testgen_all!()` macros registered
- `TestRuntime` aliases for `#[cfg(test_runtime_tt_metal)]`
- Workspace dependency wiring complete

### ✅ Tests — Binding Level

- `mesh_buffer_write_read_round_trip` — verifies device buffer I/O round-trip on hardware
  (Run: `TT_METAL_RUN_HARDWARE_TESTS=1 cargo test -p libtt-metal-cxx mesh_buffer`)

## What's NOT Implemented Yet

### Compiler / Code Generation (HIGH priority)

The dialect emits compilable C++ skeleton code but does NOT yet generate correct
TT-Metal kernel source from CubeCL IR. The critical missing pieces:

1. **Reader/dataflow kernel codegen**: Must emit a valid TT reader (dataflow) kernel
   that reads from DRAM buffers into circular buffers. Template: `#include "dataflow_api.h"`,
   `noc_async_read_tile`, `cb_reserve_back`/`cb_push_back`.
   — **New file needed**: `cubecl-cpp/src/tt_metal/reader.rs`

2. **Writer/dataflow kernel codegen**: Must emit a valid TT writer kernel that reads
   output from circular buffers and writes to DRAM. Template: `cb_wait_front`,
   `noc_async_write_tile`, `cb_pop_front`.
   — **New file needed**: `cubecl-cpp/src/tt_metal/writer.rs`

3. **DialectInstructions (compute kernel body)**: Must map CubeCL IR ops to TT
   compute API calls:
   - Arithmetic: `add_tiles(cb_in0, cb_in1, 0, 0, dst_reg)`, `mul_tiles(...)`, etc.
   - Element-wise: `sin_tile(dst_reg)`, `exp_tile(dst_reg)`, etc.
   - Data movement: `copy_tile(cb, 0, dst_reg)`, `pack_tile(dst_reg, cb_out, 0)`
   - CB management: `cb_wait_front`, `cb_pop_front`, `cb_reserve_back`, `cb_push_back`
   - Register management: `tile_regs_acquire`, `tile_regs_commit`, `tile_regs_wait`, `tile_regs_release`

4. **TtComputeKernel compilation struct**: The existing `ComputeKernel<D>` only
   carries a single `source` string. TT needs three sources (reader, compute, writer).
   Either extend the existing type or create a TT-specific variant.
   — **New type needed** in `cubecl-cpp/src/tt_metal/`

### Runtime / Execution (HIGH priority)

5. **`TtContext::compile_kernel()`**: The compilation entry point. Must:
   - Call `CppCompiler` to generate C++ source from CubeCL IR
   - Create `Program`, circular buffers, compile all three kernels
   - Set runtime args (buffer addresses, tile counts)
   - Store compiled kernel in the cache

6. **`Command::write_to_gpu`** / **`write_to_cpu`** (read): Host↔device data
   transfer using `MeshDevice::write_mesh_buffer` / `read_mesh_buffer` via
   the `libtt-metal-cxx` bindings.

7. **`Command::kernel`** (launch): After compilation, creates `MeshWorkload`,
   adds the compiled program, and enqueues it on the mesh device.

8. **`TtStorage::alloc`** and **`TtStorage::get`**: Implement actual buffer
   allocation via `libtt_metal_cxx::MeshBuffer::create_replicated` and
   resource retrieval.

### Multi-Core Support (MEDIUM priority)

Current code assumes execution on a single Tensix core `(0, 0)`. Multi-core
expansion points (annotated as `// MULTI_CORE:` in code where applicable):

| What changes | Where |
|-------------|-------|
| `TtContext::compile_kernel` — distribute kernels across `CoreRangeSet` instead of single `LogicalCore(0,0)` | `context.rs` |
| `DialectCubeBuiltins` — `blockIdx` maps to core coordinate, `cubeCount` → grid dimensions | `cubecl-cpp/src/tt_metal/dialect.rs` |
| Reader/writer — per-core tile ranges, inter-core semaphores | Reader/writer codegen |
| `TtStorage` — sharded `MeshBuffer` for distributed tensors | `storage/gpu.rs` |

### Data Movement & Tile Handling (MEDIUM priority)

9. **Data tilization**: TT-Metal requires data in 32×32 tile format. Input tensors
   must be tilized before transfer and untilized after readback. TT-Metal's
   `tilize` and `untilize` operations can handle this on-device, but the backend
   currently doesn't call them.

10. **Circular buffer sizing**: CB sizes must be derived from kernel parameters
    (tile count per operation, data format, number of inputs/outputs).

### WMMA / Matrix Operations (LOW priority)

TT's FPU (matrix engine) can perform matrix multiplications, but the interface
differs from CUDA/HIP WMMA. Currently `TtNoWmma` returns zero supported
combinations. Future: `TtFpuMma` using `mm_init`/`matmul_tiles` API.

## Next Steps — Implementation Order

### Phase 2: Compile & Launch an Empty Kernel
1. Implement reader/writer codegen (`reader.rs`, `writer.rs` in the dialect)
2. Implement `TtContext::compile_kernel()` — compile CubeCL IR → TT-Metal program
3. Implement `Command::kernel()` — create MeshWorkload and enqueue
4. Test: compile and launch an empty kernel on TT hardware

### Phase 3: Memory & Data Transfer
5. Implement `TtStorage::alloc` / `TtStorage::get` with MeshBuffer
6. Implement `Command::write_to_gpu` / read via MeshDevice I/O
7. Implement `TtServer::write` / `read` methods
8. Test: allocate buffer, transfer data, verify round-trip

### Phase 4: Element-Wise Ops
9. Implement `DialectInstructions` compute kernel body generation
   (start with `add_tiles` for element-wise add)
10. Implement needed compute API includes in `DialectIncludes`
11. Test: simple element-wise add kernel on hardware

### Phase 5: Test Suite
12. Enable `testgen!` / `testgen_all!` tests with hardware
13. Fix issues found by the test suite
14. Address data tilization for non-trivial tensor shapes

## Building & Testing

### Prerequisites

- TT-Metal installation providing:
  - `tt-metalium/host_api.hpp` header
  - `libtt_metal.so` shared library
  - Runtime root containing `tt_metal/` kernels
- Environment: `TT_METAL_RUNTIME_ROOT` or `TT_METAL_HOME` pointing to runtime root
- Hardware: Tenstorrent Wormhole or Blackhole device

### Build

```bash
# Build just the TT-Metal runtime crate
cargo build -p cubecl-tt-metal

# Build the full CubeCL umbrella with TT-Metal support
cargo build -p cubecl --features tt_metal

# Build the CXX bindings
cargo build -p libtt-metal-cxx
```

### Test

```bash
# Hardware-backed tests (requires TT device)
TT_METAL_RUN_HARDWARE_TESTS=1 cargo test -p libtt-metal-cxx mesh_buffer

# All hardware-backed integration tests
TT_METAL_RUN_HARDWARE_TESTS=1 cargo test -p libtt-metal-cxx

# CubeCL backend tests (once execution is implemented)
TT_METAL_RUN_HARDWARE_TESTS=1 cargo test -p cubecl-tt-metal
```

## Key Files Reference

| File | Purpose | Status |
|------|---------|--------|
| `cubecl-cpp/src/tt_metal/arch.rs` | TT architecture constants | ✅ |
| `cubecl-cpp/src/tt_metal/dialect.rs` | C++ codegen stubs | ⚠️ Needs compute kernel body |
| `cubecl-cpp/src/tt_metal/wmma.rs` | No-op WMMA compiler | ✅ |
| `src/runtime.rs` | Runtime trait + device init | ✅ |
| `src/device.rs` | Device identity | ✅ |
| `src/compute/context.rs` | Mesh + compilation management | ⚠️ Needs compile_kernel() |
| `src/compute/server.rs` | ComputeServer impl | ⚠️ Needs read/write/launch |
| `src/compute/command.rs` | Memory + kernel operations | ⚠️ Needs I/O + launch |
| `src/compute/stream.rs` | EventStreamBackend (synchronous) | ✅ |
| `src/compute/fence.rs` | Fence (no-op) | ✅ |
| `src/compute/storage/gpu.rs` | ComputeStorage (buffer lifecycle) | ⚠️ Needs alloc/get |

## Design Decisions

### Why synchronous execution?
TT-Metal's execution model doesn't have traditional CUDA-style async streams.
Programs are enqueued via `MeshWorkload` and synchronisation happens through
`Finish(cq)` or `blocking=true`. The MVP uses blocking operations exclusively.
Async support could be added later by deferring mesh workload enqueue to a
background thread.

### Why ContiguousMemoryLayoutPolicy instead of PitchedMemoryLayoutPolicy?
TT-Metal uses tile-aligned contiguous memory layouts (32×32 tiles). Row-pitched
layouts from CUDA/HIP are not a good fit.

### Why a no-op WMMA compiler?
TT's FPU matrix unit is architecturally different from NVIDIA/AMD tensor cores.
Instead of WMMA, TT uses the `mm_init`/`matmul_tiles` compute API. A future
`TtFpuMma` can expose matrix operations through the existing WMMA trait.

### How does multi-core dispatch work?
The CubeCL `CubeCount` maps to `blockIdx → core coordinate`. Each Tensix core
runs the same kernel but with different runtime args (tile offsets). TT-Metal's
SPMD model is natural for this — the program is broadcast to all cores in the
mesh via `MeshCoordinateRange`.

## Related Documentation

- [TT Architecture and Metalium Guide](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md)
- [TT-Metal Programming Examples](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/programming_examples)
- [libtt-metal-cxx README](../../libtt-metal-cxx/README.md)
- [CubeCL HIP Backend](../cubecl-hip/) — reference implementation
