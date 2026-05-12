# TODO — Implementation Roadmap

This file tracks what's done, what's next, and lessons learned for the
`cubecl-tt-metal` CubCL Tenstorrent backend.

---

## ✅ Complete — Phase 0: `libtt-metal-cxx` Bindings

Added `MeshBuffer` support for host↔device I/O:

| File | Change |
|------|--------|
| `libtt-metal-cxx/include/tt_metal_cxx/mesh_buffer.hpp` | `MeshBufferHandle` C++ class |
| `libtt-metal-cxx/src/tt_metal_cxx/mesh_buffer.cc` | Replicated buffer creation via `MeshBuffer::create` |
| `libtt-metal-cxx/src/ffi.rs` | CXX bridge: `MeshBufferHandle` opaque type + I/O functions |
| `libtt-metal-cxx/src/mesh_buffer.rs` | Rust `MeshBuffer` struct wrapper |
| `libtt-metal-cxx/src/distributed.rs` | `MeshDevice::write_mesh_buffer` / `read_mesh_buffer` |
| `libtt-metal-cxx/tests/device_management.rs` | `mesh_buffer_write_read_round_trip` test |

Run: `TT_METAL_RUN_HARDWARE_TESTS=1 cargo test -p libtt-metal-cxx mesh_buffer`

---

## ✅ Complete — Phase 1: Scaffolding

- `cubecl-cpp/Cargo.toml` — added `tt_metal = []` feature
- `cubecl-cpp/src/tt_metal/` — module with `arch.rs`, `dialect.rs`, `wmma.rs`, `mod.rs`
- `cubecl-tt-metal/` — new crate with `Cargo.toml`, full `src/` tree
- `cubecl/Cargo.toml` — `tt_metal` feature + dependency
- `cubecl/src/lib.rs` — `#[cfg(feature = "tt_metal")]` re-export
- `Cargo.toml` (workspace) — `libtt-metal-cxx` workspace dep
- All crates compile: `cargo check -p cubecl --features tt_metal`

---

## ✅ Complete — Phase 2: End-to-End Kernel Execution

The full compilation+execution pipeline works on hardware:

| Component | File | What it does |
|-----------|------|--------------|
| Reader codegen | `cubecl-cpp/src/tt_metal/reader.rs` | Generates dataflow kernel C++ using `TensorAccessorArgs` + `noc_async_read_tile` |
| Writer codegen | `cubecl-cpp/src/tt_metal/writer.rs` | Generates dataflow kernel C++ using `noc_async_write_tile` |
| `TtKernelSources` | `cubecl-cpp/src/tt_metal/kernel.rs` | Struct holding all 3 kernel sources + compile-time args |
| Compute kernel | `cubecl-cpp/src/tt_metal/writer.rs` | Hardcoded `copy_tile` compute kernel (will become IR-driven) |
| Dialect includes | `cubecl-cpp/src/tt_metal/dialect.rs` | Emits compute kernel API headers |
| MeshDevice ownership | `src/compute/server.rs` | `TtServer` owns `MeshDevice`; `TtContext` + `TtStorage` access via raw pointer |
| `TtStorage` | `src/compute/storage/gpu.rs` | Allocates `MeshBuffer::create_replicated`, stores in HashMap |
| `TtContext::compile_kernel` | `src/compute/context.rs` | Creates Program → CBs → compiles 3 kernels → sets runtime args |
| `Command::write_to_gpu` | `src/compute/command.rs` | Host→device via `MeshDevice::write_mesh_buffer` |
| `Command::read` | `src/compute/command.rs` | Device→host via `MeshDevice::read_mesh_buffer` |
| `Command::kernel` | `src/compute/command.rs` | Compile → MeshWorkload → enqueue (blocking) |
| `TtStreamBackend` | `src/compute/stream.rs` | Synchronous; passes mesh_ptr to TtStorage |
| Test | `src/lib.rs` | `copy_tile_round_trip` — allocates buffers, compiles kernels, executes on hardware |

Test passes: `TT_METAL_RUN_HARDWARE_TESTS=1 LD_LIBRARY_PATH=/usr/local/lib cargo test -p cubecl-tt-metal copy_tile_round_trip -- --nocapture`

---

## 🔴 Next — Phase 3: Data Tilization

**Why**: TT-Metal stores data in tilized format (32×32 tile layout, rearranged from row-major). Without tilization, tile operations interpret row-major data incorrectly — the copy test runs but output bytes don't match input.

**What to implement**:

### 3a. Host-side tilization
TT-Metal provides `tilize_nfaces` / `untilize_nfaces` functions in its host API:

```cpp
// Before writing input to device
src0_vec = tilize_nfaces(src0_vec, M, K);  // row-major → tile layout

// After reading output from device
result_vec = untilize_nfaces(result_vec, M, N);  // tile layout → row-major
```

These are not yet bound in `libtt-metal-cxx`. Options:
- **A**: Add `tilize` / `untilize` Rust wrappers to `libtt-metal-cxx`
- **B**: Implement tilization in pure Rust (the format is documented in TT-Metal source)
- **C**: Use TT-Metal's `tilize` / `untilize` device-side operations (kernels that do the conversion on-device)

Recommendation: **Option A** — add thin wrappers. The TT-Metal headers provide these as host utility functions.

### 3b. Wire tilization into `Command::write_to_gpu` / read
- `write_to_gpu`: tilize the host data before calling `write_mesh_buffer`
- Read path: call `untilize_nfaces` after `read_mesh_buffer`

### 3c. Tile-aligned tensor dimensions
CubeCL tensor shapes must be padded to tile boundaries (multiples of 32). The `TtStorage::alloc` already rounds up, but shape metadata needs updating.

### Test
Extend `copy_tile_round_trip` to verify output bytes match input after tilization/untilization.

**Resources**:
- TT-Metal docs — matmul example shows `tilize_nfaces` / `untilize_nfaces` usage: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/matmul_single_core.html
- TT-Metal source for tilization: `tt_metal/tt_metal/impl/` — look for `tilize_nfaces`
- Understanding TT tile format: https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md#native-tile-based-computing

---

## 🔴 Next — Phase 4: IR-Driven Compute Kernel

**Current state**: The compute kernel is a hardcoded `copy_tile` template (`writer.rs::generate_copy_compute_source`). The CubeCL IR pipeline is not used.

**Goal**: A CubeCL `#[cube]` kernel written in Rust compiles through `CppCompiler<TtMetalDialect>` and runs on TT hardware.

### 4a. `DialectBindings::compile_kernel_signature`
Currently emits `void kernel_main()`. Must emit proper kernel entry point with runtime args via `get_arg_val<uint32_t>(N)`. The `kernel_main` signature is standard for TT kernels.

### 4b. `DialectInstructions` — op mapping
Map CubeCL IR ops to TT compute API calls. The first ops to implement:

| CubeCL IR | TT Compute API | CB/Register management |
|-----------|---------------|----------------------|
| `Load(global_ptr)` | `copy_tile(cb_in, 0, dst_reg)` | `cb_wait_front` → copy → `cb_pop_front` |
| `Store(global_ptr, val)` | `pack_tile(dst_reg, cb_out)` | `pack_tile` → `cb_push_back` |
| Binary add (`+`) | `add_tiles(cb_in0, cb_in1, 0, 0, dst_reg)` | `tile_regs_acquire` → add → `tile_regs_commit`/`wait` |
| Binary mul (`*`) | `mul_tiles(cb_in0, cb_in1, 0, 0, dst_reg)` | Same pattern |
| Unary sin | `sin_tile(dst_reg)` | `copy_tile` → `sin_tile` → `pack_tile` |

The CB/register management must wrap each op:
```cpp
cb_wait_front(cb_in, 1);
tile_regs_acquire();
copy_tile(cb_in, 0, dst_reg);  // or add_tiles, mul_tiles, sin_tile...
tile_regs_commit();
tile_regs_wait();
cb_pop_front(cb_in, 1);
cb_reserve_back(cb_out, 1);
pack_tile(dst_reg, cb_out);
cb_push_back(cb_out, 1);
tile_regs_release();
```

### 4c. `DialectCubeBuiltins` — SIMT → SPSD mapping
CubeCL uses `UNIT_POS_X` (thread index within a block). TT has no threads — it processes tiles sequentially on a single RISC-V. The `cube_dim` maps to tile count. Implementation:
- `UNIT_POS_X` → loop variable `i`
- `CUBE_DIM_X` → `num_tiles` from runtime arg
- `CUBE_POS_X` → `0` (single core)
- `CUBE_COUNT_X` → `1` (single core)

### 4d. `DialectIncludes` — compute API headers
Already stubbed. Must include the right headers based on which ops are used:
- `#include "compute_kernel_api.h"` — always
- `#include "compute_kernel_api/eltwise_binary.h"` — for `add_tiles`, `mul_tiles`, etc.
- `#include "compute_kernel_api/tile_move_copy.h"` — for `copy_tile`
- `#include "compute_kernel_api/eltwise_unary/sfpu_trigonometry.h"` — for SFPU functions like `sin_tile`

### 4e. `KernelDefinition` → `TtKernelSources`
Wire `CppCompiler<TtMetalDialect>` to produce a `TtKernelSources` from a CubeCL `KernelDefinition`. Currently the compiler produces a single `ComputeKernel` with one source string. We need to:
- Call `CppCompiler::compile()` to get the compute kernel source
- Generate reader/writer sources from templates (parameterized by input/output count from the IR analysis)
- Return a `TtKernelSources`

### Test
A CubeCL kernel (`#[cube]`) that adds two tensors, compiled through the full pipeline and executed on hardware.

**Resources**:
- HIP compute kernel generation: `crates/cubecl-cpp/src/hip/dialect.rs` — reference for how `DialectInstructions` methods are structured
- `crates/cubecl-cpp/src/shared/dialect.rs` — trait definitions for all `Dialect*` methods
- TT compute API reference: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/compute/index.html
- VecAdd compute kernel (cleanest example of TT compute API): https://raw.githubusercontent.com/tenstorrent/tt-metal/refs/heads/main/tt_metal/programming_examples/contributed/vecadd/kernels/add.cpp

---

## 🔴 Next — Phase 5: IR Pipeline Integration

### 5a. `TtServer::launch` — IR-driven
Currently `TtServer::launch` is a stub. Must:
- Accept `Box<dyn CubeTask<TtCompiler>>` (the CubeCL kernel)
- Extract the `KernelDefinition` from the task
- Compile through `CppCompiler` → `TtKernelSources`
- Create Program, CBs, kernels via `TtContext::compile_kernel`
- Enqueue workload

### 5b. `TtServer::read` / `write` — through CubeCL memory model
Currently the read/write in `Command` uses raw `MeshBuffer` addresses from `TtResource`. Must integrate with CubeCL's `Binding` / `CopyDescriptor` / `MemoryManagement` system so that data flows:
```
host → ComputeClient.write() → TtServer.write() → Command.write_to_gpu → MeshDevice.write_mesh_buffer
```

### 5c. Enable `testgen!()` macros
Uncomment the `testgen!()` / `testgen_all!()` macros in `src/lib.rs` once the IR pipeline works end-to-end. These run the CubeCL standard test suite against the TT backend.

---

## 🔴 Next — Multi-Core Support

**Current state**: Single Tensix core at `(0, 0)`.

### Changes needed

| Component | Single-core (current) | Multi-core change |
|-----------|----------------------|-------------------|
| `TtContext::compile_kernel` | `LogicalCore::new(0, 0)` | `CoreRangeSet` spanning the compute grid |
| `DialectCubeBuiltins` | `CUBE_POS_X` → `0` | Map `blockIdx.x` to core coordinate offset |
| Reader/writer codegen | Single tile loop over all tiles | Per-core tile ranges; each core reads its own slice |
| `CubeCount` → core mapping | `(1, 1, 1)` always | `CubeCount.x` = number of columns, `.y` = number of rows |
| `TtStorage` | Single replicated `MeshBuffer` | `ShardedBufferConfig` for sharded tensors across cores |
| Runtime args | Same args for all cores | Per-core runtime args with different tile offsets |

**Resources**:
- TT distributed program dispatch: https://raw.githubusercontent.com/tenstorrent/tt-metal/75c7960c01a87e2ba11834f918d0ca25927d3ee6/tt_metal/programming_examples/distributed/1_distributed_program_dispatch/distributed_program_dispatch.cpp
- TT distributed eltwise add (multi-core vecadd): https://raw.githubusercontent.com/tenstorrent/tt-metal/75c7960c01a87e2ba11834f918d0ca25927d3ee6/tt_metal/programming_examples/distributed/3_distributed_eltwise_add/distributed_eltwise_add.cpp
- Metalium Guide SPMD section: https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md#spmd-in-metalium

---

## ⬜ Future: `libtt-metal-cxx` Binding Gaps

APIs not yet wrapped that the backend will need:

| Feature | Priority | Notes |
|---------|----------|-------|
| `tilize_nfaces` / `untilize_nfaces` | HIGH | Needed for Phase 3 |
| `TensorAccessorArgs` Rust wrapper | MEDIUM | Currently passed as raw `u32` compile args |
| Multi-device mesh (non-unit) | MEDIUM | Currently only `MeshDevice::create_unit_mesh` |
| `CommandQueue` / async enqueue | LOW | MVP uses blocking; async for perf |
| Profiler APIs | LOW | `ReadMeshDeviceProfilerResults`, etc. |

---

## Quick Reference: How to Run

```bash
# Build everything
cargo build -p cubecl --features tt_metal

# Run the copy kernel test (requires TT hardware)
TT_METAL_RUN_HARDWARE_TESTS=1 \
LD_LIBRARY_PATH=/usr/local/lib \
cargo test -p cubecl-tt-metal copy_tile_round_trip -- --nocapture

# Run libtt-metal-cxx binding tests
TT_METAL_RUN_HARDWARE_TESTS=1 cargo test -p libtt-metal-cxx mesh_buffer

# Check compilation (no hardware needed)
cargo check -p cubecl-tt-metal
cargo check -p cubecl-cpp --features tt_metal
cargo check -p cubecl --features tt_metal
```

## Key Learnings

### TensorAccessorArgs compile-time args
Each `TensorAccessorArgs<N>` in a kernel reads 2 compile-time args:
- `args[CTA_OFFSET]`: `ArgConfig` flags (`Sharded=1`, `IsDram=2`, `None=0`)
- `args[CTA_OFFSET+1]`: `AlignedPageSize` (tile size in bytes)

For non-sharded interleaved DRAM: pass `[2, tile_size_bytes]` via `DataMovementKernelConfig::add_compile_arg()`.

These are defined in:
- `tt_metal/hw/inc/api/tensor/tensor_accessor_args.h` (kernel side)
- `tt_metal/hostdevcommon/api/hostdevcommon/tensor_accessor/arg_config.hpp` (ArgConfig enum)

### MeshDevice ownership
`MeshDevice` wraps a `cxx::UniquePtr` and cannot be cloned or shared. The current pattern stores it in `TtServer` and uses raw pointers (`*const MeshDevice`) in `TtContext` and `TtStorage`. These pointers are safe because `TtServer` outlives all its components.

### CB indices
- Input CBs: indices `0, 1, 2, ...`
- Output CBs: indices `16, 17, 18, ...`
- The compute kernel uses `tt::CBIndex::c_0` etc. for CB references

### Data format
TT-Metal's `DataFormat::Float16_b` (value 5) is bfloat16 — the default for most compute operations.

### JIT caching
TT-Metal caches compiled kernels in `~/.cache/tt_metal/`. Kernel sources are hashed; identical sources get cache hits. Cached kernels show as "JIT cache stats: 8/8 hits (100.0%)".

---

## Helpful Resources

### TT-Metal Documentation
- Architecture & Programming Model: https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md
- Single-core matmul example: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/matmul_single_core.html
- Compute API reference: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/compute/index.html

### TT-Metal Source (on disk at `../tt-metal/`)
- `tt_metal/hw/inc/api/tensor/tensor_accessor_args.h` — TensorAccessorArgs template (kernel side)
- `tt_metal/hostdevcommon/api/hostdevcommon/tensor_accessor/arg_config.hpp` — ArgConfig enum
- `tt_metal/hw/inc/api/tensor/tensor_accessor.h` — TensorAccessor constructors
- `tt_metal/api/tt-metalium/distributed.hpp` — `EnqueueWriteMeshBuffer`, `EnqueueReadMeshBuffer`
- `tt_metal/api/tt-metalium/mesh_buffer.hpp` — MeshBuffer API
- `tt_metal/programming_examples/` — working reference examples

### CubeCL Reference
- HIP backend: `crates/cubecl-hip/` — closest architectural analogue (C++ dialect + FFI runtime)
- `cubecl-cpp/src/shared/dialect.rs` — all `Dialect*` trait definitions
- WGPU backend: `crates/cubecl-wgpu/` — alternative runtime pattern (no C++ dialect)
