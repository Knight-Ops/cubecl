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
| Test | `src/lib.rs` | `kernel_compile_and_execute` — allocates buffers, compiles kernels, executes on hardware |

---

## ✅ Complete — Phase 3: Tilization Bindings + Reader/Writer Fix

### 3a. Tilization bindings (`libtt-metal-cxx`)

| File | Change |
|------|--------|
| `include/tt_metal_cxx/tilize.hpp` | C++ header: `tilize()` / `untilize()` taking `elem_size` |
| `src/tt_metal_cxx/tilize.cc` | C++ impl: dispatches by elem_size to `tilize_nfaces<bfloat16>` or `<float>` |
| `src/ffi.rs` | CXX bridge: `fn tilize(data, m, n, elem_size) -> Vec<u8>` etc. |
| `src/tilize.rs` | Rust wrappers re-exported at crate root |
| `build.rs` + `include/tt_metal_cxx.hpp` | Build integration |

Supported `elem_size`: **2** (bfloat16/uint16/f16) and **4** (float32/uint32/i32).
`1` is not pre-instantiated in `libtt_metal.so` and would require adding template instantiations.

### 3b. 2-arg TensorAccessor fix — 0xBF80 root cause resolved

The TT-Metal interleaved memory documentation uses the **3-arg** constructor,
but the WORKING `dram_loopback` example uses **2-arg**. The 3-arg version
computes incorrect DRAM addresses (produces uniform `0xBF80`). Switched back to 2-arg:

```cpp
// 2-arg — WORKS (matches dram_loopback example)
const auto in0 = TensorAccessor(in0_args, in0_addr);

// 3-arg — BROKEN (shown in memory docs but produces 0xBF80)
const auto in0 = TensorAccessor(in0_args, addr, get_tile_size(cb_id));
```

- `cubecl-cpp/src/tt_metal/reader.rs` — 2-arg `TensorAccessor(args, addr)`
- `cubecl-cpp/src/tt_metal/writer.rs` — 2-arg `TensorAccessor(args, addr)`

### 3c. Tests

All 9 tests pass: `TT_METAL_RUN_HARDWARE_TESTS=1 LD_LIBRARY_PATH=/usr/local/lib cargo test -p cubecl-tt-metal -- --nocapture --test-threads=1`

| Test | What it verifies | Hardware |
|------|-----------------|----------|
| `tilize_untilize_round_trip` | Host-side tilize→untilize preserves data | No |
| `buffer_write_read_round_trip` | Raw MeshBuffer write→read round-trips | Yes |
| `dram_loopback_round_trip` | Single-kernel DRAM→CB→DRAM with verification | Yes |
| `kernel_copy_round_trip` | Single-kernel copy with data verify | Yes |
| `kernel_copy_tilized_round_trip` | Single-kernel copy tilized with data verify | Yes |
| `kernel_compile_and_execute` | Three-kernel pipeline compiles + executes | Yes |
| `two_kernel_passthrough` | Reader + writer sharing CB 0 (no compute) ✅ | Yes |
| `three_kernel_copy_raw` | Reader + compute(copy_tile) + writer, raw data, verified ✅ | Yes |
| `three_kernel_copy_tilized` | Reader + compute(copy_tile) + writer, tilized data, verified ✅ | Yes |

### 3d. Three-kernel pipeline — RESOLVED ✅

**Root cause**: The compute kernel was missing `binary_op_init_common(cb_in0, cb_in0, cb_out0)`.
This function configures the Unpack/Math/Pack cores for the CB data format.
Without it, `copy_tile` produces garbage (0xBF80) regardless of tilization.

**Fix**: Added `binary_op_init_common` + `copy_tile_init` before the compute loop
in `generate_copy_compute_source()`. Both `generate_add_compute_source()` already
had the correct init pattern (`binary_op_init_common` + `add_tiles_init`).

**Tests verifying the fix**:
| Test | What it verifies |
|------|-----------------|
| `two_kernel_passthrough` | Reader + writer sharing CB 0 (no compute) — proves reader/writer work |
| `three_kernel_copy_raw` | Reader + compute(copy_tile) + writer with raw data — ✅ passes |
| `three_kernel_copy_tilized` | Same with tilized data — ✅ passes |

**Key takeaway**: Every TT-Metal compute kernel MUST call an init function
(`binary_op_init_common`, `mm_init`, etc.) before the compute loop to configure
the hardware engines for the CB data format.

---

## ✅ Complete — Phase 4: Targeted `cubecl_core` Runtime Coverage

**Current state**: The TT backend now compiles and runs a documented TT-local,
hardware-gated `cubecl_core` subset through the real CubeCL IR pipeline.

**Green hardware subset**:
- launch basics (`with_generics`, `without_generics`, `with_comptime_tag`)
- properties and constant-array coverage
- metadata/addressing for `AddressType::U32` and `AddressType::U64`
- different-rank tensor behavior
- loop-free `assign` coverage (`assign_scalar`, `add_assign_array`)
- `u32` comparisons

**Still intentionally deferred in the TT harness**:
- `numeric::*` until the remaining TT correctness/environment issues are isolated
- `index::test_assign_index` until sliced-array out-of-bounds semantics are fixed
- `branch::test_select_*` until scalar kernel arguments are supported
- `assign::test_kernel_add_assign_vector` until unrolled loop lowering is supported

**Key takeaway**: Phase 4 is now a correctness-focused, documented subset rather
than an optimistic mirror of the broad upstream generated suite.

---

## 🟡 Next — Phase 5: Burn Downstream Validation

**Current state**: The TT-local Phase 4 hardware subset is green. The next milestone is proving the backend against a real Burn workload rather than widening local wrapper coverage blindly.

### 5a. Burn smoke entrypoint
- Add a hardware-gated TT smoke integration test in the Burn repo, not this repo.
- Validate tensor allocation/read-write, a simple forward op, one backward pass, one optimizer step, and one tiny training iteration.

### 5b. Backfill downstream gaps
- Classify each Burn failure as a `cubecl-tt-metal backend gap`, `Burn runtime/backend glue gap`, or `environment/tooling issue`.
- Add a reduced TT regression in this repo for every backend gap when practical.

### 5c. Environment note
- Burn work is currently blocked until a Burn checkout is available alongside this repo.

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
| `tilize_nfaces` / `untilize_nfaces` | ✅ DONE | Phase 3 — `libtt_metal_cxx::tilize()` / `untilize()` |
| `MeshBuffer::compile_args()` (TensorAccessorArgs auto-gen) | ✅ DONE | Phase 4 prereq — `MeshBuffer::compile_args()` |
| `generate_dram_loopback_source()` | ✅ DONE | Single-kernel DRAM→CB→DRAM copy template |
| Reader+compute+writer with tilization | 🔴 | For operations needing compute kernel (add, mul, etc.) |
| Multi-device mesh (non-unit) | MEDIUM | Currently only `MeshDevice::create_unit_mesh` |
| `CommandQueue` / async enqueue | LOW | MVP uses blocking; async for perf |
| Profiler APIs | LOW | `ReadMeshDeviceProfilerResults`, etc. |

---

## Quick Reference: How to Run

```bash
# Build everything
cargo build -p cubecl --features tt_metal

# Run all TT backend tests (requires TT hardware)
TT_METAL_RUN_HARDWARE_TESTS=1 \
LD_LIBRARY_PATH=/usr/local/lib \
cargo test -p cubecl-tt-metal -- --nocapture

# Run libtt-metal-cxx binding tests
TT_METAL_RUN_HARDWARE_TESTS=1 cargo test -p libtt-metal-cxx mesh_buffer

# Check compilation (no hardware needed)
cargo check -p cubecl-tt-metal
cargo check -p cubecl-cpp --features tt_metal
cargo check -p cubecl --features tt_metal
```

## Key Learnings

### 2-arg vs 3-arg TensorAccessor — THE ROOT CAUSE OF 0xBF80

The TT-Metal interleaved memory documentation uses the **3-arg** constructor:
```cpp
TensorAccessor(args, addr, get_tile_size(cb_id));  // BROKEN for our case
```
But the WORKING `dram_loopback` example uses the **2-arg** version:
```cpp
TensorAccessor(args, addr);  // WORKS
```
**The 3-arg version computes wrong DRAM addresses**, causing `noc_async_read_tile`
to read uniform `0xBF80`. The 2-arg version (relying on `AlignedPageSize` from
compile-time args) works correctly.

All reader/writer kernels now use 2-arg. The dram_loopback test proves
data round-trips correctly.

### `copy_tile` / tile compute ops transform data

Even with correct DRAM addressing (2-arg TensorAccessor), the compute kernel's
`copy_tile` for `Float16B` format rearranges bytes within tiles (face swizzling /
tile format conversion). **Data must be tilized** before passing through tile
compute operations, and untilized after.

The simplest working pattern (dram_loopback) avoids compute ops entirely:
- Single data-movement kernel, no compute kernel
- `noc_async_read_tile` → CB scratch → `noc_async_write_tile`
- No `copy_tile`/`pack_tile`/`tile_regs_acquire`
- Processor: `RISCV_0`, NOC: `RISCV_0_default`
- 2-arg `TensorAccessor(args, addr)`

### MeshDevice ownership
`MeshDevice` wraps a `cxx::UniquePtr` and cannot be cloned or shared. The
current pattern stores it in `TtServer` and uses raw pointers
(`*const MeshDevice`) in `TtContext` and `TtStorage`. These pointers are safe
because `TtServer` outlives all its components.

### CB indices
- Input CBs: indices `0, 1, 2, ...`
- Output CBs: indices `16, 17, 18, ...`
- The compute kernel uses `tt::CBIndex::c_0` etc. for CB references

### Data format
TT-Metal's `DataFormat::Float16_b` (value 5) is bfloat16 — the default for
most compute operations.

### JIT caching
TT-Metal caches compiled kernels in `~/.cache/tt_metal/`. Kernel sources are
hashed; identical sources get cache hits. Cached kernels show as
"JIT cache stats: 8/8 hits (100.0%)".

### Data round-trip diagnostic findings (0xBF80 issue) — RESOLVED

**Root cause**: The 3-arg `TensorAccessor(args, addr, get_tile_size(cb))` constructor
computes incorrect DRAM addresses. Switching to the 2-arg version
`TensorAccessor(args, addr)` fixes the issue.

Secondary issue: `copy_tile` for `Float16B` format rearranges bytes within tiles.
Data must be tilized before compute operations.

**Working test**: `dram_loopback_round_trip` — single-kernel DRAM→CB→DRAM copy
proves data round-trips correctly with 2-arg TensorAccessor and no compute ops.

**How we found it**: The TT-Metal source at `tt_metal/programming_examples/loopback/`
uses a single data-movement kernel with `TensorAccessor(args, addr)` (2-arg) —
NOT the 3-arg version shown in the interleaved memory documentation. Mirroring
this exact pattern in our bindings solved the issue.

Reference: `../tt-metal/tt_metal/programming_examples/loopback/loopback.cpp` + `kernels/loopback_dram_copy.cpp`

---

## Helpful Resources

### TT-Metal Documentation
- Architecture & Programming Model: https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md
- Single-core matmul example: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/examples/matmul_single_core.html
- Compute API reference: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis/compute/index.html
- Advanced Topics: tiles, memory, compute engines: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/index.html
- Memory for kernel developers (interleaved, TensorAccessor): https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/advanced_topics/memory_for_kernel_developers.html

### TT-Metal Source (on disk at `../tt-metal/`)
- `tt_metal/hw/inc/api/tensor/tensor_accessor_args.h` — TensorAccessorArgs template (kernel side)
- `tt_metal/hostdevcommon/api/hostdevcommon/tensor_accessor/arg_config.hpp` — ArgConfig enum
- `tt_metal/hw/inc/api/tensor/tensor_accessor.h` — TensorAccessor constructors
- `tt_metal/api/tt-metalium/distributed.hpp` — `EnqueueWriteMeshBuffer`, `EnqueueReadMeshBuffer`
- `tt_metal/api/tt-metalium/mesh_buffer.hpp` — MeshBuffer API
- `tt_metal/api/tt-metalium/tilize_utils.hpp` — `tilize_nfaces` / `untilize_nfaces` declarations
- `tt_metal/impl/data_format/tilize_utils.cpp` — tilize/untilize implementation + explicit instantiations
- `tt_metal/programming_examples/` — working reference examples

### CubeCL Reference
- HIP backend: `crates/cubecl-hip/` — closest architectural analogue (C++ dialect + FFI runtime)
- `cubecl-cpp/src/shared/dialect.rs` — all `Dialect*` trait definitions
- WGPU backend: `crates/cubecl-wgpu/` — alternative runtime pattern (no C++ dialect)
