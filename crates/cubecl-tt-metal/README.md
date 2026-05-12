# cubecl-tt-metal — CubeCL Tenstorrent Backend

A [CubeCL](https://github.com/tracel-ai/cubecl) runtime backend targeting
Tenstorrent hardware through the [TT-Metalium](https://github.com/tenstorrent/tt-metal)
programming model.

For detailed implementation status and roadmap, see **[TODO.md](TODO.md)**.

## Overview

Tenstorrent architectures (Wormhole, Blackhole) differ fundamentally from
CUDA/HIP GPUs: no SIMT grid launch, no warp-based threading, no async streams.
Instead, each **Tensix core** uses three cooperating kernels (reader, compute,
writer) communicating through circular buffers in L1 SRAM, operating on
native 32×32 tiles.

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
- **Phase 3**: Data tilization (tilize_nfaces/untilize_nfaces) — 🔴 planned
- **Phase 4**: IR-driven compute kernel generation — 🔴 planned
- **Phase 5**: Full CubeCL pipeline integration + test suite — 🔴 planned

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

- **Synchronous execution**: TT-Metal has no CUDA-style async streams. MVP uses blocking enqueue.
- **ContiguousMemoryLayoutPolicy**: TT uses tile-aligned contiguous layouts, not row-pitched.
- **No-op WMMA compiler**: TT's FPU matrix unit uses `mm_init`/`matmul_tiles`, not WMMA intrinsics.
- **MeshDevice raw pointers**: `TtContext`/`TtStorage` access the `MeshDevice` via `*const` pointers, safe because `TtServer` owns and outlives them.

## Related Documentation

- [TODO.md](TODO.md) — implementation roadmap and detailed status
- [TT Architecture and Metalium Guide](https://github.com/tenstorrent/tt-metal/blob/main/METALIUM_GUIDE.md)
- [TT-Metal Programming Examples](https://github.com/tenstorrent/tt-metal/tree/main/tt_metal/programming_examples)
- [libtt-metal-cxx README](../../libtt-metal-cxx/README.md)
- [CubeCL HIP Backend](../cubecl-hip/) — reference implementation
