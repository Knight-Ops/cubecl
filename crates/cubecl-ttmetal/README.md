# CubeCL TT-Metalium Backend

This crate contains the first-stage TT-Metalium integration for CubeCL.

It does not provide a runnable Tenstorrent runtime yet. Instead, it gives CubeCL a
TT-Metalium-aware backend surface made of:

- `MetaliumRuntime` / `MetaliumDevice`
- a placeholder `MetaliumServer` with normal CubeCL memory-management semantics
- a `MetaliumBridge` trait where a future `cxx` + `tt_metal` host bridge will plug in
- a compiler that emits a host-side program skeleton plus reader / compute / writer kernels

Why a scaffold first?

CubeCL kernels are expressed like conventional GPU kernels over global buffers plus
cube/thread builtins. TT-Metalium exposes a different execution model:

- host code explicitly creates `Program`s, buffers, circular buffers, and kernels
- data movement and compute are separated into different kernels
- efficient execution is tile- or page-oriented rather than pointer-thread oriented

Because of that mismatch, a correct backend needs an explicit lowering pass from
CubeCL IR into TT-Metalium host/dataflow/compute programs. This crate makes that
split concrete in-tree so the next implementation steps have a clear integration point.

Current launch behavior:

- CubeCL kernels compile into a `MetaliumSourceBundle`
- the server caches that bundle
- launch stops at the bridge boundary with a clear runtime error until the real
  `tt_metal` C++ bridge is implemented
