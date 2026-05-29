# TT-Metal Stream Blocker

## Summary

The main remaining single-device blocker for broader CubeCL parity on TT-Metal is cross-stream runtime behavior.

Current state:
- The current TT hardware-gated suite is green in the serial lane.
- `TT_METAL_RUN_HARDWARE_TESTS=1 LD_LIBRARY_PATH=/usr/local/lib cargo test -p cubecl-tt-metal -- --test-threads=1` passed with `286 passed; 0 failed`.
- Most math, control-flow, topology, barrier, atomic, plane, and block-float slices that are currently enabled are hardware-validated.
- The remaining single-device runtime gap is `stream`: even a reduced cross-stream reproducer still hangs on TT hardware.

This means the blocker is no longer ordinary kernel lowering. It is a runtime/execution-order problem at the stream boundary.

## Why This Matters

`stream` is still in the parity lane in [PHASES.md](./PHASES.md) and [CHECKLIST.md](./CHECKLIST.md) because a truthful TT backend needs to handle:
- producer work submitted on one logical stream
- a consumer read or dependent operation submitted on another logical stream
- correct ordering and visibility for shared bindings across those streams

Without that, we cannot honestly claim broader single-device stream semantics, even if most of the rest of the suite is green.

## Concrete Reproducer

The focused reproducer lives in [stream.rs](../cubecl-core/src/runtime_tests/stream.rs).

Key helpers:
- `test_stream_small`
- `test_stream_chained`

Current reduced reproducer shape:
- `len = 32`
- `rounds = 1`
- `num_loop = 32`
- producer stream: `StreamId { value: 10000 }`
- consumer stream: `StreamId { value: 10001 }`

The reduced reproducer still hangs on TT hardware, which is important because it rules out the earlier theory that the upstream stream test was simply too large or too expensive.

## What Has Already Been Tried

### 1. Wrapper-level promotion of upstream `stream`

A TT wrapper for the upstream stream runtime test was added temporarily in [lib.rs](./src/lib.rs), compiled successfully, and then hardware-tested.

Result:
- compile lane passed
- hardware execution hung

The wrapper was intentionally removed again so the live TT suite remains truthful.

### 2. Smaller reproducer in `cubecl_core`

A smaller stream reproducer was added in [crates/cubecl-core/src/runtime_tests/stream.rs](../cubecl-core/src/runtime_tests/stream.rs):
- `test_stream_small`
- `test_stream_chained`

The helper was reduced aggressively from the original large looped test to a minimal cross-stream dependency.

Result:
- the hang still reproduced

This is the strongest evidence that the blocker is a real cross-stream runtime seam, not just workload size.

### 3. Runtime/code-path inspection

The following paths were inspected to narrow the issue:
- [crates/cubecl-runtime/src/stream/event.rs](../cubecl-runtime/src/stream/event.rs)
- [crates/cubecl-runtime/src/client.rs](../cubecl-runtime/src/client.rs)
- [crates/cubecl-common/src/device/handle/channel.rs](../cubecl-common/src/device/handle/channel.rs)
- [crates/cubecl-tt-metal/src/compute/server.rs](./src/compute/server.rs)
- [crates/cubecl-tt-metal/src/compute/command.rs](./src/compute/command.rs)
- [crates/cubecl-tt-metal/src/compute/stream.rs](./src/compute/stream.rs)

That inspection ruled out a few earlier guesses and narrowed the likely fault to the stream/runtime boundary.

## Likely Failure Seam

The most likely issue is one of these closely related problems:

1. Producer work is submitted on one logical stream but not flushed at the point where the consumer stream expects visibility.
2. Cross-stream dependency analysis in `MultiStream` resolves ordering metadata, but the underlying device-runner queue behavior does not guarantee the producer submission is actually visible before the consumer-side read path blocks.
3. Cross-stream resource resolution is using the right binding stream metadata, but the effective ordering between host queue submission and TT-side readback still is not fully enforced.

The key code paths are:

### `ComputeClient`
In [crates/cubecl-runtime/src/client.rs](../cubecl-runtime/src/client.rs):
- `create_from_slice`, `empty`, and `launch` use `device.submit(...)`
- `read_one_unchecked` reaches `read_async` -> `do_read` -> `submit_blocking(...)`

That means producer work can be enqueued asynchronously while the consumer-side read becomes a blocking call later.

### `MultiStream`
In [crates/cubecl-runtime/src/stream/event.rs](../cubecl-runtime/src/stream/event.rs):
- `resolve`
- `update_shared_bindings`
- `apply_analysis`

This layer detects shared bindings and attempts to align streams by flushing origin streams and waiting on events.

### TT stream backend
In [crates/cubecl-tt-metal/src/compute/stream.rs](./src/compute/stream.rs):
- `flush` is a no-op event producer
- `wait_event` is a no-op event wait
- TT is modeled as a synchronous backend

That model has worked well for a lot of single-stream behavior, but cross-stream visibility appears to be where the simplification stops being sufficient.

## Why This Is Tricky on TT

TT is not being treated as a CUDA-like async stream backend. The backend is structured around explicit reader/compute/writer dataflow kernels and a mostly synchronous host-side submission model.

Relevant implementation context:
- [crates/cubecl-cpp/src/tt_metal/writer.rs](../cubecl-cpp/src/tt_metal/writer.rs)
- [crates/cubecl-cpp/src/tt_metal/reader.rs](../cubecl-cpp/src/tt_metal/reader.rs)
- [crates/cubecl-tt-metal/src/compute/command.rs](./src/compute/command.rs)

So the problem is not just “make events work like GPU events.”
It is more specifically:
- define truthful ordering semantics for CubeCL logical streams
- map that onto TT’s submission/runtime model
- make cross-stream shared bindings visible without inventing fake async guarantees

## Documentation References

### CubeCL local references
- Stream reproducer: [crates/cubecl-core/src/runtime_tests/stream.rs](../cubecl-core/src/runtime_tests/stream.rs)
- Stream/event alignment runtime: [crates/cubecl-runtime/src/stream/event.rs](../cubecl-runtime/src/stream/event.rs)
- Compute client submission/read flow: [crates/cubecl-runtime/src/client.rs](../cubecl-runtime/src/client.rs)
- Device-handle queue flushing: [crates/cubecl-common/src/device/handle/channel.rs](../cubecl-common/src/device/handle/channel.rs)
- TT server entrypoints: [src/compute/server.rs](./src/compute/server.rs)
- TT command/resource access path: [src/compute/command.rs](./src/compute/command.rs)
- TT stream backend: [src/compute/stream.rs](./src/compute/stream.rs)
- Live parity status: [PHASES.md](./PHASES.md)
- Execution checklist: [CHECKLIST.md](./CHECKLIST.md)

### Tenstorrent documentation
- TT-Metal Kernel APIs: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis.html
- The kernel API docs describe TT’s primitive model in terms of data movement, compute, circular buffers, and synchronization primitives inside kernels. That is the right mental model for the backend, but it does not by itself solve CubeCL host-side logical stream semantics.

## Attempted Explanations That Did Not Hold Up

### “The upstream stream test is just too large”
No.
The reduced reproducer still hangs.

### “This is a math kernel bug”
No.
The rest of the TT math surface is broadly green in the live hardware lane, including native BF16/F32 math and direct/native block-float characterization.

### “This is just a generic TT device bring-up failure”
Not primarily.
There is a known separate TT UMD startup/TLB-window issue that can affect isolated subprocess runs, but the stream blocker is different: the reduced stream reproducer was the reason the wrapper was backed out even when the broader suite remained healthy.

## Current Practical Decision

The TT harness intentionally does not keep a live `stream` wrapper enabled.

That is deliberate.
The correct state today is:
- keep the smaller reproducer in `cubecl_core`
- keep backend-only stream recovery tests green
- keep `stream` documented as blocked
- do not claim wrapper-level single-device stream parity until the hang is fixed

## Recommended Next Debugging Steps

1. Instrument the submission/flush path around producer and consumer streams.
   Focus on:
   - `ComputeClient::launch`
   - `ComputeClient::read_async` / `do_read`
   - `DeviceHandle::submit` vs `submit_blocking`
   - `DeviceHandle::flush_queue`

2. Add focused logging around `MultiStream::resolve`, `update_shared_bindings`, and `apply_analysis`.
   We want to know:
   - whether the shared binding is detected
   - whether the origin stream is flushed
   - whether the consumer stream actually observes that state transition

3. Validate whether a consumer-side blocking read needs an explicit producer-queue flush on the device-handle side before relying on TT stream alignment.

4. Only after the minimal reproducer is green should the TT wrapper for `cubecl_core::runtime_tests::stream::test_stream_small` be restored.

5. Only after that smaller wrapper is green should the broader upstream `test_stream` wrapper be reconsidered.

## Exit Criteria For Clearing This Blocker

This blocker should only be considered resolved when all of the following are true:
- the reduced cross-stream reproducer in `cubecl_core::runtime_tests::stream` is green on TT hardware
- a TT wrapper for that reduced reproducer is green in isolation
- the full serial TT hardware lane remains green with the wrapper enabled
- only then is broader stream-facing parity eligible to move out of the parity lane

## Bottom Line

The remaining single-device blocker is a real cross-stream runtime semantics problem.

It is already narrowed to a small reproducer, it is documented honestly in the parity docs, and it should be solved at the runtime/stream boundary rather than by adding more wrapper logic or pretending TT streams are GPU-like async streams.
