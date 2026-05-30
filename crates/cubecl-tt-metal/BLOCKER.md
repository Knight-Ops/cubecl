# TT-Metal Stream Blocker

## Summary

The main remaining stream-related blocker for broader CubeCL parity on TT-Metal is no longer a total cross-stream hang, but broader stream parity beyond the reduced hardware-validated wrapper.

Current state:
- The current TT hardware-gated suite is green in the serial lane.
- `TT_METAL_RUN_HARDWARE_TESTS=1 LD_LIBRARY_PATH=/usr/local/lib cargo test -p cubecl-tt-metal -- --test-threads=1` now passes with `288 passed; 0 failed`.
- Most math, control-flow, topology, barrier, atomic, plane, block-float, and the reduced/medium TT-local stream slices are hardware-validated.
- The reduced cross-stream wrappers `tests::cubecl_core_wrappers::stream::test_stream_small` and `tests::cubecl_core_wrappers::stream::test_stream_medium` are now green on TT hardware.
- The remaining stream work is the full upstream-sized stream workload and wider stream-facing runtime validation.
- After removing TT hot-path debug printing from the launch/allocation/resource path, the broad workload no longer presents like a zero-progress cross-stream deadlock; it now behaves like a throughput/performance blocker on the current generic TT path.

This means the blocker is no longer ordinary kernel lowering. It is now about how far we can generalize the repaired cross-stream path, not whether any cross-stream path works at all.

## Why This Matters

`stream` is still in the parity lane in [PHASES.md](./PHASES.md) and [CHECKLIST.md](./CHECKLIST.md) because a truthful TT backend needs to handle:
- producer work submitted on one logical stream
- a consumer read or dependent operation submitted on another logical stream
- correct ordering and visibility for shared bindings across those streams

Without that, we cannot honestly claim broad single-device stream semantics, even though the reduced cross-stream path is now green.

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

The reduced reproducer is now green on TT hardware after the cross-stream ownership fix and output-initialization cleanup, which shows that the stream path is repairable without inventing fake TT-side async behavior.

The broad workload was rerun after also removing TT hot-path debug printing from allocation, resource lookup, write, and launch paths. In that state it no longer looked like a parked dependency bug: process-state sampling showed an active worker thread burning CPU instead of a cold stalled test, which points to throughput cost on the current generic TT path rather than the original stream-ordering break.

## What Has Already Been Tried

### 1. Wrapper-level promotion of upstream `stream`

A TT wrapper for the upstream stream runtime test was added temporarily in [lib.rs](./src/lib.rs), compiled successfully, and then hardware-tested.

Result:
- compile lane passed
- before the owner-stream fix, reduced cross-stream hardware execution hung
- after the owner-stream fix and hot-path print removal, the full upstream-sized workload still does not complete in a reasonable validation window, but it now runs CPU-hot instead of parking like the old broken cross-stream path

The wrapper was intentionally removed again so the live TT suite remains truthful.

### 2. Smaller reproducer in `cubecl_core`

A smaller stream reproducer was added in [crates/cubecl-core/src/runtime_tests/stream.rs](../cubecl-core/src/runtime_tests/stream.rs):
- `test_stream_small`
- `test_stream_chained`

The helper was reduced aggressively from the original large looped test to a minimal cross-stream dependency.

Result before the fix:
- the hang still reproduced

This was the strongest evidence that the blocker was a real cross-stream runtime seam, not just workload size.

### 3. Runtime/code-path inspection

The following paths were inspected to narrow the issue:
- [crates/cubecl-runtime/src/stream/event.rs](../cubecl-runtime/src/stream/event.rs)
- [crates/cubecl-runtime/src/client.rs](../cubecl-runtime/src/client.rs)
- [crates/cubecl-common/src/device/handle/channel.rs](../cubecl-common/src/device/handle/channel.rs)
- [crates/cubecl-tt-metal/src/compute/server.rs](./src/compute/server.rs)
- [crates/cubecl-tt-metal/src/compute/command.rs](./src/compute/command.rs)
- [crates/cubecl-tt-metal/src/compute/stream.rs](./src/compute/stream.rs)

That inspection ruled out a few earlier guesses and narrowed the likely fault to the stream/runtime boundary.

### 4. Cross-stream resource-ownership fix

The key repair that cleared the reduced stream hang was not a synthetic event scheduler. It was fixing cross-stream resource ownership in the TT command path.

Changes landed in:
- [crates/cubecl-tt-metal/src/compute/storage/gpu.rs](./src/compute/storage/gpu.rs)
- [crates/cubecl-tt-metal/src/compute/command.rs](./src/compute/command.rs)
- [crates/cubecl-tt-metal/src/lib.rs](./src/lib.rs)
- [crates/cubecl-core/src/runtime_tests/stream.rs](../cubecl-core/src/runtime_tests/stream.rs)

What changed:
- `TtResource` now carries its owning `StreamId`
- cross-stream reads and writes resolve the backing `MeshBuffer` through the resource owner's storage, not the current stream's storage
- the reduced stream reproducer now zero-initializes its output buffer so it measures stream ordering rather than accumulation into an uninitialized output allocation

Outcome:
- `tests::cubecl_core_wrappers::stream::test_stream_small` is green on TT hardware
- `tests::cubecl_core_wrappers::stream::test_stream_medium` is green on TT hardware
- the full serial TT lane is green with those wrappers enabled

## Likely Failure Seam

The most likely issue was a combination of these closely related problems:

1. Cross-stream resource resolution was retrieving the `TtResource` from the producing stream, but the actual `MeshBuffer` lookup in the TT command path was still going through the current stream's storage.
2. Producer work is submitted on one logical stream while consumer reads are initiated from another, so any remaining broadened stream parity work still needs careful validation of host-side ordering behavior.
3. Cross-stream dependency analysis in `MultiStream` may still need stronger guarantees for broader workloads even though the reduced reproducer is now green.

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

The TT harness now intentionally keeps only a reduced live `stream` wrapper enabled.

That is deliberate.
The correct state today is:
- keep the smaller reproducer in `cubecl_core`
- keep `tests::cubecl_core_wrappers::stream::test_stream_small` green
- keep backend-only stream recovery tests green
- do not claim broad upstream stream parity until larger stream workloads are validated

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

4. Keep the reduced TT wrapper green in both isolated hardware runs and the full serial TT lane.

5. Only after that smaller wrapper remains stable should the broader upstream `test_stream` wrapper be reconsidered.

## Exit Criteria For Clearing This Blocker

The original “no cross-stream path works” blocker is now resolved. Broader stream parity should only be considered resolved when all of the following are true:
- the reduced and medium cross-stream reproducers in `cubecl_core::runtime_tests::stream` stay green on TT hardware
- the TT wrappers for those reproducers stay green in isolation
- the full serial TT hardware lane remains green with those wrappers enabled
- the full upstream-sized stream workload is promoted and stays green within a reasonable hardware validation window
- only then should `stream` move from partial TT-local support toward full parity

## Bottom Line

The stream blocker is now narrowed and partially resolved.

A real cross-stream runtime path works on hardware after fixing resource ownership and cleaning up the reduced reproducer. The remaining work is broadening that repaired path into honest upstream stream parity without pretending TT streams are GPU-like async streams.


## Current Broader-Workload Blocker

After the ownership fix, `tests::cubecl_core_wrappers::stream::test_stream_small` and `tests::cubecl_core_wrappers::stream::test_stream_medium` both pass on TT hardware, but the full upstream-sized workload still does not complete.

Observed behavior:
- the broad TT wrapper for `cubecl_core::runtime_tests::stream::test_stream::<TestRuntime, f32>` was re-enabled temporarily
- the process remained alive for multiple minutes without test output
- process-state sampling showed the test binary blocked in `futex_wait_queue` rather than consuming CPU like an active long-running kernel chain

This is the current blocker for full stream parity.
