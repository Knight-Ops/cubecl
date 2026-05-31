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

## Concrete Backtrace Findings

## Throughput Experiments After The Backtrace

Two targeted experiments were run after the debugger pass to verify that the blocker is about submission/completion throughput rather than a broken dependency graph.

### 1. Non-blocking TT workload submission

A targeted backend change was made in [command.rs](./src/compute/command.rs):
- `mesh.enqueue_workload(&mut workload, true)` -> `mesh.enqueue_workload(&mut workload, false)`

At the same time, the remaining hot-path debug printing was removed from:
- [src/compute/context.rs](./src/compute/context.rs)
- [src/compute/server.rs](./src/compute/server.rs)
- [src/compute/command.rs](./src/compute/command.rs)
- [src/compute/storage/gpu.rs](./src/compute/storage/gpu.rs)

Outcome:
- the normal TT hardware lane stayed green
- the broad upstream-sized stream workload still did not finish within the validation window
- this means per-launch blocking `enqueue_workload(true)` was part of the cost, but removing it was not sufficient by itself to promote full stream parity

### 2. Host queue-depth diagnostic

A temporary diagnostic changed `CHANNEL_MAX_TASK` in [channel.rs](../cubecl-common/src/device/handle/channel.rs):
- `32` -> `1024`

Outcome:
- the broad workload still did not complete within the validation window
- however, the observable process shape changed: the newest producer test thread no longer showed the earlier CPU-burning backoff behavior, which means the larger queue reduced producer-side back-pressure pressure
- because the workload still did not finish, queue depth is not the only remaining bottleneck

Interpretation:
- host queue saturation is real, but it is not sufficient to explain the whole blocker
- the remaining cost still sits in TT-side write / submission / completion behavior on the generic path

### 3. In-order queued-op batching on the TT stream backend

A deeper batching pass was implemented directly in the TT stream backend:
- TT workload submission stays non-blocking
- replicated logical writes are queued on the logical stream instead of being submitted immediately
- CubeTask launches are also queued on the logical stream
- flush points drain queued writes and queued one-program mesh workloads in original stream order
- the low-level direct-source launch path was intentionally left immediate, because direct TT characterization tests read device buffers straight from `server.mesh()` and need immediate visibility

Relevant code paths:
- [src/compute/stream.rs](./src/compute/stream.rs)
- [src/compute/command.rs](./src/compute/command.rs)
- [../../libtt-metal-cxx/src/distributed.rs](../../libtt-metal-cxx/src/distributed.rs)
- [../../libtt-metal-cxx/src/tt_metal_cxx/distributed.cc](../../libtt-metal-cxx/src/tt_metal_cxx/distributed.cc)

Why this mattered:
- the first launch-only batching attempt was incorrect because writes were still being submitted immediately, which destroyed the intended `write -> launch -> write -> launch` ordering of the broad stream workload
- the second pass fixed that by batching writes and CubeTask launches together in order
- a temporary over-broad version also batched the direct-source `kernel()` path and regressed `cubetask_compile_pipeline`; that was corrected by narrowing batching back to `kernel_cube()` only

Broad-workload result after the ordered batching pass:
- the full upstream-sized stream wrapper now runs in a much more truthful state than before
- the worker thread stays CPU-hot on real TT hardware instead of immediately parking in the old producer-side enqueue bottleneck
- however, even after the batching split was corrected, the workload still does not complete within a reasonable validation window, so broad stream parity still cannot be promoted into the live TT suite

What this tells us:
- the backend is no longer blocked on the original cross-stream correctness bug
- it is also no longer blocked on the naive per-launch submission path alone
- the remaining problem is honest throughput of the broad stress shape under TT host submission and execution costs

### Practical takeaway

The best current interpretation is:
- reduced and medium stream parity are functionally correct
- ordered TT-side batching improves the stream implementation materially, but the broad workload is still throughput-limited
- the remaining cost is now a combination of TT host submission, TT host buffer writes, and TT command-queue completion latency under the upstream-sized stress shape
- simple queue-size tuning, a single `blocking=false` switch, or launch-only batching are not enough on their own

That leaves the next meaningful optimization work as:
- coalescing multiple launches into fewer TT workload submissions where ordering allows
- reducing TT host write frequency for patterns like repeated zero-initialized output staging
- or adding a more explicit pending-work submission/drain model in the TT server instead of treating each launch as a standalone host round trip


A debugger pass on the live broad workload produced the most important missing evidence.

Broad workload used for the capture:
- `TT_METAL_RUN_HARDWARE_TESTS=1 LD_LIBRARY_PATH=/usr/local/lib cargo test -p cubecl-tt-metal tests::cubecl_core_wrappers::stream::test_stream -- --exact --nocapture --test-threads=1`

What the waiting side is doing:
- The parent cargo process is just in `waitpid`, waiting on the test subprocess.
- Inside the active hardware-test subprocess, the test harness main thread is parked waiting for `CompletedTest` results from the Rust test runner.
- The interesting blocked producer thread is the test thread named `tests::cubecl_c...`; it is not waiting on a TT event or on `read_async`. It is blocked in `cubecl_common::device::handle::channel::custom_channel::DeviceClient::enqueue` at [channel.rs](../cubecl-common/src/device/handle/channel.rs), inside the launch submission path:
  - `ComputeClient::launch_inner` in [client.rs](../cubecl-runtime/src/client.rs)
  - `DeviceHandle::submit`
  - `ChannelDeviceHandle::submit_inner`
  - `DeviceClient::enqueue`
  - then `std::thread::sleep` from the channel backoff loop

What the TT server side is doing:
- The active `DSD-0-0` server worker thread is blocked in TT-Metal's blocking enqueue path, not in CubeCL `MultiStream` logic:
  - `cubecl_tt_metal::compute::command::Command::kernel_cube`
  - `libtt_metal_cxx::distributed::MeshDevice::enqueue_workload(..., blocking=true)`
  - `tt::tt_metal::distributed::FDMeshCommandQueue::enqueue_mesh_workload`
  - `tt::tt_metal::distributed::FDMeshCommandQueue::finish_nolock`
  - waiting on a pthread condition variable inside `libtt_metal.so`
- A separate TT completion-queue thread (`DSD-0-0`) is CPU-hot in:
  - `tt::tt_metal::distributed::FDMeshCommandQueue::read_completion_queue`
  - `tt::tt_metal::SystemMemoryManager::completion_queue_wait_front`
  - `tt::umd::LocalChip::read_from_sysmem`

This combination matters:
- the producer thread is stalled by host-side back-pressure in the custom channel
- the server thread is stalled inside blocking TT workload submission/completion waiting
- the TT completion thread is actively polling completions

That is not the shape of a logical `MultiStream` dependency cycle. It is the shape of a bounded queue feeding a very expensive blocking server operation.

What this says about queue growth:
- The custom device channel in [channel.rs](../cubecl-common/src/device/handle/channel.rs) is bounded at `CHANNEL_MAX_TASK = 32`.
- `DeviceClient::enqueue` blocks once `available_index >= CHANNEL_MAX_TASK`.
- So the queue is not growing unbounded; it is saturating at a small fixed capacity and applying back-pressure to the producer.

Practical interpretation:
- The broad stream workload is currently bottlenecked by throughput / blocking submission semantics on the TT generic path.
- The evidence does not point to a remaining cross-stream ownership bug.
- The evidence also does not point to an obvious `MultiStream::apply_analysis` cycle in this run, because the producer is blocked before the test even reaches the final consumer-side read.

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
