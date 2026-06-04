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

## Newly Landed Stream-State Improvements

The latest implementation pass tightened the TT stream/runtime seam in a more
resource-centric way instead of adding a second scheduler:

- `EventStreamBackend::handle_cursor` for TT is no longer a dummy `0`; it now
  surfaces a real binding cursor using CubeCL runtime memory metadata and TT's
  per-resource latest-sequence tracking.
- The TT stream backend now tracks the latest queued sequence for each owned
  `StorageId`, so shared-binding analysis can see newer producer work on an
  already-bound allocation.
- TT readback and partial-write paths now complete only through the relevant
  owned resource lineage instead of pessimistically fencing the entire owner
  stream.
- TT `kernel_cube()` launches now mark their output resources with the queued
  workload sequence, so later cross-stream use sees the freshest producer state
  through existing CubeCL runtime machinery.

This is a better fit for both Rust and CubeCL than inventing a second dependency
tracker because it reuses the runtime's existing shared-binding analysis and
turns TT-specific ownership/completion knowledge into a backend-native cursor.

Result:
- `test_stream_small` is still green on hardware
- `test_stream_medium` is still green on hardware
- `cubetask_compile_pipeline` is still green on hardware
- the full serial TT lane is still green at `288 passed; 0 failed`
- the broad upstream-sized `stream` workload still timed out under a bounded
  180-second hardware probe, including a re-run after enabling TT `MeshDevice`
  program cache at device startup

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

## Relevant `tt-lang` Findings

The review of `../tt-lang` does not immediately remove the current TT stream
throughput blocker, but it does sharpen how we should think about the remaining
runtime and memory-model work:

- **Receiver-owned destination storage is the right default mental model.**
  `tt-lang`'s PipeNet semantics explicitly say the pipe has no hidden payload
  DFB; the receiver reserves destination storage and the sender writes directly
  into that receiver-owned block. That lines up with the owner-stream fix we had
  to make in this backend and is a good reference for future stream/tensormap
  work. Source: `../tt-lang/docs/development/PipeNets.md`.
- **DFB/block semantics matter more than generic async-stream intuition.**
  Their dataflow-buffer model is explicit about blocking `reserve`/`wait`,
  non-blocking `push`/`pop`, and double buffering as the default. That supports
  our current approach of modeling TT stream semantics at the host/runtime
  boundary rather than trying to imitate CUDA events. Source:
  `../tt-lang/docs/sphinx/tour/dataflow-buffers.md`.
- **Launch shape is not the same as active participants.** Their verifier keeps
  launch grid separate from the active work extent and requires guards around
  communication work. That is relevant to future collective/cluster work and to
  making non-1D/multi-core TT behavior honest. Source:
  `../tt-lang/docs/development/PipeNets.md`.
- **Performance work should target sync regions and block shape.** Their DST
  planning and scheduling work reinforces that TT throughput is often about
  tiles-per-sync-region, block shape, and eliminating redundant init/sync costs
  rather than about inventing finer-grained logical concurrency. Sources:
  `../tt-lang/docs/development/DST_Utilization.md`,
  `../tt-lang/lib/Dialect/TTL/Transforms/TTLScheduleOperations.cpp`.

## TT Program Cache Follow-up

We confirmed that a meaningful chunk of TT launch overhead still sits below the
current CubeCL cache layer:

- `TtContext` already caches CubeCL-side lowering and generated TT sources, but
  `compile_kernel()` still rebuilds a fresh TT `Program`, recreates CBs, and
  recreates reader/writer/compute kernels for each launch. Source: [src/compute/context.rs](./src/compute/context.rs).
- TT-Metal's distributed `MeshDevice` already exposes a native program cache via
  `enable_program_cache`, `clear_program_cache`, `disable_and_clear_program_cache`,
  and `num_program_cache_entries`.
- That API is now exposed through `libtt-metal-cxx` and enabled at TT device
  startup in [src/runtime.rs](./src/runtime.rs).
- There is now a focused TT characterization test,
  `tt_program_cache_populates_for_cubetask_pipeline`, but it is kept manual/
  ignored because the current raw `Program`/`launch_from_sources` path did not
  increase `MeshDevice::num_program_cache_entries()` even with program cache
  enabled. That result is useful, but it is not suitable as a normal green-suite
  assertion yet.

What this means:
- We now have both CubeCL-side source caching and TT-side program caching enabled
  on the ordinary runtime path.
- This is not the same as reusing a prepared `Program` object directly in
  `cubecl-tt-metal`; it relies on TT-Metal's own program-cache layer under the
  current launch path.
- The first low-level characterization showed that the raw
  `Program`/`launch_from_sources` path did not increase
  `MeshDevice::num_program_cache_entries()`, so the device cache may not be the
  relevant reuse layer for this part of the backend.
- That means the broad upstream-sized `stream` throughput blocker should still
  be treated primarily as a submission/completion-path problem until a broader
  runtime-path measurement proves otherwise.

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
- the TT stream backend now has explicit queued/submitted/completed sequence tracking, page-sized host completion fences, conditional completion at readback/sync boundaries, in-order queued-op batching, adjacent full-mesh program coalescing, and a `CubeTask` launch/source cache in `TtContext`
- `tests::cubecl_core_wrappers::stream::test_stream_small` and `tests::cubecl_core_wrappers::stream::test_stream_medium` remain green on real hardware after those changes
- `tests::cubetask_compile_pipeline` also remains green, which confirms the direct-source `kernel()` path stayed immediate
- the broad workload still remained alive for tens of seconds without test completion and process-state sampling still showed an active CPU-hot worker thread rather than a parked dependency deadlock
- a later structural probe added worker-core partitioning for the generic `kernel_cube()` runtime path while keeping the direct-source `kernel()` path synchronous; `test_stream_small`, `test_stream_medium`, and `cubetask_compile_pipeline` all stayed green, but the broad wrapper still timed out at the same bounded 180s probe
- a newer staged-input generic path now preserves globally-indexed reads from a single input by staging the full logical input buffer into TT L1/CB storage for small one-input cases; `test_stream_small` and `test_stream_medium` remain green with that path enabled, and a focused 4096-element single-round probe no longer fails immediately with a wrong value but instead runs CPU-hot until a bounded 120s timeout, which suggests the remaining blocker has shifted back to generic scalar runtime throughput rather than the earlier multi-page input-correctness bug

This is the current blocker for full stream parity: the remaining cost is now in the broad workload's host write / submission / completion throughput, not in cross-stream correctness, TT event semantics, repeated CubeTask codegen, or simple same-kernel worker-core fan-out alone.

## Burn Smoke Blocker

### Summary

The next downstream milestone was a Burn smoke test in the sibling repo at `/home/carl/projects/burn`, wired against the local CubeCL/TT backend instead of Burn's pinned git revision.

That work is currently blocked by broad API drift between:
- Burn's pinned `burn-cubecl` / `cubek` stack
- the current local CubeCL head in this repo

This is a build-time compatibility problem, not a TT hardware-runtime correctness problem.

### What was attempted

The Burn repo was patched to consume local CubeCL crates:
- `cubecl`
- `cubecl-common`
- `cubecl-zspace`

TT-specific backend plumbing was also added on the Burn side so a smoke test could target:
- `burn_cubecl::CubeBackend<cubecl_tt_metal::TtRuntime>`
- `burn_autodiff::Autodiff<CubeBackend<TtRuntime>>`

Two smoke-test directions were tried:
- a higher-level backend smoke using Burn backend traits
- a reduced "minimal mode" attempt to compile `burn-cubecl` without `cubek`, keeping only the elementwise/autodiff path needed for a scalar training-step smoke

### First blocker: `cubek` version skew

With Burn patched to the local CubeCL crates, the first build failure came from Burn's pinned `cubek` revision:
- `cubek` at `006a0ed2e9fce76ca1a87fe7687227a0db49e9cd`
- Burn's pinned `cubecl` revision at `1d628d7e8fa6ac27c67195b35736de0b63cc2839`

That `cubek` revision does not compile against the local CubeCL head.

### Second blocker: `burn-cubecl` itself is also out of sync

Even after trying to bypass `cubek` with a reduced Burn-side feature mode, `burn-cubecl` itself still has broad frontend/kernel API drift against the local CubeCL head.

Concrete incompatibility classes from the compile:
- missing view aliases and changed signatures:
  - `cubecl::std::tensor::layout::linear::LinearViewMut`
- changed frontend helpers and intrinsic methods:
  - `Vector::mod_floor`
  - `Vector::extract`
  - `Vector::vec_and`
  - `Sequence::reversed`
  - `Shared::new_slice`
  - `TensorArg::into_buffer_arg`
- changed expansion helpers and generated-method names:
  - `__expand_as_type`
  - `__expand_as_mut_slice_method`
  - related cube-macro generated compatibility surface
- changed kernel expectations around comparisons and references
- structural coupling where `Backend` supertraits still require module/quantized ops to exist, even for a reduced smoke path

This means the Burn smoke is not blocked on one or two missing methods. It is blocked on a wider Burn `burn-cubecl` to CubeCL compatibility layer that no longer matches current CubeCL.

### Current assessment

The TT backend is healthy enough for a downstream smoke in principle:
- the local TT hardware suite remains green
- reduced and medium stream parity are green
- the single-device backend surface is broad and hardware-validated

But the Burn smoke cannot honestly be completed until one of these happens:
1. Burn's `burn-cubecl` is forward-ported to the current CubeCL frontend/kernel API.
2. The local CubeCL branch grows a compatibility shim surface for the older Burn `burn-cubecl` expectations.
3. Burn is checked out at a revision that already matches the local CubeCL branch.

### Best next step

The best next step is not more TT runtime debugging.

It is choosing a compatibility strategy for Burn:
- either pin Burn to a CubeCL-compatible revision
- or intentionally port Burn's `burn-cubecl` to the current CubeCL APIs

Until that is decided, Burn smoke work is blocked by frontend/backend API skew rather than by TT-specific execution bugs.


## Upstream Architecture Guidance

A useful maintainer conversation clarified a few architectural points that should guide future TT work:
- TT does **not** need a fake GPU-style grid model to fit CubeCL.
- The recommended launch reference is the **CPU runtime**:
  - `CubeDim` is host-side concurrency
  - `CubeCount` is schedulable work injected by the runtime
  - generated kernel bodies should stay sequential
- TT-specific vectorization should represent **real SIMD/tile behavior**, not logical launch geometry.
- Upcoming upstream **tile abstractions** are expected to help map CubeCL semantics onto TT’s native 32×32 tiled execution model.
- TT data movement remains a real constraint: moving data across cores means going through the NoC, and each Tensix core is better thought of as coordinated reader / decode / math / encode / writer roles than as a conventional GPU lane group.

Actionable consequence:
- when future blockers involve `ABSOLUTE_POS`, `CUBE_POS`, non-1D launch geometry, or multi-core scheduling, the first question should be whether the backend is drifting away from this CPU-style launch model rather than whether it is matching CUDA/HIP behavior.
