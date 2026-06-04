# TT-Metal Runtime Throughput Blockers

## Current Status

The main remaining runtime-throughput blocker on TT-Metal is **not** basic stream correctness anymore.

What is working today:
- The TT backend has a safe single-device stream baseline.
- The reduced cross-stream hardware wrappers are green:
  - `tests::cubecl_core_wrappers::stream::test_stream_small`
  - `tests::cubecl_core_wrappers::stream::test_stream_medium`
- The TT compile pipeline regression guard is green:
  - `tests::cubetask_compile_pipeline`
- The direct source / characterization path remains immediate and must stay that way.
- The broad stream stress shape is now available as a **manual ignored characterization** in the TT harness:
  - `tests::cubecl_core_wrappers::stream::test_stream_broad_single_round_manual`
- The TT program-cache characterization is also intentionally manual/ignored:
  - `tests::tt_program_cache_populates_for_cubetask_pipeline`

What is still blocked:
- The full upstream-sized `cubecl_core::runtime_tests::stream::test_stream::<TestRuntime, f32>` workload is still not promotable into the live TT suite.
- The blocker is currently best described as **runtime throughput on the generic scalar TT path under a long cross-stream chain**, not a total cross-stream visibility failure.

## Why This Still Matters

For honest CubeCL parity on TT, a single-device backend still needs to handle:
- producer work submitted on one logical stream
- consumer reads or dependent work submitted on another logical stream
- correct host-visible ordering for shared bindings across those streams
- enough throughput that the upstream stream stress shape finishes in a reasonable validation window

The remaining problem is no longer “does any cross-stream path work?” It is “can the TT runtime and generic kernel path sustain the broad upstream stream workload without timing out?”

## The Real Stream Workloads

The authoritative workload definitions live in:
- [crates/cubecl-core/src/runtime_tests/stream.rs](../cubecl-core/src/runtime_tests/stream.rs)

### Reduced TT wrappers that are currently green

`test_stream_small`:
- `len = 32`
- `rounds = 1`
- `num_loop = 32`
- producer stream: `StreamId { value: 10000 }`
- consumer stream: `StreamId { value: 10001 }`

`test_stream_medium`:
- `len = 256`
- `rounds = 12`
- `num_loop = 256`
- same two-stream producer/consumer pattern

### Broad upstream-sized stream workload that is still blocked

`test_stream`:
- `len = 4096`
- `rounds = 300`
- `num_loop = 4096`
- same two-stream producer/consumer pattern
- every round launches the same `big_task` kernel and then feeds the output into the next round

Kernel shape:
```rust
#[cube(launch)]
pub fn big_task<F: Float>(input: &Array<u32>, output: &mut Array<F>, num_loop: usize) {
    if ABSOLUTE_POS > output.len() {
        terminate!()
    }

    for i in 0..num_loop {
        let pos = i % input.len();
        output[ABSOLUTE_POS] += F::cast_from(input[pos]) / F::cast_from(num_loop);
    }
}
```

This matters because the broad case is not just a stream-ordering test. It is also a very heavy generic scalar runtime shape:
- 4096 logical outputs
- each output iterates 4096 times
- 300 launches in one chain

That means the remaining blocker can be a mix of:
- stream submission/completion overhead
- repeated host-side launch/setup cost
- raw compute throughput of the generic scalar TT path

## Current TT Runtime Design Relevant To The Blocker

Relevant code paths:
- [src/compute/stream.rs](./src/compute/stream.rs)
- [src/compute/command.rs](./src/compute/command.rs)
- [src/compute/context.rs](./src/compute/context.rs)
- [src/runtime.rs](./src/runtime.rs)
- [../cubecl-runtime/src/stream/event.rs](../cubecl-runtime/src/stream/event.rs)
- [../cubecl-runtime/src/client.rs](../cubecl-runtime/src/client.rs)
- [../cubecl-common/src/device/handle/channel.rs](../cubecl-common/src/device/handle/channel.rs)

### TT stream model today

The TT backend is deliberately host-driven, not a fake CUDA event model.

Current TT stream backend state includes:
- per-stream queued/submitted/completed sequence tracking
- per-resource latest-sequence tracking (`resource_latest_seq`)
- ordered pending op batching for:
  - replicated writes
  - `kernel_cube()` launches
- conditional completion through `sync_through(target_seq)`
- resource-lineage-aware read/write completion on the owner stream

Current batching thresholds in [src/compute/stream.rs](./src/compute/stream.rs):
- `TT_STREAM_PENDING_OP_THRESHOLD = 2048`
- `TT_STREAM_IN_FLIGHT_THRESHOLD = 4096`

### Important path split

There are two very different launch paths and they must stay separate:

1. `kernel()` in [src/compute/command.rs](./src/compute/command.rs)
- direct source / characterization path
- synchronous and immediate
- required by low-level hardware characterization tests like `cubetask_compile_pipeline`

2. `kernel_cube()` in [src/compute/command.rs](./src/compute/command.rs)
- normal CubeTask runtime path
- uses queued stream batching
- this is the path relevant to the stream throughput blocker

## What We Fixed Successfully

### 1. Cross-stream ownership bug

This was the biggest real correctness fix.

Changed files:
- [src/compute/storage/gpu.rs](./src/compute/storage/gpu.rs)
- [src/compute/command.rs](./src/compute/command.rs)
- [../cubecl-core/src/runtime_tests/stream.rs](../cubecl-core/src/runtime_tests/stream.rs)

What changed:
- `TtResource` now carries `owner_stream`
- cross-stream resource resolution uses the **owner stream’s** storage for the backing `MeshBuffer`
- the reduced stream tests zero-initialize outputs so they measure ordering instead of accumulation into uninitialized memory

What it proved:
- the original blocker really did include a cross-stream ownership seam
- fixing ownership was enough to make reduced cross-stream tests work on hardware

### 2. Resource-centric completion tracking

Changed files:
- [src/compute/stream.rs](./src/compute/stream.rs)
- [src/compute/command.rs](./src/compute/command.rs)
- [../cubecl-runtime/src/memory_management/memory_manage.rs](../cubecl-runtime/src/memory_management/memory_manage.rs)

What changed:
- TT now tracks latest queued sequence per `StorageId`
- `handle_cursor` is no longer a dummy `0`
- readback and partial-write paths only force completion through the relevant owner/resource lineage
- `kernel_cube()` records output-resource sequences so consumers see the freshest producer state

What it proved:
- TT stream behavior can align with CubeCL’s shared-binding machinery without inventing fake device-side events
- reduced/medium cross-stream correctness remained green after the change

### 3. Ordered TT-side batching for the runtime path

Changed files:
- [src/compute/stream.rs](./src/compute/stream.rs)
- [src/compute/command.rs](./src/compute/command.rs)
- bridge support in `../libtt-metal-cxx`

What changed:
- queued writes and `kernel_cube()` launches preserve stream order
- flush points drain queued ops in-order
- low-level `kernel()` stayed immediate after narrowing the batching scope

What it proved:
- batching for the runtime path is safe if it is scoped carefully
- batching `kernel()` was a regression and had to be backed out

### 4. TT MeshDevice program cache enablement

Changed files:
- [src/runtime.rs](./src/runtime.rs)
- `../libtt-metal-cxx` bridge for program-cache controls

What changed:
- exposed and enabled:
  - `enable_program_cache`
  - `clear_program_cache`
  - `disable_and_clear_program_cache`
  - `num_program_cache_entries`

What it proved:
- TT device-level program cache can be turned on from this backend
- but it is **not yet proven** to be the primary reuse mechanism for the raw `Program` / `launch_from_sources` path we currently use

## What We Tried That Did **Not** Clear The Blocker

### 1. Simply re-promoting the full upstream `stream` wrapper

Approach:
- temporarily wire the upstream broad `stream::test_stream` into the TT harness
- run bounded hardware probes
- remove it again if still blocked

Result:
- repeatedly timed out under bounded runs (most recently 180s)
- no longer looked like the old cold deadlock after the ownership fix and log cleanup
- still not acceptable for live-suite promotion

Conclusion:
- broad stream parity is still blocked
- reduced/medium success does not automatically generalize to the upstream-sized stress workload

### 2. Raising batching thresholds alone

Approach:
- allow much larger pending/in-flight batches on TT streams before forcing submission/completion

Result:
- did not clear the broad stream timeout

Conclusion:
- threshold tuning alone is not enough
- the blocker is deeper than “the batch was too small”

### 3. Treating the problem as a pure event/wait-event issue

Approach:
- inspect `MultiStream` and TT stream backend as if the missing piece were mainly GPU-style event semantics

Result:
- the reduced cross-stream bug turned out to be ownership/resource resolution first
- after the ownership fix, the remaining blocker did not behave like a simple logical dependency deadlock

Conclusion:
- fake CUDA-style event emulation is not the right main fix
- the problem is better understood as host-driven ownership/completion plus throughput

### 4. Staged full-input generic runtime promotion

Approach:
- reuse the existing staged-input reader/writer support in the generic TT path
- enable it in `prepare_cube_task_launch()` when the whole first input fits in the staged tile budget

Why it looked promising:
- the broad stream shape repeatedly scans the entire first input
- for the broad stream case, the full 4096-element `u32` input is only 8 generic pages at the default 2048-byte page size, which fits under the old 16-tile staging budget

Result:
- this regressed even `test_stream_small`
- the path is not yet safe to promote on the runtime generic path
- it was reverted immediately

Conclusion:
- staged full-input support exists in source generation, but its runtime reader/writer contract is still not safe for general promotion here
- this is not the next lever to push without deeper dedicated debugging

### 5. TT program-cache characterization as the main fix

Approach:
- enable TT program cache
- add manual characterization test `tt_program_cache_populates_for_cubetask_pipeline`
- try broad stream again

Result:
- TT program cache is now wired and observable
- but the first characterization showed the raw `Program` / `launch_from_sources` path did not obviously increase `MeshDevice::num_program_cache_entries()`
- broad stream still timed out afterward

Conclusion:
- program cache is worth keeping enabled, but it did not remove the broad blocker
- the remaining problem is still more likely host submission/completion cost and/or generic scalar compute cost than “no program cache”

## Key Diagnostics We Ran And What They Proved

### GDB / backtrace diagnosis

We attached `gdb` to the active child process during broad stream hangs.

What we found:
- the producer-side test thread blocked in `DeviceClient::enqueue` in [../cubecl-common/src/device/handle/channel.rs](../cubecl-common/src/device/handle/channel.rs)
- the TT server worker blocked in TT-Metal submission/completion paths
- TT completion polling stayed active
- this did **not** look like a `MultiStream` dependency cycle

Conclusion:
- the broad blocker was not primarily a logical deadlock
- it was consistent with submission-latency/back-pressure and/or expensive runtime execution

### Queue-depth diagnostic

Approach:
- temporarily raise `CHANNEL_MAX_TASK` from 32 to 1024 for diagnosis

Result:
- changed the shape of producer pressure
- did **not** make the broad workload finish

Conclusion:
- bounded host channel depth is part of the symptom, not the root cause by itself

### TT hot-path logging cleanup

Approach:
- remove TT hot-path debug prints from allocation/resource/launch paths

Result:
- broad workload no longer looked like a cold zero-progress hang
- process-state sampling showed an active CPU-hot worker instead

Conclusion:
- the remaining blocker is more consistent with throughput cost than with a frozen dependency loop

## What The Current Evidence Says

The current best interpretation is:

1. **Basic cross-stream correctness is repaired** for the reduced and medium shapes.
2. **The remaining broad blocker is not primarily a stream-ordering bug.**
3. The broad workload is now most likely limited by one or both of:
   - host write / submission / completion churn across a long chain of launches
   - the cost of executing the broad `big_task` shape on the fully generic scalar TT runtime path
4. Because `big_task` is a 4096-by-4096 scalar loop shape repeated 300 times, it is plausible that a large share of the timeout is now **generic compute throughput**, not just stream infrastructure.

## Current Manual Characterization Hooks

These are intentionally **ignored/manual** so the live TT suite stays honest:

- `tests::cubecl_core_wrappers::stream::test_stream_broad_single_round_manual`
  - one round of the broad 4096-element stream shape
  - this is the next best probe to separate “single launch is already too expensive” from “the 300-round chain is the main issue”

- `tests::tt_program_cache_populates_for_cubetask_pipeline`
  - focused TT program-cache characterization
  - useful for reuse analysis, not part of the normal green lane

## What To Do Next

The next useful work should focus on **measurement and specialization**, not another generic event emulation pass.

Recommended order:

1. Run the one-round broad manual probe:
```bash
TT_METAL_RUN_HARDWARE_TESTS=1 LD_LIBRARY_PATH=/usr/local/lib cargo test -p cubecl-tt-metal tests::cubecl_core_wrappers::stream::test_stream_broad_single_round_manual -- --ignored --exact --nocapture --test-threads=1
```

2. Interpret that result carefully:
- if the one-round probe is already slow, the main blocker is the generic scalar kernel path itself
- if the one-round probe is fine but 300 rounds still time out, the blocker is more strongly in launch/submission/completion churn across the long chain

3. Based on that result, take one of two paths:
- **Compute-path path:** design a more specialized runtime path for this repeated full-input reduction-style shape instead of pushing it through the fully generic scalar writer forever
- **Runtime-throughput path:** reduce repeated host-side setup, write, submit, and completion churn for long chains of identical `kernel_cube()` launches

## References

### CubeCL local references
- Stream workload definitions: [../cubecl-core/src/runtime_tests/stream.rs](../cubecl-core/src/runtime_tests/stream.rs)
- Runtime stream/event alignment: [../cubecl-runtime/src/stream/event.rs](../cubecl-runtime/src/stream/event.rs)
- Compute client submission/read flow: [../cubecl-runtime/src/client.rs](../cubecl-runtime/src/client.rs)
- Device-handle queueing: [../cubecl-common/src/device/handle/channel.rs](../cubecl-common/src/device/handle/channel.rs)
- TT stream backend: [src/compute/stream.rs](./src/compute/stream.rs)
- TT command path: [src/compute/command.rs](./src/compute/command.rs)
- TT launch preparation and source caching: [src/compute/context.rs](./src/compute/context.rs)
- TT runtime/device setup: [src/runtime.rs](./src/runtime.rs)
- Live parity status: [PHASES.md](./PHASES.md)
- Execution checklist: [CHECKLIST.md](./CHECKLIST.md)

### Tenstorrent references
- TT-Metal Kernel APIs: https://docs.tenstorrent.com/tt-metal/latest/tt-metalium/tt_metal/apis/kernel_apis.html

### Related design guidance captured locally
- `tt-lang` findings already summarized in [README.md](./README.md), [PHASES.md](./PHASES.md), and [CHECKLIST.md](./CHECKLIST.md)
