use std::{
    sync::{Mutex, OnceLock},
    time::Duration,
};

#[derive(Debug, Default, Clone)]
pub(crate) struct LaunchProfileSnapshot {
    pub kernel_cube_launches: u64,
    pub queued_launches: u64,
    pub immediate_launches: u64,
    pub prepare_calls: u64,
    pub prepare_ns: u64,
    pub bridge_calls: u64,
    pub bridge_ns: u64,
    pub compile_calls: u64,
    pub compile_ns: u64,
    pub immediate_submit_calls: u64,
    pub immediate_submit_ns: u64,
    pub pending_submit_batches: u64,
    pub pending_submit_ops: u64,
    pub pending_submit_workloads: u64,
    pub pending_submit_writes: u64,
    pub pending_submit_ns: u64,
    pub completion_wait_calls: u64,
    pub completion_wait_ns: u64,
}

fn enabled() -> bool {
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| std::env::var_os("CUBECL_TT_METAL_STREAM_PROFILE").is_some())
}

fn snapshot_store() -> &'static Mutex<LaunchProfileSnapshot> {
    static STORE: OnceLock<Mutex<LaunchProfileSnapshot>> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(LaunchProfileSnapshot::default()))
}

fn nanos(duration: Duration) -> u64 {
    u64::try_from(duration.as_nanos()).unwrap_or(u64::MAX)
}

fn mutate(f: impl FnOnce(&mut LaunchProfileSnapshot)) {
    if !enabled() {
        return;
    }
    let mut snapshot = snapshot_store()
        .lock()
        .expect("TT launch profile mutex should not be poisoned");
    f(&mut snapshot);
}

#[allow(dead_code)]
pub(crate) fn reset_launch_profile() {
    let mut snapshot = snapshot_store()
        .lock()
        .expect("TT launch profile mutex should not be poisoned");
    *snapshot = LaunchProfileSnapshot::default();
}

#[allow(dead_code)]
pub(crate) fn snapshot_launch_profile() -> LaunchProfileSnapshot {
    snapshot_store()
        .lock()
        .expect("TT launch profile mutex should not be poisoned")
        .clone()
}

pub(crate) fn record_kernel_cube_launch(queued: bool) {
    mutate(|snapshot| {
        snapshot.kernel_cube_launches = snapshot.kernel_cube_launches.saturating_add(1);
        if queued {
            snapshot.queued_launches = snapshot.queued_launches.saturating_add(1);
        } else {
            snapshot.immediate_launches = snapshot.immediate_launches.saturating_add(1);
        }
    });
}

pub(crate) fn record_prepare(duration: Duration) {
    mutate(|snapshot| {
        snapshot.prepare_calls = snapshot.prepare_calls.saturating_add(1);
        snapshot.prepare_ns = snapshot.prepare_ns.saturating_add(nanos(duration));
    });
}

pub(crate) fn record_bridge(duration: Duration) {
    mutate(|snapshot| {
        snapshot.bridge_calls = snapshot.bridge_calls.saturating_add(1);
        snapshot.bridge_ns = snapshot.bridge_ns.saturating_add(nanos(duration));
    });
}

pub(crate) fn record_compile(duration: Duration) {
    mutate(|snapshot| {
        snapshot.compile_calls = snapshot.compile_calls.saturating_add(1);
        snapshot.compile_ns = snapshot.compile_ns.saturating_add(nanos(duration));
    });
}

pub(crate) fn record_immediate_submit(duration: Duration) {
    mutate(|snapshot| {
        snapshot.immediate_submit_calls = snapshot.immediate_submit_calls.saturating_add(1);
        snapshot.immediate_submit_ns = snapshot.immediate_submit_ns.saturating_add(nanos(duration));
    });
}

pub(crate) fn record_pending_submit(
    batch_size: usize,
    workload_ops: usize,
    write_ops: usize,
    duration: Duration,
) {
    mutate(|snapshot| {
        snapshot.pending_submit_batches = snapshot.pending_submit_batches.saturating_add(1);
        snapshot.pending_submit_ops = snapshot
            .pending_submit_ops
            .saturating_add(batch_size as u64);
        snapshot.pending_submit_workloads = snapshot
            .pending_submit_workloads
            .saturating_add(workload_ops as u64);
        snapshot.pending_submit_writes = snapshot
            .pending_submit_writes
            .saturating_add(write_ops as u64);
        snapshot.pending_submit_ns = snapshot.pending_submit_ns.saturating_add(nanos(duration));
    });
}

pub(crate) fn record_completion_wait(duration: Duration) {
    mutate(|snapshot| {
        snapshot.completion_wait_calls = snapshot.completion_wait_calls.saturating_add(1);
        snapshot.completion_wait_ns = snapshot.completion_wait_ns.saturating_add(nanos(duration));
    });
}
