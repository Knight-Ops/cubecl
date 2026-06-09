use cubecl_core::{
    MemoryConfiguration,
    backtrace::BackTrace,
    ir::MemoryDeviceProperties,
    server::{Binding, ServerError, StreamErrorMode},
};
use cubecl_runtime::{
    memory_management::{MemoryManagement, MemoryManagementOptions},
    storage::StorageId,
    stream::EventStreamBackend,
};
use libtt_metal_cxx::{MeshBuffer, MeshDevice, MeshWorkload, Program};
use std::collections::HashMap;
use std::time::Instant;

use crate::{
    compute::profile,
    compute::storage::gpu::TtStorage,
    runtime::{TT_DEFAULT_BUFFER_PAGE_SIZE_BYTES, TT_MEMORY_ALIGNMENT},
};

// The broad upstream stream stress is a long same-stream producer chain followed by
// one cross-stream read. Keep TT-side batching permissive enough to let that chain
// build almost entirely on the host before we force submission or completion.
const TT_STREAM_PENDING_OP_THRESHOLD: usize = 2048;
const TT_STREAM_IN_FLIGHT_THRESHOLD: u64 = 4096;
const TT_STREAM_FENCE_SIZE_BYTES: u64 = TT_DEFAULT_BUFFER_PAGE_SIZE_BYTES;

#[derive(Debug)]
enum PendingOp {
    Workload {
        seq: u64,
        workload: MeshWorkload,
    },
    WriteReplicated {
        seq: u64,
        storage_id: StorageId,
        staged: Vec<u8>,
    },
}

#[derive(Debug, Default, Clone)]
pub struct StreamProgress {
    pub queued: u64,
    pub submitted: u64,
    pub completed: u64,
    pub producer_flushes: u64,
    pub readback_completions: u64,
    pub max_pending_batch_size: usize,
    pub submit_calls: u64,
    pub completion_waits: u64,
}

#[derive(Debug, Clone, Copy)]
pub struct TtEvent {
    completed_seq: u64,
}

/// Stream for TT-Metal.
///
/// The TT backend keeps ordinary submission non-blocking and uses explicit host-side completion
/// fences only at dependency, readback, and sync boundaries. Throughput-sensitive long same-stream
/// chains should be allowed to build much larger batches before forcing a completion fence.
#[derive(Debug)]
pub struct Stream {
    mesh_ptr: *const MeshDevice,
    pub memory_management_gpu: MemoryManagement<TtStorage>,
    pub errors: Vec<ServerError>,
    pending_ops: Vec<PendingOp>,
    last_queued_seq: u64,
    last_submitted_seq: u64,
    last_completed_seq: u64,
    fence_buffer: Option<MeshBuffer>,
    fence_staging: Vec<u8>,
    fence_value: u32,
    resource_latest_seq: HashMap<StorageId, u64>,
    progress: StreamProgress,
}

impl Stream {
    fn mesh(&self) -> &MeshDevice {
        assert!(!self.mesh_ptr.is_null(), "mesh_ptr not set on TT stream");
        unsafe { &*self.mesh_ptr }
    }

    fn next_sequence(&mut self) -> u64 {
        self.last_queued_seq = self.last_queued_seq.saturating_add(1);
        self.progress.queued = self.last_queued_seq;
        self.last_queued_seq
    }

    fn in_flight_count(&self) -> u64 {
        self.last_submitted_seq
            .saturating_sub(self.last_completed_seq)
    }

    fn maybe_apply_backpressure(&mut self) -> Result<(), ServerError> {
        if self.pending_ops.len() >= TT_STREAM_PENDING_OP_THRESHOLD {
            self.flush_pending_workload()?;
        }

        if self.in_flight_count() >= TT_STREAM_IN_FLIGHT_THRESHOLD {
            self.complete_submitted_work()?;
        }

        Ok(())
    }

    pub fn last_queued_seq(&self) -> u64 {
        self.last_queued_seq
    }

    pub fn last_completed_seq(&self) -> u64 {
        self.last_completed_seq
    }

    pub fn mark_dependency_flush(&mut self) {
        self.progress.producer_flushes = self.progress.producer_flushes.saturating_add(1);
    }

    pub fn mark_readback_completion(&mut self) {
        self.progress.readback_completions = self.progress.readback_completions.saturating_add(1);
    }

    pub fn progress(&self) -> &StreamProgress {
        &self.progress
    }

    fn note_resource_sequence(&mut self, storage_id: StorageId, seq: u64) {
        self.resource_latest_seq
            .entry(storage_id)
            .and_modify(|current| *current = (*current).max(seq))
            .or_insert(seq);
    }

    pub fn latest_resource_seq(&self, storage_id: StorageId) -> u64 {
        self.resource_latest_seq
            .get(&storage_id)
            .copied()
            .unwrap_or_default()
    }

    pub fn binding_cursor(&self, handle: &Binding) -> u64 {
        let binding_cursor = self
            .memory_management_gpu
            .get_cursor(handle.memory.clone())
            .unwrap_or_default();
        let storage_cursor = self
            .memory_management_gpu
            .get_storage_id(handle.memory.clone())
            .ok()
            .map(|storage_id| self.latest_resource_seq(storage_id))
            .unwrap_or_default();

        binding_cursor.max(storage_cursor)
    }

    pub fn enqueue_replicated_write(&mut self, storage_id: StorageId, staged: Vec<u8>) {
        let seq = self.next_sequence();
        self.note_resource_sequence(storage_id, seq);
        self.pending_ops.push(PendingOp::WriteReplicated {
            seq,
            storage_id,
            staged,
        });
        if let Err(err) = self.maybe_apply_backpressure() {
            self.error(err);
        }
    }

    pub fn enqueue_full_mesh_program(&mut self, program: Program) -> Result<u64, ServerError> {
        let mesh_ptr = self.mesh_ptr;
        let mesh = unsafe { &*mesh_ptr };
        let mut workload = MeshWorkload::new();
        workload
            .add_program_to_full_mesh(mesh, program)
            .map_err(|err| ServerError::Generic {
                reason: format!("TT-Metal add_program_to_full_mesh failed: {}", err.what()),
                backtrace: BackTrace::capture(),
            })?;
        let seq = self.next_sequence();
        self.pending_ops.push(PendingOp::Workload { seq, workload });
        self.maybe_apply_backpressure()?;
        Ok(seq)
    }

    pub fn note_workload_outputs(
        &mut self,
        seq: u64,
        storage_ids: impl IntoIterator<Item = StorageId>,
    ) {
        for storage_id in storage_ids {
            self.note_resource_sequence(storage_id, seq);
        }
    }

    pub fn flush_pending_workload(&mut self) -> Result<(), ServerError> {
        if self.pending_ops.is_empty() {
            return Ok(());
        }
        let mut ops = core::mem::take(&mut self.pending_ops);
        let batch_size = ops.len();
        let mut workload_ops = 0usize;
        let mut write_ops = 0usize;
        self.progress.max_pending_batch_size = self.progress.max_pending_batch_size.max(batch_size);
        self.progress.submit_calls = self.progress.submit_calls.saturating_add(1);
        let mut max_seq = self.last_submitted_seq;
        let submit_started = Instant::now();

        for op in ops.drain(..) {
            match op {
                PendingOp::Workload { seq, mut workload } => {
                    workload_ops = workload_ops.saturating_add(1);
                    max_seq = max_seq.max(seq);
                    if workload.program_count() == 0 {
                        continue;
                    }
                    self.mesh()
                        .enqueue_workload(&mut workload, false)
                        .map_err(|err| ServerError::Generic {
                            reason: format!("TT-Metal enqueue_workload failed: {}", err.what()),
                            backtrace: BackTrace::capture(),
                        })?;
                }
                PendingOp::WriteReplicated {
                    seq,
                    storage_id,
                    staged,
                } => {
                    write_ops = write_ops.saturating_add(1);
                    max_seq = max_seq.max(seq);
                    let mesh_ptr = self.mesh_ptr;
                    let mesh_buffer_ptr =
                        self.memory_management_gpu
                            .storage()
                            .get_mesh_buffer(storage_id) as *const _;
                    let mesh = unsafe { &*mesh_ptr };
                    let mesh_buffer = unsafe { &*mesh_buffer_ptr };
                    mesh.write_mesh_buffer_with_mode(mesh_buffer, &staged, false)
                        .map_err(|err| ServerError::Generic {
                            reason: format!("TT-Metal write_mesh_buffer failed: {}", err.what()),
                            backtrace: BackTrace::capture(),
                        })?;
                }
            }
        }

        self.last_submitted_seq = max_seq;
        profile::record_pending_submit(
            batch_size,
            workload_ops,
            write_ops,
            submit_started.elapsed(),
        );
        Ok(())
    }

    pub fn complete_submitted_work(&mut self) -> Result<(), ServerError> {
        if self.last_completed_seq >= self.last_submitted_seq {
            return Ok(());
        }

        let wait_started = Instant::now();
        self.fence_value = self.fence_value.wrapping_add(1);
        self.fence_staging[..4].copy_from_slice(&self.fence_value.to_le_bytes());
        let fence_buffer = self
            .fence_buffer
            .as_ref()
            .ok_or_else(|| ServerError::Generic {
                reason: "TT stream completion fence unavailable without a live mesh".into(),
                backtrace: BackTrace::capture(),
            })?;
        self.mesh()
            .write_mesh_buffer_with_mode(fence_buffer, &self.fence_staging, true)
            .map_err(|err| ServerError::Generic {
                reason: format!("TT-Metal stream completion fence failed: {}", err.what()),
                backtrace: BackTrace::capture(),
            })?;
        self.last_completed_seq = self.last_submitted_seq;
        self.progress.completed = self.last_completed_seq;
        self.progress.completion_waits = self.progress.completion_waits.saturating_add(1);
        profile::record_completion_wait(wait_started.elapsed());
        Ok(())
    }

    pub fn sync_through(&mut self, target_seq: u64) -> Result<(), ServerError> {
        if target_seq <= self.last_completed_seq {
            return Ok(());
        }

        if target_seq > self.last_submitted_seq {
            self.flush_pending_workload()?;
        }

        if target_seq > self.last_completed_seq {
            self.complete_submitted_work()?;
        }

        Ok(())
    }

    pub fn flush_errors(&mut self, mode: StreamErrorMode) -> Result<(), ServerError> {
        if mode.flush {
            let errors = core::mem::take(&mut self.errors);
            if !mode.ignore && !errors.is_empty() {
                return Err(ServerError::ServerUnhealthy {
                    errors,
                    backtrace: BackTrace::capture(),
                });
            }
        } else if !mode.ignore && !self.errors.is_empty() {
            return Err(ServerError::ServerUnhealthy {
                errors: self.errors.clone(),
                backtrace: BackTrace::capture(),
            });
        }

        Ok(())
    }

    pub fn error(&mut self, error: ServerError) {
        self.errors.push(error);
    }

    pub fn is_healthy(&self) -> bool {
        self.errors.is_empty()
    }
}

/// Host-driven stream backend for TT-Metal.
///
/// TT does not expose CUDA-style device events here, so the backend models ordering with
/// host-visible completion points. Ordinary submission stays non-blocking; dependency waits,
/// CPU readbacks, and explicit syncs fence completion.
#[derive(Debug)]
pub struct TtStreamBackend {
    mesh_ptr: *const libtt_metal_cxx::MeshDevice,
    mem_props: MemoryDeviceProperties,
    #[allow(dead_code)]
    mem_alignment: usize,
}

// SAFETY: mesh_ptr is set during TtServer construction and lives as long as the server.
unsafe impl Send for TtStreamBackend {}

fn create_fence_buffer(mesh_ptr: *const MeshDevice) -> Option<MeshBuffer> {
    if mesh_ptr.is_null() {
        return None;
    }
    let mesh = unsafe { &*mesh_ptr };
    Some(
        MeshBuffer::create_replicated(
            mesh,
            TT_STREAM_FENCE_SIZE_BYTES,
            TT_DEFAULT_BUFFER_PAGE_SIZE_BYTES,
            0,
        )
        .expect("TT stream completion fence allocation should succeed"),
    )
}

impl TtStreamBackend {
    pub fn new(
        mesh_ptr: *const libtt_metal_cxx::MeshDevice,
        mem_props: MemoryDeviceProperties,
        _mem_config: MemoryConfiguration,
    ) -> Self {
        Self {
            mesh_ptr,
            mem_alignment: TT_MEMORY_ALIGNMENT as usize,
            mem_props,
        }
    }
}

impl EventStreamBackend for TtStreamBackend {
    type Stream = Stream;
    type Event = TtEvent;

    fn create_stream(&self) -> Self::Stream {
        let mut storage = TtStorage::new();
        storage.set_mesh_ptr(self.mesh_ptr);
        let memory_management_gpu = MemoryManagement::from_configuration(
            storage,
            &self.mem_props,
            MemoryConfiguration::ExclusivePages,
            Default::default(),
            MemoryManagementOptions::new("Main GPU Memory"),
        );
        Stream {
            mesh_ptr: self.mesh_ptr,
            memory_management_gpu,
            errors: Vec::new(),
            pending_ops: Vec::new(),
            last_queued_seq: 0,
            last_submitted_seq: 0,
            last_completed_seq: 0,
            fence_buffer: create_fence_buffer(self.mesh_ptr),
            fence_staging: vec![0u8; TT_STREAM_FENCE_SIZE_BYTES as usize],
            fence_value: 0,
            resource_latest_seq: HashMap::new(),
            progress: StreamProgress::default(),
        }
    }

    fn handle_cursor(stream: &Self::Stream, handle: &Binding) -> u64 {
        stream.binding_cursor(handle)
    }

    fn is_healthy(stream: &Self::Stream) -> bool {
        stream.is_healthy()
    }

    fn flush(stream: &mut Self::Stream) -> Self::Event {
        let target_seq = stream.last_queued_seq();
        stream.mark_dependency_flush();
        if let Err(err) = stream.sync_through(target_seq) {
            stream.error(err);
        }
        TtEvent {
            completed_seq: stream.last_completed_seq(),
        }
    }

    fn wait_event(stream: &mut Self::Stream, event: Self::Event) {
        if event.completed_seq > stream.last_completed_seq {
            stream.last_completed_seq = event.completed_seq;
            stream.progress.completed = stream.last_completed_seq;
        }
    }

    fn wait_event_sync(_event: Self::Event) -> Result<(), ServerError> {
        Ok(())
    }
}
