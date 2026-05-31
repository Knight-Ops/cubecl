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
use libtt_metal_cxx::{MeshDevice, MeshWorkload, Program};

use crate::{compute::storage::gpu::TtStorage, runtime::TT_MEMORY_ALIGNMENT};

#[derive(Debug)]
enum PendingOp {
    Workload(MeshWorkload),
    WriteReplicated { storage_id: StorageId, staged: Vec<u8> },
}

/// Stream for TT-Metal (synchronous execution).
#[derive(Debug)]
pub struct Stream {
    mesh_ptr: *const MeshDevice,
    pub memory_management_gpu: MemoryManagement<TtStorage>,
    pub errors: Vec<ServerError>,
    pending_ops: Vec<PendingOp>,
}

impl Stream {
    fn mesh(&self) -> &MeshDevice {
        assert!(!self.mesh_ptr.is_null(), "mesh_ptr not set on TT stream");
        unsafe { &*self.mesh_ptr }
    }

    pub fn enqueue_replicated_write(&mut self, storage_id: StorageId, staged: Vec<u8>) {
        self.pending_ops
            .push(PendingOp::WriteReplicated { storage_id, staged });
    }

    pub fn enqueue_full_mesh_program(&mut self, program: Program) -> Result<(), ServerError> {
        let mesh_ptr = self.mesh_ptr;
        let mesh = unsafe { &*mesh_ptr };
        let mut workload = MeshWorkload::new();
        workload
            .add_program_to_full_mesh(mesh, program)
            .map_err(|err| ServerError::Generic {
                reason: format!("TT-Metal add_program_to_full_mesh failed: {}", err.what()),
                backtrace: BackTrace::capture(),
            })?;
        self.pending_ops.push(PendingOp::Workload(workload));
        Ok(())
    }

    pub fn flush_pending_workload(&mut self) -> Result<(), ServerError> {
        if self.pending_ops.is_empty() {
            return Ok(());
        }
        let mut ops = core::mem::take(&mut self.pending_ops);
        for op in ops.drain(..) {
            match op {
                PendingOp::Workload(mut workload) => {
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
                PendingOp::WriteReplicated { storage_id, staged } => {
                    let mesh_ptr = self.mesh_ptr;
                    let mesh_buffer_ptr = self
                        .memory_management_gpu
                        .storage()
                        .get_mesh_buffer(storage_id)
                        as *const _;
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

/// Synchronous stream backend for TT-Metal.
///
/// TT-Metal does not have traditional async streams/events in the CUDA/HIP sense.
/// All operations are synchronous (blocking enqueue).
/// This backend provides no-op implementations.
#[derive(Debug)]
pub struct TtStreamBackend {
    mesh_ptr: *const libtt_metal_cxx::MeshDevice,
    mem_props: MemoryDeviceProperties,
    mem_config: MemoryConfiguration,
    #[allow(dead_code)]
    mem_alignment: usize,
}

// SAFETY: mesh_ptr is set during TtServer construction and lives as long as the server.
unsafe impl Send for TtStreamBackend {}

impl TtStreamBackend {
    pub fn new(
        mesh_ptr: *const libtt_metal_cxx::MeshDevice,
        mem_props: MemoryDeviceProperties,
        mem_config: MemoryConfiguration,
    ) -> Self {
        Self {
            mesh_ptr,
            mem_alignment: TT_MEMORY_ALIGNMENT as usize,
            mem_props,
            mem_config,
        }
    }
}

impl EventStreamBackend for TtStreamBackend {
    type Stream = Stream;
    type Event = ();

    fn create_stream(&self) -> Self::Stream {
        let mut storage = TtStorage::new();
        storage.set_mesh_ptr(self.mesh_ptr);
        // TT-Metal kernels currently operate on full hardware pages, so sharing a single
        // backing MeshBuffer between logical tensor slices causes page overlap on device.
        // Force exclusive pages until the backend learns page-safe sub-allocation semantics.
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
        }
    }

    fn handle_cursor(_stream: &Self::Stream, _handle: &Binding) -> u64 {
        0
    }

    fn is_healthy(stream: &Self::Stream) -> bool {
        stream.is_healthy()
    }

    fn flush(stream: &mut Self::Stream) -> Self::Event {
        if let Err(err) = stream.flush_pending_workload() {
            stream.error(err);
        }
    }

    fn wait_event(_stream: &mut Self::Stream, _event: Self::Event) {}

    fn wait_event_sync(_event: Self::Event) -> Result<(), ServerError> {
        Ok(())
    }
}
