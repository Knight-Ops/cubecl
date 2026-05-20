use cubecl_core::{
    MemoryConfiguration,
    backtrace::BackTrace,
    ir::MemoryDeviceProperties,
    server::{Binding, ServerError, StreamErrorMode},
};
use cubecl_runtime::{
    memory_management::{MemoryManagement, MemoryManagementOptions},
    stream::EventStreamBackend,
};

use crate::{compute::storage::gpu::TtStorage, runtime::TT_MEMORY_ALIGNMENT};

/// Stream for TT-Metal (synchronous execution).
#[derive(Debug)]
pub struct Stream {
    pub memory_management_gpu: MemoryManagement<TtStorage>,
    pub errors: Vec<ServerError>,
}

impl Stream {
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
            memory_management_gpu,
            errors: Vec::new(),
        }
    }

    fn handle_cursor(_stream: &Self::Stream, _handle: &Binding) -> u64 {
        0
    }

    fn is_healthy(stream: &Self::Stream) -> bool {
        stream.is_healthy()
    }

    fn flush(_stream: &mut Self::Stream) -> Self::Event {}

    fn wait_event(_stream: &mut Self::Stream, _event: Self::Event) {}

    fn wait_event_sync(_event: Self::Event) -> Result<(), ServerError> {
        Ok(())
    }
}
