use cubecl_core::{
    MemoryConfiguration,
    ir::MemoryDeviceProperties,
    server::{Binding, ServerError},
};
use cubecl_runtime::{
    memory_management::{MemoryManagement, MemoryManagementOptions},
    stream::EventStreamBackend,
};

use crate::compute::storage::gpu::TtStorage;

/// Stream for TT-Metal (synchronous execution).
#[derive(Debug)]
pub struct Stream {
    pub memory_management_gpu: MemoryManagement<TtStorage>,
    pub errors: Vec<ServerError>,
}

/// Synchronous stream backend for TT-Metal.
///
/// TT-Metal does not have traditional async streams/events in the CUDA/HIP sense.
/// All operations are synchronous (blocking enqueue).
/// This backend provides no-op implementations.
#[derive(Debug)]
pub struct TtStreamBackend {
    mem_props: MemoryDeviceProperties,
    mem_config: MemoryConfiguration,
    mem_alignment: usize,
}

impl TtStreamBackend {
    pub fn new() -> Self {
        Self {
            mem_props: MemoryDeviceProperties {
                max_page_size: 16 * 1024 * 1024,
                alignment: 32,
            },
            mem_config: MemoryConfiguration::default(),
            mem_alignment: 32,
        }
    }
}

impl Default for TtStreamBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl EventStreamBackend for TtStreamBackend {
    type Stream = Stream;
    type Event = ();

    fn create_stream(&self) -> Self::Stream {
        let storage = TtStorage::new();
        let memory_management_gpu = MemoryManagement::from_configuration(
            storage,
            &self.mem_props,
            self.mem_config.clone(),
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
        stream.errors.is_empty()
    }

    fn flush(_stream: &mut Self::Stream) -> Self::Event {}

    fn wait_event(_stream: &mut Self::Stream, _event: Self::Event) {}

    fn wait_event_sync(_event: Self::Event) -> Result<(), ServerError> {
        Ok(())
    }
}
