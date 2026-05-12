use crate::compute::{context::TtContext, stream::TtStreamBackend};
use cubecl_core::server::{Binding, IoError, ServerError};
use cubecl_runtime::memory_management::ManagedMemoryHandle;
use cubecl_runtime::stream::ResolvedStreams;

pub(crate) struct Command<'a> {
    ctx: &'a mut TtContext,
    pub(crate) streams: ResolvedStreams<'a, TtStreamBackend>,
}

impl<'a> Command<'a> {
    pub(crate) fn new(ctx: &'a mut TtContext, streams: ResolvedStreams<'a, TtStreamBackend>) -> Self {
        Self { ctx, streams }
    }

    pub fn reserve(&mut self, size: u64) -> Result<ManagedMemoryHandle, IoError> {
        self.streams.current().memory_management_gpu.reserve(size)
    }

    pub fn error(&mut self, error: ServerError) {
        self.streams.current().errors.push(error);
    }

    pub fn resource(
        &mut self,
        binding: Binding,
    ) -> Result<crate::compute::storage::gpu::TtResource, IoError> {
        self.streams
            .get(&binding.stream)
            .memory_management_gpu
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
    }
}
