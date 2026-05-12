use crate::compute::{context::TtContext, stream::TtStreamBackend};
use cubecl_common::backtrace::BackTrace;
use cubecl_common::bytes::Bytes;
use cubecl_core::server::{Binding, CopyDescriptor, IoError, LaunchError, ServerError};
use cubecl_runtime::id::KernelId;
use cubecl_runtime::logging::ServerLogger;
use cubecl_runtime::memory_management::ManagedMemoryHandle;
use cubecl_runtime::stream::ResolvedStreams;
use std::sync::Arc;

use cubecl_cpp::tt_metal::TtKernelSources;

pub(crate) struct Command<'a> {
    ctx: &'a mut TtContext,
    pub(crate) streams: ResolvedStreams<'a, TtStreamBackend>,
}

impl<'a> Command<'a> {
    pub(crate) fn new(
        ctx: &'a mut TtContext,
        streams: ResolvedStreams<'a, TtStreamBackend>,
    ) -> Self {
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

    /// Write host data to a device buffer (blocking).
    pub fn write_to_gpu(&mut self, descriptor: CopyDescriptor, data: Bytes) -> Result<(), IoError> {
        let resource = self.resource(descriptor.handle.clone())?;
        let stream = self.streams.current();
        let storage = &mut stream.memory_management_gpu;
        let mesh_buffer = storage.storage().get_mesh_buffer(resource.storage_id);
        let mesh = self.ctx.mesh();

        let num_bytes: usize = descriptor.shape.iter().product::<usize>() * descriptor.elem_size;
        let slice = if data.len() >= num_bytes {
            &data[..num_bytes]
        } else {
            &data[..]
        };

        mesh.write_mesh_buffer(mesh_buffer, slice)
            .map_err(|e| IoError::Unknown {
                backtrace: BackTrace::capture(),
                description: format!("write_mesh_buffer failed: {}", e.what()),
            })?;

        Ok(())
    }

    /// Read device data to host (blocking).
    pub fn write_to_cpu(&mut self, descriptor: CopyDescriptor) -> Result<Bytes, IoError> {
        let resource = self.resource(descriptor.handle.clone())?;
        let stream = self.streams.current();
        let storage = &mut stream.memory_management_gpu;
        let mesh_buffer = storage.storage().get_mesh_buffer(resource.storage_id);
        let mesh = self.ctx.mesh();

        let num_bytes: usize = descriptor.shape.iter().product::<usize>() * descriptor.elem_size;
        let mut data = vec![0u8; num_bytes];

        mesh.read_mesh_buffer(mesh_buffer, &mut data)
            .map_err(|e| IoError::Unknown {
                backtrace: BackTrace::capture(),
                description: format!("read_mesh_buffer failed: {}", e.what()),
            })?;

        Ok(Bytes::from_bytes_vec(data))
    }

    /// Compile and launch a TT-Metal kernel.
    ///
    /// For Phase 2, the kernel is a copy operation that reads from the
    /// input buffer and writes to the output buffer tile-by-tile.
    pub fn kernel(
        &mut self,
        kernel_id: KernelId,
        input_addr: u32,
        output_addr: u32,
        num_tiles: u32,
        tile_size_bytes: u32,
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        // Generate kernel sources (copy kernel)
        let sources = TtKernelSources::copy_kernel(num_tiles, tile_size_bytes);

        // Compile the kernels into a Program
        let compiled =
            self.ctx
                .compile_kernel(&kernel_id, &sources, input_addr, output_addr, logger)?;

        // Create MeshWorkload and enqueue
        let mut workload = libtt_metal_cxx::MeshWorkload::new();
        workload
            .add_program_to_full_mesh(self.ctx.mesh(), compiled.program)
            .map_err(|e| LaunchError::Unknown {
                reason: format!("add_program_to_full_mesh failed: {}", e.what()),
                backtrace: BackTrace::capture(),
            })?;

        self.ctx
            .mesh()
            .enqueue_workload(&mut workload, true)
            .map_err(|e| LaunchError::Unknown {
                reason: format!("enqueue_workload failed: {}", e.what()),
                backtrace: BackTrace::capture(),
            })?;

        Ok(())
    }
}
