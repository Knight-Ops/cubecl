use crate::compute::{context::TtContext, stream::TtStreamBackend};
use crate::runtime::TtCompiler;
use cubecl_common::backtrace::BackTrace;
use cubecl_common::bytes::Bytes;
use cubecl_core::server::{
    Binding, CopyDescriptor, ExecutionMode, IoError, LaunchError, ServerError,
};
use cubecl_core::zspace::striding::has_pitched_row_major_strides;
use cubecl_runtime::compiler::CubeTask;
use cubecl_runtime::logging::ServerLogger;
use cubecl_runtime::memory_management::ManagedMemoryHandle;
use cubecl_runtime::stream::ResolvedStreams;
use std::sync::Arc;

use cubecl_cpp::tt_metal::TtKernelSources;

pub(crate) struct Command<'a> {
    ctx: &'a mut TtContext,
    mesh: &'a libtt_metal_cxx::MeshDevice,
    pub(crate) streams: ResolvedStreams<'a, TtStreamBackend>,
}

impl<'a> Command<'a> {
    pub(crate) fn new(
        ctx: &'a mut TtContext,
        mesh: &'a libtt_metal_cxx::MeshDevice,
        streams: ResolvedStreams<'a, TtStreamBackend>,
    ) -> Self {
        Self { ctx, mesh, streams }
    }

    pub fn reserve(&mut self, size: u64) -> Result<ManagedMemoryHandle, IoError> {
        self.streams.current().memory_management_gpu.reserve(size)
    }

    /// Bind a reserved memory allocation to a managed memory handle.
    pub fn bind(&mut self, reserved: ManagedMemoryHandle, new: ManagedMemoryHandle) {
        let cursor = self.cursor();
        self.streams
            .current()
            .memory_management_gpu
            .bind(reserved, new, cursor)
            .unwrap();
    }

    /// Allocate host-side staging memory (for CPU↔GPU transfers).
    pub fn reserve_cpu(&mut self, size: usize) -> Bytes {
        Bytes::from_bytes_vec(vec![0u8; size])
    }

    pub fn cursor(&self) -> u64 {
        self.streams.cursor
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

        // Validate strides: only contiguous row-major tensors are supported
        if !has_pitched_row_major_strides(&descriptor.shape, &descriptor.strides) {
            return Err(IoError::UnsupportedStrides {
                backtrace: BackTrace::capture(),
            });
        }

        let num_bytes: usize = descriptor.shape.iter().product::<usize>() * descriptor.elem_size;
        if data.len() < num_bytes {
            return Err(IoError::Unknown {
                backtrace: BackTrace::capture(),
                description: format!(
                    "write_to_gpu: data size {} < expected {}",
                    data.len(),
                    num_bytes
                ),
            });
        }

        self.mesh
            .write_mesh_buffer(mesh_buffer, &data[..num_bytes])
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

        let num_bytes: usize = descriptor.shape.iter().product::<usize>() * descriptor.elem_size;
        let mut data = vec![0u8; num_bytes];

        self.mesh
            .read_mesh_buffer(mesh_buffer, &mut data)
            .map_err(|e| IoError::Unknown {
                backtrace: BackTrace::capture(),
                description: format!("read_mesh_buffer failed: {}", e.what()),
            })?;

        Ok(Bytes::from_bytes_vec(data))
    }

    /// Compile and launch a TT-Metal kernel.
    pub fn kernel(
        &mut self,
        sources: &TtKernelSources,
        input_addrs: &[u32],
        output_addrs: &[u32],
        reader_compile_args: &[u32],
        writer_compile_args: &[u32],
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        // Compile the kernels into a Program
        let compiled = self.ctx.compile_kernel(
            sources,
            input_addrs,
            output_addrs,
            reader_compile_args,
            writer_compile_args,
            logger,
        )?;

        // Create MeshWorkload and enqueue
        let mut workload = libtt_metal_cxx::MeshWorkload::new();
        workload
            .add_program_to_full_mesh(self.mesh, compiled.program)
            .map_err(|e| LaunchError::Unknown {
                reason: format!("add_program_to_full_mesh failed: {}", e.what()),
                backtrace: BackTrace::capture(),
            })?;

        self.mesh
            .enqueue_workload(&mut workload, true)
            .map_err(|e| LaunchError::Unknown {
                reason: format!("enqueue_workload failed: {}", e.what()),
                backtrace: BackTrace::capture(),
            })?;

        Ok(())
    }

    /// Compile and launch a `CubeTask` kernel.
    pub fn kernel_cube(
        &mut self,
        cube_kernel: Box<dyn CubeTask<TtCompiler>>,
        mode: ExecutionMode,
        input_addrs: &[u32],
        output_addrs: &[u32],
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        let compiled =
            self.ctx
                .compile_cube_task(cube_kernel, mode, input_addrs, output_addrs, logger)?;

        let mut workload = libtt_metal_cxx::MeshWorkload::new();
        workload
            .add_program_to_full_mesh(self.mesh, compiled.program)
            .map_err(|e| LaunchError::Unknown {
                reason: format!("add_program_to_full_mesh: {}", e.what()),
                backtrace: BackTrace::capture(),
            })?;
        self.mesh
            .enqueue_workload(&mut workload, true)
            .map_err(|e| LaunchError::Unknown {
                reason: format!("enqueue_workload: {}", e.what()),
                backtrace: BackTrace::capture(),
            })?;
        Ok(())
    }
}
