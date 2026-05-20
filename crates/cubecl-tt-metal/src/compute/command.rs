use crate::compute::{
    context::TtContext,
    storage::gpu::{TtBufferLayout, TtResource},
    stream::TtStreamBackend,
};
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TtLaunchTarget {
    FullMesh,
}

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
            .expect("bind should succeed for reserved TT allocations");
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

    pub fn resource(&mut self, binding: Binding) -> Result<TtResource, IoError> {
        println!("[resource] getting resource");
        self.streams
            .get(&binding.stream)
            .memory_management_gpu
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)
    }

    /// Write host data to a device buffer (blocking).
    pub fn write_to_gpu(&mut self, descriptor: CopyDescriptor, data: Bytes) -> Result<(), IoError> {
        println!("[write_to_gpu] enter");
        let resource = self.resource(descriptor.handle.clone())?;
        if resource.layout != TtBufferLayout::Replicated {
            return Err(IoError::Unknown {
                backtrace: BackTrace::capture(),
                description: "write_to_gpu only supports replicated TT buffers today".into(),
            });
        }
        let stream = self.streams.current();
        let storage = &mut stream.memory_management_gpu;
        let mesh_buffer = storage.storage().get_mesh_buffer(resource.storage_id);

        if !has_pitched_row_major_strides(&descriptor.shape, &descriptor.strides) {
            return Err(IoError::UnsupportedStrides {
                backtrace: BackTrace::capture(),
            });
        }

        let num_bytes: usize = descriptor.shape.iter().product::<usize>() * descriptor.elem_size;
        if num_bytes as u64 > resource.size {
            return Err(IoError::Unknown {
                backtrace: BackTrace::capture(),
                description: format!(
                    "write_to_gpu: logical write {} exceeds resource size {}",
                    num_bytes, resource.size
                ),
            });
        }
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

        let mut staged = vec![0u8; resource.allocation_size as usize];
        let start = resource.allocation_offset as usize;
        let end = start + num_bytes;

        if resource.allocation_offset != 0 || num_bytes as u64 != resource.allocation_size {
            let read_existing = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.mesh.read_mesh_buffer(mesh_buffer, &mut staged)
            }));
            match read_existing {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    return Err(IoError::Unknown {
                        backtrace: BackTrace::capture(),
                        description: format!("read_mesh_buffer before partial write failed: {}", e.what()),
                    });
                }
                Err(_panic) => {
                    return Err(IoError::Unknown {
                        backtrace: BackTrace::capture(),
                        description: "read_mesh_buffer before partial write panicked".into(),
                    });
                }
            }
        }

        staged[start..end].copy_from_slice(&data[..num_bytes]);

        println!(
            "[write_to_gpu] calling write_mesh_buffer {} bytes ({} staged)",
            num_bytes,
            staged.len()
        );
        let write_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.mesh.write_mesh_buffer(mesh_buffer, &staged)
        }));
        match write_result {
            Ok(Ok(())) => println!("[write_to_gpu] write_mesh_buffer ok"),
            Ok(Err(e)) => {
                println!("[write_to_gpu] write_mesh_buffer CXX error: {}", e.what());
                return Err(IoError::Unknown {
                    backtrace: BackTrace::capture(),
                    description: format!("write_mesh_buffer failed: {}", e.what()),
                });
            }
            Err(_panic) => {
                println!("[write_to_gpu] write_mesh_buffer PANICKED");
                return Err(IoError::Unknown {
                    backtrace: BackTrace::capture(),
                    description: "write_mesh_buffer panicked".into(),
                });
            }
        }
        println!("[write_to_gpu] done");
        Ok(())
    }

    /// Read device data to host (blocking).
    pub fn write_to_cpu(&mut self, descriptor: CopyDescriptor) -> Result<Bytes, IoError> {
        let resource = self.resource(descriptor.handle.clone())?;
        if resource.layout != TtBufferLayout::Replicated {
            return Err(IoError::Unknown {
                backtrace: BackTrace::capture(),
                description: "write_to_cpu only supports replicated TT buffers today".into(),
            });
        }
        let stream = self.streams.current();
        let storage = &mut stream.memory_management_gpu;
        let mesh_buffer = storage.storage().get_mesh_buffer(resource.storage_id);

        let num_bytes: usize = descriptor.shape.iter().product::<usize>() * descriptor.elem_size;
        if num_bytes as u64 > resource.size {
            return Err(IoError::Unknown {
                backtrace: BackTrace::capture(),
                description: format!(
                    "write_to_cpu: logical read {} exceeds resource size {}",
                    num_bytes, resource.size
                ),
            });
        }
        let mut staged = vec![0u8; resource.allocation_size as usize];

        let read_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            self.mesh.read_mesh_buffer(mesh_buffer, &mut staged)
        }));
        match read_result {
            Ok(Ok(())) => {}
            Ok(Err(e)) => {
                return Err(IoError::Unknown {
                    backtrace: BackTrace::capture(),
                    description: format!("read_mesh_buffer failed: {}", e.what()),
                });
            }
            Err(_panic) => {
                return Err(IoError::Unknown {
                    backtrace: BackTrace::capture(),
                    description: "read_mesh_buffer panicked".into(),
                });
            }
        }

        let start = resource.allocation_offset as usize;
        let end = start + num_bytes;
        Ok(Bytes::from_bytes_vec(staged[start..end].to_vec()))
    }

    fn launch_target_for_resources(
        &self,
        resources: &[TtResource],
    ) -> Result<TtLaunchTarget, LaunchError> {
        if resources
            .iter()
            .any(|resource| resource.layout != TtBufferLayout::Replicated)
        {
            return Err(LaunchError::Unknown {
                reason: "TT-Metal sharded/distributed buffer layouts are not launchable yet".into(),
                backtrace: BackTrace::capture(),
            });
        }

        Ok(TtLaunchTarget::FullMesh)
    }

    fn add_program_to_workload(
        &self,
        workload: &mut libtt_metal_cxx::MeshWorkload,
        program: libtt_metal_cxx::Program,
        target: TtLaunchTarget,
    ) -> Result<(), LaunchError> {
        match target {
            TtLaunchTarget::FullMesh => catch_launch_panic("add_program_to_full_mesh failed", || {
                workload.add_program_to_full_mesh(self.mesh, program)
            }),
        }
    }

    /// Compile and launch a TT-Metal kernel.
    pub fn kernel(
        &mut self,
        sources: &TtKernelSources,
        input_addrs: &[u32],
        output_addrs: &[u32],
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        let compiled = self
            .ctx
            .compile_kernel(sources, input_addrs, output_addrs, logger)?;
        let target = TtLaunchTarget::FullMesh;

        let mut workload = libtt_metal_cxx::MeshWorkload::new();
        self.add_program_to_workload(&mut workload, compiled.program, target)?;

        catch_launch_panic("enqueue_workload failed", || {
            self.mesh.enqueue_workload(&mut workload, true)
        })?;

        Ok(())
    }

    /// Compile and launch a `CubeTask` kernel.
    pub fn kernel_cube(
        &mut self,
        cube_kernel: Box<dyn CubeTask<TtCompiler>>,
        mode: ExecutionMode,
        resources: &[TtResource],
        info: &cubecl_runtime::server::MetadataBindingInfo,
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        let _ = println!(
            "[kernel_cube] enter
"
        );
        let compiled = self
            .ctx
            .compile_cube_task(cube_kernel, mode, resources, info, logger)?;
        let target = self.launch_target_for_resources(resources)?;
        let _ = println!(
            "[kernel_cube] compile done
"
        );

        let mut workload = libtt_metal_cxx::MeshWorkload::new();
        self.add_program_to_workload(&mut workload, compiled.program, target)?;
        catch_launch_panic("enqueue_workload failed", || {
            self.mesh.enqueue_workload(&mut workload, true)
        })?;
        let _ = println!(
            "[kernel_cube] exit
"
        );
        Ok(())
    }
}

fn catch_launch_panic<T>(
    context: &'static str,
    f: impl FnOnce() -> Result<T, libtt_metal_cxx::Exception>,
) -> Result<T, LaunchError> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(f)) {
        Ok(Ok(value)) => Ok(value),
        Ok(Err(err)) => Err(LaunchError::Unknown {
            reason: format!("{context}: {}", err.what()),
            backtrace: BackTrace::capture(),
        }),
        Err(_) => Err(LaunchError::Unknown {
            reason: format!("{context}: TT-Metal host call panicked"),
            backtrace: BackTrace::capture(),
        }),
    }
}
