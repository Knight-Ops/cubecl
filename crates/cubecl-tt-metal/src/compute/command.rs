use crate::compute::{
    context::{PreparedLaunch, TtContext},
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
use cubecl_runtime::server::CubeCount;
use cubecl_runtime::stream::ResolvedStreams;
use std::sync::Arc;

use cubecl_cpp::tt_metal::TtKernelSources;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum TtLaunchTarget {
    FullMesh,
}

#[derive(Debug)]
struct TtTiledTempBuffer {
    mesh_buffer: libtt_metal_cxx::MeshBuffer,
    address: u32,
    compile_args: Vec<u32>,
}

#[derive(Debug)]
struct TtTiledOutputBridge {
    temp: TtTiledTempBuffer,
    original: TtResource,
    logical_size_bytes: usize,
}

#[derive(Debug)]
struct TtTiledLaunchBridge {
    row_major_rows: u32,
    row_major_cols: u32,
    logical_elem_size_bytes: u32,
    data_format_tt: u8,
    _inputs: Vec<TtTiledTempBuffer>,
    outputs: Vec<TtTiledOutputBridge>,
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
        let mut resource = self
            .streams
            .get(&binding.stream)
            .memory_management_gpu
            .get_resource(binding.memory, binding.offset_start, binding.offset_end)?;
        resource.owner_stream = binding.stream;
        Ok(resource)
    }

    fn flush_pending_stream(&mut self, stream_id: &cubecl_common::stream_id::StreamId) -> Result<(), IoError> {
        self.streams
            .get(stream_id)
            .flush_pending_workload()
            .map_err(|err| IoError::Unknown {
                backtrace: BackTrace::capture(),
                description: format!("failed to flush pending TT workload on stream {stream_id}: {err:?}"),
            })
    }

    fn flush_current_pending_workload(&mut self) -> Result<(), LaunchError> {
        self.streams
            .current()
            .flush_pending_workload()
            .map_err(|err| LaunchError::Unknown {
                reason: format!("failed to flush pending TT workload: {err:?}"),
                backtrace: BackTrace::capture(),
            })
    }

    pub(crate) fn read_resource_bytes(
        &mut self,
        resource: &TtResource,
        num_bytes: usize,
    ) -> Result<Vec<u8>, IoError> {
        if resource.layout != TtBufferLayout::Replicated {
            return Err(IoError::Unknown {
                backtrace: BackTrace::capture(),
                description: "read_resource_bytes only supports replicated TT buffers today".into(),
            });
        }
        if num_bytes as u64 > resource.size {
            return Err(IoError::Unknown {
                backtrace: BackTrace::capture(),
                description: format!(
                    "read_resource_bytes: logical read {} exceeds resource size {}",
                    num_bytes, resource.size
                ),
            });
        }

        self.flush_pending_stream(&resource.owner_stream)?;

        let stream = self.streams.get(&resource.owner_stream);
        let storage = &mut stream.memory_management_gpu;
        let mesh_buffer = storage.storage().get_mesh_buffer(resource.storage_id);
        let mut staged = vec![0u8; resource.allocation_size as usize];
        mesh_read_buffer(
            self.mesh,
            mesh_buffer,
            &mut staged,
            "read_mesh_buffer failed",
            true,
        )?;

        let start = resource.allocation_offset as usize;
        let end = start + num_bytes;
        Ok(staged[start..end].to_vec())
    }

    pub(crate) fn write_resource_bytes(
        &mut self,
        resource: &TtResource,
        bytes: &[u8],
    ) -> Result<(), IoError> {
        if resource.layout != TtBufferLayout::Replicated {
            return Err(IoError::Unknown {
                backtrace: BackTrace::capture(),
                description: "write_resource_bytes only supports replicated TT buffers today"
                    .into(),
            });
        }
        if bytes.len() as u64 > resource.size {
            return Err(IoError::Unknown {
                backtrace: BackTrace::capture(),
                description: format!(
                    "write_resource_bytes: logical write {} exceeds resource size {}",
                    bytes.len(),
                    resource.size
                ),
            });
        }

        if resource.allocation_offset != 0 || bytes.len() as u64 != resource.allocation_size {
            self.flush_pending_stream(&resource.owner_stream)?;
        }

        let stream = self.streams.get(&resource.owner_stream);
        let storage = &mut stream.memory_management_gpu;
        let mesh_buffer = storage.storage().get_mesh_buffer(resource.storage_id);

        let mut staged = vec![0u8; resource.allocation_size as usize];
        let start = resource.allocation_offset as usize;
        let end = start + bytes.len();

        if resource.allocation_offset != 0 || bytes.len() as u64 != resource.allocation_size {
            mesh_read_buffer(
                self.mesh,
                mesh_buffer,
                &mut staged,
                "read_mesh_buffer before partial write failed",
                true,
            )?;
        }

        staged[start..end].copy_from_slice(bytes);
        self.streams
            .get(&resource.owner_stream)
            .enqueue_replicated_write(resource.storage_id, staged);
        Ok(())
    }

    fn create_temp_tiled_buffer(
        &self,
        staged: &[u8],
        page_size: u64,
    ) -> Result<TtTiledTempBuffer, LaunchError> {
        let mesh_buffer = catch_launch_panic("temporary MeshBuffer allocation failed", || {
            libtt_metal_cxx::MeshBuffer::create_replicated(
                self.mesh,
                staged.len() as u64,
                page_size,
                0,
            )
        })?;
        let compile_args = mesh_buffer
            .compile_args()
            .map_err(map_tt_exception_to_launch(
                "temporary MeshBuffer compile args failed",
            ))?;
        let address = u32::try_from(mesh_buffer.address()).map_err(|_| LaunchError::Unknown {
            reason: "temporary MeshBuffer address exceeds 32-bit DRAM address space".into(),
            backtrace: BackTrace::capture(),
        })?;
        if !staged.is_empty() {
            let write_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.mesh.write_mesh_buffer_with_mode(&mesh_buffer, staged, false)
            }));
            match write_result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    return Err(LaunchError::Unknown {
                        reason: format!("temporary MeshBuffer write failed: {}", e.what()),
                        backtrace: BackTrace::capture(),
                    });
                }
                Err(_) => {
                    return Err(LaunchError::Unknown {
                        reason: "temporary MeshBuffer write panicked".into(),
                        backtrace: BackTrace::capture(),
                    });
                }
            }
        }
        Ok(TtTiledTempBuffer {
            mesh_buffer,
            address,
            compile_args,
        })
    }

    fn bridge_prepared_launch(
        &mut self,
        prepared: PreparedLaunch,
        resources: &[TtResource],
    ) -> Result<(PreparedLaunch, Option<TtTiledLaunchBridge>), LaunchError> {
        if !prepared.sources.requires_tiled_io() {
            return Ok((prepared, None));
        }
        if resources.len() != prepared.bindings.len() {
            return Err(LaunchError::Unknown {
                reason: format!(
                    "TT tiled I/O bridge expected {} resources but got {}",
                    prepared.bindings.len(),
                    resources.len()
                ),
                backtrace: BackTrace::capture(),
            });
        }

        let num_tiles = prepared.sources.num_tiles.max(1);
        let row_major_rows = 32u32;
        let row_major_cols = num_tiles * 32;
        let logical_elem_size_bytes = logical_host_elem_size_bytes(
            prepared.sources.data_format_tt,
            prepared.sources.native_scalar_size_bytes(),
        );
        if logical_elem_size_bytes == 0 {
            return Err(LaunchError::Unknown {
                reason: "TT tiled I/O bridge computed a zero scalar element width".into(),
                backtrace: BackTrace::capture(),
            });
        }
        let logical_padded_bytes =
            row_major_rows as usize * row_major_cols as usize * logical_elem_size_bytes as usize;

        let mut input_addrs = Vec::new();
        let mut output_addrs = Vec::new();
        let mut reader_compile_args = Vec::new();
        let mut writer_compile_args = Vec::new();
        let mut temp_inputs = Vec::new();
        let mut temp_outputs = Vec::new();

        for (binding, resource) in prepared.bindings.iter().zip(resources.iter()) {
            match binding.visibility {
                cubecl_runtime::kernel::Visibility::Read => {
                    let logical = self
                        .read_resource_bytes(resource, binding.logical_size_bytes as usize)
                        .map_err(map_io_err_to_launch("TT tiled input bridge read failed"))?;
                    let mut padded = vec![0u8; logical_padded_bytes];
                    padded[..logical.len()].copy_from_slice(&logical);
                    let tilized =
                        if is_native_block_float_format(prepared.sources.data_format_tt) {
                            libtt_metal_cxx::tilize_with_data_format(
                                &padded,
                                row_major_rows,
                                row_major_cols,
                                prepared.sources.data_format_tt,
                            )
                        } else {
                            libtt_metal_cxx::tilize(
                                &padded,
                                row_major_rows,
                                row_major_cols,
                                logical_elem_size_bytes,
                            )
                        }
                        .map_err(map_tt_exception_to_launch("TT tiled input tilize failed"))?;
                    let temp = self.create_temp_tiled_buffer(
                        &tilized,
                        u64::from(prepared.sources.tile_size_bytes),
                    )?;
                    input_addrs.push(temp.address);
                    reader_compile_args.extend(temp.compile_args.iter().copied());
                    temp_inputs.push(temp);
                }
                cubecl_runtime::kernel::Visibility::ReadWrite => {
                    let temp = self.create_temp_tiled_buffer(
                        &vec![0u8; num_tiles as usize * prepared.sources.tile_size_bytes as usize],
                        u64::from(prepared.sources.tile_size_bytes),
                    )?;
                    output_addrs.push(temp.address);
                    writer_compile_args.extend(temp.compile_args.iter().copied());
                    temp_outputs.push(TtTiledOutputBridge {
                        temp,
                        original: resource.clone(),
                        logical_size_bytes: binding.logical_size_bytes as usize,
                    });
                }
            }
        }

        let adapted = PreparedLaunch {
            sources: prepared
                .sources
                .clone()
                .with_compile_args(reader_compile_args, writer_compile_args),
            input_addrs,
            output_addrs,
            bindings: prepared.bindings,
        };
        let bridge = TtTiledLaunchBridge {
            row_major_rows,
            row_major_cols,
            logical_elem_size_bytes,
            data_format_tt: prepared.sources.data_format_tt,
            _inputs: temp_inputs,
            outputs: temp_outputs,
        };
        Ok((adapted, Some(bridge)))
    }

    fn finish_tiled_launch_bridge(
        &mut self,
        bridge: TtTiledLaunchBridge,
    ) -> Result<(), LaunchError> {
        for output in bridge.outputs {
            let mut staged = vec![0u8; output.temp.mesh_buffer.size() as usize];
            let read_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                self.mesh
                    .read_mesh_buffer_with_mode(&output.temp.mesh_buffer, &mut staged, true)
            }));
            match read_result {
                Ok(Ok(())) => {}
                Ok(Err(e)) => {
                    return Err(LaunchError::Unknown {
                        reason: format!("TT tiled output read failed: {}", e.what()),
                        backtrace: BackTrace::capture(),
                    });
                }
                Err(_) => {
                    return Err(LaunchError::Unknown {
                        reason: "TT tiled output read panicked".into(),
                        backtrace: BackTrace::capture(),
                    });
                }
            }

            let untilized = if is_native_block_float_format(bridge.data_format_tt) {
                libtt_metal_cxx::untilize_with_data_format(
                    &staged,
                    bridge.row_major_rows,
                    bridge.row_major_cols,
                    bridge.data_format_tt,
                )
            } else {
                libtt_metal_cxx::untilize(
                    &staged,
                    bridge.row_major_rows,
                    bridge.row_major_cols,
                    bridge.logical_elem_size_bytes,
                )
            }
            .map_err(map_tt_exception_to_launch(
                "TT tiled output untilize failed",
            ))?;
            if untilized.len() < output.logical_size_bytes {
                return Err(LaunchError::Unknown {
                    reason: format!(
                        "TT tiled output bridge produced {} bytes, expected at least {}",
                        untilized.len(),
                        output.logical_size_bytes
                    ),
                    backtrace: BackTrace::capture(),
                });
            }
            self.write_resource_bytes(&output.original, &untilized[..output.logical_size_bytes])
                .map_err(map_io_err_to_launch(
                    "TT tiled output write-back to logical buffer failed",
                ))?;
        }
        Ok(())
    }

    /// Write host data to a device buffer (blocking).
    pub fn write_to_gpu(&mut self, descriptor: CopyDescriptor, data: Bytes) -> Result<(), IoError> {
        let resource = self.resource(descriptor.handle.clone())?;
        if resource.layout != TtBufferLayout::Replicated {
            return Err(IoError::Unknown {
                backtrace: BackTrace::capture(),
                description: "write_to_gpu only supports replicated TT buffers today".into(),
            });
        }

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

        self.write_resource_bytes(&resource, &data[..num_bytes])?;
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

        let num_bytes: usize = descriptor.shape.iter().product::<usize>() * descriptor.elem_size;
        Ok(Bytes::from_bytes_vec(
            self.read_resource_bytes(&resource, num_bytes)?,
        ))
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
            TtLaunchTarget::FullMesh => {
                catch_launch_panic("add_program_to_full_mesh failed", || {
                    workload.add_program_to_full_mesh(self.mesh, program)
                })
            }
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
            self.mesh.enqueue_workload(&mut workload, false)
        })?;

        Ok(())
    }

    /// Compile and launch a `CubeTask` kernel.
    pub fn kernel_cube(
        &mut self,
        cube_kernel: Box<dyn CubeTask<TtCompiler>>,
        count: CubeCount,
        mode: ExecutionMode,
        resources: &[TtResource],
        info: &cubecl_runtime::server::MetadataBindingInfo,
        logger: Arc<ServerLogger>,
    ) -> Result<(), LaunchError> {
        let prepared =
            self.ctx
                .prepare_cube_task_launch(cube_kernel, mode, count, resources, info)?;
        let (prepared, bridge) = self.bridge_prepared_launch(prepared, resources)?;
        let compiled = self.ctx.compile_kernel(
            &prepared.sources,
            &prepared.input_addrs,
            &prepared.output_addrs,
            logger,
        )?;
        let target = self.launch_target_for_resources(resources)?;

        if let Some(bridge) = bridge {
            self.flush_current_pending_workload()?;
            let mut workload = libtt_metal_cxx::MeshWorkload::new();
            self.add_program_to_workload(&mut workload, compiled.program, target)?;
            catch_launch_panic("enqueue_workload failed", || {
                self.mesh.enqueue_workload(&mut workload, false)
            })?;
            self.finish_tiled_launch_bridge(bridge)?;
        } else {
            match target {
                TtLaunchTarget::FullMesh => self
                    .streams
                    .current()
                    .enqueue_full_mesh_program(compiled.program)
                    .map_err(|err| LaunchError::Unknown {
                        reason: format!("failed to queue TT workload program: {err:?}"),
                        backtrace: BackTrace::capture(),
                    })?,
            }
        }
        Ok(())
    }
}

fn mesh_write_buffer(
    mesh: &libtt_metal_cxx::MeshDevice,
    mesh_buffer: &libtt_metal_cxx::MeshBuffer,
    staged: &[u8],
    context: &'static str,
    blocking: bool,
) -> Result<(), IoError> {
    let write_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        mesh.write_mesh_buffer_with_mode(mesh_buffer, staged, blocking)
    }));
    match write_result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(IoError::Unknown {
            backtrace: BackTrace::capture(),
            description: format!("{context}: {}", e.what()),
        }),
        Err(_panic) => Err(IoError::Unknown {
            backtrace: BackTrace::capture(),
            description: format!("{context}: TT-Metal host call panicked"),
        }),
    }
}

fn mesh_read_buffer(
    mesh: &libtt_metal_cxx::MeshDevice,
    mesh_buffer: &libtt_metal_cxx::MeshBuffer,
    staged: &mut [u8],
    context: &'static str,
    blocking: bool,
) -> Result<(), IoError> {
    let read_result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        mesh.read_mesh_buffer_with_mode(mesh_buffer, staged, blocking)
    }));
    match read_result {
        Ok(Ok(())) => Ok(()),
        Ok(Err(e)) => Err(IoError::Unknown {
            backtrace: BackTrace::capture(),
            description: format!("{context}: {}", e.what()),
        }),
        Err(_panic) => Err(IoError::Unknown {
            backtrace: BackTrace::capture(),
            description: format!("{context}: TT-Metal host call panicked"),
        }),
    }
}

fn is_native_block_float_format(data_format_tt: u8) -> bool {
    matches!(data_format_tt, 6 | 7 | 15)
}

fn logical_host_elem_size_bytes(data_format_tt: u8, native_scalar_size_bytes: u32) -> u32 {
    match data_format_tt {
        6 | 7 | 15 => 4,
        _ => native_scalar_size_bytes,
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

fn map_io_err_to_launch(context: &'static str) -> impl FnOnce(IoError) -> LaunchError {
    move |err| LaunchError::Unknown {
        reason: format!("{context}: {err:?}"),
        backtrace: BackTrace::capture(),
    }
}

fn map_tt_exception_to_launch(
    context: &'static str,
) -> impl FnOnce(libtt_metal_cxx::Exception) -> LaunchError {
    move |err| LaunchError::Unknown {
        reason: format!("{context}: {}", err.what()),
        backtrace: BackTrace::capture(),
    }
}
