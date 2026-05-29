use super::storage::gpu::{TtResource, TtStorage};
use crate::{
    compute::{command::Command, context::TtContext, stream::TtStreamBackend},
    runtime::{TtCompiler, tt_memory_properties},
};
use cubecl_cpp::tt_metal::TtKernelSources;

use cubecl_common::bytes::Bytes;
use cubecl_common::future::DynFut;
use cubecl_common::profile::ProfileDuration;
use cubecl_common::stream_id::StreamId;
use cubecl_core::{
    MemoryConfiguration,
    backtrace::BackTrace,
    ir::{BarrierLevel, MemoryDeviceProperties, OpaqueType, features::TypeUsage},
    prelude::*,
    server::{
        Binding, CopyDescriptor, ExecutionMode, KernelArguments, ProfileError, ProfilingToken,
        ServerCommunication, ServerError, ServerUtilities, StreamErrorMode,
    },
};
use cubecl_runtime::{
    allocator::ContiguousMemoryLayoutPolicy,
    compiler::CubeTask,
    config::{CubeClRuntimeConfig, RuntimeConfig},
    logging::ServerLogger,
    memory_management::{ManagedMemoryHandle, MemoryAllocationMode, MemoryUsage},
    server::ComputeServer,
    storage::ManagedResource,
    stream::MultiStream,
};
use std::sync::Arc;

#[derive(Debug)]
pub struct TtServer {
    mesh: &'static libtt_metal_cxx::MeshDevice,
    ctx: TtContext,
    streams: MultiStream<TtStreamBackend>,
    utilities: Arc<ServerUtilities<Self>>,
}

// SAFETY: `TtServer` is only accessed from one thread at a time via the `DeviceHandle`.
unsafe impl Send for TtServer {}

impl ComputeServer for TtServer {
    type Kernel = Box<dyn CubeTask<TtCompiler>>;
    type Storage = TtStorage;
    type MemoryLayoutPolicy = ContiguousMemoryLayoutPolicy;
    type Info = ();

    fn logger(&self) -> Arc<ServerLogger> {
        self.streams.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn staging(&mut self, sizes: &[usize], stream_id: StreamId) -> Result<Vec<Bytes>, ServerError> {
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;
        Ok(sizes
            .iter()
            .map(|size| command.reserve_cpu(*size))
            .collect())
    }

    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, stream_id: StreamId) {
        println!("[initialize_memory] size={size}");
        let mut command = match self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        ) {
            Ok(val) => val,
            Err(err) => unreachable!("{err:?}"),
        };
        match command.reserve(size) {
            Ok(reserved) => {
                command.bind(reserved, memory);
                println!("[initialize_memory] done");
            }
            Err(err) => command.error(ServerError::Io(err)),
        }
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        match self.command(
            stream_id,
            descriptors.iter().map(|d| &d.handle),
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
        ) {
            Ok(mut command) => {
                let results: Result<Vec<_>, _> = descriptors
                    .into_iter()
                    .map(|d| {
                        command.write_to_cpu(d).map_err(|e| ServerError::Generic {
                            reason: format!("{e:?}"),
                            backtrace: BackTrace::capture(),
                        })
                    })
                    .collect();
                Box::pin(async { results })
            }
            Err(err) => Box::pin(async move { Err(err) }),
        }
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, stream_id: StreamId) {
        println!("[write] enter {} descriptors", descriptors.len());
        let mut command = match self.command(
            stream_id,
            descriptors.iter().map(|desc| &desc.0.handle),
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        ) {
            Ok(val) => val,
            Err(err) => unreachable!("{err:?}"),
        };

        for (descriptor, data) in descriptors {
            if let Err(err) = command.write_to_gpu(descriptor, data) {
                command.error(ServerError::Generic {
                    reason: format!("{err:?}"),
                    backtrace: BackTrace::capture(),
                });
                return;
            }
        }
        println!("[write] done");
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) {
        let _ = println!("[launch] enter\n");
        if let Err(err) = self.launch_checked(kernel, count, bindings, mode, stream_id) {
            let mut stream = match self.streams.resolve(stream_id, [].into_iter(), false) {
                Ok(stream) => stream,
                Err(err) => unreachable!("{err:?}"),
            };
            stream.current().error(err);
        }
        let _ = println!("[launch] exit\n");
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.flush_stream_errors(
            stream_id,
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
        )
    }

    fn sync(&mut self, stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        let result = self.flush_stream_errors(
            stream_id,
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
        );
        Box::pin(async move { result })
    }

    fn start_profile(&mut self, _stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        Ok(self.ctx.timestamps.start())
    }

    fn end_profile(
        &mut self,
        _stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        self.ctx.timestamps.stop(token)
    }

    fn get_resource(
        &mut self,
        binding: Binding,
        stream_id: StreamId,
    ) -> Result<ManagedResource<TtResource>, ServerError> {
        let mut command = self.command(
            stream_id,
            [&binding].into_iter(),
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;
        let memory = binding.memory.clone();
        let resource = command
            .resource(binding)
            .map_err(|e| ServerError::Generic {
                reason: format!("{e:?}"),
                backtrace: BackTrace::capture(),
            })?;
        Ok(ManagedResource::new(memory, resource))
    }

    fn memory_usage(&mut self, stream_id: StreamId) -> Result<MemoryUsage, ServerError> {
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: false,
                flush: false,
            },
        )?;
        Ok(command
            .streams
            .current()
            .memory_management_gpu
            .memory_usage())
    }

    fn memory_cleanup(&mut self, stream_id: StreamId) {
        if let Ok(mut command) = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        ) {
            command
                .streams
                .current()
                .memory_management_gpu
                .cleanup(true);
        }
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, stream_id: StreamId) {
        if let Ok(mut command) = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        ) {
            command.streams.current().memory_management_gpu.mode(mode);
        }
    }
}

impl ServerCommunication for TtServer {
    const SERVER_COMM_ENABLED: bool = false;

    fn sync_collective(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        self.flush_stream_errors(
            stream_id,
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
        )
    }

    fn comm_init(&mut self, device_ids: Vec<cubecl_common::device::DeviceId>) -> Result<(), ServerError> {
        if device_ids.len() <= 1 {
            return Ok(());
        }

        Err(ServerError::Generic {
            reason: format!(
                "TT-Metal collective communication currently only supports single-device identity all_reduce; requested {} devices",
                device_ids.len()
            ),
            backtrace: BackTrace::capture(),
        })
    }

    fn all_reduce(
        &mut self,
        src: Binding,
        dst: Binding,
        _dtype: cubecl_core::ir::ElemType,
        stream_id: StreamId,
        _op: cubecl_runtime::server::ReduceOperation,
        device_ids: Vec<cubecl_common::device::DeviceId>,
    ) -> Result<(), ServerError> {
        if device_ids.len() > 1 {
            return Err(ServerError::Generic {
                reason: format!(
                    "TT-Metal all_reduce currently only supports single-device identity semantics; requested {} devices",
                    device_ids.len()
                ),
                backtrace: BackTrace::capture(),
            });
        }

        let mut command = self.command(
            stream_id,
            [&src, &dst].into_iter(),
            StreamErrorMode {
                ignore: false,
                flush: false,
            },
        )?;

        let src_resource = command.resource(src.clone()).map_err(|e| ServerError::Generic {
            reason: format!("all_reduce source resource: {e:?}"),
            backtrace: BackTrace::capture(),
        })?;
        let dst_resource = command.resource(dst.clone()).map_err(|e| ServerError::Generic {
            reason: format!("all_reduce destination resource: {e:?}"),
            backtrace: BackTrace::capture(),
        })?;

        let logical_size = src.size_in_used() as usize;
        if dst.size_in_used() < src.size_in_used() {
            return Err(ServerError::Generic {
                reason: format!(
                    "TT-Metal single-device all_reduce requires destination size {} >= source size {}",
                    dst.size_in_used(),
                    src.size_in_used()
                ),
                backtrace: BackTrace::capture(),
            });
        }

        let bytes = command
            .read_resource_bytes(&src_resource, logical_size)
            .map_err(|e| ServerError::Generic {
                reason: format!("single-device all_reduce read failed: {e:?}"),
                backtrace: BackTrace::capture(),
            })?;
        command
            .write_resource_bytes(&dst_resource, &bytes)
            .map_err(|e| ServerError::Generic {
                reason: format!("single-device all_reduce write failed: {e:?}"),
                backtrace: BackTrace::capture(),
            })
    }
}

impl TtServer {
    /// Create a `TtServer` using the process-level `MeshDevice` singleton.
    ///
    /// This is a convenience for tests that need a `TtServer` without going
    /// through `DeviceService::init`. The underlying `MeshDevice` is obtained
    /// from `crate::runtime::get_mesh()`.
    pub fn from_singleton() -> Self {
        let mesh = crate::runtime::get_mesh();

        use cubecl_common::profile::TimingMethod;
        use cubecl_core::ir::{DeviceProperties, HardwareProperties, VectorSize};
        use cubecl_cpp::shared::{Architecture, CompilationOptions, CppSupportedFeatures};
        use cubecl_cpp::tt_metal::TtArchitecture;

        let arch = TtArchitecture::Wormhole;
        let warp_size = arch.warp_size();

        let topology = HardwareProperties {
            load_width: 128,
            plane_size_min: warp_size,
            plane_size_max: warp_size,
            max_bindings: 16,
            max_shared_memory_size: 1_500_000,
            max_cube_count: (i32::MAX as u32, u16::MAX as u32, u16::MAX as u32),
            max_units_per_cube: warp_size * 32,
            max_cube_dim: (u32::MAX, warp_size * 32, 1),
            num_streaming_multiprocessors: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
            num_cpu_cores: None,
            max_vector_size: VectorSize::MAX,
        };

        let mem_properties = tt_memory_properties();

        let mut device_props = DeviceProperties::new(
            Default::default(),
            mem_properties.clone(),
            topology,
            TimingMethod::System,
        );

        cubecl_cpp::register_supported_types(&mut device_props);
        cubecl_cpp::shared::register_wmma_features(Vec::new(), &mut device_props);
        device_props.register_type_usage(OpaqueType::Barrier(BarrierLevel::Unit), TypeUsage::Buffer);
        device_props.register_type_usage(OpaqueType::Barrier(BarrierLevel::Cube), TypeUsage::Buffer);

        let comp_opts = CompilationOptions {
            warp_size: arch.warp_size(),
            supports_features: CppSupportedFeatures {
                fast_math: true,
                ..Default::default()
            },
        };

        let ctx = TtContext::new(comp_opts, device_props.clone());
        let logger = Arc::new(ServerLogger::default());
        let policy = ContiguousMemoryLayoutPolicy::new(device_props.memory.alignment as usize);
        let utilities = ServerUtilities::new(device_props, logger, (), policy);

        TtServer::new(
            mesh,
            ctx,
            mem_properties,
            MemoryConfiguration::default(),
            utilities,
        )
    }

    pub(crate) fn new(
        mesh: &'static libtt_metal_cxx::MeshDevice,
        ctx: TtContext,
        mem_props: MemoryDeviceProperties,
        mem_config: MemoryConfiguration,
        utilities: ServerUtilities<Self>,
    ) -> Self {
        println!("[TtServer::new] enter");
        let config = CubeClRuntimeConfig::get();
        let max_streams = config.streaming.max_streams;

        // Take a raw pointer to the mesh. Since mesh is &'static,
        // the pointed-to value lives for the entire program lifetime.
        let mesh_ptr: *const libtt_metal_cxx::MeshDevice = mesh;
        let backend = TtStreamBackend::new(mesh_ptr, mem_props, mem_config.clone());

        println!("[TtServer::new] creating MultiStream");
        let server = Self {
            mesh,
            ctx,
            streams: MultiStream::new(utilities.logger.clone(), backend, max_streams),
            utilities: Arc::new(utilities),
        };
        println!("[TtServer::new] done");
        server
    }

    /// Access the underlying `MeshDevice`.
    pub fn mesh(&self) -> &libtt_metal_cxx::MeshDevice {
        self.mesh
    }

    /// Compile a `CubeTask` and launch it on the device.
    fn launch_checked(
        &mut self,
        kernel: Box<dyn CubeTask<TtCompiler>>,
        count: CubeCount,
        bindings: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) -> Result<(), ServerError> {
        let _ = println!("[launch_checked] enter\n");
        let logger = self.streams.logger.clone();
        let mut command = self.command(
            stream_id,
            bindings.buffers.iter(),
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;

        let resources: Vec<_> = bindings
            .buffers
            .iter()
            .map(|b| {
                command
                    .resource(b.clone())
                    .map_err(|e| ServerError::Generic {
                        reason: format!("resource: {e:?}"),
                        backtrace: BackTrace::capture(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let _ = println!("[launch_checked] calling kernel_cube\n");
        command
            .kernel_cube(kernel, count, mode, &resources, &bindings.info, logger)
            .map_err(ServerError::Launch)
    }

    /// Launch a kernel from pre-built TT-Metal sources.
    pub fn launch_from_sources(
        &mut self,
        sources: &TtKernelSources,
        input_addrs: &[u32],
        output_addrs: &[u32],
        reader_compile_args: &[u32],
        writer_compile_args: &[u32],
        stream_id: StreamId,
    ) -> Result<(), ServerError> {
        let logger = self.streams.logger.clone();
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;
        let sources = sources
            .clone()
            .with_compile_args(reader_compile_args.to_vec(), writer_compile_args.to_vec());
        command
            .kernel(&sources, input_addrs, output_addrs, logger)
            .map_err(ServerError::Launch)
    }

    fn command_no_inputs(
        &mut self,
        stream_id: StreamId,
        mode: StreamErrorMode,
    ) -> Result<Command<'_>, ServerError> {
        self.command(stream_id, [].into_iter(), mode)
    }

    fn flush_stream_errors(
        &mut self,
        stream_id: StreamId,
        mode: StreamErrorMode,
    ) -> Result<(), ServerError> {
        let mut streams = self.streams.resolve(stream_id, [].into_iter(), false)?;
        streams.current().flush_errors(mode)
    }

    fn command<'a>(
        &mut self,
        stream_id: StreamId,
        handles: impl Iterator<Item = &'a Binding>,
        mode: StreamErrorMode,
    ) -> Result<Command<'_>, ServerError> {
        let streams = self.streams.resolve(stream_id, handles, !mode.ignore)?;
        Ok(Command::new(&mut self.ctx, self.mesh, streams))
    }

    pub(crate) fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }
}
