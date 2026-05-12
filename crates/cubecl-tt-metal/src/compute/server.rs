use super::storage::gpu::{TtResource, TtStorage};
use crate::{
    compute::{command::Command, context::TtContext, stream::TtStreamBackend},
    runtime::TtCompiler,
};
use cubecl_common::bytes::Bytes;
use cubecl_common::future::DynFut;
use cubecl_common::profile::ProfileDuration;
use cubecl_common::stream_id::StreamId;
use cubecl_core::{
    MemoryConfiguration,
    backtrace::BackTrace,
    future,
    ir::MemoryDeviceProperties,
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
    mesh: libtt_metal_cxx::MeshDevice,
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
            .map(|size| {
                command
                    .reserve(*size as u64)
                    .map(|_| Bytes::from_bytes_vec(vec![0u8; *size]))
            })
            .collect::<Result<Vec<_>, _>>()
            .unwrap_or_default())
    }

    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, stream_id: StreamId) {
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
        let reserved = command.reserve(size).unwrap();
        // Bind the reserved memory to the managed memory handle
        // TODO: proper binding
        let _ = (reserved, memory);
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
    }

    unsafe fn launch(
        &mut self,
        _kernel: Self::Kernel,
        _count: CubeCount,
        _bindings: KernelArguments,
        _mode: ExecutionMode,
        _stream_id: StreamId,
    ) {
        // TODO: IR-driven launch in Phase 4+
    }

    fn flush(&mut self, _stream_id: StreamId) -> Result<(), ServerError> {
        Ok(())
    }

    fn sync(&mut self, _stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        Box::pin(async { Ok(()) })
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

    fn memory_usage(&mut self, _stream_id: StreamId) -> Result<MemoryUsage, ServerError> {
        Ok(MemoryUsage {
            number_allocs: 0,
            bytes_in_use: 0,
            bytes_padding: 0,
            bytes_reserved: 0,
        })
    }

    fn memory_cleanup(&mut self, _stream_id: StreamId) {}

    fn allocation_mode(&mut self, _mode: MemoryAllocationMode, _stream_id: StreamId) {}
}

impl ServerCommunication for TtServer {
    const SERVER_COMM_ENABLED: bool = false;
}

impl TtServer {
    /// Create a TtServer from an existing MeshDevice (for testing).
    pub fn from_mesh(mesh: libtt_metal_cxx::MeshDevice) -> Self {
        use cubecl_common::profile::TimingMethod;
        use cubecl_core::ir::{
            DeviceProperties, HardwareProperties, MemoryDeviceProperties, VectorSize,
        };
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
            max_cube_count: (1, 1, 1),
            max_units_per_cube: warp_size * 32,
            max_cube_dim: (u32::MAX, 1, 1),
            num_streaming_multiprocessors: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
            num_cpu_cores: None,
            max_vector_size: VectorSize::MAX,
        };

        let mem_properties = MemoryDeviceProperties {
            max_page_size: 16 * 1024 * 1024,
            alignment: 32,
        };

        let mut device_props = DeviceProperties::new(
            Default::default(),
            mem_properties.clone(),
            topology,
            TimingMethod::System,
        );

        cubecl_cpp::register_supported_types(&mut device_props);
        cubecl_cpp::shared::register_wmma_features(Vec::new(), &mut device_props);

        let comp_opts = CompilationOptions {
            warp_size: arch.warp_size(),
            supports_features: CppSupportedFeatures {
                fast_math: true,
                ..Default::default()
            },
        };

        let mut ctx = TtContext::new(comp_opts, device_props.clone());
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
        mesh: libtt_metal_cxx::MeshDevice,
        mut ctx: TtContext,
        _mem_props: MemoryDeviceProperties,
        _mem_config: MemoryConfiguration,
        utilities: ServerUtilities<Self>,
    ) -> Self {
        let config = CubeClRuntimeConfig::get();
        let max_streams = config.streaming.max_streams;

        let mesh_ptr: *const libtt_metal_cxx::MeshDevice = &mesh;
        ctx.set_mesh_ptr(mesh_ptr);

        let backend = TtStreamBackend::new(mesh_ptr);

        Self {
            mesh,
            ctx,
            streams: MultiStream::new(utilities.logger.clone(), backend, max_streams),
            utilities: Arc::new(utilities),
        }
    }

    /// Launch a copy kernel directly (Phase 2 test harness).
    pub fn launch_copy_kernel(
        &mut self,
        input_addr: u32,
        output_addr: u32,
        num_tiles: u32,
        tile_size_bytes: u32,
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
        let kernel_id = KernelId::new::<()>();
        command
            .kernel(
                kernel_id,
                input_addr,
                output_addr,
                num_tiles,
                tile_size_bytes,
                logger,
            )
            .map_err(|e| ServerError::Generic {
                reason: format!("{e:?}"),
                backtrace: BackTrace::capture(),
            })
    }

    fn command_no_inputs(
        &mut self,
        stream_id: StreamId,
        mode: StreamErrorMode,
    ) -> Result<Command<'_>, ServerError> {
        self.command(stream_id, [].into_iter(), mode)
    }

    fn command<'a>(
        &mut self,
        stream_id: StreamId,
        handles: impl Iterator<Item = &'a Binding>,
        mode: StreamErrorMode,
    ) -> Result<Command<'_>, ServerError> {
        let streams = self.streams.resolve(stream_id, handles, !mode.ignore)?;
        Ok(Command::new(&mut self.ctx, streams))
    }

    pub(crate) fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }
}
