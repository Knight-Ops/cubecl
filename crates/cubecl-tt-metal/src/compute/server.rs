use super::storage::gpu::{TtResource, TtStorage};
use crate::{
    compute::{command::Command, context::TtContext, stream::TtStreamBackend},
    runtime::TtCompiler,
};
use cubecl_cpp::tt_metal::TtKernelSources;

use cubecl_common::bytes::Bytes;
use cubecl_common::future::DynFut;
use cubecl_common::profile::ProfileDuration;
use cubecl_common::stream_id::StreamId;
use cubecl_core::{
    MemoryConfiguration,
    backtrace::BackTrace,
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
    mesh: Box<libtt_metal_cxx::MeshDevice>,
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
        command.bind(reserved, memory);
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
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) {
        if let Err(err) = self.launch_checked(kernel, count, bindings, mode, stream_id) {
            let mut stream = match self.streams.resolve(stream_id, [].into_iter(), false) {
                Ok(stream) => stream,
                Err(err) => unreachable!("{err:?}"),
            };
            stream.current().errors.push(err);
        }
    }

    fn flush(&mut self, stream_id: StreamId) -> Result<(), ServerError> {
        let mut command = self.command_no_inputs(
            stream_id,
            StreamErrorMode {
                ignore: false,
                flush: true,
            },
        )?;
        // Flush GPU storage (synchronous — no-op for TT)
        let _current = command.streams.current();
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
}

impl TtServer {
    /// Create a `TtServer` from an already-boxed `MeshDevice` (for testing).
    ///
    /// The `MeshDevice` must be in a `Box` so its address is stable
    /// and raw pointers to it remain valid after `TtServer` is moved.
    pub fn from_mesh_boxed(mesh: Box<libtt_metal_cxx::MeshDevice>) -> Self {
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
        mesh: Box<libtt_metal_cxx::MeshDevice>,
        ctx: TtContext,
        _mem_props: MemoryDeviceProperties,
        _mem_config: MemoryConfiguration,
        utilities: ServerUtilities<Self>,
    ) -> Self {
        let config = CubeClRuntimeConfig::get();
        let max_streams = config.streaming.max_streams;

        // Take a raw pointer to the heap-allocated MeshDevice.
        // Box guarantees the allocation address is stable even if TtServer is moved.
        let mesh_ptr: *const libtt_metal_cxx::MeshDevice = &*mesh;
        let backend = TtStreamBackend::new(mesh_ptr);

        Self {
            mesh,
            ctx,
            streams: MultiStream::new(utilities.logger.clone(), backend, max_streams),
            utilities: Arc::new(utilities),
        }
    }

    /// Access the underlying `MeshDevice`.
    pub fn mesh(&self) -> &libtt_metal_cxx::MeshDevice {
        &self.mesh
    }

    /// Compile a `CubeTask` and launch it on the device.
    fn launch_checked(
        &mut self,
        kernel: Box<dyn CubeTask<TtCompiler>>,
        _count: CubeCount,
        bindings: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) -> Result<(), ServerError> {
        let logger = self.streams.logger.clone();
        let mut command = self.command(
            stream_id,
            bindings.buffers.iter(),
            StreamErrorMode {
                ignore: true,
                flush: false,
            },
        )?;

        // Resolve buffer addresses from bindings
        // Standard CubeCL convention: last buffer is output, preceding are inputs
        let resources: Vec<_> = bindings
            .buffers
            .iter()
            .map(|b| {
                command
                    .resource(b.clone())
                    .map(|r| r.address)
                    .map_err(|e| ServerError::Generic {
                        reason: format!("resource: {e:?}"),
                        backtrace: BackTrace::capture(),
                    })
            })
            .collect::<Result<Vec<_>, _>>()?;

        let (input_addrs, output_addrs) = if resources.len() <= 1 {
            (resources.as_slice(), [].as_slice())
        } else {
            let split = resources.len() - 1;
            (&resources[..split], &resources[split..])
        };

        command
            .kernel_cube(kernel, mode, input_addrs, output_addrs, logger)
            .map_err(|e| ServerError::Generic {
                reason: format!("{e:?}"),
                backtrace: BackTrace::capture(),
            })
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
        command
            .kernel(
                sources,
                input_addrs,
                output_addrs,
                reader_compile_args,
                writer_compile_args,
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
        Ok(Command::new(&mut self.ctx, &self.mesh, streams))
    }

    pub(crate) fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }
}
