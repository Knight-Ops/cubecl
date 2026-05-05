use crate::{MetaliumCompilationOptions, MetaliumCompiler, MetaliumExecutable};
use cubecl_core::zspace::{Shape, Strides};
use cubecl_core::{
    CompilationError, CubeCount, ExecutionMode, MemoryConfiguration, MemoryUsage,
    backtrace::BackTrace,
    bytes::Bytes,
    future::DynFut,
    ir::MemoryDeviceProperties,
    profile::ProfileDuration,
    server::{
        Binding, ComputeServer, CopyDescriptor, IoError, KernelArguments, LaunchError,
        ProfileError, ProfilingToken, ServerCommunication, ServerError, ServerUtilities,
    },
    stream_id::StreamId,
};
use cubecl_runtime::{
    allocator::ContiguousMemoryLayoutPolicy,
    compiler::CubeTask,
    id::KernelId,
    logging::ServerLogger,
    memory_management::{
        ManagedMemoryHandle, MemoryAllocationMode, MemoryManagement, MemoryManagementOptions,
    },
    storage::{BytesStorage, ComputeStorage, ManagedResource},
    timestamp_profiler::TimestampProfiler,
};
use std::{collections::HashMap, sync::Arc};

#[derive(Debug, Clone, Copy)]
pub struct MetaliumRuntimeInfo {
    pub bridge_name: &'static str,
    pub ffi_available: bool,
}

#[derive(Debug, Clone)]
pub struct MetaliumLaunchDescriptor {
    pub kernel_id: KernelId,
    pub kernel_name: String,
    pub cube_count: [u32; 3],
    pub buffer_count: usize,
    pub tensor_map_count: usize,
    pub source_bundle: MetaliumExecutable,
}

pub trait MetaliumBridge: Send + Sync + std::fmt::Debug + 'static {
    fn submit(&mut self, launch: MetaliumLaunchDescriptor) -> Result<(), ServerError>;
}

#[derive(Debug, Default)]
pub struct UnavailableBridge;

impl UnavailableBridge {
    pub const NAME: &'static str = "unavailable";
}

impl MetaliumBridge for UnavailableBridge {
    fn submit(&mut self, launch: MetaliumLaunchDescriptor) -> Result<(), ServerError> {
        Err(ServerError::Launch(LaunchError::CompilationError(
            cubecl_core::CompilationError::Generic {
                reason: format!(
                    "TT-Metalium bundle for `{}` was generated, but no host API bridge is linked yet. \
                Next step: connect this launch path to a `cxx` bridge over `tt_metal::Program`, \
                `CreateKernel`, buffer setup, and command queue submission.\n\nGenerated sections: {}",
                    launch.kernel_name,
                    launch
                        .source_bundle
                        .sections
                        .iter()
                        .map(|section| section.name)
                        .collect::<Vec<_>>()
                        .join(", ")
                ),
                backtrace: BackTrace::capture(),
            },
        )))
    }
}

#[derive(Debug)]
pub struct MetaliumServer {
    memory_management: MemoryManagement<BytesStorage>,
    timestamps: TimestampProfiler,
    utilities: Arc<ServerUtilities<Self>>,
    bridge: Box<dyn MetaliumBridge>,
    compilation_cache: HashMap<KernelId, MetaliumExecutable>,
    pending_error: Option<ServerError>,
}

impl MetaliumServer {
    pub(crate) fn new(
        memory_properties: MemoryDeviceProperties,
        memory_config: MemoryConfiguration,
        utilities: Arc<ServerUtilities<Self>>,
        bridge: Box<dyn MetaliumBridge>,
    ) -> Self {
        let memory_management = MemoryManagement::from_configuration(
            BytesStorage::default(),
            &memory_properties,
            memory_config,
            utilities.logger.clone(),
            MemoryManagementOptions::new("Metalium Host Memory"),
        );

        Self {
            memory_management,
            timestamps: TimestampProfiler::default(),
            utilities,
            bridge,
            compilation_cache: HashMap::new(),
            pending_error: None,
        }
    }

    pub(crate) fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn ensure_healthy(&self) -> Result<(), ServerError> {
        if let Some(error) = &self.pending_error {
            return Err(ServerError::ServerUnhealthy {
                errors: vec![error.clone()],
                backtrace: BackTrace::capture(),
            });
        }

        Ok(())
    }

    fn set_error(&mut self, error: ServerError) {
        self.pending_error = Some(error);
    }

    fn compile_bundle(
        &mut self,
        kernel: &dyn CubeTask<MetaliumCompiler>,
        mode: ExecutionMode,
    ) -> Result<MetaliumExecutable, CompilationError> {
        if let Some(bundle) = self.compilation_cache.get(&kernel.id()) {
            return Ok(bundle.clone());
        }

        let compiled = kernel.compile(
            &mut MetaliumCompiler,
            &MetaliumCompilationOptions::default(),
            mode,
            kernel.address_type(),
        )?;

        self.logger().log_compilation(&compiled);

        let bundle = compiled
            .repr
            .expect("MetaliumCompiler always returns a source bundle representation");
        self.compilation_cache.insert(kernel.id(), bundle.clone());
        Ok(bundle)
    }

    fn resolve_cube_count(&mut self, count: CubeCount) -> Result<[u32; 3], ServerError> {
        match count {
            CubeCount::Static(x, y, z) => Ok([x, y, z]),
            CubeCount::Dynamic(binding) => {
                let resource = self.memory_management.get_resource(
                    binding.memory,
                    binding.offset_start,
                    binding.offset_end,
                )?;
                let bytes = resource.read();

                if bytes.len() < 12 {
                    return Err(ServerError::Io(IoError::NotFound {
                        backtrace: BackTrace::capture(),
                        reason: "Dynamic cube-count buffer must contain at least 3 u32 values"
                            .into(),
                    }));
                }

                Ok([
                    u32::from_ne_bytes(bytes[0..4].try_into().unwrap()),
                    u32::from_ne_bytes(bytes[4..8].try_into().unwrap()),
                    u32::from_ne_bytes(bytes[8..12].try_into().unwrap()),
                ])
            }
        }
    }

    fn bind_with_data(
        &mut self,
        data: &[u8],
        handle: cubecl_runtime::server::Handle,
        stream_id: StreamId,
    ) {
        let shape: Shape = [data.len()].into();
        let strides: Strides = [1].into();

        self.initialize_memory(handle.memory.clone(), handle.size(), stream_id);
        self.write(
            vec![(
                CopyDescriptor::new(handle.binding(), shape, strides, 1),
                Bytes::from_bytes_vec(data.to_vec()),
            )],
            stream_id,
        );
    }
}

impl ComputeServer for MetaliumServer {
    type Kernel = Box<dyn CubeTask<MetaliumCompiler>>;
    type Storage = BytesStorage;
    type MemoryLayoutPolicy = ContiguousMemoryLayoutPolicy;
    type Info = MetaliumRuntimeInfo;

    fn logger(&self) -> Arc<ServerLogger> {
        self.utilities.logger.clone()
    }

    fn utilities(&self) -> Arc<ServerUtilities<Self>> {
        self.utilities.clone()
    }

    fn initialize_memory(&mut self, memory: ManagedMemoryHandle, size: u64, _stream_id: StreamId) {
        let reserved = self.memory_management.reserve(size).unwrap();
        self.memory_management.bind(reserved, memory, 0).unwrap();
    }

    fn staging(
        &mut self,
        _sizes: &[usize],
        _stream_id: StreamId,
    ) -> Result<Vec<Bytes>, ServerError> {
        self.ensure_healthy()?;
        Err(IoError::UnsupportedIoOperation {
            backtrace: BackTrace::capture(),
        }
        .into())
    }

    fn read(
        &mut self,
        descriptors: Vec<CopyDescriptor>,
        _stream_id: StreamId,
    ) -> DynFut<Result<Vec<Bytes>, ServerError>> {
        if let Err(error) = self.ensure_healthy() {
            return Box::pin(async move { Err(error) });
        }

        let mut output = Vec::with_capacity(descriptors.len());
        for descriptor in descriptors {
            if descriptor.strides != contiguous_strides(&descriptor.shape) {
                return Box::pin(async move {
                    Err(ServerError::Io(IoError::UnsupportedStrides {
                        backtrace: BackTrace::capture(),
                    }))
                });
            }

            let resource = match self.memory_management.get_resource(
                descriptor.handle.memory.clone(),
                descriptor.handle.offset_start,
                descriptor.handle.offset_end,
            ) {
                Ok(resource) => resource,
                Err(error) => return Box::pin(async move { Err(ServerError::Io(error)) }),
            };
            let size = descriptor.handle.size_in_used() as usize;
            output.push(Bytes::from_bytes_vec(resource.read()[0..size].to_vec()));
        }

        Box::pin(async move { Ok(output) })
    }

    fn write(&mut self, descriptors: Vec<(CopyDescriptor, Bytes)>, _stream_id: StreamId) {
        if self.ensure_healthy().is_err() {
            return;
        }

        for (descriptor, data) in descriptors {
            if descriptor.strides != contiguous_strides(&descriptor.shape) {
                self.set_error(ServerError::Io(IoError::UnsupportedStrides {
                    backtrace: BackTrace::capture(),
                }));
                return;
            }

            let mut resource = match self.memory_management.get_resource(
                descriptor.handle.memory,
                descriptor.handle.offset_start,
                descriptor.handle.offset_end,
            ) {
                Ok(resource) => resource,
                Err(error) => {
                    self.set_error(ServerError::Io(error));
                    return;
                }
            };
            resource.write()[..data.len()].copy_from_slice(&data);
        }
    }

    fn sync(&mut self, _stream_id: StreamId) -> DynFut<Result<(), ServerError>> {
        let result = self.ensure_healthy();
        Box::pin(async move { result })
    }

    fn get_resource(
        &mut self,
        binding: Binding,
        _stream_id: StreamId,
    ) -> Result<ManagedResource<<Self::Storage as ComputeStorage>::Resource>, ServerError> {
        self.ensure_healthy()?;
        let memory = binding.memory.clone();
        let resource = self.memory_management.get_resource(
            binding.memory,
            binding.offset_start,
            binding.offset_end,
        )?;
        Ok(ManagedResource::new(memory, resource))
    }

    unsafe fn launch(
        &mut self,
        kernel: Self::Kernel,
        count: CubeCount,
        bindings: KernelArguments,
        mode: ExecutionMode,
        stream_id: StreamId,
    ) {
        if self.ensure_healthy().is_err() {
            return;
        }

        let info_bytes = bindings
            .info
            .data
            .iter()
            .flat_map(|value| value.to_ne_bytes())
            .collect::<Vec<_>>();
        let info_handle = cubecl_runtime::server::Handle::new(stream_id, info_bytes.len() as u64);
        self.bind_with_data(&info_bytes, info_handle, stream_id);

        let cube_count = match self.resolve_cube_count(count) {
            Ok(count) => count,
            Err(error) => {
                self.set_error(error);
                return;
            }
        };

        let bundle = match self.compile_bundle(kernel.as_ref(), mode) {
            Ok(bundle) => bundle,
            Err(error) => {
                self.set_error(ServerError::Launch(LaunchError::CompilationError(error)));
                return;
            }
        };

        let launch = MetaliumLaunchDescriptor {
            kernel_id: kernel.id(),
            kernel_name: kernel.name().to_string(),
            cube_count,
            buffer_count: bindings.buffers.len(),
            tensor_map_count: bindings.tensor_maps.len(),
            source_bundle: bundle,
        };

        if let Err(error) = self.bridge.submit(launch) {
            self.set_error(error);
        }
    }

    fn flush(&mut self, _stream_id: StreamId) -> Result<(), ServerError> {
        self.ensure_healthy()
    }

    fn memory_usage(&mut self, _stream_id: StreamId) -> Result<MemoryUsage, ServerError> {
        Ok(self.memory_management.memory_usage())
    }

    fn memory_cleanup(&mut self, _stream_id: StreamId) {
        self.memory_management.cleanup(true);
    }

    fn start_profile(&mut self, _stream_id: StreamId) -> Result<ProfilingToken, ServerError> {
        self.ensure_healthy()?;
        Ok(self.timestamps.start())
    }

    fn end_profile(
        &mut self,
        _stream_id: StreamId,
        token: ProfilingToken,
    ) -> Result<ProfileDuration, ProfileError> {
        if let Err(error) = self.ensure_healthy() {
            self.timestamps
                .error(ProfileError::Server(Box::new(error.clone())));
            return Err(ProfileError::Server(Box::new(error)));
        }

        self.timestamps.stop(token)
    }

    fn allocation_mode(&mut self, mode: MemoryAllocationMode, _stream_id: StreamId) {
        self.memory_management.mode(mode);
    }
}

impl ServerCommunication for MetaliumServer {
    const SERVER_COMM_ENABLED: bool = false;
}

fn contiguous_strides(shape: &Shape) -> Strides {
    let rank = shape.len();
    let mut strides = cubecl_core::zspace::strides![1; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::MetaliumCompiler;
    use cubecl_ir::{ElemType, Scope, StorageType, UIntKind};
    use cubecl_runtime::{
        compiler::{CompilationError, Compiler},
        kernel::{CompiledKernel, KernelDefinition, KernelMetadata},
    };

    #[derive(Debug)]
    struct TestTask;

    impl KernelMetadata for TestTask {
        fn id(&self) -> KernelId {
            KernelId::new::<Self>()
        }

        fn address_type(&self) -> StorageType {
            StorageType::Scalar(ElemType::UInt(UIntKind::U32))
        }
    }

    impl CubeTask<MetaliumCompiler> for TestTask {
        fn compile(
            &self,
            compiler: &mut MetaliumCompiler,
            compilation_options: &<MetaliumCompiler as Compiler>::CompilationOptions,
            mode: ExecutionMode,
            addr_type: StorageType,
        ) -> Result<CompiledKernel<MetaliumCompiler>, CompilationError> {
            let definition = KernelDefinition {
                buffers: vec![],
                tensor_maps: vec![],
                scalars: vec![],
                cube_dim: cubecl_core::CubeDim::new_single(),
                body: Scope::root(false),
                options: Default::default(),
            };
            let repr = compiler.compile(definition, compilation_options, mode, addr_type)?;
            Ok(CompiledKernel {
                entrypoint_name: "test".to_string(),
                debug_name: Some("TestTask"),
                source: repr.to_string(),
                repr: Some(repr),
                cube_dim: cubecl_core::CubeDim::new_single(),
                debug_info: None,
            })
        }
    }

    #[test]
    fn launch_surfaces_missing_bridge() {
        let logger = Arc::new(ServerLogger::default());
        let mem_properties = MemoryDeviceProperties {
            max_page_size: 1024 * 1024,
            alignment: 64,
        };
        let utilities = Arc::new(ServerUtilities::new(
            cubecl_core::ir::DeviceProperties::new(
                Default::default(),
                mem_properties.clone(),
                cubecl_core::ir::HardwareProperties {
                    load_width: 64,
                    plane_size_min: 1,
                    plane_size_max: 1,
                    max_bindings: 32,
                    max_shared_memory_size: 1024,
                    max_cube_count: (1, 1, 1),
                    max_units_per_cube: 1,
                    max_cube_dim: (1, 1, 1),
                    num_streaming_multiprocessors: None,
                    num_tensor_cores: None,
                    min_tensor_cores_dim: None,
                    num_cpu_cores: None,
                    max_vector_size: cubecl_core::ir::VectorSize::MAX,
                },
                cubecl_core::profile::TimingMethod::System,
            ),
            logger,
            MetaliumRuntimeInfo {
                bridge_name: UnavailableBridge::NAME,
                ffi_available: false,
            },
            ContiguousMemoryLayoutPolicy::new(64),
        ));
        let mut server = MetaliumServer::new(
            mem_properties,
            MemoryConfiguration::default(),
            utilities,
            Box::<UnavailableBridge>::default(),
        );

        let stream = StreamId::current();

        unsafe {
            server.launch(
                Box::new(TestTask),
                CubeCount::new_single(),
                KernelArguments::default(),
                ExecutionMode::Checked,
                stream,
            );
        }

        let result = cubecl_core::reader::read_sync(server.sync(stream));
        assert!(result.is_err());
    }
}
