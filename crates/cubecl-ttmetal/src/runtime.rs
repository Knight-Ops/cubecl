use crate::{
    MetaliumCompiler, MetaliumDevice, MetaliumRuntimeInfo, MetaliumServer, UnavailableBridge,
};
use cubecl_core::{
    MemoryConfiguration, Runtime,
    client::ComputeClient,
    device::{DeviceId, DeviceService, ServerUtilitiesHandle},
    ir::{
        DeviceProperties, HardwareProperties, MemoryDeviceProperties, TargetProperties, VectorSize,
        features::Features,
    },
    server::{ComputeServer, ServerUtilities},
    zspace::{Shape, Strides},
};
use cubecl_runtime::{allocator::ContiguousMemoryLayoutPolicy, logging::ServerLogger};
use std::sync::Arc;

#[derive(Default)]
pub struct RuntimeOptions {
    /// Configures the memory management used by the placeholder server.
    pub memory_config: MemoryConfiguration,
}

#[derive(Debug, Clone)]
pub struct MetaliumRuntime;

impl DeviceService for MetaliumServer {
    fn init(_device_id: cubecl_core::device::DeviceId) -> Self {
        let options = RuntimeOptions::default();

        let hardware = HardwareProperties {
            load_width: 64,
            plane_size_min: 1,
            plane_size_max: 1,
            max_bindings: 32,
            max_shared_memory_size: 1024 * 1024,
            max_cube_count: (u16::MAX as u32, u16::MAX as u32, 1),
            max_units_per_cube: 1,
            max_cube_dim: (1, 1, 1),
            num_streaming_multiprocessors: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: Some(32),
            num_cpu_cores: None,
            max_vector_size: VectorSize::MAX,
        };

        let mem_properties = MemoryDeviceProperties {
            max_page_size: 128 * 1024 * 1024,
            alignment: 64,
        };

        let device_props = DeviceProperties::new(
            Features {
                alignment: true,
                ..Default::default()
            },
            mem_properties.clone(),
            hardware,
            cubecl_core::profile::TimingMethod::System,
        );

        let logger = Arc::new(ServerLogger::default());
        let utilities = Arc::new(ServerUtilities::new(
            device_props,
            logger,
            MetaliumRuntimeInfo {
                bridge_name: UnavailableBridge::NAME,
                ffi_available: false,
            },
            ContiguousMemoryLayoutPolicy::new(mem_properties.alignment as usize),
        ));

        MetaliumServer::new(
            mem_properties,
            options.memory_config,
            utilities,
            Box::<UnavailableBridge>::default(),
        )
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        self.utilities() as ServerUtilitiesHandle
    }
}

impl Runtime for MetaliumRuntime {
    type Compiler = MetaliumCompiler;
    type Server = MetaliumServer;
    type Device = MetaliumDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self> {
        ComputeClient::load(device)
    }

    fn name(_client: &ComputeClient<Self>) -> &'static str {
        "metalium"
    }

    fn require_array_lengths() -> bool {
        true
    }

    fn max_cube_count() -> (u32, u32, u32) {
        (u16::MAX as u32, u16::MAX as u32, 1)
    }

    fn can_read_tensor(shape: &Shape, strides: &Strides) -> bool {
        if shape.is_empty() {
            return true;
        }

        contiguous_strides(shape) == *strides
    }

    fn target_properties() -> TargetProperties {
        TargetProperties {
            mma: Default::default(),
        }
    }

    fn enumerate_devices(_: u16, _: &<Self::Server as ComputeServer>::Info) -> Vec<DeviceId> {
        vec![DeviceId::new(0, 0)]
    }
}

fn contiguous_strides(shape: &Shape) -> Strides {
    let rank = shape.len();
    let mut strides = cubecl_core::zspace::strides![1; rank];
    for i in (0..rank.saturating_sub(1)).rev() {
        strides[i] = strides[i + 1] * shape[i + 1];
    }
    strides
}
