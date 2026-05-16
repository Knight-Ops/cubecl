use crate::TtWmmaCompiler;
use crate::compute::{context::TtContext, server::TtServer};
use crate::device::TtDevice;
use cubecl_common::{
    device::{Device, DeviceService},
    profile::TimingMethod,
};
use cubecl_core::{
    MemoryConfiguration, Runtime,
    device::{DeviceId, ServerUtilitiesHandle},
    ir::{
        DeviceProperties, HardwareProperties, MatrixLayout, MemoryDeviceProperties, MmaProperties,
        TargetProperties, VectorSize, features::Plane,
    },
    server::ServerUtilities,
    zspace::{Shape, Strides, striding::has_pitched_row_major_strides},
};
use cubecl_cpp::{
    register_supported_types,
    shared::register_wmma_features,
    shared::{Architecture, CompilationOptions, CppCompiler, CppSupportedFeatures},
    tt_metal::TtMetalDialect,
};
use cubecl_runtime::{
    allocator::ContiguousMemoryLayoutPolicy, client::ComputeClient, logging::ServerLogger,
};
use std::sync::Arc;

/// The values that control how a TT-Metal Runtime will perform its calculations.
#[derive(Default)]
pub struct RuntimeOptions {
    /// Configures the memory management.
    pub memory_config: MemoryConfiguration,
}

#[derive(Debug, Clone)]
pub struct TtRuntime;

pub type TtCompiler = CppCompiler<TtMetalDialect<TtWmmaCompiler>>;

/// Standard tile constants for Tenstorrent hardware.
pub const TILE_WIDTH: u32 = 32;
pub const TILE_HEIGHT: u32 = 32;
pub const TILE_ELEMENTS: u32 = TILE_WIDTH * TILE_HEIGHT;

/// Maximum number of kernel bindings supported by TT-Metal.
pub const TT_MAX_BINDINGS: u32 = 16;

impl DeviceService for TtServer {
    fn init(device_id: cubecl_common::device::DeviceId) -> Self {
        let device = TtDevice::from_id(device_id);

        let mesh = match libtt_metal_cxx::MeshDevice::create_unit_mesh(device.index as i32) {
            Ok(m) => m,
            Err(e) => panic!("Failed to open TT device {}: {}", device.index, e.what()),
        };

        let arch = cubecl_cpp::tt_metal::TtArchitecture::Wormhole;
        let warp_size = arch.warp_size();
        let grid_rows = mesh.num_rows().unwrap_or(1) as u32;
        let grid_cols = mesh.num_cols().unwrap_or(1) as u32;
        let num_cores = grid_rows * grid_cols;

        let topology = HardwareProperties {
            load_width: 128,
            plane_size_min: warp_size,
            plane_size_max: warp_size,
            max_bindings: TT_MAX_BINDINGS,
            max_shared_memory_size: 1_500_000,
            max_cube_count: (num_cores, 1, 1),
            max_units_per_cube: warp_size * TILE_HEIGHT,
            max_cube_dim: (u32::MAX, 1, 1),
            num_streaming_multiprocessors: Some(num_cores as u32),
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
        register_supported_types(&mut device_props);
        register_wmma_features(Vec::new(), &mut device_props);
        device_props.features.memory_reinterpret = true;
        device_props.features.alignment = true;
        device_props.features.plane.insert(Plane::Ops);

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
        let options = RuntimeOptions::default();

        TtServer::new(Box::new(mesh), ctx, mem_properties, options.memory_config, utilities)
    }

    fn utilities(&self) -> ServerUtilitiesHandle {
        self.utilities() as ServerUtilitiesHandle
    }
}

impl Runtime for TtRuntime {
    type Compiler = TtCompiler;
    type Server = TtServer;
    type Device = TtDevice;

    fn client(device: &Self::Device) -> ComputeClient<Self> {
        ComputeClient::load(device)
    }

    fn name(_client: &ComputeClient<Self>) -> &'static str {
        "tt_metal"
    }

    fn require_array_lengths() -> bool {
        true
    }

    fn max_cube_count() -> (u32, u32, u32) {
        (i32::MAX as u32, u16::MAX as u32, u16::MAX as u32)
    }

    fn can_read_tensor(shape: &Shape, strides: &Strides) -> bool {
        if shape.is_empty() {
            return true;
        }
        has_pitched_row_major_strides(shape, strides)
    }

    fn target_properties() -> TargetProperties {
        TargetProperties {
            mma: MmaProperties {
                register_size_bits: 32,
                const_plane_size: 32,
                register_layout_a: MatrixLayout::RowMajor,
                register_layout_b: MatrixLayout::ColMajor,
                register_layout_acc: MatrixLayout::ColMajor,
                register_duplication_a: 1,
                register_duplication_b: 1,
                register_duplication_acc: 1,
                contiguous_elements: Default::default(),
            },
        }
    }

    fn enumerate_devices(
        _: u16,
        _: &<Self::Server as cubecl_core::server::ComputeServer>::Info,
    ) -> Vec<DeviceId> {
        let count = libtt_metal_cxx::available_device_count().unwrap_or(0);
        (0..count).map(|i| DeviceId::new(0, i as u16)).collect()
    }
}
