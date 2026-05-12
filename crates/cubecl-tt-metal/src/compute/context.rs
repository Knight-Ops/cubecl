use cubecl_common::cache::CacheOption;
use cubecl_common::hash::StableHash;
use cubecl_core::backtrace::BackTrace;
use cubecl_core::compilation_cache::CompilationCache;
use cubecl_core::ir::DeviceProperties;
use cubecl_core::server::LaunchError;
use cubecl_cpp::shared::CompilationOptions;
use cubecl_cpp::tt_metal::TtKernelSources;
use cubecl_runtime::id::KernelId;
use cubecl_runtime::timestamp_profiler::TimestampProfiler;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use cubecl_runtime::logging::ServerLogger;
use libtt_metal_cxx::{
    CircularBufferConfig, ComputeKernelConfig, CoreRangeSet, DataFormat, DataMovementKernelConfig,
    DataMovementProcessor, KernelBuildOptLevel, LogicalCore, MathFidelity, MeshDevice,
    MeshWorkload, Program,
};

#[derive(Debug)]
pub(crate) struct TtContext {
    mesh_ptr: *const MeshDevice,
    pub compiled_sources: HashMap<KernelId, TtKernelSources>,
    pub timestamps: TimestampProfiler,
    pub compilation_options: CompilationOptions,
    pub properties: DeviceProperties,
    pub compilation_cache: Option<CompilationCache<StableHash, CompilationCacheEntry>>,
}

// SAFETY: mesh_ptr is set during TtServer::new() and lives as long as TtServer.
unsafe impl Send for TtContext {}

pub struct TtCompiledKernel {
    pub program: Program,
    pub reader_kernel_id: u32,
    pub compute_kernel_id: u32,
    pub writer_kernel_id: u32,
    pub cb_in_id: usize,
    pub cb_out_id: usize,
}

impl core::fmt::Debug for TtCompiledKernel {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TtCompiledKernel")
            .field("program", &"<Program>")
            .field("reader_kernel_id", &self.reader_kernel_id)
            .field("compute_kernel_id", &self.compute_kernel_id)
            .field("writer_kernel_id", &self.writer_kernel_id)
            .field("cb_in_id", &self.cb_in_id)
            .field("cb_out_id", &self.cb_out_id)
            .finish()
    }
}

#[derive(Debug, Serialize, Deserialize, PartialEq, Eq, Clone)]
pub struct CompilationCacheEntry {
    pub entrypoint_name: String,
    pub shared_mem_bytes: usize,
    pub binary: Vec<i8>,
}

impl TtContext {
    pub fn new(compilation_options: CompilationOptions, properties: DeviceProperties) -> Self {
        Self {
            mesh_ptr: std::ptr::null(),
            compiled_sources: HashMap::new(),
            timestamps: TimestampProfiler::default(),
            compilation_options,
            compilation_cache: {
                use cubecl_runtime::config::RuntimeConfig;
                let config = cubecl_runtime::config::CubeClRuntimeConfig::get();
                if let Some(cache) = &config.compilation.cache {
                    let root = cache.root();
                    Some(CompilationCache::new(
                        "tt-metal-kernel",
                        CacheOption::default().name("tt_metal").root(root),
                    ))
                } else {
                    None
                }
            },
            properties,
        }
    }

    pub fn set_mesh_ptr(&mut self, ptr: *const MeshDevice) {
        self.mesh_ptr = ptr;
    }

    pub fn mesh(&self) -> &MeshDevice {
        assert!(!self.mesh_ptr.is_null(), "mesh_ptr not set");
        unsafe { &*self.mesh_ptr }
    }

    /// Compile a kernel from TT-Metal sources into a Program with kernels and CBs.
    ///
    /// This does NOT use the CubeCL IR pipeline yet. It uses pre-generated
    /// TtKernelSources (reader/compute/writer C++ strings) and compiles them
    /// through TT-Metal's host compiler.
    pub fn compile_kernel(
        &mut self,
        kernel_id: &KernelId,
        sources: &TtKernelSources,
        input_addr: u32,
        output_addr: u32,
        _logger: Arc<ServerLogger>,
    ) -> Result<TtCompiledKernel, LaunchError> {
        let core = LogicalCore::new(0, 0);
        let core_range = CoreRangeSet::from_core(core);

        // Create the Program
        let mut program = Program::new();

        // Configure Circular Buffers
        let cb_tiles = 2u32; // double buffering
        let cb_size = cb_tiles * sources.tile_size_bytes;
        let cb_format = data_format_to_tt(sources.data_format_tt);

        // Input CB at index 0
        let mut cb_in_config = CircularBufferConfig::new(cb_size);
        cb_in_config
            .index(0)
            .set_data_format(cb_format)
            .set_page_size(sources.tile_size_bytes);
        let cb_in_id = program
            .create_circular_buffer(&core_range, &cb_in_config)
            .map_err(|e| LaunchError::Unknown {
                reason: format!("failed to create input CB: {}", e.what()),
                backtrace: BackTrace::capture(),
            })?;

        // Output CB at index 16
        let mut cb_out_config = CircularBufferConfig::new(cb_size);
        cb_out_config
            .index(16)
            .set_data_format(cb_format)
            .set_page_size(sources.tile_size_bytes);
        let cb_out_id = program
            .create_circular_buffer(&core_range, &cb_out_config)
            .map_err(|e| LaunchError::Unknown {
                reason: format!("failed to create output CB: {}", e.what()),
                backtrace: BackTrace::capture(),
            })?;

        // Create kernels
        let mut reader_config =
            DataMovementKernelConfig::reader().map_err(|e| LaunchError::Unknown {
                reason: format!("failed to create reader config: {}", e.what()),
                backtrace: BackTrace::capture(),
            })?;
        reader_config
            .set_processor(DataMovementProcessor::Riscv1)
            .set_opt_level(KernelBuildOptLevel::O3);
        for &arg in &sources.reader_compile_args {
            reader_config.add_compile_arg(arg);
        }
        let reader_id = program
            .create_data_movement_kernel_from_string_with_config(
                &sources.reader_source,
                core,
                &reader_config,
            )
            .map_err(|e| LaunchError::Unknown {
                reason: format!("failed to create reader kernel: {}", e.what()),
                backtrace: BackTrace::capture(),
            })?;

        let mut writer_config =
            DataMovementKernelConfig::writer().map_err(|e| LaunchError::Unknown {
                reason: format!("failed to create writer config: {}", e.what()),
                backtrace: BackTrace::capture(),
            })?;
        writer_config
            .set_processor(DataMovementProcessor::Riscv0)
            .set_opt_level(KernelBuildOptLevel::O3);
        for &arg in &sources.writer_compile_args {
            writer_config.add_compile_arg(arg);
        }
        let writer_id = program
            .create_data_movement_kernel_from_string_with_config(
                &sources.writer_source,
                core,
                &writer_config,
            )
            .map_err(|e| LaunchError::Unknown {
                reason: format!("failed to create writer kernel: {}", e.what()),
                backtrace: BackTrace::capture(),
            })?;

        let mut compute_config = ComputeKernelConfig::new();
        compute_config
            .set_math_fidelity(MathFidelity::HiFi4)
            .set_opt_level(KernelBuildOptLevel::O3);
        let compute_id = program
            .create_compute_kernel_from_string_with_config(
                &sources.compute_source,
                core,
                &compute_config,
            )
            .map_err(|e| LaunchError::Unknown {
                reason: format!("failed to create compute kernel: {}", e.what()),
                backtrace: BackTrace::capture(),
            })?;

        // Set runtime args
        program
            .set_runtime_args(reader_id, core, &[input_addr, sources.num_tiles])
            .map_err(|e| LaunchError::Unknown {
                reason: format!("failed to set reader runtime args: {}", e.what()),
                backtrace: BackTrace::capture(),
            })?;
        program
            .set_runtime_args(writer_id, core, &[output_addr, sources.num_tiles])
            .map_err(|e| LaunchError::Unknown {
                reason: format!("failed to set writer runtime args: {}", e.what()),
                backtrace: BackTrace::capture(),
            })?;
        program
            .set_runtime_args(compute_id, core, &[sources.num_tiles])
            .map_err(|e| LaunchError::Unknown {
                reason: format!("failed to set compute runtime args: {}", e.what()),
                backtrace: BackTrace::capture(),
            })?;

        Ok(TtCompiledKernel {
            program,
            reader_kernel_id: reader_id,
            compute_kernel_id: compute_id,
            writer_kernel_id: writer_id,
            cb_in_id,
            cb_out_id,
        })
    }
}

/// Map our internal DataFormat value to the TT DataFormat enum.
/// Values: 0=Float32, 1=Float16, 5=Float16_b, etc.
fn data_format_to_tt(fmt: u8) -> DataFormat {
    match fmt {
        0 => DataFormat::Float32,
        1 => DataFormat::Float16,
        5 => DataFormat::Float16B,
        _ => DataFormat::Float16B, // default
    }
}
