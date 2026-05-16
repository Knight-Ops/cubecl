use cubecl_common::cache::CacheOption;
use cubecl_common::hash::StableHash;
use cubecl_core::backtrace::BackTrace;
use cubecl_core::compilation_cache::CompilationCache;
use cubecl_core::ir::DeviceProperties;
use cubecl_core::server::{ExecutionMode, LaunchError};
use cubecl_cpp::shared::CompilationOptions;
use cubecl_cpp::tt_metal::TtKernelSources;
use cubecl_runtime::compiler::CubeTask;
use cubecl_runtime::id::KernelId;
use cubecl_runtime::timestamp_profiler::TimestampProfiler;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::runtime::TtCompiler;
use cubecl_runtime::logging::ServerLogger;
use libtt_metal_cxx::{
    CircularBufferConfig, ComputeKernelConfig, CoreRangeSet, DataFormat, DataMovementKernelConfig,
    DataMovementProcessor, KernelBuildOptLevel, LogicalCore, MathFidelity, Program,
};

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct TtContext {
    pub compiled_sources: HashMap<KernelId, TtKernelSources>,
    pub timestamps: TimestampProfiler,
    pub compilation_options: CompilationOptions,
    pub properties: DeviceProperties,
    pub compilation_cache: Option<CompilationCache<StableHash, CompilationCacheEntry>>,
}

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

    /// Compile a kernel from TT-Metal sources into a Program with kernels and CBs.
    pub fn compile_kernel(
        &mut self,
        sources: &TtKernelSources,
        input_addrs: &[u32],
        output_addrs: &[u32],
        reader_compile_args: &[u32],
        writer_compile_args: &[u32],
        _logger: Arc<ServerLogger>,
    ) -> Result<TtCompiledKernel, LaunchError> {
        let core = LogicalCore::new(0, 0);
        let core_range = CoreRangeSet::from_core(core);

        let mut program = Program::new();

        let cb_tiles = 2u32;
        let cb_size = cb_tiles * sources.tile_size_bytes;
        let cb_format = data_format_to_tt(sources.data_format_tt);

        // Input CBs at indices 0, 1, 2, ...
        let mut cb_in_ids = Vec::new();
        for i in 0..sources.num_inputs {
            let mut cb_config = CircularBufferConfig::new(cb_size);
            cb_config
                .index(i as u8)
                .set_data_format(cb_format)
                .set_page_size(sources.tile_size_bytes);
            let id = program
                .create_circular_buffer(&core_range, &cb_config)
                .map_err(map_launch_err("input CB create"))?;
            cb_in_ids.push(id);
        }

        // Output CBs at indices 16, 17, 18, ...
        let mut cb_out_ids = Vec::new();
        for i in 0..sources.num_outputs {
            let mut cb_config = CircularBufferConfig::new(cb_size);
            cb_config
                .index(16u8 + i as u8)
                .set_data_format(cb_format)
                .set_page_size(sources.tile_size_bytes);
            let id = program
                .create_circular_buffer(&core_range, &cb_config)
                .map_err(map_launch_err("output CB create"))?;
            cb_out_ids.push(id);
        }

        // Reader kernel
        let mut reader_config =
            DataMovementKernelConfig::reader().map_err(map_launch_err("reader config"))?;
        reader_config
            .set_processor(DataMovementProcessor::Riscv1)
            .set_opt_level(KernelBuildOptLevel::O3);
        for &arg in reader_compile_args {
            reader_config.add_compile_arg(arg);
        }
        let reader_id = program
            .create_data_movement_kernel_from_string_with_config(
                &sources.reader_source,
                core,
                &reader_config,
            )
            .map_err(map_launch_err("reader kernel"))?;

        // Writer kernel
        let mut writer_config =
            DataMovementKernelConfig::writer().map_err(map_launch_err("writer config"))?;
        writer_config
            .set_processor(DataMovementProcessor::Riscv0)
            .set_opt_level(KernelBuildOptLevel::O3);
        for &arg in writer_compile_args {
            writer_config.add_compile_arg(arg);
        }
        let writer_id = program
            .create_data_movement_kernel_from_string_with_config(
                &sources.writer_source,
                core,
                &writer_config,
            )
            .map_err(map_launch_err("writer kernel"))?;

        // Compute kernel
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
            .map_err(map_launch_err("compute kernel"))?;

        // Runtime args: reader gets [addr_0, ..., num_tiles, tile_size]
        let mut reader_args: Vec<u32> = input_addrs.to_vec();
        reader_args.push(sources.num_tiles);
        reader_args.push(sources.tile_size_bytes);
        program
            .set_runtime_args(reader_id, core, &reader_args)
            .map_err(map_launch_err("reader runtime args"))?;

        let mut writer_args: Vec<u32> = output_addrs.to_vec();
        writer_args.push(sources.num_tiles);
        writer_args.push(sources.tile_size_bytes);
        program
            .set_runtime_args(writer_id, core, &writer_args)
            .map_err(map_launch_err("writer runtime args"))?;

        program
            .set_runtime_args(compute_id, core, &[sources.num_tiles])
            .map_err(map_launch_err("compute runtime args"))?;

        Ok(TtCompiledKernel {
            program,
            reader_kernel_id: reader_id,
            compute_kernel_id: compute_id,
            writer_kernel_id: writer_id,
            cb_in_id: cb_in_ids.first().copied().unwrap_or(0),
            cb_out_id: cb_out_ids.first().copied().unwrap_or(0),
        })
    }

    /// Compile a `CubeTask` into a TT-Metal program.
    pub fn compile_cube_task(
        &mut self,
        cube_kernel: Box<dyn CubeTask<TtCompiler>>,
        mode: ExecutionMode,
        input_addrs: &[u32],
        output_addrs: &[u32],
        logger: Arc<ServerLogger>,
    ) -> Result<TtCompiledKernel, LaunchError> {
        let mut compiler: TtCompiler = Default::default();
        let compiled = cube_kernel
            .compile(
                &mut compiler,
                &self.compilation_options,
                mode,
                cube_kernel.address_type(),
            )
            .map_err(|e| LaunchError::Unknown {
                reason: format!("CubeCL compilation failed: {e:?}"),
                backtrace: BackTrace::capture(),
            })?;

        let repr = compiled.repr.as_ref().ok_or_else(|| LaunchError::Unknown {
            reason: "no IR representation".into(),
            backtrace: BackTrace::capture(),
        })?;

        let num_tiles = repr.cube_dim.x.max(1);
        let tile_size_bytes: u32 = 32 * 32 * 2;

        let op_kind = detect_op_from_body(&repr.body, repr.buffers.len() as u32);
        let sources = match op_kind {
            TtOpKind::Copy | TtOpKind::Unknown => {
                TtKernelSources::copy_kernel(num_tiles, tile_size_bytes)
            }
            TtOpKind::EltwiseBinaryAdd => TtKernelSources::add_kernel(num_tiles, tile_size_bytes),
        };

        self.compile_kernel(
            &sources,
            input_addrs,
            output_addrs,
            &[2, tile_size_bytes],
            &[2, tile_size_bytes],
            logger,
        )
    }
}

use cubecl_cpp::Dialect;
use cubecl_cpp::tt_metal::TtOpKind;

fn detect_op_from_body<D: Dialect>(
    body: &cubecl_cpp::shared::Body<D>,
    _num_buffers: u32,
) -> TtOpKind {
    for inst in &body.instructions {
        if matches!(
            inst,
            cubecl_cpp::shared::Instruction::Add(_)
                | cubecl_cpp::shared::Instruction::Mul(_)
                | cubecl_cpp::shared::Instruction::Sub(_)
        ) {
            return TtOpKind::EltwiseBinaryAdd;
        }
    }
    TtOpKind::Copy
}

fn data_format_to_tt(fmt: u8) -> DataFormat {
    match fmt {
        0 => DataFormat::Float32,
        1 => DataFormat::Float16,
        5 => DataFormat::Float16B,
        _ => DataFormat::Float16B,
    }
}

fn map_launch_err(context: &'static str) -> impl FnOnce(libtt_metal_cxx::Exception) -> LaunchError {
    move |e| LaunchError::Unknown {
        reason: format!("{context}: {}", e.what()),
        backtrace: BackTrace::capture(),
    }
}
