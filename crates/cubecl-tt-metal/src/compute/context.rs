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
use cubecl_runtime::kernel::Visibility;
use cubecl_runtime::server::CubeCount;
use cubecl_runtime::timestamp_profiler::TimestampProfiler;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;

use crate::compute::storage::gpu::TtResource;
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

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct RuntimeBinding {
    pub visibility: Visibility,
    pub address: u32,
    pub logical_size_bytes: u64,
    pub allocation_size_bytes: u64,
    pub compile_args: Vec<u32>,
    pub item_size_bytes: u32,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub(crate) struct PreparedLaunch {
    pub sources: TtKernelSources,
    pub input_addrs: Vec<u32>,
    pub output_addrs: Vec<u32>,
    pub bindings: Vec<RuntimeBinding>,
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
        _logger: Arc<ServerLogger>,
    ) -> Result<TtCompiledKernel, LaunchError> {
        let _ = println!(
            "[compile_kernel] enter
"
        );
        let core = LogicalCore::new(0, 0);
        let core_range = CoreRangeSet::from_core(core);

        if sources.num_inputs > 0 && sources.reader_compile_args.is_empty() {
            return Err(kernel_build_error(
                "reader compile args missing for TT-Metal input buffers",
            ));
        }
        if sources.num_outputs > 0 && sources.writer_compile_args.is_empty() {
            return Err(kernel_build_error(
                "writer compile args missing for TT-Metal output buffers",
            ));
        }

        let _ = println!(
            "[compile_kernel] Program::new
"
        );
        let mut program = Program::new();

        let cb_tiles = 2u32;
        let cb_size = cb_tiles * sources.tile_size_bytes;
        let cb_format = data_format_to_tt(sources.data_format_tt)?;

        let mut cb_in_ids = Vec::new();
        for i in 0..sources.num_inputs {
            let _ = println!(
                "[compile_kernel] create_circular_buffer in
"
            );
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

        let mut cb_out_ids = Vec::new();
        for i in 0..sources.num_outputs {
            let _ = println!(
                "[compile_kernel] create_circular_buffer out
"
            );
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

        let _ = println!(
            "[compile_kernel] create_data_movement reader
"
        );
        let mut reader_config =
            DataMovementKernelConfig::reader().map_err(map_launch_err("reader config"))?;
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
            .map_err(map_kernel_build_err("reader kernel"))?;

        let _ = println!(
            "[compile_kernel] create_data_movement writer
"
        );
        let mut writer_config =
            DataMovementKernelConfig::writer().map_err(map_launch_err("writer config"))?;
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
            .map_err(map_kernel_build_err("writer kernel"))?;

        let _ = println!(
            "[compile_kernel] create_compute_kernel
"
        );
        let mut compute_config = ComputeKernelConfig::new();
        compute_config
            .set_math_fidelity(MathFidelity::HiFi4)
            .set_opt_level(KernelBuildOptLevel::O3);
        if matches!(cb_format, DataFormat::Float32) {
            compute_config
                .set_fp32_dest_acc_en(true)
                .set_dst_full_sync_en(true);
        }
        let compute_id = program
            .create_compute_kernel_from_string_with_config(
                &sources.compute_source,
                core,
                &compute_config,
            )
            .map_err(map_kernel_build_err("compute kernel"))?;

        let _ = println!(
            "[compile_kernel] set_runtime_args
"
        );
        let mut reader_args: Vec<u32> = input_addrs.to_vec();
        reader_args.push(sources.num_tiles);
        program
            .set_runtime_args(reader_id, core, &reader_args)
            .map_err(map_launch_err("reader runtime args"))?;

        let mut writer_args =
            Vec::with_capacity(output_addrs.len() + 1 + sources.writer_runtime_args.len());
        writer_args.extend(output_addrs.iter().copied());
        writer_args.push(sources.num_tiles);
        writer_args.extend(sources.writer_runtime_args.iter().copied());
        println!("[compile_kernel] writer_args={:?}", writer_args);
        program
            .set_runtime_args(writer_id, core, &writer_args)
            .map_err(map_launch_err("writer runtime args"))?;

        let mut compute_args = Vec::with_capacity(1 + sources.compute_runtime_args.len());
        compute_args.push(sources.num_tiles);
        compute_args.extend(sources.compute_runtime_args.iter().copied());
        program
            .set_runtime_args(compute_id, core, &compute_args)
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

    /// Compile a `CubeTask` into a prepared TT-Metal launch description.
    pub fn prepare_cube_task_launch(
        &mut self,
        cube_kernel: Box<dyn CubeTask<TtCompiler>>,
        mode: ExecutionMode,
        count: CubeCount,
        resources: &[TtResource],
        info: &cubecl_runtime::server::MetadataBindingInfo,
    ) -> Result<PreparedLaunch, LaunchError> {
        let _ = println!(
            "[compile_cube_task] enter
"
        );
        let mut compiler: TtCompiler = Default::default();
        let _ = println!(
            "[compile_cube_task] calling cube_kernel.compile()
"
        );
        let compiled = cube_kernel
            .compile(
                &mut compiler,
                &self.compilation_options,
                mode,
                cube_kernel.address_type(),
            )
            .map_err(LaunchError::CompilationError)?;
        let _ = println!(
            "[compile_cube_task] ir compiled
"
        );

        let repr = compiled.repr.as_ref().ok_or_else(|| {
            LaunchError::CompilationError(cubecl_runtime::compiler::CompilationError::Generic {
                reason: "no IR representation".into(),
                backtrace: BackTrace::capture(),
            })
        })?;

        let sources = match cubecl_cpp::tt_metal::compile::sources_from_repr(repr, 1) {
            Ok(sources) => sources,
            Err(err) => {
                eprintln!("[compile_cube_task] sources_from_repr failed: {err:?}");
                return Err(LaunchError::CompilationError(err));
            }
        };
        match prepare_launch(repr, sources, resources, info, count) {
            Ok(prepared) => Ok(prepared),
            Err(err) => {
                eprintln!("[compile_cube_task] prepare_launch failed: {err:?}");
                Err(err)
            }
        }
    }

    /// Compile a `CubeTask` into a TT-Metal program.
    pub fn compile_cube_task(
        &mut self,
        cube_kernel: Box<dyn CubeTask<TtCompiler>>,
        mode: ExecutionMode,
        resources: &[TtResource],
        info: &cubecl_runtime::server::MetadataBindingInfo,
        logger: Arc<ServerLogger>,
    ) -> Result<TtCompiledKernel, LaunchError> {
        let prepared = self.prepare_cube_task_launch(
            cube_kernel,
            mode,
            CubeCount::Static(1, 1, 1),
            resources,
            info,
        )?;

        let _ = println!(
            "[compile_cube_task] calling compile_kernel
"
        );
        let result = self.compile_kernel(
            &prepared.sources,
            &prepared.input_addrs,
            &prepared.output_addrs,
            logger,
        );
        let _ = println!(
            "[compile_cube_task] compile_kernel done
"
        );
        result
    }
}

pub(crate) fn data_format_to_tt(fmt: u8) -> Result<DataFormat, LaunchError> {
    match fmt {
        0 => Ok(DataFormat::Float32),
        1 => Ok(DataFormat::Float16),
        5 => Ok(DataFormat::Float16B),
        8 => Ok(DataFormat::Int32),
        9 => Ok(DataFormat::UInt16),
        24 => Ok(DataFormat::UInt32),
        30 => Ok(DataFormat::UInt8),
        _ => Err(kernel_build_error(&format!(
            "unsupported TT data format code: {fmt}"
        ))),
    }
}

fn kernel_build_error(reason: &str) -> LaunchError {
    LaunchError::CompilationError(cubecl_runtime::compiler::CompilationError::Generic {
        reason: reason.into(),
        backtrace: BackTrace::capture(),
    })
}

fn map_launch_err(context: &'static str) -> impl FnOnce(libtt_metal_cxx::Exception) -> LaunchError {
    move |e| LaunchError::Unknown {
        reason: format!("{context}: {}", e.what()),
        backtrace: BackTrace::capture(),
    }
}

fn map_kernel_build_err(
    context: &'static str,
) -> impl FnOnce(libtt_metal_cxx::Exception) -> LaunchError {
    move |e| kernel_build_error(&format!("{context}: {}", e.what()))
}

pub(crate) fn prepare_launch(
    repr: &cubecl_cpp::shared::ComputeKernel<
        cubecl_cpp::tt_metal::TtMetalDialect<crate::TtWmmaCompiler>,
    >,
    mut sources: TtKernelSources,
    resources: &[TtResource],
    info: &cubecl_runtime::server::MetadataBindingInfo,
    count: CubeCount,
) -> Result<PreparedLaunch, LaunchError> {
    if resources.len() != repr.buffers.len() {
        return Err(kernel_build_error(&format!(
            "resource/buffer arity mismatch: {} runtime resources for {} compiled buffers",
            resources.len(),
            repr.buffers.len()
        )));
    }

    let bindings = repr
        .buffers
        .iter()
        .zip(resources.iter())
        .map(|(binding, resource)| RuntimeBinding {
            visibility: binding.vis,
            address: resource.address,
            logical_size_bytes: resource.size,
            allocation_size_bytes: resource.allocation_size,
            compile_args: resource.compile_args.clone(),
            item_size_bytes: binding.item.size() as u32,
        })
        .collect::<Vec<_>>();

    let launched_num_units = match count {
        CubeCount::Static(x, y, z) => x
            .saturating_mul(y)
            .saturating_mul(z)
            .saturating_mul(repr.cube_dim.num_elems())
            .max(1),
        CubeCount::Dynamic(_) => 0,
    };

    let launched_cube_count = match count {
        CubeCount::Static(x, y, z) => [x, y, z],
        CubeCount::Dynamic(_) => [0, 0, 0],
    };

    let input_binding_indices =
        if sources.input_binding_indices.is_empty() && sources.num_inputs > 0 {
            bindings
                .iter()
                .enumerate()
                .filter_map(|(index, binding)| {
                    matches!(binding.visibility, Visibility::Read).then_some(index)
                })
                .collect::<Vec<_>>()
        } else {
            sources.input_binding_indices.clone()
        };
    let output_binding_indices =
        if sources.output_binding_indices.is_empty() && sources.num_outputs > 0 {
            bindings
                .iter()
                .enumerate()
                .filter_map(|(index, binding)| {
                    matches!(binding.visibility, Visibility::ReadWrite).then_some(index)
                })
                .collect::<Vec<_>>()
        } else {
            sources.output_binding_indices.clone()
        };
    let input_bindings = input_binding_indices
        .iter()
        .map(|&index| {
            bindings.get(index).ok_or_else(|| {
                kernel_build_error(&format!(
                    "reader binding index {} out of range for {} runtime bindings",
                    index,
                    bindings.len(),
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let output_bindings = output_binding_indices
        .iter()
        .map(|&index| {
            bindings.get(index).ok_or_else(|| {
                kernel_build_error(&format!(
                    "writer binding index {} out of range for {} runtime bindings",
                    index,
                    bindings.len(),
                ))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    if sources.num_inputs != input_bindings.len() as u32 {
        return Err(kernel_build_error(&format!(
            "reader source expects {} inputs but runtime provided {}",
            sources.num_inputs,
            input_bindings.len()
        )));
    }
    if sources.num_outputs != output_bindings.len() as u32 {
        return Err(kernel_build_error(&format!(
            "writer source expects {} outputs but runtime provided {}",
            sources.num_outputs,
            output_bindings.len()
        )));
    }

    let input_addrs = input_bindings
        .iter()
        .map(|binding| binding.address)
        .collect::<Vec<_>>();
    let output_addrs = output_bindings
        .iter()
        .map(|binding| binding.address)
        .collect::<Vec<_>>();
    let rewrite_accessor_page_size = |compile_args: &[u32]| -> Vec<u32> {
        let mut adjusted = compile_args.to_vec();
        if !sources.requires_tiled_io() {
            if let Some(page_size) = adjusted.get_mut(1) {
                *page_size = sources.tile_size_bytes;
            }
        }
        adjusted
    };
    let reader_compile_args = input_bindings
        .iter()
        .flat_map(|binding| rewrite_accessor_page_size(&binding.compile_args))
        .collect::<Vec<_>>();
    let writer_compile_args = output_bindings
        .iter()
        .flat_map(|binding| rewrite_accessor_page_size(&binding.compile_args))
        .collect::<Vec<_>>();
    sources = sources.with_compile_args(reader_compile_args, writer_compile_args);

    if !sources.buffer_item_sizes.is_empty() {
        if sources.buffer_item_sizes.len() != bindings.len() {
            return Err(kernel_build_error(&format!(
                "static metadata expects {} buffer item sizes but runtime provided {} resources",
                sources.buffer_item_sizes.len(),
                bindings.len(),
            )));
        }

        for (index, (expected, binding)) in sources
            .buffer_item_sizes
            .iter()
            .zip(bindings.iter())
            .enumerate()
        {
            if *expected != binding.item_size_bytes {
                return Err(kernel_build_error(&format!(
                    "buffer item size mismatch at binding {index}: compiler expected {} bytes but runtime binding uses {} bytes",
                    expected, binding.item_size_bytes,
                )));
            }
        }
    }

    let logical_num_units = if sources.requires_tiled_io() {
        let Some(first_binding) = bindings.first() else {
            return Err(kernel_build_error(
                "TT-Metal tiled I/O path requires at least one runtime binding",
            ));
        };
        let item_size = u64::from(sources.native_scalar_size_bytes().max(1));
        for (index, binding) in bindings.iter().enumerate() {
            if binding.item_size_bytes != first_binding.item_size_bytes {
                return Err(kernel_build_error(&format!(
                    "TT-Metal tiled I/O path requires a uniform element size across bindings; binding 0 uses {} bytes but binding {index} uses {} bytes",
                    first_binding.item_size_bytes, binding.item_size_bytes,
                )));
            }
            if binding.logical_size_bytes != first_binding.logical_size_bytes {
                return Err(kernel_build_error(&format!(
                    "TT-Metal tiled I/O path currently requires matching logical buffer sizes across bindings; binding 0 uses {} bytes but binding {index} uses {} bytes",
                    first_binding.logical_size_bytes, binding.logical_size_bytes,
                )));
            }
        }

        let tile_elems = u64::from(sources.tile_size_bytes) / item_size;
        let logical_elems = first_binding.logical_size_bytes.div_ceil(item_size);
        let native_tiles = logical_elems.div_ceil(tile_elems).max(1) as u32;
        sources = sources.with_num_tiles(native_tiles);
        logical_elems as u32
    } else {
        let unit_size = u64::from(sources.unit_item_size_bytes.max(1));
        let logical_units = output_bindings
            .iter()
            .map(|binding| binding.logical_size_bytes.div_ceil(unit_size) as u32)
            .max()
            .unwrap_or_else(|| {
                bindings
                    .iter()
                    .map(|binding| binding.logical_size_bytes.div_ceil(unit_size) as u32)
                    .max()
                    .unwrap_or(1)
            })
            .max(1);
        let dispatched_units = if launched_num_units == 0 {
            logical_units
        } else {
            launched_num_units
        };
        let tile_units = (sources.tile_size_bytes / sources.unit_item_size_bytes.max(1)).max(1);
        let dispatched_tiles = dispatched_units.div_ceil(tile_units).max(1);
        sources = sources.with_num_tiles(dispatched_tiles);
        dispatched_units
    };

    if !info.data.is_empty() {
        let packed_info_words = bytemuck::cast_slice::<u64, u32>(&info.data).to_vec();
        if sources.requires_tiled_io() {
            sources = sources.with_writer_runtime_args(packed_info_words);
        } else {
            let mut writer_runtime_args = Vec::with_capacity(4 + packed_info_words.len());
            writer_runtime_args.push(logical_num_units);
            writer_runtime_args.extend_from_slice(&launched_cube_count);
            writer_runtime_args.extend(packed_info_words);
            sources = sources.with_writer_runtime_args(writer_runtime_args);
        }
    } else {
        if repr.body.has_dynamic_meta {
            return Err(kernel_build_error(
                "TT-Metal generic kernels require metadata info runtime args for dynamic tensor metadata",
            ));
        }

        if sources.info_static_len > 0 {
            if sources.buffer_item_sizes.len() != bindings.len() {
                return Err(kernel_build_error(&format!(
                    "cannot derive static metadata for {} bindings from {} item sizes",
                    bindings.len(),
                    sources.buffer_item_sizes.len(),
                )));
            }

            let logical_lengths = bindings
                .iter()
                .zip(sources.buffer_item_sizes.iter())
                .map(|(binding, item_size)| {
                    let item_size = u64::from((*item_size).max(1));
                    binding.logical_size_bytes.div_ceil(item_size) as u32
                })
                .collect::<Vec<_>>();
            let mut compute_runtime_args = Vec::with_capacity(logical_lengths.len() * 2);
            compute_runtime_args.extend(logical_lengths.iter().copied());
            compute_runtime_args.extend(logical_lengths.iter().copied());
            if compute_runtime_args.len() != sources.info_static_len {
                return Err(kernel_build_error(&format!(
                    "compute runtime args mismatch: derived {} static args for info_static_len {}",
                    compute_runtime_args.len(),
                    sources.info_static_len,
                )));
            }
            sources = sources.with_compute_runtime_args(compute_runtime_args.clone());
            if sources.requires_tiled_io() {
                sources = sources.with_writer_runtime_args(compute_runtime_args);
            } else {
                let mut writer_runtime_args = Vec::with_capacity(4 + compute_runtime_args.len());
                writer_runtime_args.push(logical_num_units);
                writer_runtime_args.extend_from_slice(&launched_cube_count);
                writer_runtime_args.extend(compute_runtime_args);
                sources = sources.with_writer_runtime_args(writer_runtime_args);
            }
        } else if !sources.requires_tiled_io() {
            sources = sources.with_writer_runtime_args(vec![
                logical_num_units,
                launched_cube_count[0],
                launched_cube_count[1],
                launched_cube_count[2],
            ]);
        }
    }

    Ok(PreparedLaunch {
        sources,
        input_addrs,
        output_addrs,
        bindings,
    })
}
