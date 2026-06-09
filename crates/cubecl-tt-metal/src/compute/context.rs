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
use std::collections::{HashMap, hash_map::Entry};
use std::sync::Arc;

use crate::compute::storage::gpu::TtResource;
use crate::runtime::TtCompiler;
use cubecl_runtime::logging::ServerLogger;
use libtt_metal_cxx::{
    CircularBufferConfig, ComputeKernelConfig, CoreRange, CoreRangeSet, DataFormat,
    DataMovementKernelConfig, DataMovementProcessor, KernelBuildOptLevel, LogicalCore,
    MathFidelity, MeshDevice, Program,
};

const TT_RUNTIME_FULL_INPUT_STAGING_MAX_TILES: u32 = 16;
const TT_STREAM_DEBUG_ENV: &str = "CUBECL_TT_METAL_STREAM_DEBUG";

#[derive(Debug)]
#[allow(dead_code)]
pub(crate) struct TtContext {
    pub compiled_sources: HashMap<KernelId, CachedCubeTaskLaunch>,
    pub timestamps: TimestampProfiler,
    pub compilation_options: CompilationOptions,
    pub properties: DeviceProperties,
    pub compilation_cache: Option<CompilationCache<StableHash, CompilationCacheEntry>>,
}

#[derive(Debug, Clone)]
pub(crate) struct CachedCubeTaskLaunch {
    pub repr: cubecl_cpp::shared::ComputeKernel<
        cubecl_cpp::tt_metal::TtMetalDialect<crate::TtWmmaCompiler>,
    >,
    pub sources: TtKernelSources,
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
    pub tensor_output_bridge: Option<TensorOutputBridgePlan>,
}

#[derive(Debug, Clone)]
pub(crate) struct TensorOutputBridgePlan {
    pub temp_size_bytes: u64,
    pub logical_shape: Vec<usize>,
    pub logical_strides: Vec<usize>,
    pub logical_elem_size_bytes: usize,
    pub padded_row_bytes: usize,
    pub writer_runtime_args: Vec<u32>,
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

fn enumerate_core_range_row_wise(range: CoreRange) -> Vec<LogicalCore> {
    let start = range.start;
    let end = range.end;
    let mut cores = Vec::new();
    for y in start.y..=end.y {
        for x in start.x..=end.x {
            cores.push(LogicalCore::new(x, y));
        }
    }
    cores
}

fn split_cores_row_wise(target_num_cores: u32, grid_x: u32, grid_y: u32) -> Vec<CoreRange> {
    if target_num_cores == 0 || grid_x == 0 || grid_y == 0 {
        return Vec::new();
    }

    // Keep the current generic TT path on a single logical row. The multi-row
    // CoreRangeSet launch path is not yet behaving truthfully for these raw
    // generic kernels on hardware, while a single-row partition still lets us
    // distribute multiple tiles per core correctly.
    let take = target_num_cores.min(grid_x);
    vec![CoreRange::new(
        LogicalCore::new(0, 0),
        LogicalCore::new(take - 1, 0),
    )]
}

fn partitioned_tt_cores(
    mesh: &MeshDevice,
    sources: &TtKernelSources,
    allow_core_partitioning: bool,
) -> Result<(Vec<LogicalCore>, CoreRangeSet), LaunchError> {
    let fallback_core = LogicalCore::new(0, 0);
    let fallback_range = CoreRange::from_core(fallback_core);
    if !allow_core_partitioning || !sources.supports_core_partitioning || sources.num_tiles <= 1 {
        return Ok((
            vec![fallback_core],
            CoreRangeSet::from_range(fallback_range),
        ));
    }

    let (grid_x, grid_y) = mesh
        .compute_with_storage_grid_size()
        .map_err(map_launch_err("mesh compute_with_storage_grid_size"))?;
    let available = grid_x.saturating_mul(grid_y).max(1);
    let target = sources.num_tiles.min(available).max(1);
    if target <= 1 {
        return Ok((
            vec![fallback_core],
            CoreRangeSet::from_range(fallback_range),
        ));
    }

    // Keep the generic TT path conservative until uneven multi-core tile
    // partitioning is fully trustworthy. The linearized topology lane can
    // otherwise drop the remainder tail and leave stale output at the end.
    if sources.num_tiles % target != 0 {
        return Ok((
            vec![fallback_core],
            CoreRangeSet::from_range(fallback_range),
        ));
    }

    let core_ranges = split_cores_row_wise(target, grid_x, grid_y);
    let logical_cores = core_ranges
        .iter()
        .copied()
        .flat_map(enumerate_core_range_row_wise)
        .collect::<Vec<_>>();
    let core_range_set = CoreRangeSet::from_ranges(core_ranges.iter().copied());

    Ok((logical_cores, core_range_set))
}

fn full_input_staging_tiles(resources: &[TtResource], sources: &TtKernelSources) -> Option<u32> {
    if sources.requires_tiled_io() || !sources.supports_core_partitioning || sources.num_inputs != 1
    {
        return None;
    }

    let input_binding_index = *sources.input_binding_indices.first()?;
    let resource = resources.get(input_binding_index)?;
    if resource.allocation_offset != 0 {
        return None;
    }

    let page_size_bytes = u64::from(*resource.compile_args.get(1)?);
    if page_size_bytes == 0 {
        return None;
    }

    let staged_tiles = resource.size.div_ceil(page_size_bytes) as u32;
    (2..=TT_RUNTIME_FULL_INPUT_STAGING_MAX_TILES)
        .contains(&staged_tiles)
        .then_some(staged_tiles)
}

pub(crate) fn effective_generic_dispatch_unit_size_bytes(
    repr: &cubecl_cpp::shared::ComputeKernel<
        cubecl_cpp::tt_metal::TtMetalDialect<crate::TtWmmaCompiler>,
    >,
    resources: &[TtResource],
    info: &cubecl_runtime::server::MetadataBindingInfo,
    count: &CubeCount,
) -> Option<u32> {
    let (cube_count_x, launched_num_units) = match count {
        CubeCount::Static(x, y, z) => (
            *x,
            x.saturating_mul(*y)
                .saturating_mul(*z)
                .saturating_mul(repr.cube_dim.num_elems())
                .max(1),
        ),
        CubeCount::Dynamic(_) => return None,
    };
    let total_units_x = cube_count_x.saturating_mul(repr.cube_dim.x).max(1);
    let dynamic_meta_base_words =
        info.dynamic_metadata_offset * (core::mem::size_of::<u64>() / core::mem::size_of::<u32>());
    let info_words = bytemuck::cast_slice::<u64, u32>(&info.data);

    repr.buffers
        .iter()
        .zip(resources.iter())
        .enumerate()
        .filter(|(_, (binding, _))| {
            matches!(binding.vis, Visibility::ReadWrite)
                && binding.item.size() > binding.item.elem().unpacked().size()
        })
        .map(|(binding_index, (binding, resource))| {
            let scalar_size = binding.item.elem().unpacked().size() as u64;
            let mut bytes_per_unit = resource
                .size
                .div_ceil(u64::from(launched_num_units))
                .max(scalar_size);
            let metadata = &repr.info.metadata;
            let rank = info_words
                .get(metadata.rank_index(binding_index as u32) as usize)
                .copied()
                .unwrap_or(0) as usize;
            if rank > 0 {
                let shape_offset = info_words
                    .get(metadata.shape_offset_index(binding_index as u32) as usize)
                    .copied()
                    .unwrap_or(0) as usize;
                let stride_offset = info_words
                    .get(metadata.stride_offset_index(binding_index as u32) as usize)
                    .copied()
                    .unwrap_or(0) as usize;
                let last_dim = rank - 1;
                let logical_row_len = info_words
                    .get(dynamic_meta_base_words + shape_offset + last_dim)
                    .copied()
                    .unwrap_or(0) as u64;
                let logical_row_stride = info_words
                    .get(dynamic_meta_base_words + stride_offset + last_dim)
                    .copied()
                    .unwrap_or(0);
                if logical_row_len > 0 && logical_row_stride == 1 {
                    bytes_per_unit = bytes_per_unit.max(
                        logical_row_len
                            .saturating_mul(scalar_size)
                            .div_ceil(u64::from(total_units_x)),
                    );
                }
            }
            bytes_per_unit as u32
        })
        .max()
}

fn tensor_output_bridge_plan(
    repr: &cubecl_cpp::shared::ComputeKernel<
        cubecl_cpp::tt_metal::TtMetalDialect<crate::TtWmmaCompiler>,
    >,
    info: &cubecl_runtime::server::MetadataBindingInfo,
    count: &CubeCount,
) -> Option<TensorOutputBridgePlan> {
    if !repr.body.has_dynamic_meta
        || repr
            .buffers
            .iter()
            .any(|binding| matches!(binding.vis, Visibility::Read))
    {
        return None;
    }

    let (cube_count_x, cube_count_y, cube_count_z) = match count {
        CubeCount::Static(x, y, z) => (*x as usize, *y as usize, *z as usize),
        CubeCount::Dynamic(_) => return None,
    };
    if cube_count_z != 1 {
        return None;
    }

    let output_binding_index = repr
        .buffers
        .iter()
        .enumerate()
        .find_map(|(index, binding)| {
            matches!(binding.vis, Visibility::ReadWrite).then_some(index)
        })?;
    let output_binding = repr.buffers.get(output_binding_index)?;
    let vectorization = output_binding.item.size() / output_binding.item.elem().size();
    if vectorization <= 1 {
        return None;
    }

    let packed_info_words = bytemuck::cast_slice::<u64, u32>(&info.data);
    let static_meta_base_words = repr
        .info
        .sized_meta
        .as_ref()
        .map(|field| field.offset / core::mem::size_of::<u32>())
        .unwrap_or(0);
    let dynamic_meta_base_words =
        info.dynamic_metadata_offset * (core::mem::size_of::<u64>() / core::mem::size_of::<u32>());
    let metadata = &repr.info.metadata;
    let rank = packed_info_words
        .get(static_meta_base_words + metadata.rank_index(output_binding_index as u32) as usize)
        .copied()? as usize;
    if rank != 2 {
        return None;
    }

    let shape_offset = packed_info_words
        .get(
            static_meta_base_words
                + metadata.shape_offset_index(output_binding_index as u32) as usize,
        )
        .copied()? as usize;
    let stride_offset = packed_info_words
        .get(
            static_meta_base_words
                + metadata.stride_offset_index(output_binding_index as u32) as usize,
        )
        .copied()? as usize;

    let logical_shape = (0..rank)
        .map(|dim| {
            packed_info_words
                .get(dynamic_meta_base_words + shape_offset + dim)
                .copied()
                .map(|value| value as usize)
        })
        .collect::<Option<Vec<_>>>()?;
    let logical_strides = (0..rank)
        .map(|dim| {
            packed_info_words
                .get(dynamic_meta_base_words + stride_offset + dim)
                .copied()
                .map(|value| value as usize)
        })
        .collect::<Option<Vec<_>>>()?;

    if logical_shape[1] == 0
        || logical_shape[0] == 0
        || logical_strides[1] != 1
        || logical_shape[1] % vectorization != 0
    {
        return None;
    }

    let launched_width = cube_count_x.checked_mul(repr.cube_dim.x as usize)?;
    let launched_height = cube_count_y.checked_mul(repr.cube_dim.y as usize)?;
    if launched_width == 0
        || launched_height == 0
        || logical_shape[0] > launched_height
        || logical_shape[1] > launched_width.checked_mul(vectorization)?
    {
        return None;
    }

    let logical_elem_size_bytes = output_binding.item.elem().unpacked().size();
    let padded_row_scalars = launched_width.checked_mul(vectorization)?;
    let padded_row_bytes = launched_width.checked_mul(output_binding.item.size())?;
    let temp_size_bytes = launched_height.checked_mul(padded_row_bytes)? as u64;

    let mut bridge_info_words = packed_info_words.to_vec();
    bridge_info_words[dynamic_meta_base_words + shape_offset] = logical_shape[0] as u32;
    bridge_info_words[dynamic_meta_base_words + shape_offset + 1] = logical_shape[1] as u32;
    bridge_info_words[dynamic_meta_base_words + stride_offset] = padded_row_scalars as u32;
    bridge_info_words[dynamic_meta_base_words + stride_offset + 1] = 1;
    if let Some(gap_scalar) = repr.info.scalars.first() {
        let scalar_word_offset = gap_scalar.offset / core::mem::size_of::<u32>();
        let expected_gap = logical_strides[0].saturating_add(1) as u32;
        if bridge_info_words.get(scalar_word_offset).copied() == Some(expected_gap) {
            bridge_info_words[scalar_word_offset] = padded_row_scalars.saturating_add(1) as u32;
        }
    }

    let launched_num_units = (cube_count_x as u32)
        .saturating_mul(cube_count_y as u32)
        .saturating_mul(cube_count_z as u32)
        .saturating_mul(repr.cube_dim.num_elems())
        .max(1);
    let writer_runtime_args = [
        vec![
            launched_num_units,
            cube_count_x as u32,
            cube_count_y as u32,
            cube_count_z as u32,
        ],
        bridge_info_words,
    ]
    .concat();

    Some(TensorOutputBridgePlan {
        temp_size_bytes,
        logical_shape,
        logical_strides,
        logical_elem_size_bytes,
        padded_row_bytes,
        writer_runtime_args,
    })
}

fn stream_debug_enabled() -> bool {
    std::env::var_os(TT_STREAM_DEBUG_ENV).is_some()
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
        mesh: &MeshDevice,
        sources: &TtKernelSources,
        input_addrs: &[u32],
        output_addrs: &[u32],
        _logger: Arc<ServerLogger>,
        allow_core_partitioning: bool,
    ) -> Result<TtCompiledKernel, LaunchError> {
        let (logical_cores, core_range_set) =
            partitioned_tt_cores(mesh, sources, allow_core_partitioning)?;
        let core = logical_cores
            .first()
            .copied()
            .unwrap_or_else(|| LogicalCore::new(0, 0));

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

        let mut program = Program::new();

        let cb_tiles = 2u32;
        let _cb_size = cb_tiles * sources.tile_size_bytes;
        let cb_format = data_format_to_tt(sources.data_format_tt)?;

        let mut cb_in_ids = Vec::new();
        for i in 0..sources.num_inputs {
            let input_cb_bytes = if sources.full_input_staging_tiles > 0 && i == 0 {
                sources.full_input_staging_tiles.max(1) * sources.tile_size_bytes
            } else {
                cb_tiles * sources.tile_size_bytes
            };
            let input_cb_page_size = sources.tile_size_bytes;
            let mut cb_config = CircularBufferConfig::new(input_cb_bytes);
            cb_config
                .index(i as u8)
                .set_data_format(cb_format)
                .set_page_size(input_cb_page_size);
            let id = program
                .create_circular_buffer(&core_range_set, &cb_config)
                .map_err(map_launch_err("input CB create"))?;
            cb_in_ids.push(id);
        }

        let mut cb_out_ids = Vec::new();
        for i in 0..sources.num_outputs {
            let mut cb_config = CircularBufferConfig::new(cb_tiles * sources.tile_size_bytes);
            cb_config
                .index(16u8 + i as u8)
                .set_data_format(cb_format)
                .set_page_size(sources.tile_size_bytes);
            let id = program
                .create_circular_buffer(&core_range_set, &cb_config)
                .map_err(map_launch_err("output CB create"))?;
            cb_out_ids.push(id);
        }

        let mut reader_config =
            DataMovementKernelConfig::reader().map_err(map_launch_err("reader config"))?;
        reader_config
            .set_processor(DataMovementProcessor::Riscv1)
            .set_opt_level(KernelBuildOptLevel::O3);
        for &arg in &sources.reader_compile_args {
            reader_config.add_compile_arg(arg);
        }
        let reader_id = if logical_cores.len() == 1 {
            program
                .create_data_movement_kernel_from_string_with_config(
                    &sources.reader_source,
                    core,
                    &reader_config,
                )
                .map_err(map_kernel_build_err("reader kernel"))?
        } else {
            program
                .create_data_movement_kernel_from_string_with_config_ranges(
                    &core_range_set,
                    &sources.reader_source,
                    &reader_config,
                )
                .map_err(map_kernel_build_err("reader kernel"))?
        };

        let mut writer_config =
            DataMovementKernelConfig::writer().map_err(map_launch_err("writer config"))?;
        writer_config
            .set_processor(DataMovementProcessor::Riscv0)
            .set_opt_level(KernelBuildOptLevel::O3);
        for &arg in &sources.writer_compile_args {
            writer_config.add_compile_arg(arg);
        }
        let writer_id = if logical_cores.len() == 1 {
            program
                .create_data_movement_kernel_from_string_with_config(
                    &sources.writer_source,
                    core,
                    &writer_config,
                )
                .map_err(map_kernel_build_err("writer kernel"))?
        } else {
            program
                .create_data_movement_kernel_from_string_with_config_ranges(
                    &core_range_set,
                    &sources.writer_source,
                    &writer_config,
                )
                .map_err(map_kernel_build_err("writer kernel"))?
        };

        let mut compute_config = ComputeKernelConfig::new();
        compute_config
            .set_math_fidelity(MathFidelity::HiFi4)
            .set_opt_level(KernelBuildOptLevel::O3);
        if matches!(cb_format, DataFormat::Float32) {
            compute_config
                .set_fp32_dest_acc_en(true)
                .set_dst_full_sync_en(true);
        }
        let compute_id = if logical_cores.len() == 1 {
            program
                .create_compute_kernel_from_string_with_config(
                    &sources.compute_source,
                    core,
                    &compute_config,
                )
                .map_err(map_kernel_build_err("compute kernel"))?
        } else {
            program
                .create_compute_kernel_from_string_with_config_ranges(
                    &core_range_set,
                    &sources.compute_source,
                    &compute_config,
                )
                .map_err(map_kernel_build_err("compute kernel"))?
        };

        let core_count = logical_cores.len() as u32;
        let base_tiles = sources.num_tiles / core_count.max(1);
        let extra_tiles = sources.num_tiles % core_count.max(1);
        let mut start_tile = 0u32;
        for (core_index, logical_core) in logical_cores.iter().copied().enumerate() {
            let local_tiles = base_tiles + u32::from((core_index as u32) < extra_tiles);

            let mut reader_args: Vec<u32> = input_addrs.to_vec();
            reader_args.push(local_tiles);
            // TT reader sources now consistently accept `start_tile`, even on
            // the single-core path. Native tiled kernels simply receive `0`.
            reader_args.push(start_tile);
            reader_args.extend(sources.reader_runtime_args.iter().copied());
            program
                .set_runtime_args(reader_id, logical_core, &reader_args)
                .map_err(map_launch_err("reader runtime args"))?;

            let mut writer_args = Vec::with_capacity(
                output_addrs.len()
                    + 1
                    + usize::from(!sources.requires_tiled_io())
                    + sources.writer_runtime_args.len(),
            );
            writer_args.extend(output_addrs.iter().copied());
            writer_args.push(local_tiles);
            if !sources.requires_tiled_io() {
                writer_args.push(start_tile);
            }
            writer_args.extend(sources.writer_runtime_args.iter().copied());
            program
                .set_runtime_args(writer_id, logical_core, &writer_args)
                .map_err(map_launch_err("writer runtime args"))?;

            let mut compute_args = Vec::with_capacity(1 + sources.compute_runtime_args.len());
            compute_args.push(local_tiles);
            compute_args.extend(sources.compute_runtime_args.iter().copied());
            program
                .set_runtime_args(compute_id, logical_core, &compute_args)
                .map_err(map_launch_err("compute runtime args"))?;

            start_tile = start_tile.saturating_add(local_tiles);
        }

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
        let kernel_id = cube_kernel.id();
        let cached = match self.compiled_sources.entry(kernel_id) {
            Entry::Occupied(entry) => entry.into_mut(),
            Entry::Vacant(entry) => {
                let mut compiler: TtCompiler = Default::default();
                let compiled = cube_kernel
                    .compile(
                        &mut compiler,
                        &self.compilation_options,
                        mode,
                        cube_kernel.address_type(),
                    )
                    .map_err(LaunchError::CompilationError)?;

                let repr = compiled.repr.ok_or_else(|| {
                    LaunchError::CompilationError(
                        cubecl_runtime::compiler::CompilationError::Generic {
                            reason: "no IR representation".into(),
                            backtrace: BackTrace::capture(),
                        },
                    )
                })?;

                let sources =
                    match cubecl_cpp::tt_metal::compile::runtime_sources_from_repr(&repr, 1) {
                        Ok(sources) => sources,
                        Err(err) => {
                            eprintln!(
                                "[compile_cube_task] runtime_sources_from_repr failed: {err:?}"
                            );
                            return Err(LaunchError::CompilationError(err));
                        }
                    };

                entry.insert(CachedCubeTaskLaunch { repr, sources })
            }
        };

        let launch_sources = if cached.sources.supports_core_partitioning
            && !cached.sources.requires_tiled_io()
        {
            let tensor_output_bridge = tensor_output_bridge_plan(&cached.repr, info, &count);
            let generic_page_size = tensor_output_bridge
                .as_ref()
                .map(|plan| plan.padded_row_bytes as u32)
                .or_else(|| {
                    resources
                        .iter()
                        .find_map(|resource| resource.compile_args.get(1).copied())
                })
                .unwrap_or(cached.sources.tile_size_bytes);
            let full_input_staging_tiles = full_input_staging_tiles(resources, &cached.sources);
            // Vectorized tensor outputs need dispatch units sized in scalar elements so
            // the generic writer doesn't synthesize extra output pages that the logical
            // tensor never owned. Keep this metadata-backed override scoped to dynamic
            // tensor kernels for now; array-style generic kernels don't expose enough
            // shape information here to safely reinterpret their unit width.
            let dispatch_unit_size_override = if tensor_output_bridge.is_some() {
                None
            } else {
                cached
                    .repr
                    .body
                    .has_dynamic_meta
                    .then(|| {
                        effective_generic_dispatch_unit_size_bytes(
                            &cached.repr,
                            resources,
                            info,
                            &count,
                        )
                    })
                    .flatten()
            };
            match cubecl_cpp::tt_metal::compile::runtime_sources_from_repr_with_page_size_and_staging_and_unit_size(
                &cached.repr,
                1,
                generic_page_size,
                full_input_staging_tiles.is_some(),
                dispatch_unit_size_override,
            ) {
                Ok(mut sources) => {
                    if tensor_output_bridge.is_some() {
                        sources = sources.with_core_partitioning(false);
                    }
                    if let Some(staged_tiles) = full_input_staging_tiles {
                        sources
                            .with_reader_runtime_args(vec![staged_tiles, generic_page_size])
                            .with_full_input_staging_tiles(staged_tiles)
                    } else {
                        sources
                    }
                }
                Err(err) => {
                    eprintln!(
                        "[compile_cube_task] runtime_sources_from_repr_with_page_size_and_staging failed: {err:?}"
                    );
                    return Err(LaunchError::CompilationError(err));
                }
            }
        } else {
            cached.sources.clone()
        };

        match prepare_launch(&cached.repr, launch_sources, resources, info, count) {
            Ok(prepared) => {
                if stream_debug_enabled() && prepared.sources.full_input_staging_tiles > 0 {
                    let packed_info_words = bytemuck::cast_slice::<u64, u32>(&info.data).to_vec();
                    eprintln!(
                        "[tt-stream-debug] staged launch: reader_runtime_args={:?} writer_runtime_args_prefix={:?} logical_sizes={:?} allocation_sizes={:?} item_sizes={:?} packed_info_words={:?}",
                        prepared.sources.reader_runtime_args,
                        &prepared.sources.writer_runtime_args
                            [..prepared.sources.writer_runtime_args.len().min(12)],
                        prepared
                            .bindings
                            .iter()
                            .map(|binding| binding.logical_size_bytes)
                            .collect::<Vec<_>>(),
                        prepared
                            .bindings
                            .iter()
                            .map(|binding| binding.allocation_size_bytes)
                            .collect::<Vec<_>>(),
                        prepared
                            .bindings
                            .iter()
                            .map(|binding| binding.item_size_bytes)
                            .collect::<Vec<_>>(),
                        packed_info_words,
                    );
                }
                Ok(prepared)
            }
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

        let mesh = crate::runtime::get_mesh();
        let result = self.compile_kernel(
            &mesh,
            &prepared.sources,
            &prepared.input_addrs,
            &prepared.output_addrs,
            logger,
            true,
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
            let mut writer_runtime_args = Vec::with_capacity(
                4 + usize::from(sources.full_input_staging_tiles > 0) + packed_info_words.len(),
            );
            writer_runtime_args.push(logical_num_units);
            writer_runtime_args.extend_from_slice(&launched_cube_count);
            if sources.full_input_staging_tiles > 0 {
                writer_runtime_args.push(sources.full_input_staging_tiles);
            }
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
                let mut writer_runtime_args = Vec::with_capacity(
                    4 + usize::from(sources.full_input_staging_tiles > 0)
                        + compute_runtime_args.len(),
                );
                writer_runtime_args.push(logical_num_units);
                writer_runtime_args.extend_from_slice(&launched_cube_count);
                if sources.full_input_staging_tiles > 0 {
                    writer_runtime_args.push(sources.full_input_staging_tiles);
                }
                writer_runtime_args.extend(compute_runtime_args);
                sources = sources.with_writer_runtime_args(writer_runtime_args);
            }
        } else if !sources.requires_tiled_io() {
            let mut writer_runtime_args = vec![
                logical_num_units,
                launched_cube_count[0],
                launched_cube_count[1],
                launched_cube_count[2],
            ];
            if sources.full_input_staging_tiles > 0 {
                writer_runtime_args.push(sources.full_input_staging_tiles);
            }
            sources = sources.with_writer_runtime_args(writer_runtime_args);
        }
    }

    let tensor_output_bridge = if !sources.requires_tiled_io() {
        if let Some(plan) = tensor_output_bridge_plan(repr, info, &count) {
            sources = sources.with_writer_runtime_args(plan.writer_runtime_args.clone());
            Some(plan)
        } else {
            None
        }
    } else {
        None
    };

    Ok(PreparedLaunch {
        sources,
        input_addrs,
        output_addrs,
        bindings,
        tensor_output_bridge,
    })
}
