use cubecl_common::cache::CacheOption;
use cubecl_common::hash::StableHash;
use cubecl_core::compilation_cache::CompilationCache;
use cubecl_core::ir::DeviceProperties;
use cubecl_cpp::shared::CompilationOptions;
use cubecl_runtime::id::KernelId;
use cubecl_runtime::timestamp_profiler::TimestampProfiler;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

#[derive(Debug)]
pub(crate) struct TtContext {
    pub mesh: libtt_metal_cxx::MeshDevice,
    pub compiled_kernels: HashMap<KernelId, TtCompiledKernel>,
    pub timestamps: TimestampProfiler,
    pub compilation_options: CompilationOptions,
    pub properties: DeviceProperties,
    pub compilation_cache: Option<CompilationCache<StableHash, CompilationCacheEntry>>,
}

pub struct TtCompiledKernel {
    pub program: libtt_metal_cxx::Program,
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
    pub fn new(
        mesh: libtt_metal_cxx::MeshDevice,
        compilation_options: CompilationOptions,
        properties: DeviceProperties,
    ) -> Self {
        Self {
            mesh,
            compiled_kernels: HashMap::new(),
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
}
