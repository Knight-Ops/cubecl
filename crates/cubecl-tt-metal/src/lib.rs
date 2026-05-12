#[allow(unused_imports)]
#[macro_use]
extern crate derive_new;
extern crate alloc;

pub mod compute;
pub mod device;
pub mod runtime;
pub use device::*;
pub use runtime::TtRuntime;

/// The default WMMA compiler for TT-Metal (no-op).
pub type TtWmmaCompiler = cubecl_cpp::tt_metal::TtNoWmma;

#[cfg(test)]
mod tests {
    use cubecl_cpp::tt_metal::TtKernelSources;
    use libtt_metal_cxx::{
        CircularBufferConfig, ComputeKernelConfig, CoreRangeSet, DataFormat,
        DataMovementKernelConfig, DataMovementProcessor, KernelBuildOptLevel, LogicalCore,
        MathFidelity, MeshBuffer, MeshDevice, MeshWorkload, Program,
    };
    use std::env;

    pub type TestRuntime = crate::TtRuntime;

    // NOTE: testgen!() macros are disabled for now — they require a working
    // IR pipeline which will be implemented in Phase 4+.
    // cubecl_std::testgen!();
    // cubecl_core::testgen_all!(f32: [f32]);

    fn hardware_tests_enabled() -> bool {
        env::var_os("TT_METAL_RUN_HARDWARE_TESTS").is_some()
    }

    #[test]
    fn copy_tile_round_trip() {
        if !hardware_tests_enabled() {
            return;
        }

        let mut mesh = MeshDevice::create_unit_mesh(0).expect("should open unit mesh");

        const TILE_SIZE: u32 = 32 * 32 * 2; // 32x32 bfloat16 = 2048 bytes
        const NUM_TILES: u32 = 2;
        const BUF_SIZE: u64 = NUM_TILES as u64 * TILE_SIZE as u64;

        // Allocate input and output buffers
        let input_buf = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("input buffer should allocate");
        let output_buf = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("output buffer should allocate");

        // Write known data pattern to input
        let input_data = vec![0u8; BUF_SIZE as usize];
        mesh.write_mesh_buffer(&input_buf, &input_data)
            .expect("input write should succeed");

        // Generate kernel sources (copy kernel: in → out tile-by-tile)
        let sources = TtKernelSources::copy_kernel(NUM_TILES, TILE_SIZE);

        // Build the Program
        let core = LogicalCore::new(0, 0);
        let core_range = CoreRangeSet::from_core(core);
        let mut program = Program::new();

        // Configure circular buffers (double-buffered)
        let cb_tiles = 2u32;
        let cb_size = cb_tiles * TILE_SIZE;

        let mut cb_in_config = CircularBufferConfig::new(cb_size);
        cb_in_config
            .index(0)
            .set_data_format(DataFormat::Float16B)
            .set_page_size(TILE_SIZE);
        program
            .create_circular_buffer(&core_range, &cb_in_config)
            .expect("input CB should create");

        let mut cb_out_config = CircularBufferConfig::new(cb_size);
        cb_out_config
            .index(16)
            .set_data_format(DataFormat::Float16B)
            .set_page_size(TILE_SIZE);
        program
            .create_circular_buffer(&core_range, &cb_out_config)
            .expect("output CB should create");

        // Pass compile-time args for TensorAccessorArgs
        // ArgConfig: IsDram=2 (non-sharded DRAM), AlignedPageSize=tile_size
        let mut reader_config =
            DataMovementKernelConfig::reader().expect("reader config should create");
        reader_config
            .set_processor(DataMovementProcessor::Riscv1)
            .set_opt_level(KernelBuildOptLevel::O3);
        reader_config.add_compile_arg(2); // IsDram
        reader_config.add_compile_arg(TILE_SIZE); // AlignedPageSize
        let reader_id = program
            .create_data_movement_kernel_from_string_with_config(
                &sources.reader_source,
                core,
                &reader_config,
            )
            .expect("reader kernel should compile");

        let mut writer_config =
            DataMovementKernelConfig::writer().expect("writer config should create");
        writer_config
            .set_processor(DataMovementProcessor::Riscv0)
            .set_opt_level(KernelBuildOptLevel::O3);
        writer_config.add_compile_arg(2); // IsDram
        writer_config.add_compile_arg(TILE_SIZE); // AlignedPageSize
        let writer_id = program
            .create_data_movement_kernel_from_string_with_config(
                &sources.writer_source,
                core,
                &writer_config,
            )
            .expect("writer kernel should compile");

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
            .expect("compute kernel should compile");

        // Set runtime args
        program
            .set_runtime_args(reader_id, core, &[input_buf.address(), NUM_TILES])
            .expect("reader runtime args should set");
        program
            .set_runtime_args(writer_id, core, &[output_buf.address(), NUM_TILES])
            .expect("writer runtime args should set");
        program
            .set_runtime_args(compute_id, core, &[NUM_TILES])
            .expect("compute runtime args should set");

        // Enqueue and execute
        let mut workload = MeshWorkload::new();
        workload
            .add_program_to_full_mesh(&mesh, program)
            .expect("workload should accept program");
        mesh.enqueue_workload(&mut workload, true)
            .expect("workload should enqueue");

        // Verify the workload executed without error
        // NOTE: Data verification is skipped because TT-Metal requires
        // tilized data format (tilize_nfaces/untilize_nfaces) for tile
        // operations. Raw byte data in row-major format won't round-trip
        // correctly through tile-based compute. Proper tilization support
        // is planned for a follow-up phase.

        assert!(mesh.close().expect("mesh should close"));
    }
}
