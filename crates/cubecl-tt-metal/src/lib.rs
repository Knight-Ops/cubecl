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

    pub type TestRuntime = crate::runtime::TtRuntime;

    // NOTE: testgen!() macros are disabled for now — they require a working
    // IR pipeline which will be implemented in Phase 4+.
    // cubecl_std::testgen!();
    // cubecl_core::testgen_all!(f32: [f32]);

    fn hardware_tests_enabled() -> bool {
        env::var_os("TT_METAL_RUN_HARDWARE_TESTS").is_some()
    }

    // ── Standalone tilization test (no GPU needed) ────────────────────────

    #[test]
    fn tilize_untilize_round_trip() {
        const M: u32 = 64;
        const N: u32 = 64;
        const ELEM_SIZE: u32 = 2;
        let num_elements = (M * N) as usize;

        let mut input = vec![0u16; num_elements];
        for r in 0..M as usize {
            for c in 0..N as usize {
                input[r * N as usize + c] = 0x3E00u16 | ((r * N as usize + c) as u16 & 0xFF);
            }
        }

        let bytes = bytemuck::cast_slice(&input);
        let tilized =
            libtt_metal_cxx::tilize(bytes, M, N, ELEM_SIZE).expect("tilize should succeed");
        assert_eq!(
            tilized.len(),
            bytes.len(),
            "tilized size should match input for tile-aligned dims"
        );

        let untilized =
            libtt_metal_cxx::untilize(&tilized, M, N, ELEM_SIZE).expect("untilize should succeed");
        let output: &[u16] = bytemuck::cast_slice(&untilized);
        assert_eq!(
            input, output,
            "tilize/untilize round-trip should preserve data"
        );
    }

    // ── Raw buffer I/O test ───────────────────────────────────────────────

    #[test]
    fn buffer_write_read_round_trip() {
        if !hardware_tests_enabled() {
            return;
        }
        let mut mesh = MeshDevice::create_unit_mesh(0).expect("should open unit mesh");
        const BUF_SIZE: u64 = 4096;
        let buf = MeshBuffer::create_replicated(&mesh, BUF_SIZE, BUF_SIZE, 0)
            .expect("buffer should allocate");

        let mut input = vec![0u8; BUF_SIZE as usize];
        for i in 0..BUF_SIZE as usize {
            input[i] = (i % 251 + 1) as u8;
        }
        mesh.write_mesh_buffer(&buf, &input).expect("write");
        let mut output = vec![0u8; BUF_SIZE as usize];
        mesh.read_mesh_buffer(&buf, &mut output).expect("read");
        assert_eq!(input, output, "raw buffer write/read round-trip");
        assert!(mesh.close().expect("mesh should close"));
    }

    // ── Kernel compile + execute test (data round-trip NOT verified) ──────
    //
    // KNOWN ISSUE: Data round-trip through kernels produces uniform 0xBF80
    // (bf16 -1.0) regardless of input. This occurs because noc_async_read_tile
    // (using TensorAccessor) addresses interleaved DRAM differently from
    // EnqueueWriteMeshBuffer. The readback of the input buffer via
    // read_mesh_buffer confirms data IS written correctly (0 mismatches).
    // The kernels compile and execute without error (JIT cache 8/8 hits).
    // Debugging the TensorAccessor address computation vs write_mesh_buffer
    // address mapping requires deeper TT-Metal DRAM interleaving investigation.
    // See TODO.md for details.

    #[test]
    fn kernel_compile_and_execute() {
        if !hardware_tests_enabled() {
            return;
        }
        let mut mesh = MeshDevice::create_unit_mesh(0).expect("should open unit mesh");

        const TILE_SIZE: u32 = 32 * 32 * 2;
        const NUM_TILES: u32 = 2;
        const BUF_SIZE: u64 = NUM_TILES as u64 * TILE_SIZE as u64;

        let input_buf = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("input buffer should allocate");
        let output_buf = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("output buffer should allocate");

        // Write test pattern and verify it's in DRAM
        let num_u16 = BUF_SIZE as usize / 2;
        let mut input = vec![0u16; num_u16];
        for i in 0..num_u16 {
            input[i] = 0x3E00u16 | (i as u16 & 0xFF);
        }
        mesh.write_mesh_buffer(&input_buf, bytemuck::cast_slice(&input))
            .expect("input write");
        let mut verify = vec![0u8; BUF_SIZE as usize];
        mesh.read_mesh_buffer(&input_buf, &mut verify)
            .expect("verify read");
        let verify_u16: &[u16] = bytemuck::cast_slice(&verify);
        assert_eq!(
            input.as_slice(),
            verify_u16,
            "input data written correctly to DRAM"
        );

        // Build and launch kernel
        let sources = TtKernelSources::copy_kernel(NUM_TILES, TILE_SIZE);
        let core = LogicalCore::new(0, 0);
        let core_range = CoreRangeSet::from_core(core);
        let mut program = Program::new();
        let cb_tiles = 2u32;
        let cb_size = cb_tiles * TILE_SIZE;

        let mut cb_in_config = CircularBufferConfig::new(cb_size);
        cb_in_config
            .index(0)
            .set_data_format(DataFormat::Float16B)
            .set_page_size(TILE_SIZE);
        program
            .create_circular_buffer(&core_range, &cb_in_config)
            .expect("input CB");
        let mut cb_out_config = CircularBufferConfig::new(cb_size);
        cb_out_config
            .index(16)
            .set_data_format(DataFormat::Float16B)
            .set_page_size(TILE_SIZE);
        program
            .create_circular_buffer(&core_range, &cb_out_config)
            .expect("output CB");

        let mut reader_config = DataMovementKernelConfig::reader().expect("reader config");
        reader_config
            .set_processor(DataMovementProcessor::Riscv1)
            .set_opt_level(KernelBuildOptLevel::O3);
        reader_config.add_compile_arg(2);
        reader_config.add_compile_arg(TILE_SIZE);
        let reader_id = program
            .create_data_movement_kernel_from_string_with_config(
                &sources.reader_source,
                core,
                &reader_config,
            )
            .expect("reader");
        let mut writer_config = DataMovementKernelConfig::writer().expect("writer config");
        writer_config
            .set_processor(DataMovementProcessor::Riscv0)
            .set_opt_level(KernelBuildOptLevel::O3);
        writer_config.add_compile_arg(2);
        writer_config.add_compile_arg(TILE_SIZE);
        let writer_id = program
            .create_data_movement_kernel_from_string_with_config(
                &sources.writer_source,
                core,
                &writer_config,
            )
            .expect("writer");
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
            .expect("compute");

        program
            .set_runtime_args(reader_id, core, &[input_buf.address(), NUM_TILES])
            .expect("reader args");
        program
            .set_runtime_args(writer_id, core, &[output_buf.address(), NUM_TILES])
            .expect("writer args");
        program
            .set_runtime_args(compute_id, core, &[NUM_TILES])
            .expect("compute args");

        let mut workload = MeshWorkload::new();
        workload
            .add_program_to_full_mesh(&mesh, program)
            .expect("workload");
        mesh.enqueue_workload(&mut workload, true).expect("enqueue");

        assert!(mesh.close().expect("mesh should close"));
    }
}
