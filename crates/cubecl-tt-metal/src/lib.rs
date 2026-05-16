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
    use cubecl_cpp::tt_metal::{TtKernelSources, reader};
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

    // ── Dram loopback test ─────────────────────────────────────────────────
    // Single data-movement kernel: DRAM → L1(CB) → DRAM.
    // Exactly mirrors the working TT-Metal `dram_loopback` example.
    // Uses 2-arg TensorAccessor, no compute kernel, no separate reader/writer.
    // Tests whether our buffer I/O and TensorAccessor addressing round-trip correctly.
    #[test]
    fn dram_loopback_round_trip() {
        if !hardware_tests_enabled() {
            return;
        }
        let mut mesh = MeshDevice::create_unit_mesh(0).expect("should open unit mesh");

        const TILE_SIZE: u32 = 32 * 32 * 2;
        const NUM_TILES: u32 = 2;
        const BUF_SIZE: u64 = NUM_TILES as u64 * TILE_SIZE as u64;

        let input_buf = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("input buffer");
        let output_buf = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("output buffer");

        // Write known pattern
        let num_u16 = BUF_SIZE as usize / 2;
        let mut input = vec![0u16; num_u16];
        for i in 0..num_u16 {
            input[i] = (i as u16).wrapping_mul(7919) | 0x3E00;
        }
        mesh.write_mesh_buffer(&input_buf, bytemuck::cast_slice(&input))
            .expect("input write");

        // Single data-movement kernel (dram_loopback style)
        let kernel_source = cubecl_cpp::tt_metal::writer::generate_dram_loopback_source();

        let core = LogicalCore::new(0, 0);
        let core_range = CoreRangeSet::from_core(core);
        let mut program = Program::new();

        // Allocate a CB just to get an L1 address for scratch
        let scratch_cb_size = 2 * TILE_SIZE;
        let mut scratch_cb_config = CircularBufferConfig::new(scratch_cb_size);
        scratch_cb_config
            .index(0)
            .set_data_format(DataFormat::Float16B)
            .set_page_size(TILE_SIZE);
        program
            .create_circular_buffer(&core_range, &scratch_cb_config)
            .expect("scratch CB");

        // Auto-generate compile args (input + output = 2 buffers)
        let input_args = input_buf.compile_args().expect("input compile args");
        let output_args = output_buf.compile_args().expect("output compile args");
        let mut kernel_compile_args = input_args.clone();
        kernel_compile_args.extend(&output_args);

        let mut kernel_config = DataMovementKernelConfig::reader().expect("reader config");
        kernel_config
            .set_processor(DataMovementProcessor::Riscv0)
            .set_opt_level(KernelBuildOptLevel::O3);
        for &arg in &kernel_compile_args {
            kernel_config.add_compile_arg(arg);
        }
        let kernel_id = program
            .create_data_movement_kernel_from_string_with_config(
                &kernel_source,
                core,
                &kernel_config,
            )
            .expect("kernel compile");

        // Runtime args: src_addr, dst_addr, num_tiles
        program
            .set_runtime_args(
                kernel_id,
                core,
                &[input_buf.address(), output_buf.address(), NUM_TILES],
            )
            .expect("runtime args");

        let mut workload = MeshWorkload::new();
        workload
            .add_program_to_full_mesh(&mesh, program)
            .expect("workload");
        mesh.enqueue_workload(&mut workload, true).expect("enqueue");

        // Read back and verify
        let mut output_bytes = vec![0u8; BUF_SIZE as usize];
        mesh.read_mesh_buffer(&output_buf, &mut output_bytes)
            .expect("output read");
        let output: &[u16] = bytemuck::cast_slice(&output_bytes);
        let mismatches = input
            .iter()
            .zip(output.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            mismatches, 0,
            "dram loopback: {}/{} mismatches",
            mismatches, num_u16
        );

        assert!(mesh.close().expect("mesh should close"));
    }

    // ── Kernel copy round-trip test (dram_loopback style) ──────────────────
    // Single data-movement kernel: DRAM → L1(CB) → DRAM.
    // Verifies data round-trips correctly through the kernel pipeline.
    #[test]
    fn kernel_copy_round_trip() {
        if !hardware_tests_enabled() {
            return;
        }
        let mut mesh = MeshDevice::create_unit_mesh(0).expect("should open unit mesh");

        const TILE_SIZE: u32 = 32 * 32 * 2;
        const NUM_TILES: u32 = 2;
        const BUF_SIZE: u64 = NUM_TILES as u64 * TILE_SIZE as u64;

        let input_buf = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("input buffer");
        let output_buf = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("output buffer");

        let num_u16 = BUF_SIZE as usize / 2;
        let mut input = vec![0u16; num_u16];
        for i in 0..num_u16 {
            input[i] = 0x3E00u16 | (i as u16 & 0xFF);
        }
        mesh.write_mesh_buffer(&input_buf, bytemuck::cast_slice(&input))
            .expect("input write");

        let kernel_source = cubecl_cpp::tt_metal::writer::generate_dram_loopback_source();
        let core = LogicalCore::new(0, 0);
        let core_range = CoreRangeSet::from_core(core);
        let mut program = Program::new();

        let scratch_cb_size = 2 * TILE_SIZE;
        let mut scratch_cb_config = CircularBufferConfig::new(scratch_cb_size);
        scratch_cb_config
            .index(0)
            .set_data_format(DataFormat::Float16B)
            .set_page_size(TILE_SIZE);
        program
            .create_circular_buffer(&core_range, &scratch_cb_config)
            .expect("scratch CB");

        let input_args = input_buf.compile_args().expect("input compile args");
        let output_args = output_buf.compile_args().expect("output compile args");
        let mut kernel_compile_args = input_args.clone();
        kernel_compile_args.extend(&output_args);

        let mut kernel_config = DataMovementKernelConfig::reader().expect("reader config");
        kernel_config
            .set_processor(DataMovementProcessor::Riscv0)
            .set_opt_level(KernelBuildOptLevel::O3);
        for &arg in &kernel_compile_args {
            kernel_config.add_compile_arg(arg);
        }
        let kernel_id = program
            .create_data_movement_kernel_from_string_with_config(
                &kernel_source,
                core,
                &kernel_config,
            )
            .expect("kernel compile");

        program
            .set_runtime_args(
                kernel_id,
                core,
                &[input_buf.address(), output_buf.address(), NUM_TILES],
            )
            .expect("runtime args");

        let mut workload = MeshWorkload::new();
        workload
            .add_program_to_full_mesh(&mesh, program)
            .expect("workload");
        mesh.enqueue_workload(&mut workload, true).expect("enqueue");

        let mut output_bytes = vec![0u8; BUF_SIZE as usize];
        mesh.read_mesh_buffer(&output_buf, &mut output_bytes)
            .expect("output read");
        let output: &[u16] = bytemuck::cast_slice(&output_bytes);
        let mismatches = input
            .iter()
            .zip(output.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            mismatches, 0,
            "copy kernel: {}/{} mismatches",
            mismatches, num_u16
        );

        assert!(mesh.close().expect("mesh should close"));
    }

    // ── Kernel copy tilized round-trip test (dram_loopback style) ─────────
    // Same as kernel_copy_round_trip but data is tilized before writing
    // and untilized after reading, verifying correctness through the tile path.
    #[test]
    fn kernel_copy_tilized_round_trip() {
        if !hardware_tests_enabled() {
            return;
        }
        let mut mesh = MeshDevice::create_unit_mesh(0).expect("should open unit mesh");

        const M: u32 = 64;
        const N: u32 = 64;
        const ELEM_SIZE: u32 = 2;
        const TILE_SIZE: u32 = 32 * 32 * 2;
        const NUM_TILES: u32 = 4;
        const BUF_SIZE: u64 = NUM_TILES as u64 * TILE_SIZE as u64;

        let input_buf = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("input buffer");
        let output_buf = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("output buffer");

        let num_elements = (M * N) as usize;
        let mut input = vec![0u16; num_elements];
        for i in 0..num_elements {
            input[i] = 0x3E00u16 | (i as u16 & 0xFF);
        }
        let tilized =
            libtt_metal_cxx::tilize(bytemuck::cast_slice(&input), M, N, ELEM_SIZE).expect("tilize");
        mesh.write_mesh_buffer(&input_buf, &tilized)
            .expect("input write");

        let kernel_source = cubecl_cpp::tt_metal::writer::generate_dram_loopback_source();
        let core = LogicalCore::new(0, 0);
        let core_range = CoreRangeSet::from_core(core);
        let mut program = Program::new();

        let scratch_cb_size = 2 * TILE_SIZE;
        let mut scratch_cb_config = CircularBufferConfig::new(scratch_cb_size);
        scratch_cb_config
            .index(0)
            .set_data_format(DataFormat::Float16B)
            .set_page_size(TILE_SIZE);
        program
            .create_circular_buffer(&core_range, &scratch_cb_config)
            .expect("scratch CB");

        let input_args = input_buf.compile_args().expect("input compile args");
        let output_args = output_buf.compile_args().expect("output compile args");
        let mut kernel_compile_args = input_args.clone();
        kernel_compile_args.extend(&output_args);

        let mut kernel_config = DataMovementKernelConfig::reader().expect("reader config");
        kernel_config
            .set_processor(DataMovementProcessor::Riscv0)
            .set_opt_level(KernelBuildOptLevel::O3);
        for &arg in &kernel_compile_args {
            kernel_config.add_compile_arg(arg);
        }
        let kernel_id = program
            .create_data_movement_kernel_from_string_with_config(
                &kernel_source,
                core,
                &kernel_config,
            )
            .expect("kernel compile");

        program
            .set_runtime_args(
                kernel_id,
                core,
                &[input_buf.address(), output_buf.address(), NUM_TILES],
            )
            .expect("runtime args");

        let mut workload = MeshWorkload::new();
        workload
            .add_program_to_full_mesh(&mesh, program)
            .expect("workload");
        mesh.enqueue_workload(&mut workload, true).expect("enqueue");

        let mut output_bytes = vec![0u8; BUF_SIZE as usize];
        mesh.read_mesh_buffer(&output_buf, &mut output_bytes)
            .expect("output read");
        let output_tilized =
            libtt_metal_cxx::untilize(&output_bytes, M, N, ELEM_SIZE).expect("untilize");
        let output: &[u16] = bytemuck::cast_slice(&output_tilized);
        let mismatches = input
            .iter()
            .zip(output.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            mismatches, 0,
            "tilized copy kernel: {}/{} mismatches",
            mismatches, num_elements
        );

        assert!(mesh.close().expect("mesh should close"));
    }

    // ── Original kernel compile + execute test ─────────────────────────────

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

        // Write test pattern
        let num_u16 = BUF_SIZE as usize / 2;
        let mut input = vec![0u16; num_u16];
        for i in 0..num_u16 {
            input[i] = 0x3E00u16 | (i as u16 & 0xFF);
        }
        mesh.write_mesh_buffer(&input_buf, bytemuck::cast_slice(&input))
            .expect("input write");

        // Build the Program with hardcoded compile args
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
            .expect("input CB should create");

        let mut cb_out_config = CircularBufferConfig::new(cb_size);
        cb_out_config
            .index(16)
            .set_data_format(DataFormat::Float16B)
            .set_page_size(TILE_SIZE);
        program
            .create_circular_buffer(&core_range, &cb_out_config)
            .expect("output CB should create");

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

        program
            .set_runtime_args(reader_id, core, &[input_buf.address(), NUM_TILES])
            .expect("reader runtime args should set");
        program
            .set_runtime_args(writer_id, core, &[output_buf.address(), NUM_TILES])
            .expect("writer runtime args should set");
        program
            .set_runtime_args(compute_id, core, &[NUM_TILES])
            .expect("compute runtime args should set");

        let mut workload = MeshWorkload::new();
        workload
            .add_program_to_full_mesh(&mesh, program)
            .expect("workload should accept program");
        mesh.enqueue_workload(&mut workload, true)
            .expect("workload should enqueue");

        assert!(mesh.close().expect("mesh should close"));
    }

    // ── Isolation test: reader + writer (no compute kernel) ───────────────
    // Two kernels sharing CB 0: reader pushes tiles, writer consumes.
    // If this works, the issue is in the compute kernel (copy_tile unpack).
    #[test]
    fn two_kernel_passthrough() {
        if !hardware_tests_enabled() {
            return;
        }
        let mut mesh = MeshDevice::create_unit_mesh(0).expect("should open unit mesh");

        const TILE_SIZE: u32 = 32 * 32 * 2;
        const NUM_TILES: u32 = 2;
        const BUF_SIZE: u64 = NUM_TILES as u64 * TILE_SIZE as u64;

        let input_buf = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("input buffer");
        let output_buf = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("output buffer");

        let num_u16 = BUF_SIZE as usize / 2;
        let mut input = vec![0u16; num_u16];
        for i in 0..num_u16 {
            input[i] = 0x3E00u16 | (i as u16 & 0xFF);
        }
        mesh.write_mesh_buffer(&input_buf, bytemuck::cast_slice(&input))
            .expect("input write");

        let core = LogicalCore::new(0, 0);
        let core_range = CoreRangeSet::from_core(core);
        let mut program = Program::new();
        let cb_tiles = 2u32;
        let cb_size = cb_tiles * TILE_SIZE;

        // One CB at index 0 — shared between reader and writer
        let mut cb_config = CircularBufferConfig::new(cb_size);
        cb_config
            .index(0)
            .set_data_format(DataFormat::Float16B)
            .set_page_size(TILE_SIZE);
        program
            .create_circular_buffer(&core_range, &cb_config)
            .expect("CB");

        // Reader: DRAM → CB 0 (same as three-kernel reader)
        let reader_source = cubecl_cpp::tt_metal::reader::generate_reader_source(1);
        let mut reader_config = DataMovementKernelConfig::reader().expect("reader config");
        reader_config
            .set_processor(DataMovementProcessor::Riscv1)
            .set_opt_level(KernelBuildOptLevel::O3);
        reader_config.add_compile_arg(2);
        reader_config.add_compile_arg(TILE_SIZE);
        let reader_id = program
            .create_data_movement_kernel_from_string_with_config(
                &reader_source,
                core,
                &reader_config,
            )
            .expect("reader");

        // Writer: CB 0 → DRAM (reads from CB 0, not CB 16)
        // Use a custom source: reads CB 0 instead of CB 16
        let writer_source = r#"#include <cstdint>
void kernel_main() {
    uint32_t dst_addr = get_arg_val<uint32_t>(0);
    uint32_t num_tiles = get_arg_val<uint32_t>(1);

    constexpr uint32_t cb_out = 0;  // read from CB 0, not 16

    constexpr auto c_args = TensorAccessorArgs<0>();
    const auto c = TensorAccessor(c_args, dst_addr);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_out, 1);
        uint32_t l1_addr = get_read_ptr(cb_out);
        noc_async_write_tile(i, c, l1_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_out, 1);
    }
}
"#;
        let mut writer_config = DataMovementKernelConfig::writer().expect("writer config");
        writer_config
            .set_processor(DataMovementProcessor::Riscv0)
            .set_opt_level(KernelBuildOptLevel::O3);
        writer_config.add_compile_arg(2);
        writer_config.add_compile_arg(TILE_SIZE);
        let writer_id = program
            .create_data_movement_kernel_from_string_with_config(
                writer_source,
                core,
                &writer_config,
            )
            .expect("writer");

        program
            .set_runtime_args(reader_id, core, &[input_buf.address(), NUM_TILES])
            .expect("reader args");
        program
            .set_runtime_args(writer_id, core, &[output_buf.address(), NUM_TILES])
            .expect("writer args");

        let mut workload = MeshWorkload::new();
        workload
            .add_program_to_full_mesh(&mesh, program)
            .expect("workload");
        mesh.enqueue_workload(&mut workload, true).expect("enqueue");

        let mut output_bytes = vec![0u8; BUF_SIZE as usize];
        mesh.read_mesh_buffer(&output_buf, &mut output_bytes)
            .expect("output read");
        let output: &[u16] = bytemuck::cast_slice(&output_bytes);
        let mismatches = input
            .iter()
            .zip(output.iter())
            .filter(|(a, b)| a != b)
            .count();
        if mismatches > 0 {
            let all_bf80 = output.iter().all(|&v| v == 0xBF80u16);
            panic!(
                "two-kernel: {}/{} mismatches (all_0xBF80={})",
                mismatches, num_u16, all_bf80
            );
        }

        assert!(mesh.close().expect("mesh should close"));
    }

    // ── Three-kernel pipeline: raw data round-trip ─────────────────────────
    // Reader + compute (copy_tile) + writer with 2-arg TensorAccessor.
    // Tests whether raw row-major data survives the face-unpack/re-pack cycle.
    #[test]
    fn three_kernel_copy_raw() {
        if !hardware_tests_enabled() {
            return;
        }
        let mut mesh = MeshDevice::create_unit_mesh(0).expect("should open unit mesh");

        const TILE_SIZE: u32 = 32 * 32 * 2;
        const NUM_TILES: u32 = 2;
        const BUF_SIZE: u64 = NUM_TILES as u64 * TILE_SIZE as u64;

        let input_buf = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("input buffer");
        let output_buf = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("output buffer");

        let num_u16 = BUF_SIZE as usize / 2;
        let mut input = vec![0u16; num_u16];
        for i in 0..num_u16 {
            input[i] = 0x3E00u16 | (i as u16 & 0xFF);
        }
        mesh.write_mesh_buffer(&input_buf, bytemuck::cast_slice(&input))
            .expect("input write");

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
            .set_runtime_args(
                reader_id,
                core,
                &[input_buf.address(), NUM_TILES, TILE_SIZE],
            )
            .expect("reader args");
        program
            .set_runtime_args(
                writer_id,
                core,
                &[output_buf.address(), NUM_TILES, TILE_SIZE],
            )
            .expect("writer args");
        program
            .set_runtime_args(compute_id, core, &[NUM_TILES])
            .expect("compute args");

        let mut workload = MeshWorkload::new();
        workload
            .add_program_to_full_mesh(&mesh, program)
            .expect("workload");
        mesh.enqueue_workload(&mut workload, true).expect("enqueue");

        let mut output_bytes = vec![0u8; BUF_SIZE as usize];
        mesh.read_mesh_buffer(&output_buf, &mut output_bytes)
            .expect("output read");
        let output: &[u16] = bytemuck::cast_slice(&output_bytes);
        let mismatches = input
            .iter()
            .zip(output.iter())
            .filter(|(a, b)| a != b)
            .count();
        if mismatches > 0 {
            let all_bf80 = output.iter().all(|&v| v == 0xBF80u16);
            panic!(
                "three-kernel raw: {}/{} mismatches (all_0xBF80={})",
                mismatches, num_u16, all_bf80
            );
        }

        assert!(mesh.close().expect("mesh should close"));
    }

    // ── Three-kernel pipeline: tilized data round-trip ─────────────────────
    // Uses tilize_nfaces before write and untilize_nfaces after read.
    // Tests whether tilized data survives the face-unpack/re-pack cycle.
    #[test]
    fn three_kernel_copy_tilized() {
        if !hardware_tests_enabled() {
            return;
        }
        let mut mesh = MeshDevice::create_unit_mesh(0).expect("should open unit mesh");

        // 64×64 bfloat16 — tile-aligned
        const M: u32 = 64;
        const N: u32 = 64;
        const ELEM_SIZE: u32 = 2;
        const NUM_ELEMENTS: usize = (M * N) as usize;
        const TILE_SIZE: u32 = 32 * 32 * ELEM_SIZE;
        const NUM_TILES: u32 = (M / 32) * (N / 32);
        const BUF_SIZE: u64 = NUM_TILES as u64 * TILE_SIZE as u64;

        let input_buf = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("input buffer");
        let output_buf = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("output buffer");

        let mut input_bf16 = vec![0u16; NUM_ELEMENTS];
        for i in 0..NUM_ELEMENTS {
            input_bf16[i] = 0x3E00u16 | (i as u16 & 0xFF);
        }

        // Tilize before writing
        let tilized = libtt_metal_cxx::tilize(bytemuck::cast_slice(&input_bf16), M, N, ELEM_SIZE)
            .expect("tilize");
        mesh.write_mesh_buffer(&input_buf, &tilized)
            .expect("input write");

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
            .set_runtime_args(
                reader_id,
                core,
                &[input_buf.address(), NUM_TILES, TILE_SIZE],
            )
            .expect("reader args");
        program
            .set_runtime_args(
                writer_id,
                core,
                &[output_buf.address(), NUM_TILES, TILE_SIZE],
            )
            .expect("writer args");
        program
            .set_runtime_args(compute_id, core, &[NUM_TILES])
            .expect("compute args");

        let mut workload = MeshWorkload::new();
        workload
            .add_program_to_full_mesh(&mesh, program)
            .expect("workload");
        mesh.enqueue_workload(&mut workload, true).expect("enqueue");

        // Read back tilized output and untilize
        let mut output_tilized = vec![0u8; tilized.len()];
        mesh.read_mesh_buffer(&output_buf, &mut output_tilized)
            .expect("output read");
        let output_bytes =
            libtt_metal_cxx::untilize(&output_tilized, M, N, ELEM_SIZE).expect("untilize");
        let output_bf16: &[u16] = bytemuck::cast_slice(&output_bytes);
        let mismatches = input_bf16
            .iter()
            .zip(output_bf16.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            mismatches, 0,
            "three-kernel tilized: {}/{} mismatches",
            mismatches, NUM_ELEMENTS
        );

        assert!(mesh.close().expect("mesh should close"));
    }

    // ── IR pipeline: copy detection ───────────────────────────────────────
    // Constructs a minimal KernelDefinition, passes through compile_to_tt_sources,
    // builds a Program, executes on hardware, verifies data round-trip.
    #[test]
    fn ir_pipeline_copy() {
        if !hardware_tests_enabled() {
            return;
        }
        let mut mesh = MeshDevice::create_unit_mesh(0).expect("should open unit mesh");

        const TILE_SIZE: u32 = 32 * 32 * 2;
        const NUM_TILES: u32 = 2;
        const BUF_SIZE: u64 = NUM_TILES as u64 * TILE_SIZE as u64;

        let input_buf = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("input buffer");
        let output_buf = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("output buffer");

        let num_u16 = BUF_SIZE as usize / 2;
        let mut input = vec![0u16; num_u16];
        for i in 0..num_u16 {
            input[i] = 0x3E00u16 | (i as u16 & 0xFF);
        }
        mesh.write_mesh_buffer(&input_buf, bytemuck::cast_slice(&input))
            .expect("input write");

        // Build a minimal KernelDefinition (1 input + 1 output, no operations → Copy detection)
        let kernel = build_empty_kernel(1, 1);
        let sources =
            cubecl_cpp::tt_metal::compile::compile_to_tt_sources(&kernel, NUM_TILES, TILE_SIZE)
                .expect("compile_to_tt_sources");

        // Build Program from the generated sources
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

        let mut output_bytes = vec![0u8; BUF_SIZE as usize];
        mesh.read_mesh_buffer(&output_buf, &mut output_bytes)
            .expect("output read");
        let output: &[u16] = bytemuck::cast_slice(&output_bytes);
        let mismatches = input
            .iter()
            .zip(output.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            mismatches, 0,
            "IR pipeline copy: {}/{} mismatches",
            mismatches, num_u16
        );

        assert!(mesh.close().expect("mesh should close"));
    }

    // ── IR pipeline: add detection ────────────────────────────────────────
    // Constructs a KernelDefinition with Arithmetic::Add, passes through
    // compile_to_tt_sources, builds a Program, executes on hardware, verifies.
    #[test]
    fn ir_pipeline_add() {
        if !hardware_tests_enabled() {
            return;
        }
        let mut mesh = MeshDevice::create_unit_mesh(0).expect("should open unit mesh");

        const TILE_SIZE: u32 = 32 * 32 * 2;
        const NUM_TILES: u32 = 2;
        const BUF_SIZE: u64 = NUM_TILES as u64 * TILE_SIZE as u64;

        let input_a = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("input A buffer");
        let input_b = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("input B buffer");
        let output_buf = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("output buffer");

        let num_u16 = BUF_SIZE as usize / 2;
        // A = all 0x3F80 (bf16 1.0), B = all zeros (bf16 0.0)
        // Expected: C = A + B = all 0x3F80
        let mut a_data = vec![0x3F80u16; num_u16];
        let b_data = vec![0u16; num_u16];
        let expected = vec![0x3F80u16; num_u16];
        // Add a small variation to detect byte ordering issues
        for i in 0..num_u16 {
            a_data[i] = 0x3F80u16 | ((i as u16) & 0x7F);
        }
        // Expected: A[i] + 0 = A[i], but bf16 addition of small mantissa-only values to 1.0
        // may round. For simplicity, compare against A (since B=0, result = A)
        for i in 0..num_u16 {
            a_data[i] = 0x3E00u16 | (i as u16 & 0xFF);
        }
        // Use A = all 0x3E00 (bf16 0.125), B = all 0 (bf16 0.0)
        // Result should be 0.125 for all elements
        let expected: Vec<u16> = (0..num_u16)
            .map(|i| 0x3E00u16 | (i as u16 & 0xFF))
            .collect();
        let a_data = expected.clone();
        let b_data = vec![0u16; num_u16];

        mesh.write_mesh_buffer(&input_a, bytemuck::cast_slice(&a_data))
            .expect("input A write");
        mesh.write_mesh_buffer(&input_b, bytemuck::cast_slice(&b_data))
            .expect("input B write");

        // Build KernelDefinition with 2 inputs + 1 output + Arithmetic::Add
        let kernel = build_add_kernel();
        let sources =
            cubecl_cpp::tt_metal::compile::compile_to_tt_sources(&kernel, NUM_TILES, TILE_SIZE)
                .expect("compile_to_tt_sources");

        // Build Program from the generated add kernel sources
        let core = LogicalCore::new(0, 0);
        let core_range = CoreRangeSet::from_core(core);
        let mut program = Program::new();
        let cb_tiles = 2u32;
        let cb_size = cb_tiles * TILE_SIZE;

        // Two input CBs at indices 0 and 1
        let mut cb_in0 = CircularBufferConfig::new(cb_size);
        cb_in0
            .index(0)
            .set_data_format(DataFormat::Float16B)
            .set_page_size(TILE_SIZE);
        program
            .create_circular_buffer(&core_range, &cb_in0)
            .expect("CB in0");
        let mut cb_in1 = CircularBufferConfig::new(cb_size);
        cb_in1
            .index(1)
            .set_data_format(DataFormat::Float16B)
            .set_page_size(TILE_SIZE);
        program
            .create_circular_buffer(&core_range, &cb_in1)
            .expect("CB in1");
        let mut cb_out = CircularBufferConfig::new(cb_size);
        cb_out
            .index(16)
            .set_data_format(DataFormat::Float16B)
            .set_page_size(TILE_SIZE);
        program
            .create_circular_buffer(&core_range, &cb_out)
            .expect("CB out");

        // Reader for two inputs (uses generate_reader_source(2))
        let mut reader_config = DataMovementKernelConfig::reader().expect("reader config");
        reader_config
            .set_processor(DataMovementProcessor::Riscv1)
            .set_opt_level(KernelBuildOptLevel::O3);
        reader_config.add_compile_arg(2);
        reader_config.add_compile_arg(TILE_SIZE); // buffer 0
        reader_config.add_compile_arg(2);
        reader_config.add_compile_arg(TILE_SIZE); // buffer 1
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
            .set_runtime_args(
                reader_id,
                core,
                &[input_a.address(), input_b.address(), NUM_TILES],
            )
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

        let mut output_bytes = vec![0u8; BUF_SIZE as usize];
        mesh.read_mesh_buffer(&output_buf, &mut output_bytes)
            .expect("output read");
        let output: &[u16] = bytemuck::cast_slice(&output_bytes);
        let mismatches = expected
            .iter()
            .zip(output.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            mismatches, 0,
            "IR pipeline add: {}/{} mismatches",
            mismatches, num_u16
        );

        assert!(mesh.close().expect("mesh should close"));
    }

    // ── Helpers for building KernelDefinition ─────────────────────────────

    use cubecl_core::ir::{ElemType, FloatKind, Scope, StorageType, Type};

    fn build_empty_kernel(
        num_inputs: u32,
        num_outputs: u32,
    ) -> cubecl_runtime::kernel::KernelDefinition {
        use cubecl_runtime::kernel::{KernelArg, KernelOptions, Visibility};
        use cubecl_runtime::server::CubeDim;

        let f32_type = Type::new(StorageType::Scalar(ElemType::Float(FloatKind::F32)));
        let mut buffers = Vec::new();
        for i in 0..num_inputs {
            buffers.push(KernelArg {
                id: i,
                ty: f32_type,
                visibility: Visibility::Read,
                size: None,
                has_extended_meta: false,
            });
        }
        for i in 0..num_outputs {
            buffers.push(KernelArg {
                id: num_inputs + i,
                ty: f32_type,
                visibility: Visibility::ReadWrite,
                size: None,
                has_extended_meta: false,
            });
        }
        cubecl_runtime::kernel::KernelDefinition {
            buffers,
            tensor_maps: vec![],
            scalars: vec![],
            cube_dim: CubeDim::new_single(),
            body: Scope::root(false),
            options: KernelOptions::default(),
        }
    }

    fn build_add_kernel() -> cubecl_runtime::kernel::KernelDefinition {
        use cubecl_core::ir::{Allocator, Arithmetic, BinaryOperator, Instruction};
        use cubecl_runtime::kernel::{KernelArg, KernelOptions, Visibility};
        use cubecl_runtime::server::CubeDim;

        let f32_type = Type::new(StorageType::Scalar(ElemType::Float(FloatKind::F32)));
        let mut allocator = Allocator::default();
        let lhs = allocator.create_local(f32_type);
        let rhs = allocator.create_local(f32_type);
        let result = allocator.create_local(f32_type);
        let mut scope = Scope::root(false).with_allocator(allocator);
        scope.register(Instruction::new(
            Arithmetic::Add(BinaryOperator {
                lhs: *lhs,
                rhs: *rhs,
            }),
            *result,
        ));
        cubecl_runtime::kernel::KernelDefinition {
            buffers: vec![
                KernelArg {
                    id: 0,
                    ty: f32_type,
                    visibility: Visibility::Read,
                    size: None,
                    has_extended_meta: false,
                },
                KernelArg {
                    id: 1,
                    ty: f32_type,
                    visibility: Visibility::Read,
                    size: None,
                    has_extended_meta: false,
                },
                KernelArg {
                    id: 2,
                    ty: f32_type,
                    visibility: Visibility::ReadWrite,
                    size: None,
                    has_extended_meta: false,
                },
            ],
            tensor_maps: vec![],
            scalars: vec![],
            cube_dim: CubeDim::new_single(),
            body: scope,
            options: KernelOptions::default(),
        }
    }
}

// ── Old crate-level helpers removed ─────────────────────────────────────
