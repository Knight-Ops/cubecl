/// Holds all three kernel source strings for a TT-Metal operation.
#[derive(Debug, Clone)]
pub struct TtKernelSources {
    pub reader_source: String,
    pub compute_source: String,
    pub writer_source: String,
    pub num_inputs: u32,
    pub num_outputs: u32,
    pub num_tiles: u32,
    pub tile_size_bytes: u32,
    pub data_format_tt: u8,
    /// Compile-time args for the reader data movement kernel.
    /// Each input buffer needs 2 compile-time args: [ArgConfig flags, AlignedPageSize].
    pub reader_compile_args: Vec<u32>,
    /// Compile-time args for the writer data movement kernel.
    pub writer_compile_args: Vec<u32>,
}

impl TtKernelSources {
    /// Create sources for a simple copy kernel (1 input → 1 output).
    pub fn copy_kernel(num_tiles: u32, tile_size_bytes: u32) -> Self {
        Self {
            reader_source: super::reader::generate_reader_source(1),
            compute_source: super::writer::generate_copy_compute_source(),
            writer_source: super::writer::generate_writer_source(1),
            num_inputs: 1,
            num_outputs: 1,
            num_tiles,
            tile_size_bytes,
            data_format_tt: 5, // Float16_b
            // ArgConfig for non-sharded DRAM: IsDram = 2
            reader_compile_args: vec![2, tile_size_bytes],
            writer_compile_args: vec![2, tile_size_bytes],
        }
    }
}
