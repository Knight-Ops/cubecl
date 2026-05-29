/// Holds all three kernel source strings for a TT-Metal operation.
#[derive(Debug, Clone)]
pub struct TtKernelSources {
    pub reader_source: String,
    pub compute_source: String,
    pub writer_source: String,
    pub num_inputs: u32,
    pub num_outputs: u32,
    pub input_binding_indices: Vec<usize>,
    pub output_binding_indices: Vec<usize>,
    pub num_tiles: u32,
    pub tile_size_bytes: u32,
    pub data_format_tt: u8,
    pub unit_item_size_bytes: u32,
    pub io_layout: TtIoDataLayout,
    pub reader_compile_args: Vec<u32>,
    pub writer_compile_args: Vec<u32>,
    pub writer_runtime_args: Vec<u32>,
    pub compute_runtime_args: Vec<u32>,
    pub buffer_item_sizes: Vec<u32>,
    pub info_static_len: usize,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtIoDataLayout {
    Logical,
    Tiled,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtBinaryComputeOp {
    Add,
    Sub,
    Mul,
    Div,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtUnaryComputeOp {
    Abs,
    Sqrt,
    Rsqrt,
    Sin,
    Cos,
    Tan,
    Tanh,
    Exp,
    Log,
}

impl TtKernelSources {
    pub fn new(
        reader_source: String,
        compute_source: String,
        writer_source: String,
        num_inputs: u32,
        num_outputs: u32,
        num_tiles: u32,
        tile_size_bytes: u32,
        data_format_tt: u8,
        unit_item_size_bytes: u32,
    ) -> Self {
        Self {
            reader_source,
            compute_source,
            writer_source,
            num_inputs,
            num_outputs,
            input_binding_indices: Vec::new(),
            output_binding_indices: Vec::new(),
            num_tiles,
            tile_size_bytes,
            data_format_tt,
            unit_item_size_bytes,
            io_layout: TtIoDataLayout::Logical,
            reader_compile_args: Vec::new(),
            writer_compile_args: Vec::new(),
            writer_runtime_args: Vec::new(),
            compute_runtime_args: Vec::new(),
            buffer_item_sizes: Vec::new(),
            info_static_len: 0,
        }
    }

    pub fn with_compile_args(
        mut self,
        reader_compile_args: Vec<u32>,
        writer_compile_args: Vec<u32>,
    ) -> Self {
        self.reader_compile_args = reader_compile_args;
        self.writer_compile_args = writer_compile_args;
        self
    }

    pub fn with_writer_runtime_args(mut self, writer_runtime_args: Vec<u32>) -> Self {
        self.writer_runtime_args = writer_runtime_args;
        self
    }

    pub fn with_compute_runtime_args(mut self, compute_runtime_args: Vec<u32>) -> Self {
        self.compute_runtime_args = compute_runtime_args;
        self
    }

    pub fn with_binding_indices(
        mut self,
        input_binding_indices: Vec<usize>,
        output_binding_indices: Vec<usize>,
    ) -> Self {
        self.input_binding_indices = input_binding_indices;
        self.output_binding_indices = output_binding_indices;
        self
    }

    pub fn with_buffer_item_sizes(mut self, buffer_item_sizes: Vec<u32>) -> Self {
        self.buffer_item_sizes = buffer_item_sizes;
        self
    }

    pub fn with_info_static_len(mut self, info_static_len: usize) -> Self {
        self.info_static_len = info_static_len;
        self
    }

    pub fn with_io_layout(mut self, io_layout: TtIoDataLayout) -> Self {
        self.io_layout = io_layout;
        self
    }

    pub fn with_num_tiles(mut self, num_tiles: u32) -> Self {
        self.num_tiles = num_tiles;
        self
    }

    pub fn with_writer_runtime_args_prefix(mut self, prefix: &[u32]) -> Self {
        if prefix.is_empty() {
            return self;
        }
        let mut runtime_args = Vec::with_capacity(prefix.len() + self.writer_runtime_args.len());
        runtime_args.extend_from_slice(prefix);
        runtime_args.extend(self.writer_runtime_args);
        self.writer_runtime_args = runtime_args;
        self
    }

    pub fn requires_tiled_io(&self) -> bool {
        matches!(self.io_layout, TtIoDataLayout::Tiled)
    }

    pub fn native_scalar_size_bytes(&self) -> u32 {
        // TT native tiles always represent 32x32 unpacked scalar elements.
        self.tile_size_bytes / (32 * 32)
    }

    /// Create sources for a simple copy kernel (1 input → 1 output).
    pub fn copy_kernel(num_tiles: u32, tile_size_bytes: u32) -> Self {
        Self::copy_kernel_with_format(num_tiles, tile_size_bytes, 5, tile_size_bytes / (32 * 32))
    }

    pub fn copy_kernel_with_format(
        num_tiles: u32,
        tile_size_bytes: u32,
        data_format_tt: u8,
        unit_item_size_bytes: u32,
    ) -> Self {
        Self::new(
            super::reader::generate_reader_source(1),
            super::writer::generate_copy_compute_source(),
            super::writer::generate_writer_source(1),
            1,
            1,
            num_tiles,
            tile_size_bytes,
            data_format_tt,
            unit_item_size_bytes,
        )
        .with_io_layout(TtIoDataLayout::Tiled)
    }

    pub fn binary_kernel_with_format(
        op: TtBinaryComputeOp,
        num_tiles: u32,
        tile_size_bytes: u32,
        data_format_tt: u8,
        unit_item_size_bytes: u32,
    ) -> Self {
        Self::new(
            super::reader::generate_reader_source(2),
            super::writer::generate_binary_compute_source(op),
            super::writer::generate_writer_source(1),
            2,
            1,
            num_tiles,
            tile_size_bytes,
            data_format_tt,
            unit_item_size_bytes,
        )
        .with_io_layout(TtIoDataLayout::Tiled)
    }

    /// Create sources for an element-wise addition kernel (2 inputs → 1 output).
    pub fn add_kernel(num_tiles: u32, tile_size_bytes: u32) -> Self {
        Self::add_kernel_with_format(num_tiles, tile_size_bytes, 5, tile_size_bytes / (32 * 32))
    }

    pub fn add_kernel_with_format(
        num_tiles: u32,
        tile_size_bytes: u32,
        data_format_tt: u8,
        unit_item_size_bytes: u32,
    ) -> Self {
        Self::binary_kernel_with_format(
            TtBinaryComputeOp::Add,
            num_tiles,
            tile_size_bytes,
            data_format_tt,
            unit_item_size_bytes,
        )
    }

    pub fn sub_kernel_with_format(
        num_tiles: u32,
        tile_size_bytes: u32,
        data_format_tt: u8,
        unit_item_size_bytes: u32,
    ) -> Self {
        Self::binary_kernel_with_format(
            TtBinaryComputeOp::Sub,
            num_tiles,
            tile_size_bytes,
            data_format_tt,
            unit_item_size_bytes,
        )
    }

    pub fn mul_kernel_with_format(
        num_tiles: u32,
        tile_size_bytes: u32,
        data_format_tt: u8,
        unit_item_size_bytes: u32,
    ) -> Self {
        Self::binary_kernel_with_format(
            TtBinaryComputeOp::Mul,
            num_tiles,
            tile_size_bytes,
            data_format_tt,
            unit_item_size_bytes,
        )
    }

    pub fn div_kernel_with_format(
        num_tiles: u32,
        tile_size_bytes: u32,
        data_format_tt: u8,
        unit_item_size_bytes: u32,
    ) -> Self {
        Self::binary_kernel_with_format(
            TtBinaryComputeOp::Div,
            num_tiles,
            tile_size_bytes,
            data_format_tt,
            unit_item_size_bytes,
        )
    }

    pub fn unary_kernel_with_format(
        op: TtUnaryComputeOp,
        num_tiles: u32,
        tile_size_bytes: u32,
        data_format_tt: u8,
        unit_item_size_bytes: u32,
    ) -> Self {
        Self::new(
            super::reader::generate_reader_source(1),
            super::writer::generate_unary_compute_source(op),
            super::writer::generate_writer_source(1),
            1,
            1,
            num_tiles,
            tile_size_bytes,
            data_format_tt,
            unit_item_size_bytes,
        )
        .with_io_layout(TtIoDataLayout::Tiled)
    }

    pub fn abs_kernel_with_format(
        num_tiles: u32,
        tile_size_bytes: u32,
        data_format_tt: u8,
        unit_item_size_bytes: u32,
    ) -> Self {
        Self::unary_kernel_with_format(
            TtUnaryComputeOp::Abs,
            num_tiles,
            tile_size_bytes,
            data_format_tt,
            unit_item_size_bytes,
        )
    }

    pub fn sqrt_kernel_with_format(
        num_tiles: u32,
        tile_size_bytes: u32,
        data_format_tt: u8,
        unit_item_size_bytes: u32,
    ) -> Self {
        Self::unary_kernel_with_format(
            TtUnaryComputeOp::Sqrt,
            num_tiles,
            tile_size_bytes,
            data_format_tt,
            unit_item_size_bytes,
        )
    }

    pub fn rsqrt_kernel_with_format(
        num_tiles: u32,
        tile_size_bytes: u32,
        data_format_tt: u8,
        unit_item_size_bytes: u32,
    ) -> Self {
        Self::unary_kernel_with_format(
            TtUnaryComputeOp::Rsqrt,
            num_tiles,
            tile_size_bytes,
            data_format_tt,
            unit_item_size_bytes,
        )
    }

    pub fn exp_kernel_with_format(
        num_tiles: u32,
        tile_size_bytes: u32,
        data_format_tt: u8,
        unit_item_size_bytes: u32,
    ) -> Self {
        Self::unary_kernel_with_format(
            TtUnaryComputeOp::Exp,
            num_tiles,
            tile_size_bytes,
            data_format_tt,
            unit_item_size_bytes,
        )
    }

    pub fn log_kernel_with_format(
        num_tiles: u32,
        tile_size_bytes: u32,
        data_format_tt: u8,
        unit_item_size_bytes: u32,
    ) -> Self {
        Self::unary_kernel_with_format(
            TtUnaryComputeOp::Log,
            num_tiles,
            tile_size_bytes,
            data_format_tt,
            unit_item_size_bytes,
        )
    }
}
