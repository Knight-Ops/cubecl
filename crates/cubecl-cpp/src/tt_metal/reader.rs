/// Generate the C++ source for a TT-Metal reader (dataflow) kernel.
///
/// Uses `TensorAccessorArgs` with `noc_async_read_tile` for correct
/// interleaved DRAM tile addressing. Each input buffer needs 2 compile-time
/// args: [`ArgConfig`] flags, [`AlignedPageSize`].
pub fn generate_reader_source(num_inputs: u32) -> String {
    let mut src = String::new();
    src.push_str(
        "#include <cstdint>

",
    );
    src.push_str(
        "void kernel_main() {
",
    );

    if num_inputs == 0 {
        src.push_str(
            "    (void)get_arg_val<uint32_t>(0);
",
        );
        src.push_str(
            "}
",
        );
        return src;
    }

    // Runtime args: src_addr_0, ..., src_addr_N, num_tiles
    for i in 0..num_inputs {
        src.push_str(&format!(
            "    uint32_t src{}_addr = get_arg_val<uint32_t>({});
",
            i, i
        ));
    }
    let num_tiles_idx = num_inputs;
    src.push_str(&format!(
        "    uint32_t num_tiles = get_arg_val<uint32_t>({});

",
        num_tiles_idx
    ));

    // Circular buffer indices
    for i in 0..num_inputs {
        src.push_str(&format!(
            "    constexpr uint32_t cb_in{} = {};
",
            i, i
        ));
    }

    // Declare TensorAccessor for each input buffer.
    // Use the 2-arg constructor matching the working dram_loopback pattern.
    src.push('\n');
    for i in 0..num_inputs {
        if i == 0 {
            src.push_str(&format!(
                "    constexpr auto a{}_args = TensorAccessorArgs<0>();
",
                i
            ));
        } else {
            src.push_str(&format!(
                "    constexpr auto a{}_args = TensorAccessorArgs<a{}_args.next_compile_time_args_offset()>();
",
                i, i - 1
            ));
        }
        src.push_str(&format!(
            "    const auto a{} = TensorAccessor(a{}_args, src{}_addr);
",
            i, i, i
        ));
    }

    // Tile reading loop
    src.push_str(
        "
    for (uint32_t i = 0; i < num_tiles; i++) {
",
    );
    for idx in 0..num_inputs {
        src.push_str(&format!(
            "        cb_reserve_back(cb_in{idx}, 1);
"
        ));
    }
    src.push('\n');
    for idx in 0..num_inputs {
        src.push_str(&format!(
            "        uint32_t l1_addr_in{idx} = get_write_ptr(cb_in{idx});
"
        ));
        src.push_str(&format!(
            "        noc_async_read_tile(i, a{idx}, l1_addr_in{idx});
"
        ));
    }
    src.push_str(
        "        noc_async_read_barrier();

",
    );
    for idx in 0..num_inputs {
        src.push_str(&format!(
            "        cb_push_back(cb_in{idx}, 1);
"
        ));
    }
    src.push_str(
        "    }
",
    );
    src.push_str(
        "}
",
    );

    src
}

/// Generate the C++ source for a TT-Metal reader (dataflow) kernel that preserves
/// the raw linear page layout used by CubeCL host buffers.
///
/// Unlike the tile reader fast path, this issues raw NOC reads for each page so the
/// generic scalar writer can treat the L1 payload as a plain contiguous array.
pub fn generate_scalar_reader_source(num_inputs: u32) -> String {
    let mut src = String::new();
    src.push_str(
        "#include <cstdint>

",
    );
    src.push_str(
        "void kernel_main() {
",
    );

    if num_inputs == 0 {
        src.push_str(
            "    (void)get_arg_val<uint32_t>(0);
",
        );
        src.push_str(
            "}
",
        );
        return src;
    }

    for i in 0..num_inputs {
        src.push_str(&format!(
            "    uint32_t src{}_addr = get_arg_val<uint32_t>({});
",
            i, i
        ));
    }
    let num_tiles_idx = num_inputs;
    src.push_str(&format!(
        "    uint32_t num_tiles = get_arg_val<uint32_t>({});

",
        num_tiles_idx
    ));

    for i in 0..num_inputs {
        src.push_str(&format!(
            "    constexpr uint32_t cb_in{} = {};
",
            i, i
        ));
    }

    src.push('\n');
    for i in 0..num_inputs {
        if i == 0 {
            src.push_str(&format!(
                "    constexpr auto a{}_args = TensorAccessorArgs<0>();
",
                i
            ));
        } else {
            src.push_str(&format!(
                "    constexpr auto a{}_args = TensorAccessorArgs<a{}_args.next_compile_time_args_offset()>();
",
                i, i - 1
            ));
        }
        src.push_str(&format!(
            "    const auto a{} = TensorAccessor(a{}_args, src{}_addr);
",
            i, i, i
        ));
    }

    src.push_str(
        "
    for (uint32_t i = 0; i < num_tiles; i++) {
",
    );
    for idx in 0..num_inputs {
        src.push_str(&format!(
            "        cb_reserve_back(cb_in{idx}, 1);
"
        ));
    }
    src.push('\n');
    for idx in 0..num_inputs {
        src.push_str(&format!(
            "        uint32_t l1_addr_in{idx} = get_write_ptr(cb_in{idx});
"
        ));
        src.push_str(&format!(
            "        uint32_t page_bytes_in{idx} = get_tile_size(cb_in{idx});
"
        ));
        src.push_str(&format!(
            "        uint64_t src_noc_addr_in{idx} = a{idx}.get_noc_addr(i);
"
        ));
        src.push_str(&format!(
            "        noc_async_read(src_noc_addr_in{idx}, l1_addr_in{idx}, page_bytes_in{idx});
"
        ));
    }
    src.push_str(
        "        noc_async_read_barrier();

",
    );
    for idx in 0..num_inputs {
        src.push_str(&format!(
            "        cb_push_back(cb_in{idx}, 1);
"
        ));
    }
    src.push_str(
        "    }
",
    );
    src.push_str(
        "}
",
    );

    src
}
