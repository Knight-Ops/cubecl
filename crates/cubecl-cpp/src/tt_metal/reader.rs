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

    // Runtime args: src_addr_0, ..., src_addr_N, num_tiles, start_tile
    for i in 0..num_inputs {
        src.push_str(&format!(
            "    uint32_t src{}_addr = get_arg_val<uint32_t>({});\n",
            i, i
        ));
    }
    let num_tiles_idx = num_inputs;
    let start_tile_idx = num_tiles_idx + 1;
    src.push_str(&format!(
        "    uint32_t num_tiles = get_arg_val<uint32_t>({});\n",
        num_tiles_idx
    ));
    src.push_str(&format!(
        "    uint32_t start_tile = get_arg_val<uint32_t>({});\n\n",
        start_tile_idx
    ));

    for i in 0..num_inputs {
        src.push_str(&format!("    constexpr uint32_t cb_in{} = {};\n", i, i));
    }

    src.push('\n');
    for i in 0..num_inputs {
        if i == 0 {
            src.push_str(&format!(
                "    constexpr auto a{}_args = TensorAccessorArgs<0>();\n",
                i
            ));
        } else {
            src.push_str(&format!(
                "    constexpr auto a{}_args = TensorAccessorArgs<a{}_args.next_compile_time_args_offset()>();\n",
                i, i - 1
            ));
        }
        src.push_str(&format!(
            "    const auto a{} = TensorAccessor(a{}_args, src{}_addr);\n",
            i, i, i
        ));
    }

    src.push_str(
        "
    for (uint32_t i = 0; i < num_tiles; i++) {
",
    );
    src.push_str(
        "        uint32_t tile_idx = start_tile + i;
",
    );
    for idx in 0..num_inputs {
        src.push_str(&format!("        cb_reserve_back(cb_in{idx}, 1);\n"));
    }
    src.push('\n');
    for idx in 0..num_inputs {
        src.push_str(&format!(
            "        uint32_t l1_addr_in{idx} = get_write_ptr(cb_in{idx});\n"
        ));
        src.push_str(&format!(
            "        noc_async_read_tile(tile_idx, a{idx}, l1_addr_in{idx});\n"
        ));
    }
    src.push_str(
        "        noc_async_read_barrier();

",
    );
    for idx in 0..num_inputs {
        src.push_str(&format!("        cb_push_back(cb_in{idx}, 1);\n"));
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
pub fn generate_scalar_reader_source(num_inputs: u32, full_input_staging: bool) -> String {
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
            "    uint32_t src{}_addr = get_arg_val<uint32_t>({});\n",
            i, i
        ));
    }
    let num_tiles_idx = num_inputs;
    let start_tile_idx = num_tiles_idx + 1;
    let staging_tiles_idx = start_tile_idx + 1;
    let staging_page_bytes_idx = staging_tiles_idx + 1;
    src.push_str(&format!(
        "    uint32_t num_tiles = get_arg_val<uint32_t>({});\n",
        num_tiles_idx
    ));
    src.push_str(&format!(
        "    uint32_t start_tile = get_arg_val<uint32_t>({});\n",
        start_tile_idx
    ));
    if full_input_staging {
        src.push_str(&format!(
            "    uint32_t staged_input_tiles = get_arg_val<uint32_t>({});\n",
            staging_tiles_idx
        ));
        src.push_str(&format!(
            "    uint32_t staged_input_page_bytes = get_arg_val<uint32_t>({});\n",
            staging_page_bytes_idx
        ));
    }
    src.push('\n');

    for i in 0..num_inputs {
        src.push_str(&format!("    constexpr uint32_t cb_in{} = {};\n", i, i));
    }

    src.push('\n');
    for i in 0..num_inputs {
        if i == 0 {
            src.push_str(&format!(
                "    constexpr auto a{}_args = TensorAccessorArgs<0>();\n",
                i
            ));
        } else {
            src.push_str(&format!(
                "    constexpr auto a{}_args = TensorAccessorArgs<a{}_args.next_compile_time_args_offset()>();\n",
                i, i - 1
            ));
        }
        src.push_str(&format!(
            "    const auto a{} = TensorAccessor(a{}_args, src{}_addr);\n",
            i, i, i
        ));
    }

    if full_input_staging {
        src.push_str(
            "
    cb_reserve_back(cb_in0, staged_input_tiles);
",
        );
        src.push_str(
            "    uint32_t l1_addr_in0_base = get_write_ptr(cb_in0);
",
        );
        src.push_str(
            "    for (uint32_t staged_tile = 0; staged_tile < staged_input_tiles; ++staged_tile) {
",
        );
        src.push_str(
            "        uint64_t src_noc_addr_in0 = a0.get_noc_addr(staged_tile);
",
        );
        src.push_str("        noc_async_read(src_noc_addr_in0, l1_addr_in0_base + staged_tile * staged_input_page_bytes, staged_input_page_bytes);
");
        src.push_str(
            "    }
",
        );
        src.push_str(
            "    noc_async_read_barrier();
",
        );
        src.push_str(
            "    cb_push_back(cb_in0, staged_input_tiles);
",
        );
    }

    if num_inputs > 1 || !full_input_staging {
        src.push_str(
            "
    for (uint32_t i = 0; i < num_tiles; i++) {
",
        );
        src.push_str(
            "        uint32_t tile_idx = start_tile + i;
",
        );
        for idx in 0..num_inputs {
            if full_input_staging && idx == 0 {
                continue;
            }
            src.push_str(&format!("        cb_reserve_back(cb_in{idx}, 1);\n"));
        }
        src.push('\n');
        for idx in 0..num_inputs {
            if full_input_staging && idx == 0 {
                continue;
            }
            src.push_str(&format!(
                "        uint32_t l1_addr_in{idx} = get_write_ptr(cb_in{idx});\n"
            ));
            src.push_str(&format!(
                "        uint32_t page_bytes_in{idx} = get_tile_size(cb_in{idx});\n"
            ));
            src.push_str(&format!(
                "        uint64_t src_noc_addr_in{idx} = a{idx}.get_noc_addr(tile_idx);\n"
            ));
            src.push_str(&format!(
                "        noc_async_read(src_noc_addr_in{idx}, l1_addr_in{idx}, page_bytes_in{idx});\n"
            ));
        }
        if full_input_staging && num_inputs == 1 {
            src.push_str(
                "        (void)tile_idx;
",
            );
        } else {
            src.push_str(
                "        noc_async_read_barrier();

",
            );
        }
        for idx in 0..num_inputs {
            if full_input_staging && idx == 0 {
                continue;
            }
            src.push_str(&format!("        cb_push_back(cb_in{idx}, 1);\n"));
        }
        src.push_str(
            "    }
",
        );
    }

    src.push_str(
        "}
",
    );
    src
}
