/// Generate the C++ source for a TT-Metal reader (dataflow) kernel.
///
/// Uses `TensorAccessorArgs` with `noc_async_read_tile` for correct
/// interleaved DRAM tile addressing. Each input buffer needs 2 compile-time
/// args: [ArgConfig flags, AlignedPageSize].
pub fn generate_reader_source(num_inputs: u32) -> String {
    assert!(num_inputs >= 1, "at least one input buffer required");

    let mut src = String::new();
    src.push_str("#include <cstdint>\n\n");
    src.push_str("void kernel_main() {\n");

    // Runtime args: src_addr_0, ..., src_addr_N, num_tiles
    for i in 0..num_inputs {
        src.push_str(&format!(
            "    uint32_t src{}_addr = get_arg_val<uint32_t>({});\n",
            i, i
        ));
    }
    let num_tiles_idx = num_inputs;
    src.push_str(&format!(
        "    uint32_t num_tiles = get_arg_val<uint32_t>({});\n\n",
        num_tiles_idx
    ));

    // Circular buffer indices
    for i in 0..num_inputs {
        src.push_str(&format!("    constexpr uint32_t cb_in{} = {};\n", i, i));
    }

    // Declare TensorAccessor for each input buffer
    // Each uses 2 compile-time args: [ArgConfig, AlignedPageSize]
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

    // Tile reading loop
    src.push_str("\n    for (uint32_t i = 0; i < num_tiles; i++) {\n");
    for idx in 0..num_inputs {
        src.push_str(&format!("        cb_reserve_back(cb_in{idx}, 1);\n"));
    }
    src.push('\n');
    for idx in 0..num_inputs {
        src.push_str(&format!(
            "        uint32_t l1_addr_in{idx} = get_write_ptr(cb_in{idx});\n"
        ));
        src.push_str(&format!(
            "        noc_async_read_tile(i, a{idx}, l1_addr_in{idx});\n"
        ));
    }
    src.push_str("        noc_async_read_barrier();\n\n");
    for idx in 0..num_inputs {
        src.push_str(&format!("        cb_push_back(cb_in{idx}, 1);\n"));
    }
    src.push_str("    }\n");
    src.push_str("}\n");

    src
}
