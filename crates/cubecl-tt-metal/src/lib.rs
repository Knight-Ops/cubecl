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
    use crate::compute::context::TtContext;
    use crate::compute::context::{data_format_to_tt, prepare_launch};
    use crate::compute::stream::TtStreamBackend;
    use crate::runtime::{TtCompiler, get_mesh, tt_memory_properties};
    use cubecl::prelude::*;
    use cubecl_core as cubecl;
    use cubecl_core::Runtime;
    use cubecl_cpp::tt_metal::TtKernelSources;
    use cubecl_runtime::compiler::{CompilationError, Compiler, CubeTask};
    use cubecl_runtime::id::KernelId;
    use cubecl_runtime::kernel::{CompiledKernel, KernelMetadata};
    use cubecl_runtime::server::{CubeCount, ExecutionMode, KernelArguments};
    use cubecl_runtime::storage::StorageId;
    use cubecl_runtime::stream::EventStreamBackend;
    use libtt_metal_cxx::{
        CircularBufferConfig, ComputeKernelConfig, CoreRangeSet, DataFormat,
        DataMovementKernelConfig, DataMovementProcessor, KernelBuildOptLevel, LogicalCore,
        MathFidelity, MeshBuffer, MeshDevice, MeshWorkload, Program,
    };
    use std::env;

    #[allow(dead_code)]
    pub type TestRuntime = crate::runtime::TtRuntime;

    // NOTE: The broad upstream generated test macros stay disabled for TT.
    // They are not gated on TT hardware availability, and several coarse-grained
    // categories still bundle unsupported single-core TT semantics.
    const TT_HARDWARE_SUBPROCESS_ENV: &str = "CUBECL_TT_METAL_TEST_SUBPROCESS";

    unsafe extern "C" {
        fn _exit(status: i32) -> !;
    }

    fn with_tt_hardware_test(test: impl FnOnce()) {
        if !hardware_tests_enabled() {
            return;
        }

        if env::var_os(TT_HARDWARE_SUBPROCESS_ENV).is_some() {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(test));
            let status = if result.is_ok() { 0 } else { 101 };
            unsafe { _exit(status) }
        }

        let test_name = std::thread::current()
            .name()
            .expect("TT hardware test should have a thread name")
            .to_owned();
        let output = std::process::Command::new(
            std::env::current_exe().expect("test binary path should resolve"),
        )
        .env(TT_HARDWARE_SUBPROCESS_ENV, &test_name)
        .env("TT_METAL_RUN_HARDWARE_TESTS", "1")
        .arg("--exact")
        .arg(&test_name)
        .arg("--nocapture")
        .arg("--test-threads=1")
        .output()
        .expect("TT hardware subprocess should spawn");

        if !output.status.success() {
            panic!(
                "TT hardware subprocess failed for {test_name}
stdout:
{}
stderr:
{}",
                String::from_utf8_lossy(&output.stdout),
                String::from_utf8_lossy(&output.stderr),
            );
        }
    }

    fn with_tt_hardware_test_client(test: impl FnOnce(ComputeClient<TestRuntime>)) {
        with_tt_hardware_test(|| test(TestRuntime::client(&Default::default())));
    }

    // Shared TT test-surface matrix. Keep this summary aligned with `PHASES.md`
    // and `cargo test -p cubecl-tt-metal -- --list`.
    //
    // `cubecl_std`
    // - enabled: `event`, `reinterpret_slice`, `trigonometry`
    // - bring-up: `tensor_identity` (blocked on non-1D cube dimensions), `quantized_view`
    //   once quantized values, scales, and view metadata are characterized end-to-end on TT
    //   hardware
    //
    // `cubecl_core::runtime_tests`
    // - enabled: `assign`, `binary_untyped` (`mulhi`), `branch` (`select` only),
    //   `comparison`, `const_match`, `constants`, `debug` (helper-call subset),
    //   `different_rank`, `enums`, `file`, `index` (`test_assign_index` only), `launch`,
    //   `launch_untyped` (dynamic addressing only), `metadata`, `minifloat`
    //   (feature-gated conversion subset), `numeric`, `properties`, `saturating`
    //   (`i32`/`u32` subset), `to_client` (opportunistic when more than one device is visible),
    //   `unary_int` (integer abs/bit-operation subset)
    // - partially enabled: `slice` (all current cases except the range-loop `slice_for` test),
    //   `vector` (index, index-assign, conditional, and comparison subset)
    // - queued after the current baseline: `binary`, `topology`, `unary`
    //   direct TT-native kernels for `sub`, `mul`, and `sqrt` are now characterized on
    //   hardware, but ordinary CubeCL wrappers still need a safe row-major/logical-buffer
    //   bridge before they can use that path end-to-end
    // - control-flow bring-up after simpler kernels: remaining `branch` switch/loop cases,
    //   `sequence`, `slice_for`, `unroll`, and vector cases that depend on loop lowering or
    //   shared memory
    // - parity lane / capability-blocked: `all_reduce`, `atomic`, `barrier`, `cluster`,
    //   `cmma`, `index::test_kernel_shuffle`, `plane`, `stream`, `synchronization`,
    //   `tensor::test_tensor_coordinate`, `tensormap`
    // - helper-only inventory entry: `traits`
    mod cubecl_std_wrappers {
        use super::*;

        #[test]
        fn event_suite() {
            with_tt_hardware_test_client(|client| {
                cubecl_std::tests::event::event_test_1::<TestRuntime>(client.clone());
                cubecl_std::tests::event::event_test_2::<TestRuntime>(client.clone());
                cubecl_std::tests::event::event_test_3::<TestRuntime>(client);
            });
        }

        #[test]
        fn reinterpret_slice_suite() {
            with_tt_hardware_test_client(|client| {
                cubecl_std::tests::reinterpret_slice::run_test_read_global::<TestRuntime>(
                    client.clone(),
                    1,
                );
                cubecl_std::tests::reinterpret_slice::run_test_read_global::<TestRuntime>(
                    client.clone(),
                    2,
                );
                cubecl_std::tests::reinterpret_slice::run_test_read_global::<TestRuntime>(
                    client.clone(),
                    4,
                );
                cubecl_std::tests::reinterpret_slice::run_test_write_global::<TestRuntime>(
                    client.clone(),
                    1,
                );
                cubecl_std::tests::reinterpret_slice::run_test_write_global::<TestRuntime>(
                    client.clone(),
                    2,
                );
                cubecl_std::tests::reinterpret_slice::run_test_write_global::<TestRuntime>(
                    client,
                    4,
                );
            });
        }

        #[test]
        fn trigonometry_suite() {
            with_tt_hardware_test_client(|client| {
                cubecl_std::tests::trigonometry::test_to_degrees::<TestRuntime>(client.clone());
                cubecl_std::tests::trigonometry::test_to_radians::<TestRuntime>(client);
            });
        }

        // `tensor_identity` stays out of the enabled TT std baseline for now because
        // the upstream kernel launches with 2D cube dimensions and the current TT generic
        // path only models a single execution axis.
    }

    // Phase 4 uses TT-local wrappers around upstream core runtime tests instead
    // of `cubecl_core::testgen_all!()`. Keep the enabled subset explicit and
    // promote queued categories only after they are green in isolation.
    mod cubecl_core_wrappers {
        use super::*;

        mod launch {
            use super::*;

            #[test]
            fn test_kernel_with_generics() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::launch::test_kernel_with_generics::<
                        TestRuntime,
                        f32,
                    >(client);
                });
            }

            #[test]
            fn test_kernel_without_generics() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::launch::test_kernel_without_generics::<
                        TestRuntime,
                    >(client);
                });
            }

            #[test]
            fn test_kernel_with_comptime_tag() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::launch::test_kernel_with_comptime_tag::<
                        TestRuntime,
                    >(client);
                });
            }
        }

        mod launch_untyped {
            use super::*;

            #[test]
            fn test_dynamic_addressing_32() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::launch::test_kernel_dynamic_addressing::<
                        TestRuntime,
                    >(client, AddressType::U32);
                });
            }

            #[test]
            fn test_dynamic_addressing_64() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::launch::test_kernel_dynamic_addressing::<
                        TestRuntime,
                    >(client, AddressType::U64);
                });
            }
        }

        mod properties {
            use super::*;

            #[test]
            fn test_device_properties() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::properties::test_kernel_properties::<TestRuntime>(
                        client,
                    );
                });
            }
        }

        mod constants {
            use super::*;

            #[test]
            fn test_constant_array() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::constants::test_constant_array::<TestRuntime>(
                        client,
                    );
                });
            }
        }

        mod metadata {
            use super::*;

            #[test]
            fn test_shape() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::metadata::test_shape_dim_4::<TestRuntime>(
                        client.clone(),
                        AddressType::U32,
                    );
                    cubecl_core::runtime_tests::metadata::test_shape_dim_4::<TestRuntime>(
                        client.clone(),
                        AddressType::U64,
                    );
                    cubecl_core::runtime_tests::metadata::test_shape_different_ranks::<
                        TestRuntime,
                    >(client.clone(), AddressType::U32);
                    cubecl_core::runtime_tests::metadata::test_shape_different_ranks::<
                        TestRuntime,
                    >(client, AddressType::U64);
                });
            }

            #[test]
            fn test_stride() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::metadata::test_stride_different_ranks::<
                        TestRuntime,
                    >(client.clone(), AddressType::U32);
                    cubecl_core::runtime_tests::metadata::test_stride_different_ranks::<
                        TestRuntime,
                    >(client, AddressType::U64);
                });
            }

            #[test]
            fn test_len() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::metadata::test_len_different_ranks::<TestRuntime>(
                        client.clone(),
                        AddressType::U32,
                    );
                    cubecl_core::runtime_tests::metadata::test_len_different_ranks::<TestRuntime>(
                        client,
                        AddressType::U64,
                    );
                });
            }

            #[test]
            fn test_buffer_len_discontiguous() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::metadata::test_buffer_len_discontiguous::<
                        TestRuntime,
                    >(client.clone(), AddressType::U32);
                    cubecl_core::runtime_tests::metadata::test_buffer_len_discontiguous::<
                        TestRuntime,
                    >(client, AddressType::U64);
                });
            }

            #[test]
            fn test_buffer_len_vectorized() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::metadata::test_buffer_len_vectorized::<
                        TestRuntime,
                    >(client.clone(), AddressType::U32);
                    cubecl_core::runtime_tests::metadata::test_buffer_len_vectorized::<
                        TestRuntime,
                    >(client, AddressType::U64);
                });
            }

            #[test]
            fn test_buffer_len_offset() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::metadata::test_buffer_len_offset::<TestRuntime>(
                        client.clone(),
                        AddressType::U32,
                    );
                    cubecl_core::runtime_tests::metadata::test_buffer_len_offset::<TestRuntime>(
                        client,
                        AddressType::U64,
                    );
                });
            }
        }

        mod different_rank {
            use super::*;

            #[test]
            fn test_kernel_different_rank_first_biggest() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::different_rank::test_kernel_different_rank_first_biggest::<
                        TestRuntime,
                        f32,
                    >(client);
                });
            }

            #[test]
            fn test_kernel_different_rank_last_biggest() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::different_rank::test_kernel_different_rank_last_biggest::<
                        TestRuntime,
                        f32,
                    >(client);
                });
            }
        }

        mod assign {
            use super::*;

            #[test]
            fn test_assign_scalar() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::assign::test_kernel_assign_scalar::<
                        TestRuntime,
                        f32,
                    >(client);
                });
            }

            #[test]
            fn test_add_assign_array() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::assign::test_kernel_add_assign_array::<
                        TestRuntime,
                        f32,
                    >(client);
                });
            }

            #[test]
            fn test_add_assign_vector() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::assign::test_kernel_add_assign_vector::<
                        TestRuntime,
                        f32,
                    >(client);
                });
            }
        }

        mod comparison {
            use super::*;

            macro_rules! add_test {
                ($test_name:ident) => {
                    #[test]
                    fn $test_name() {
                        with_tt_hardware_test_client(|client| {
                            cubecl_core::runtime_tests::comparison::$test_name::<TestRuntime>(
                                client,
                            );
                        });
                    }
                };
            }

            add_test!(test_gt);
            add_test!(test_lt);
            add_test!(test_ge);
            add_test!(test_le);
            add_test!(test_eq);
            add_test!(test_ne);
        }

        mod branch {
            use super::*;

            #[test]
            fn test_select_true() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::branch::test_select::<TestRuntime, f32>(
                        client, true,
                    );
                });
            }

            #[test]
            fn test_select_false() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::branch::test_select::<TestRuntime, f32>(
                        client, false,
                    );
                });
            }
        }

        mod index {
            use super::*;

            #[test]
            fn test_assign_index() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::index::test_kernel_index_scalar::<
                        TestRuntime,
                        f32,
                    >(client);
                });
            }

            // `test_kernel_shuffle` stays off because it depends on local-array / shuffle
            // semantics that the TT single-core generic subset does not model yet.
        }

        mod numeric {
            use super::*;

            #[test]
            fn test_kernel_define() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::numeric::test_kernel_define::<TestRuntime>(client);
                });
            }

            #[test]
            fn test_kernel_define_many() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::numeric::test_kernel_define_many::<TestRuntime>(
                        client,
                    );
                });
            }
        }

        mod const_match {
            use super::*;

            #[test]
            fn test_const_match() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::const_match::test_kernel_const_match::<
                        TestRuntime,
                        f32,
                        u32,
                    >(client);
                });
            }
        }

        mod enums {
            use super::*;

            #[test]
            fn scalar_enum() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::enums::test_scalar_enum::<TestRuntime>(client);
                });
            }

            #[test]
            fn runtime_enum_empty() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::enums::test_runtime_variants_empty::<TestRuntime>(
                        client,
                    );
                });
            }

            #[test]
            fn runtime_enum_value() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::enums::test_runtime_variants_value::<TestRuntime>(
                        client,
                    );
                });
            }

            #[test]
            fn runtime_enum_empty_wildcard() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::enums::test_runtime_variants_empty_wildcard::<
                        TestRuntime,
                    >(client);
                });
            }

            #[test]
            fn array_float_int() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::enums::test_array_float_int::<TestRuntime, f32>(
                        &client, 10.0,
                    );
                    cubecl_core::runtime_tests::enums::test_array_float_int::<TestRuntime, i32>(
                        &client, 20,
                    );
                });
            }

            #[test]
            fn tuple_enum() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::enums::test_tuple_enum::<TestRuntime>(&client);
                });
            }
        }

        mod file {
            use super::*;

            #[test]
            fn test_kernel_load_file() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::file::test_file_memory::<TestRuntime>(client);
                });
            }
        }

        mod slice {
            use super::*;

            #[test]
            fn test_slice_select() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::slice::test_slice_select::<TestRuntime, f32>(
                        client,
                    );
                });
            }

            #[test]
            fn test_slice_len() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::slice::test_slice_len::<TestRuntime, f32>(client);
                });
            }

            #[test]
            fn test_slice_mut_assign() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::slice::test_slice_mut_assign::<TestRuntime, f32>(
                        client,
                    );
                });
            }

            #[test]
            fn test_slice_mut_len() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::slice::test_slice_mut_len::<TestRuntime>(client);
                });
            }

            // `test_slice_for` stays off until range-loop lowering is intentionally supported
            // by the TT single-core path.
        }

        mod saturating {
            use super::*;

            #[test]
            fn test_saturating_add_unsigned() {
                with_tt_hardware_test_client(|client| {
                    let test = cubecl_core::runtime_tests::saturating::test_saturating_add_unsigned::<
                        TestRuntime,
                        u32,
                    >;
                    test(client.clone(), 1);
                    test(client.clone(), 2);
                    test(client, 4);
                });
            }

            #[test]
            fn test_saturating_sub_unsigned() {
                with_tt_hardware_test_client(|client| {
                    let test = cubecl_core::runtime_tests::saturating::test_saturating_sub_unsigned::<
                        TestRuntime,
                        u32,
                    >;
                    test(client.clone(), 1);
                    test(client.clone(), 2);
                    test(client, 4);
                });
            }

            #[test]
            fn test_saturating_add_signed() {
                with_tt_hardware_test_client(|client| {
                    let test = cubecl_core::runtime_tests::saturating::test_saturating_add_signed::<
                        TestRuntime,
                        i32,
                    >;
                    test(client.clone(), 1);
                    test(client.clone(), 2);
                    test(client, 4);
                });
            }

            #[test]
            fn test_saturating_sub_signed() {
                with_tt_hardware_test_client(|client| {
                    let test = cubecl_core::runtime_tests::saturating::test_saturating_sub_signed::<
                        TestRuntime,
                        i32,
                    >;
                    test(client.clone(), 1);
                    test(client.clone(), 2);
                    test(client, 4);
                });
            }
        }

        mod minifloat {
            use super::*;

            #[test]
            fn test_fp8() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::minifloat::test_fp8::<TestRuntime, f32>(
                        client.clone(),
                        1,
                    );
                    cubecl_core::runtime_tests::minifloat::test_fp8::<TestRuntime, f32>(
                        client.clone(),
                        2,
                    );
                    cubecl_core::runtime_tests::minifloat::test_fp8::<TestRuntime, f32>(
                        client,
                        4,
                    );
                });
            }

            #[test]
            fn test_fp6() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::minifloat::test_fp6::<TestRuntime, f32>(
                        client.clone(),
                        1,
                    );
                    cubecl_core::runtime_tests::minifloat::test_fp6::<TestRuntime, f32>(
                        client.clone(),
                        2,
                    );
                    cubecl_core::runtime_tests::minifloat::test_fp6::<TestRuntime, f32>(
                        client,
                        4,
                    );
                });
            }

            #[test]
            fn test_fp4() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::minifloat::test_fp4::<TestRuntime, f32>(
                        client.clone(),
                        2,
                    );
                    cubecl_core::runtime_tests::minifloat::test_fp4::<TestRuntime, f32>(
                        client,
                        4,
                    );
                });
            }

            #[test]
            fn test_scale() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::minifloat::test_scale::<TestRuntime>(
                        client.clone(),
                        1,
                    );
                    cubecl_core::runtime_tests::minifloat::test_scale::<TestRuntime>(
                        client.clone(),
                        2,
                    );
                    cubecl_core::runtime_tests::minifloat::test_scale::<TestRuntime>(client, 4);
                });
            }
        }

        mod binary_untyped {
            use super::*;

            #[test]
            fn test_mulhi() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::binary::test_mulhi::<TestRuntime>(client);
                });
            }
        }

        mod vector {
            use super::*;

            #[test]
            fn test_vector_index() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::vector::test_vector_index::<TestRuntime, f32>(
                        client,
                    );
                });
            }

            #[test]
            fn test_vector_index_assign() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::vector::test_vector_index_assign::<
                        TestRuntime,
                        f32,
                    >(client);
                });
            }

            #[test]
            fn test_vector_conditional() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::vector::test_vector_conditional::<
                        TestRuntime,
                        f32,
                    >(client);
                });
            }

            macro_rules! add_cmp_test {
                ($test_name:ident) => {
                    #[test]
                    fn $test_name() {
                        with_tt_hardware_test_client(|client| {
                            cubecl_core::runtime_tests::vector::$test_name::<TestRuntime, f32>(
                                client,
                            );
                        });
                    }
                };
            }

            add_cmp_test!(test_vector_equal);
            add_cmp_test!(test_vector_not_equal);
            add_cmp_test!(test_vector_less_than);
            add_cmp_test!(test_vector_greater_than);
            add_cmp_test!(test_vector_less_equal);
            add_cmp_test!(test_vector_greater_equal);

            // `test_vector_loop_unroll` stays off until TT intentionally supports
            // unrolled loop lowering, and `test_shared_memory` stays off until the
            // shared-memory execution model is real on TT.
        }

        mod unary_int {
            use super::*;

            #[test]
            fn test_abs_int() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::unary::test_abs_int::<TestRuntime, i32>(client);
                });
            }

            #[test]
            fn test_vector_sum_int() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::unary::test_vector_sum_int::<TestRuntime, i32>(
                        client,
                    );
                });
            }

            #[test]
            fn test_count_ones() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::unary::test_count_ones::<TestRuntime, i32>(
                        client,
                    );
                });
            }

            #[test]
            fn test_reverse_bits() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::unary::test_reverse_bits::<TestRuntime, i32>(
                        client,
                    );
                });
            }

            #[test]
            fn test_leading_zeros() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::unary::test_leading_zeros::<TestRuntime, i32>(
                        client,
                    );
                });
            }

            #[test]
            fn test_trailing_zeros() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::unary::test_trailing_zeros::<TestRuntime, i32>(
                        client,
                    );
                });
            }

            #[test]
            fn test_find_first_set() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::unary::test_find_first_set::<TestRuntime, i32>(
                        client,
                    );
                });
            }
        }

        mod debug {
            use super::*;

            #[test]
            fn test_simple_call() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::debug::test_simple_call::<TestRuntime>(client);
                });
            }

            #[test]
            fn test_nested_call() {
                with_tt_hardware_test_client(|client| {
                    cubecl_core::runtime_tests::debug::test_nested_call::<TestRuntime>(client);
                });
            }

            // `test_debug_print` stays off until the TT debug-print/logging path is
            // characterized end-to-end and shown to fail cleanly when unsupported.
        }

        mod to_client {
            use super::*;

            #[test]
            fn test_to_client() {
                with_tt_hardware_test(|| {
                    cubecl_core::runtime_tests::to_client::test_to_client::<TestRuntime>();
                });
            }
        }

    }

    // cubecl_std::testgen!();
    // cubecl_core::testgen_all!(f32: [f32], i32: [i32], u32: [u32]);

    fn hardware_tests_enabled() -> bool {
        env::var_os("TT_METAL_RUN_HARDWARE_TESTS").is_some()
    }

    fn test_mesh() -> &'static MeshDevice {
        get_mesh()
    }

    #[derive(Clone)]
    struct FailingCubeTask;

    impl KernelMetadata for FailingCubeTask {
        fn id(&self) -> KernelId {
            KernelId::new::<Self>()
        }

        fn address_type(&self) -> StorageType {
            StorageType::Scalar(ElemType::UInt(UIntKind::U32))
        }
    }

    impl CubeTask<TtCompiler> for FailingCubeTask {
        fn compile(
            &self,
            _compiler: &mut TtCompiler,
            _compilation_options: &<TtCompiler as cubecl_runtime::compiler::Compiler>::CompilationOptions,
            _mode: ExecutionMode,
            _addr_type: StorageType,
        ) -> Result<CompiledKernel<TtCompiler>, CompilationError> {
            Err(CompilationError::Generic {
                reason: "intentional TT test compilation failure".into(),
                backtrace: cubecl_common::backtrace::BackTrace::capture(),
            })
        }
    }

    #[derive(CubeType)]
    struct MirrorEventUInt {
        #[cube(comptime)]
        value: u32,
    }

    #[derive(CubeType)]
    struct MirrorEventFloat {
        #[cube(comptime)]
        value: f32,
    }

    #[derive(CubeType, Clone)]
    struct MirrorEventListenerPosZero {
        items: SliceMut<f32>,
    }

    #[derive(CubeType, Clone)]
    struct MirrorEventListenerPosOne {
        items: SliceMut<f32>,
    }

    #[derive(CubeType, Clone)]
    struct MirrorEventCounter {
        #[cube(comptime)]
        value: u32,
    }

    #[derive(CubeType, Clone)]
    struct MirrorEventListenerPosTwo {
        items: SliceMut<f32>,
        times: ComptimeCell<MirrorEventCounter>,
    }

    #[cube]
    impl cubecl_std::event::EventListener for MirrorEventListenerPosZero {
        type Event = MirrorEventUInt;

        fn on_event(&mut self, event: Self::Event, bus: &mut cubecl_std::event::ComptimeEventBus) {
            if comptime!(event.value < 10) {
                bus.event::<MirrorEventUInt>(MirrorEventUInt {
                    value: comptime!(15u32 + event.value),
                });
            } else {
                self.items[0] = f32::cast_from(event.value);
            }
        }
    }

    #[cube]
    impl cubecl_std::event::EventListener for MirrorEventListenerPosOne {
        type Event = MirrorEventUInt;

        fn on_event(&mut self, event: Self::Event, _bus: &mut cubecl_std::event::ComptimeEventBus) {
            self.items[1] = (f32::cast_from(event.value) * 2.0) + self.items[1];
        }
    }

    #[cube]
    impl cubecl_std::event::EventListener for MirrorEventListenerPosTwo {
        type Event = MirrorEventFloat;

        fn on_event(&mut self, event: Self::Event, bus: &mut cubecl_std::event::ComptimeEventBus) {
            self.items[2] = event.value + self.items[2];

            let times = self.times.read();
            self.times.store(MirrorEventCounter {
                value: comptime!(times.value + 1),
            });

            if comptime!(times.value < 4) {
                bus.event::<MirrorEventFloat>(MirrorEventFloat {
                    value: comptime!(event.value * 2.0),
                });
                bus.event::<MirrorEventUInt>(MirrorEventUInt {
                    value: comptime!((event.value * 2.0) as u32),
                });
            }
        }
    }

    #[cube]
    fn mirror_event_test_1(items: SliceMut<f32>) {
        let mut bus = cubecl_std::event::ComptimeEventBus::new();
        let listener_zero = MirrorEventListenerPosZero { items };
        let listener_one = MirrorEventListenerPosOne { items };

        bus.listener::<MirrorEventListenerPosZero>(listener_zero);
        bus.listener::<MirrorEventListenerPosOne>(listener_one);
        bus.event::<MirrorEventUInt>(MirrorEventUInt { value: 5u32 });
    }

    #[cube]
    fn mirror_event_test_2(items: SliceMut<f32>) {
        let mut bus = cubecl_std::event::ComptimeEventBus::new();
        let listener_zero = MirrorEventListenerPosZero { items };
        let listener_one = MirrorEventListenerPosOne { items };

        bus.listener::<MirrorEventListenerPosZero>(listener_zero);
        bus.listener::<MirrorEventListenerPosOne>(listener_one);
        bus.event::<MirrorEventUInt>(MirrorEventUInt { value: 15u32 });
    }

    #[cube]
    fn mirror_event_test_3(items: SliceMut<f32>) {
        let mut bus = cubecl_std::event::ComptimeEventBus::new();
        let listener_zero = MirrorEventListenerPosZero { items };
        let listener_one = MirrorEventListenerPosOne { items };
        let listener_two = MirrorEventListenerPosTwo {
            items,
            times: ComptimeCell::new(MirrorEventCounter { value: 0u32 }),
        };

        bus.listener::<MirrorEventListenerPosZero>(listener_zero);
        bus.listener::<MirrorEventListenerPosOne>(listener_one);
        bus.listener::<MirrorEventListenerPosTwo>(listener_two);
        bus.event::<MirrorEventFloat>(MirrorEventFloat { value: 15.0f32 });
    }

    #[cube(launch_unchecked)]
    fn mirror_event_launch_1(output: &mut Array<f32>) {
        output[0] = 0.0;
        output[1] = 0.0;
        mirror_event_test_1(output.to_slice_mut());
    }

    #[cube(launch_unchecked)]
    fn mirror_event_launch_2(output: &mut Array<f32>) {
        output[0] = 0.0;
        output[1] = 0.0;
        mirror_event_test_2(output.to_slice_mut());
    }

    #[cube(launch_unchecked)]
    fn mirror_event_launch_3(output: &mut Array<f32>) {
        output[0] = 0.0;
        output[1] = 0.0;
        output[2] = 0.0;
        mirror_event_test_3(output.to_slice_mut());
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
        let mesh = test_mesh();
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
    }

    #[test]
    fn padded_allocation_io_preserves_logical_size() {
        if !hardware_tests_enabled() {
            return;
        }

        let client = TestRuntime::client(&Default::default());
        let input = (0u8..24).collect::<Vec<_>>();
        let handle = client.create(cubecl_common::bytes::Bytes::from_bytes_vec(input.clone()));
        let resource = client
            .get_resource(handle.clone())
            .expect("resource should resolve");

        assert_eq!(resource.resource().size, input.len() as u64);
        assert!(
            resource.resource().allocation_size
                >= crate::runtime::TT_DEFAULT_BUFFER_PAGE_SIZE_BYTES
        );

        let output = client
            .read_one(handle)
            .expect("logical read should succeed even with padded backing allocation");
        assert_eq!(output.len(), input.len());
        assert_eq!(&output[..], input.as_slice());
    }

    // ── Phase 5d: CubeTask compilation pipeline ─────────────────────────
    // Tests compile_cube_task by wrapping a KernelDefinition in a CubeTask.
    #[test]
    fn cubetask_compile_pipeline() {
        if !hardware_tests_enabled() {
            return;
        }
        let mesh = test_mesh();

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

        // Build a KernelDefinition and wrap it in a CubeTask-compatible struct
        let kernel_def = build_empty_kernel(1, 1);
        let mut server = crate::compute::server::TtServer::from_singleton();
        let stream_id = cubecl_common::stream_id::StreamId::current();

        // Compile through the CubeTask pipeline
        let sources =
            cubecl_cpp::tt_metal::compile::compile_to_tt_sources(&kernel_def, NUM_TILES, TILE_SIZE)
                .expect("compile_to_tt_sources");

        server
            .launch_from_sources(
                &sources,
                &[input_buf.address()],
                &[output_buf.address()],
                &[2, TILE_SIZE],
                &[2, TILE_SIZE],
                stream_id,
            )
            .expect("launch");

        let mut output_bytes = vec![0u8; BUF_SIZE as usize];
        server
            .mesh()
            .read_mesh_buffer(&output_buf, &mut output_bytes)
            .expect("output read");
        let output: &[u16] = bytemuck::cast_slice(&output_bytes);
        let mismatches = input
            .iter()
            .zip(output.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            mismatches, 0,
            "CubeTask pipeline: {}/{} mismatches",
            mismatches, num_u16
        );
    }

    // ── Kernel copy round-trip test (dram_loopback style) ──────────────────
    // Single data-movement kernel: DRAM → L1(CB) → DRAM.
    // Verifies data round-trips correctly through the kernel pipeline.
    #[test]
    fn kernel_copy_round_trip() {
        if !hardware_tests_enabled() {
            return;
        }
        let mesh = test_mesh();

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
    }

    // ── Kernel copy tilized round-trip test (dram_loopback style) ─────────
    // Same as kernel_copy_round_trip but data is tilized before writing
    // and untilized after reading, verifying correctness through the tile path.
    #[test]
    fn kernel_copy_tilized_round_trip() {
        if !hardware_tests_enabled() {
            return;
        }
        let mesh = test_mesh();

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
    }

    // ── Original kernel compile + execute test ─────────────────────────────

    #[test]
    fn kernel_compile_and_execute() {
        if !hardware_tests_enabled() {
            return;
        }
        let mesh = test_mesh();

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
    }

    // ── Isolation test: reader + writer (no compute kernel) ───────────────
    // Two kernels sharing CB 0: reader pushes tiles, writer consumes.
    // If this works, the issue is in the compute kernel (copy_tile unpack).
    #[test]
    fn two_kernel_passthrough() {
        if !hardware_tests_enabled() {
            return;
        }
        let mesh = test_mesh();

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
    }

    // ── Three-kernel pipeline: raw data round-trip ─────────────────────────
    // Reader + compute (copy_tile) + writer with 2-arg TensorAccessor.
    // Tests whether raw row-major data survives the face-unpack/re-pack cycle.
    #[test]
    fn three_kernel_copy_raw() {
        if !hardware_tests_enabled() {
            return;
        }
        let mesh = test_mesh();

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
    }

    // ── Three-kernel pipeline: tilized data round-trip ─────────────────────
    // Uses tilize_nfaces before write and untilize_nfaces after read.
    // Tests whether tilized data survives the face-unpack/re-pack cycle.
    #[test]
    fn three_kernel_copy_tilized() {
        if !hardware_tests_enabled() {
            return;
        }
        let mesh = test_mesh();

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
    }

    // ── IR pipeline: copy detection ───────────────────────────────────────
    // Constructs a minimal KernelDefinition, passes through compile_to_tt_sources,
    // builds a Program, executes on hardware, verifies data round-trip.
    #[test]
    fn ir_pipeline_copy() {
        if !hardware_tests_enabled() {
            return;
        }
        let mesh = test_mesh();

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
    }

    // ── IR pipeline: add detection ────────────────────────────────────────
    // Constructs a KernelDefinition with Arithmetic::Add, passes through
    // compile_to_tt_sources, builds a Program, executes on hardware, verifies.
    #[test]
    fn ir_pipeline_add() {
        if !hardware_tests_enabled() {
            return;
        }
        let mesh = test_mesh();

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
        let _b_data = vec![0u16; num_u16];
        let _expected = vec![0x3F80u16; num_u16];
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
    }

    #[test]
    fn ir_pipeline_sub() {
        if !hardware_tests_enabled() {
            return;
        }

        let lhs = tiled_bf16_pattern(&[0x3F80, 0x4000, 0x4040, 0x4080]);
        let rhs = tiled_bf16_pattern(&[0x3F00, 0x3F80, 0x4000, 0x4040]);
        let expected = tiled_bf16_pattern(&[0x3F00, 0x3F80, 0x3F80, 0x3F80]);
        run_native_binary_ir_pipeline_test("sub", build_sub_kernel(), &lhs, &rhs, &expected);
    }

    #[test]
    fn ir_pipeline_mul() {
        if !hardware_tests_enabled() {
            return;
        }

        let lhs = tiled_bf16_pattern(&[0x4000, 0x4040, 0x4080, 0x4100]);
        let rhs = tiled_bf16_pattern(&[0x3F00, 0x3F00, 0x3F80, 0x3F80]);
        let expected = tiled_bf16_pattern(&[0x3F80, 0x3FC0, 0x4080, 0x4100]);
        run_native_binary_ir_pipeline_test("mul", build_mul_kernel(), &lhs, &rhs, &expected);
    }

    #[test]
    fn ir_pipeline_sqrt() {
        if !hardware_tests_enabled() {
            return;
        }

        let input = tiled_bf16_pattern(&[0x3E80, 0x3F80, 0x4080, 0x4110]);
        let expected = tiled_bf16_pattern(&[0x3F00, 0x3F80, 0x4000, 0x4040]);
        run_native_unary_ir_pipeline_test("sqrt", build_sqrt_kernel(), &input, &expected);
    }

    fn tiled_bf16_pattern(pattern: &[u16]) -> Vec<u16> {
        assert!(!pattern.is_empty(), "pattern must not be empty");
        const TILE_SIZE: u32 = 32 * 32 * 2;
        const NUM_TILES: u32 = 2;
        let num_u16 = (NUM_TILES * TILE_SIZE / 2) as usize;
        (0..num_u16)
            .map(|i| pattern[i % pattern.len()])
            .collect()
    }

    fn run_native_binary_ir_pipeline_test(
        op_label: &str,
        kernel: cubecl_runtime::kernel::KernelDefinition,
        lhs: &[u16],
        rhs: &[u16],
        expected: &[u16],
    ) {
        let mesh = test_mesh();

        const TILE_SIZE: u32 = 32 * 32 * 2;
        const NUM_TILES: u32 = 2;
        const BUF_SIZE: u64 = NUM_TILES as u64 * TILE_SIZE as u64;

        assert_eq!(lhs.len(), expected.len(), "lhs/expected length mismatch");
        assert_eq!(rhs.len(), expected.len(), "rhs/expected length mismatch");

        let input_a = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("input A buffer");
        let input_b = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("input B buffer");
        let output_buf = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("output buffer");

        mesh.write_mesh_buffer(&input_a, bytemuck::cast_slice(lhs))
            .expect("input A write");
        mesh.write_mesh_buffer(&input_b, bytemuck::cast_slice(rhs))
            .expect("input B write");

        let sources = cubecl_cpp::tt_metal::compile::compile_to_tt_sources(&kernel, NUM_TILES, TILE_SIZE)
            .expect("compile_to_tt_sources");
        assert!(sources.compute_source.contains(&format!("{op_label}_tiles")));

        let core = LogicalCore::new(0, 0);
        let core_range = CoreRangeSet::from_core(core);
        let mut program = Program::new();
        let cb_tiles = 2u32;
        let cb_size = cb_tiles * TILE_SIZE;

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

        let mut reader_config = DataMovementKernelConfig::reader().expect("reader config");
        reader_config
            .set_processor(DataMovementProcessor::Riscv1)
            .set_opt_level(KernelBuildOptLevel::O3);
        reader_config.add_compile_arg(2);
        reader_config.add_compile_arg(TILE_SIZE);
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
            "IR pipeline {op_label}: {}/{} mismatches
first output: {:?}
first expected: {:?}",
            mismatches,
            expected.len(),
            &output[..output.len().min(16)],
            &expected[..expected.len().min(16)]
        );
    }

    fn run_native_unary_ir_pipeline_test(
        op_label: &str,
        kernel: cubecl_runtime::kernel::KernelDefinition,
        input: &[u16],
        expected: &[u16],
    ) {
        let mesh = test_mesh();

        const TILE_SIZE: u32 = 32 * 32 * 2;
        const NUM_TILES: u32 = 2;
        const BUF_SIZE: u64 = NUM_TILES as u64 * TILE_SIZE as u64;

        assert_eq!(input.len(), expected.len(), "input/expected length mismatch");

        let input_buf = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("input buffer");
        let output_buf = MeshBuffer::create_replicated(&mesh, BUF_SIZE, TILE_SIZE as u64, 0)
            .expect("output buffer");

        mesh.write_mesh_buffer(&input_buf, bytemuck::cast_slice(input))
            .expect("input write");

        let sources = cubecl_cpp::tt_metal::compile::compile_to_tt_sources(&kernel, NUM_TILES, TILE_SIZE)
            .expect("compile_to_tt_sources");
        assert!(sources.compute_source.contains(&format!("{op_label}_tile")));

        let core = LogicalCore::new(0, 0);
        let core_range = CoreRangeSet::from_core(core);
        let mut program = Program::new();
        let cb_tiles = 2u32;
        let cb_size = cb_tiles * TILE_SIZE;

        let mut cb_in0 = CircularBufferConfig::new(cb_size);
        cb_in0
            .index(0)
            .set_data_format(DataFormat::Float16B)
            .set_page_size(TILE_SIZE);
        program
            .create_circular_buffer(&core_range, &cb_in0)
            .expect("CB in0");
        let mut cb_out = CircularBufferConfig::new(cb_size);
        cb_out
            .index(16)
            .set_data_format(DataFormat::Float16B)
            .set_page_size(TILE_SIZE);
        program
            .create_circular_buffer(&core_range, &cb_out)
            .expect("CB out");

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
        let mismatches = expected
            .iter()
            .zip(output.iter())
            .filter(|(a, b)| a != b)
            .count();
        assert_eq!(
            mismatches, 0,
            "IR pipeline {op_label}: {}/{} mismatches
first output: {:?}
first expected: {:?}",
            mismatches,
            expected.len(),
            &output[..output.len().min(16)],
            &expected[..expected.len().min(16)]
        );
    }

    #[test]
    fn reader_writer_source_regressions() {
        let reader_zero = cubecl_cpp::tt_metal::reader::generate_reader_source(0);
        assert!(reader_zero.contains("get_arg_val<uint32_t>(0)"));

        let reader_two = cubecl_cpp::tt_metal::reader::generate_reader_source(2);
        assert!(
            reader_two.contains("TensorAccessorArgs<a0_args.next_compile_time_args_offset()>()")
        );

        let writer_zero = cubecl_cpp::tt_metal::writer::generate_writer_source(0);
        assert!(writer_zero.contains("get_arg_val<uint32_t>(0)"));

        let writer_two = cubecl_cpp::tt_metal::writer::generate_writer_source(2);
        assert!(
            writer_two.contains("TensorAccessorArgs<c0_args.next_compile_time_args_offset()>()")
        );
    }

    #[test]
    fn kernel_sources_bind_compile_args() {
        let sources = cubecl_cpp::tt_metal::TtKernelSources::copy_kernel(2, 2048)
            .with_compile_args(vec![2, 2048, 2, 2048], vec![2, 2048]);
        assert_eq!(sources.reader_compile_args, vec![2, 2048, 2, 2048]);
        assert_eq!(sources.writer_compile_args, vec![2, 2048]);
    }

    #[test]
    fn tt_data_format_mapping_regressions() {
        assert_eq!(data_format_to_tt(0).unwrap(), DataFormat::Float32);
        assert_eq!(data_format_to_tt(1).unwrap(), DataFormat::Float16);
        assert_eq!(data_format_to_tt(5).unwrap(), DataFormat::Float16B);
        assert_eq!(data_format_to_tt(8).unwrap(), DataFormat::Int32);
        assert_eq!(data_format_to_tt(9).unwrap(), DataFormat::UInt16);
        assert_eq!(data_format_to_tt(24).unwrap(), DataFormat::UInt32);
        assert_eq!(data_format_to_tt(30).unwrap(), DataFormat::UInt8);

        let err = data_format_to_tt(99).expect_err("unknown data format should fail");
        assert!(format!("{err:?}").contains("unsupported TT data format code"));
    }

    #[test]
    fn prepare_launch_preserves_declared_binding_order() {
        let kernel = build_interleaved_binding_kernel();
        let mut compiler: TtCompiler = Default::default();
        let repr = compiler
            .compile(
                kernel,
                &cubecl_cpp::shared::CompilationOptions::default(),
                cubecl_core::server::ExecutionMode::Checked,
                StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
            )
            .expect("kernel should compile");
        let sources = cubecl_cpp::tt_metal::compile::sources_from_repr(&repr, 1)
            .expect("TT sources should build");
        let resources = vec![
            make_test_resource(0x10, 16, 2048, vec![10, 11]),
            make_test_resource(0x20, 16, 2048, vec![20, 21]),
            make_test_resource(0x30, 16, 2048, vec![30, 31]),
        ];

        let prepared = prepare_launch(&repr, sources, &resources, &Default::default())
            .expect("launch should prepare");

        assert_eq!(prepared.input_addrs, vec![0x10, 0x30]);
        assert_eq!(prepared.output_addrs, vec![0x20]);
        assert_eq!(prepared.sources.reader_compile_args, vec![10, 11, 30, 31]);
        assert_eq!(prepared.sources.writer_compile_args, vec![20, 21]);
        assert_eq!(prepared.bindings[1].allocation_size_bytes, 2048);
    }

    #[test]
    fn prepare_launch_uses_logical_sizes_for_static_metadata() {
        let kernel = build_phase1_bounds_checked_f32_kernel();
        let mut compiler: TtCompiler = Default::default();
        let repr = compiler
            .compile(
                kernel,
                &cubecl_cpp::shared::CompilationOptions::default(),
                cubecl_core::server::ExecutionMode::Checked,
                StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
            )
            .expect("kernel should compile");
        let sources = cubecl_cpp::tt_metal::compile::sources_from_repr(&repr, 1)
            .expect("TT sources should build");
        let resources = vec![
            make_test_resource(0x100, 24, 4096, vec![2, 4096]),
            make_test_resource(0x200, 24, 4096, vec![2, 4096]),
        ];

        let prepared = prepare_launch(&repr, sources, &resources, &Default::default())
            .expect("launch should prepare");

        assert_eq!(prepared.sources.compute_runtime_args, vec![6, 6, 6, 6]);
        assert_eq!(prepared.bindings[0].logical_size_bytes, 24);
        assert_eq!(prepared.bindings[0].allocation_size_bytes, 4096);
    }

    #[test]
    fn tt_stream_flush_clears_launch_errors() {
        let backend =
            TtStreamBackend::new(std::ptr::null(), tt_memory_properties(), Default::default());
        let mut stream = <TtStreamBackend as EventStreamBackend>::create_stream(&backend);
        stream.error(cubecl_core::server::ServerError::Launch(
            cubecl_core::server::LaunchError::Unknown {
                reason: "boom".into(),
                backtrace: cubecl_common::backtrace::BackTrace::capture(),
            },
        ));

        let err = stream
            .flush_errors(cubecl_core::server::StreamErrorMode {
                ignore: false,
                flush: true,
            })
            .expect_err("flush should surface the pending launch error");
        assert!(matches!(
            err,
            cubecl_core::server::ServerError::ServerUnhealthy { .. }
        ));
        assert!(stream.errors.is_empty());

        stream
            .flush_errors(cubecl_core::server::StreamErrorMode {
                ignore: false,
                flush: true,
            })
            .expect("a second flush should succeed after clearing errors");
    }

    #[test]
    fn compile_kernel_rejects_missing_reader_compile_args() {
        let mut ctx = make_test_context();
        let err = ctx
            .compile_kernel(
                &TtKernelSources::copy_kernel(1, 2048),
                &[0x10],
                &[0x20],
                std::sync::Arc::new(cubecl_runtime::logging::ServerLogger::default()),
            )
            .expect_err("missing reader compile args should fail before TT kernel build");

        assert!(format!("{err:?}").contains("reader compile args missing"));
    }

    #[test]
    fn compile_kernel_surfaces_bad_cpp_as_launch_error() {
        let mut ctx = make_test_context();
        let sources = TtKernelSources::new(
            cubecl_cpp::tt_metal::reader::generate_reader_source(1),
            "this is not valid c++".into(),
            cubecl_cpp::tt_metal::writer::generate_writer_source(1),
            1,
            1,
            1,
            2048,
            5,
        )
        .with_compile_args(vec![2, 2048], vec![2, 2048]);

        let err = ctx
            .compile_kernel(
                &sources,
                &[0x10],
                &[0x20],
                std::sync::Arc::new(cubecl_runtime::logging::ServerLogger::default()),
            )
            .expect_err("invalid TT compute source should surface as a Rust-side launch error");

        assert!(matches!(
            err,
            cubecl_core::server::LaunchError::CompilationError(_)
                | cubecl_core::server::LaunchError::Unknown { .. }
        ));
    }

    #[test]
    fn launch_failure_can_be_flushed_and_client_recovers() {
        if !hardware_tests_enabled() {
            return;
        }

        let client = TestRuntime::client(&Default::default());
        unsafe {
            client.launch_unchecked(
                Box::new(FailingCubeTask),
                CubeCount::Static(1, 1, 1),
                KernelArguments::new(),
            );
        }

        let err = client
            .flush()
            .expect_err("failed launch should surface on flush");
        let reason = format!("{err:?}");
        assert!(reason.contains("intentional TT test compilation failure"));

        cubecl_std::tests::trigonometry::test_to_degrees::<TestRuntime>(client.clone());
        cubecl_std::tests::trigonometry::test_to_radians::<TestRuntime>(client);
    }

    #[test]
    fn mirror_event_kernel_1_characterization() {
        if !hardware_tests_enabled() {
            return;
        }

        let client = TestRuntime::client(&Default::default());
        let output = client.empty(8);
        unsafe {
            mirror_event_launch_1::launch_unchecked::<TestRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                cubecl_runtime::server::CubeDim { x: 1, y: 1, z: 1 },
                ArrayArg::from_raw_parts(output.clone(), 2),
            );
        }

        let bytes = client.read_one_unchecked(output);
        let actual = f32::from_bytes(&bytes);
        assert_eq!(actual, &[20.0, 50.0]);
    }

    #[test]
    fn mirror_event_kernel_2_characterization() {
        if !hardware_tests_enabled() {
            return;
        }

        let client = TestRuntime::client(&Default::default());
        let output = client.empty(8);
        unsafe {
            mirror_event_launch_2::launch_unchecked::<TestRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                cubecl_runtime::server::CubeDim { x: 1, y: 1, z: 1 },
                ArrayArg::from_raw_parts(output.clone(), 2),
            );
        }

        let bytes = client.read_one_unchecked(output);
        let actual = f32::from_bytes(&bytes);
        assert_eq!(actual, &[15.0, 30.0]);
    }

    #[test]
    fn mirror_event_kernel_3_characterization() {
        if !hardware_tests_enabled() {
            return;
        }

        let client = TestRuntime::client(&Default::default());
        let output = client.empty(12);
        unsafe {
            mirror_event_launch_3::launch_unchecked::<TestRuntime>(
                &client,
                CubeCount::Static(1, 1, 1),
                cubecl_runtime::server::CubeDim { x: 1, y: 1, z: 1 },
                ArrayArg::from_raw_parts(output.clone(), 3),
            );
        }

        let bytes = client.read_one_unchecked(output);
        let actual = f32::from_bytes(&bytes);
        assert_eq!(actual, &[30.0, 900.0, 465.0]);
    }

    // ── Helpers for building KernelDefinition ─────────────────────────────

    use cubecl_core::ir::{
        Allocator, Arithmetic, BinaryOperator, Branch, Builtin, Comparison, ConstantValue,
        ElemType, FloatKind, Id, If, IndexAssignOperator, IndexOperator, Instruction, IntKind,
        Metadata, Operator, RangeLoop, Scope, StorageType, Type, UIntKind, UnaryOperator, Variable,
        VariableKind,
    };

    #[test]
    fn phase1_sources_accept_bounds_checked_f32_kernel() {
        let kernel = build_phase1_bounds_checked_f32_kernel();
        let sources = cubecl_cpp::tt_metal::compile::compile_to_tt_sources(&kernel, 1, 4096)
            .expect("f32 generic kernel should compile to TT sources");

        assert_eq!(sources.num_inputs, 1);
        assert_eq!(sources.num_outputs, 1);
        assert_eq!(sources.tile_size_bytes, 4096);
        assert_eq!(sources.info_static_len, 4);
        assert!(
            sources
                .writer_source
                .contains("for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx)")
        );
        assert!(
            sources
                .writer_source
                .contains("uint32_t unit_idx = tile_idx * tile_units + i;")
        );
        assert!(sources.writer_source.contains("runtime_info_bytes"));
        assert!(
            sources
                .writer_source
                .contains("__builtin_memcpy(info.static_meta")
        );
        assert!(sources.writer_source.contains("using std::min;"));
        assert!(sources.writer_source.contains("buffer_0"));
        assert!(sources.writer_source.contains("buffer_1"));
    }

    #[test]
    fn phase1_sources_accept_global_reinterpret_kernel() {
        let kernel = build_phase1_global_reinterpret_kernel(2);
        let sources = cubecl_cpp::tt_metal::compile::compile_to_tt_sources(&kernel, 1, 2048)
            .expect("global reinterpret kernel should compile to TT sources");

        assert_eq!(sources.num_inputs, 1);
        assert_eq!(sources.num_outputs, 1);
        assert_eq!(sources.tile_size_bytes, 2048);
        assert!(sources.writer_source.contains("reinterpret_cast<const"));
        assert!(
            sources
                .writer_source
                .contains("__builtin_memcpy(&cubecl_bitcast_out")
        );
        assert!(sources.writer_source.contains("buffer_0"));
        assert!(sources.writer_source.contains("buffer_1"));
    }

    #[test]
    fn phase1_sources_reject_range_loop_kernel() {
        let kernel = build_phase1_range_loop_kernel();
        let err = cubecl_cpp::tt_metal::compile::compile_to_tt_sources(&kernel, 1, 2048)
            .expect_err("range loops should stay unsupported in Phase 1");
        let reason = format!("{err:?}");
        assert!(reason.contains("Phase 1 does not support instruction"));
        assert!(reason.contains("RangeLoop") || reason.contains("range"));
    }

    #[test]
    fn phase1_sources_reject_shared_memory_kernel() {
        let kernel = build_phase1_shared_memory_kernel();
        let err = cubecl_cpp::tt_metal::compile::compile_to_tt_sources(&kernel, 1, 2048)
            .expect_err("shared memory kernels should stay unsupported in Phase 1");
        let reason = format!("{err:?}");
        assert!(!reason.is_empty());
    }

    fn build_empty_kernel(
        num_inputs: u32,
        num_outputs: u32,
    ) -> cubecl_runtime::kernel::KernelDefinition {
        use cubecl_runtime::kernel::{KernelArg, KernelOptions, Visibility};
        use cubecl_runtime::server::CubeDim;

        let f32_type = Type::new(StorageType::Scalar(ElemType::Float(FloatKind::BF16)));
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
        use cubecl_runtime::kernel::{KernelOptions, Visibility};
        use cubecl_runtime::server::CubeDim;

        let f32_type = Type::new(StorageType::Scalar(ElemType::Float(FloatKind::BF16)));
        let allocator = Allocator::default();
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
                kernel_arg(0, f32_type, Visibility::Read),
                kernel_arg(1, f32_type, Visibility::Read),
                kernel_arg(2, f32_type, Visibility::ReadWrite),
            ],
            tensor_maps: vec![],
            scalars: vec![],
            cube_dim: CubeDim::new_single(),
            body: scope,
            options: KernelOptions::default(),
        }
    }

    fn build_sub_kernel() -> cubecl_runtime::kernel::KernelDefinition {
        use cubecl_runtime::kernel::{KernelOptions, Visibility};
        use cubecl_runtime::server::CubeDim;

        let f32_type = Type::new(StorageType::Scalar(ElemType::Float(FloatKind::BF16)));
        let allocator = Allocator::default();
        let lhs = allocator.create_local(f32_type);
        let rhs = allocator.create_local(f32_type);
        let result = allocator.create_local(f32_type);
        let mut scope = Scope::root(false).with_allocator(allocator);
        scope.register(Instruction::new(
            Arithmetic::Sub(BinaryOperator {
                lhs: *lhs,
                rhs: *rhs,
            }),
            *result,
        ));
        cubecl_runtime::kernel::KernelDefinition {
            buffers: vec![
                kernel_arg(0, f32_type, Visibility::Read),
                kernel_arg(1, f32_type, Visibility::Read),
                kernel_arg(2, f32_type, Visibility::ReadWrite),
            ],
            tensor_maps: vec![],
            scalars: vec![],
            cube_dim: CubeDim::new_single(),
            body: scope,
            options: KernelOptions::default(),
        }
    }

    fn build_mul_kernel() -> cubecl_runtime::kernel::KernelDefinition {
        use cubecl_runtime::kernel::{KernelOptions, Visibility};
        use cubecl_runtime::server::CubeDim;

        let f32_type = Type::new(StorageType::Scalar(ElemType::Float(FloatKind::BF16)));
        let allocator = Allocator::default();
        let lhs = allocator.create_local(f32_type);
        let rhs = allocator.create_local(f32_type);
        let result = allocator.create_local(f32_type);
        let mut scope = Scope::root(false).with_allocator(allocator);
        scope.register(Instruction::new(
            Arithmetic::Mul(BinaryOperator {
                lhs: *lhs,
                rhs: *rhs,
            }),
            *result,
        ));
        cubecl_runtime::kernel::KernelDefinition {
            buffers: vec![
                kernel_arg(0, f32_type, Visibility::Read),
                kernel_arg(1, f32_type, Visibility::Read),
                kernel_arg(2, f32_type, Visibility::ReadWrite),
            ],
            tensor_maps: vec![],
            scalars: vec![],
            cube_dim: CubeDim::new_single(),
            body: scope,
            options: KernelOptions::default(),
        }
    }

    fn build_sqrt_kernel() -> cubecl_runtime::kernel::KernelDefinition {
        use cubecl_runtime::kernel::{KernelOptions, Visibility};
        use cubecl_runtime::server::CubeDim;

        let f32_type = Type::new(StorageType::Scalar(ElemType::Float(FloatKind::BF16)));
        let allocator = Allocator::default();
        let input = allocator.create_local(f32_type);
        let result = allocator.create_local(f32_type);
        let mut scope = Scope::root(false).with_allocator(allocator);
        scope.register(Instruction::new(
            Arithmetic::Sqrt(UnaryOperator { input: *input }),
            *result,
        ));
        cubecl_runtime::kernel::KernelDefinition {
            buffers: vec![
                kernel_arg(0, f32_type, Visibility::Read),
                kernel_arg(1, f32_type, Visibility::ReadWrite),
            ],
            tensor_maps: vec![],
            scalars: vec![],
            cube_dim: CubeDim::new_single(),
            body: scope,
            options: KernelOptions::default(),
        }
    }

    fn build_interleaved_binding_kernel() -> cubecl_runtime::kernel::KernelDefinition {
        use cubecl_runtime::kernel::{KernelOptions, Visibility};
        use cubecl_runtime::server::CubeDim;

        let f32_type = Type::scalar(ElemType::Float(FloatKind::F32));
        let allocator = Allocator::default();
        let lhs = allocator.create_local(f32_type);
        let rhs = allocator.create_local(f32_type);
        let out = allocator.create_local(f32_type);
        let mut scope = Scope::root(false).with_allocator(allocator.clone());
        register_phase1_scope_types(&mut scope);

        let input_a = Variable::new(VariableKind::GlobalInputArray(0), f32_type);
        let output = Variable::new(VariableKind::GlobalOutputArray(1), f32_type);
        let input_b = Variable::new(VariableKind::GlobalInputArray(2), f32_type);
        let unit_pos = Variable::builtin(
            Builtin::UnitPos,
            StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
        );

        scope.register(Instruction::new(
            Operator::Index(IndexOperator {
                list: input_a,
                index: unit_pos,
                vector_size: 0,
                unroll_factor: 1,
            }),
            *lhs,
        ));
        scope.register(Instruction::new(
            Operator::Index(IndexOperator {
                list: input_b,
                index: unit_pos,
                vector_size: 0,
                unroll_factor: 1,
            }),
            *rhs,
        ));
        scope.register(Instruction::new(
            Arithmetic::Add(BinaryOperator {
                lhs: *lhs,
                rhs: *rhs,
            }),
            *out,
        ));
        scope.register(Instruction::new(
            Operator::IndexAssign(IndexAssignOperator {
                index: unit_pos,
                value: *out,
                vector_size: 0,
                unroll_factor: 1,
            }),
            output,
        ));

        cubecl_runtime::kernel::KernelDefinition {
            buffers: vec![
                kernel_arg(0, f32_type, Visibility::Read),
                kernel_arg(1, f32_type, Visibility::ReadWrite),
                kernel_arg(2, f32_type, Visibility::Read),
            ],
            tensor_maps: vec![],
            scalars: vec![],
            cube_dim: CubeDim::new_1d(4),
            body: scope,
            options: KernelOptions::default(),
        }
    }

    fn make_test_resource(
        address: u32,
        logical_size: u64,
        allocation_size: u64,
        compile_args: Vec<u32>,
    ) -> crate::compute::storage::gpu::TtResource {
        crate::compute::storage::gpu::TtResource {
            storage_id: StorageId::new(),
            address,
            size: logical_size,
            allocation_size,
            allocation_offset: 0,
            compile_args,
            layout: crate::compute::storage::gpu::TtBufferLayout::Replicated,
        }
    }

    fn build_phase1_bounds_checked_f32_kernel() -> cubecl_runtime::kernel::KernelDefinition {
        use cubecl_runtime::kernel::{KernelOptions, Visibility};
        use cubecl_runtime::server::CubeDim;

        let f32_type = Type::scalar(ElemType::Float(FloatKind::F32));
        let u32_type = Type::scalar(ElemType::UInt(UIntKind::U32));
        let bool_type = Type::scalar(ElemType::Bool);
        let allocator = Allocator::default();
        let len = allocator.create_local(u32_type);
        let cond = allocator.create_local(bool_type);
        let value = allocator.create_local(f32_type);
        let scaled = allocator.create_local(f32_type);
        let mut scope = Scope::root(false).with_allocator(allocator.clone());
        register_phase1_scope_types(&mut scope);

        let input = Variable::new(VariableKind::GlobalInputArray(0), f32_type);
        let output = Variable::new(VariableKind::GlobalOutputArray(1), f32_type);
        let unit_pos = Variable::builtin(
            Builtin::UnitPos,
            StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
        );
        let scale = Variable::constant(
            ConstantValue::Float((180.0f32 / core::f32::consts::PI) as f64),
            f32_type,
        );

        scope.register(Instruction::new(Metadata::Length { var: input }, *len));
        scope.register(Instruction::new(
            Comparison::Lower(BinaryOperator {
                lhs: unit_pos,
                rhs: *len,
            }),
            *cond,
        ));

        let mut if_scope = scope.child();
        register_phase1_scope_types(&mut if_scope);
        if_scope.register(Instruction::new(
            Operator::Index(IndexOperator {
                list: input,
                index: unit_pos,
                vector_size: 0,
                unroll_factor: 1,
            }),
            *value,
        ));
        if_scope.register(Instruction::new(
            Arithmetic::Mul(BinaryOperator {
                lhs: *value,
                rhs: scale,
            }),
            *scaled,
        ));
        if_scope.register(Instruction::new(
            Operator::IndexAssign(IndexAssignOperator {
                index: unit_pos,
                value: *scaled,
                vector_size: 0,
                unroll_factor: 1,
            }),
            output,
        ));
        scope.register(Instruction::no_out(Branch::If(Box::new(If {
            cond: *cond,
            scope: if_scope,
        }))));

        cubecl_runtime::kernel::KernelDefinition {
            buffers: vec![
                kernel_arg(0, f32_type, Visibility::Read),
                kernel_arg(1, f32_type, Visibility::ReadWrite),
            ],
            tensor_maps: vec![],
            scalars: vec![],
            cube_dim: CubeDim::new_1d(6),
            body: scope,
            options: KernelOptions::default(),
        }
    }

    fn build_phase1_global_reinterpret_kernel(
        vector_size: u8,
    ) -> cubecl_runtime::kernel::KernelDefinition {
        use cubecl_runtime::kernel::{KernelOptions, Visibility};
        use cubecl_runtime::server::CubeDim;

        let vec_i8_type =
            Type::scalar(ElemType::Int(IntKind::I8)).with_vector_size(vector_size.into());
        let f16_type = Type::scalar(ElemType::Float(FloatKind::F16));
        let u32_storage = StorageType::Scalar(ElemType::UInt(UIntKind::U32));
        let allocator = Allocator::default();
        let packed = allocator.create_local(vec_i8_type);
        let unpacked = allocator.create_local(f16_type);
        let mut scope = Scope::root(false).with_allocator(allocator);
        register_phase1_scope_types(&mut scope);

        let input = Variable::new(VariableKind::GlobalInputArray(0), vec_i8_type);
        let output = Variable::new(VariableKind::GlobalOutputArray(1), f16_type);
        let unit_pos = Variable::builtin(Builtin::UnitPos, u32_storage);

        scope.register(Instruction::new(
            Operator::Index(IndexOperator {
                list: input,
                index: unit_pos,
                vector_size: 0,
                unroll_factor: 1,
            }),
            *packed,
        ));
        scope.register(Instruction::new(
            Operator::Reinterpret(UnaryOperator { input: *packed }),
            *unpacked,
        ));
        scope.register(Instruction::new(
            Operator::IndexAssign(IndexAssignOperator {
                index: unit_pos,
                value: *unpacked,
                vector_size: 0,
                unroll_factor: 1,
            }),
            output,
        ));

        cubecl_runtime::kernel::KernelDefinition {
            buffers: vec![
                kernel_arg(0, vec_i8_type, Visibility::Read),
                kernel_arg(1, f16_type, Visibility::ReadWrite),
            ],
            tensor_maps: vec![],
            scalars: vec![],
            cube_dim: CubeDim::new_1d(1),
            body: scope,
            options: KernelOptions::default(),
        }
    }

    fn build_phase1_range_loop_kernel() -> cubecl_runtime::kernel::KernelDefinition {
        use cubecl_runtime::kernel::{KernelOptions, Visibility};
        use cubecl_runtime::server::CubeDim;

        let f32_type = Type::scalar(ElemType::Float(FloatKind::BF16));
        let u32_type = Type::scalar(ElemType::UInt(UIntKind::U32));
        let allocator = Allocator::default();
        let i = allocator.create_local_restricted(u32_type);
        let mut scope = Scope::root(false).with_allocator(allocator);
        register_phase1_scope_types(&mut scope);
        scope.register(Instruction::no_out(Branch::RangeLoop(Box::new(
            RangeLoop {
                i: *i,
                start: Variable::constant(ConstantValue::UInt(0), u32_type),
                end: Variable::constant(ConstantValue::UInt(1), u32_type),
                step: None,
                inclusive: false,
                scope: Scope::root(false),
            },
        ))));

        cubecl_runtime::kernel::KernelDefinition {
            buffers: vec![kernel_arg(0, f32_type, Visibility::ReadWrite)],
            tensor_maps: vec![],
            scalars: vec![],
            cube_dim: CubeDim::new_single(),
            body: scope,
            options: KernelOptions::default(),
        }
    }

    fn build_phase1_shared_memory_kernel() -> cubecl_runtime::kernel::KernelDefinition {
        use cubecl_runtime::kernel::{KernelOptions, Visibility};
        use cubecl_runtime::server::CubeDim;

        let f32_type = Type::scalar(ElemType::Float(FloatKind::BF16));
        let u32_type = Type::scalar(ElemType::UInt(UIntKind::U32));
        let mut scope = Scope::root(false);
        register_phase1_scope_types(&mut scope);
        let _ = scope.create_shared_array(u32_type, 1, None);

        cubecl_runtime::kernel::KernelDefinition {
            buffers: vec![kernel_arg(0, f32_type, Visibility::ReadWrite)],
            tensor_maps: vec![],
            scalars: vec![],
            cube_dim: CubeDim::new_single(),
            body: scope,
            options: KernelOptions::default(),
        }
    }

    fn kernel_arg(
        id: Id,
        ty: Type,
        visibility: cubecl_runtime::kernel::Visibility,
    ) -> cubecl_runtime::kernel::KernelArg {
        cubecl_runtime::kernel::KernelArg {
            id,
            ty,
            visibility,
            size: None,
            has_extended_meta: false,
        }
    }

    fn register_phase1_scope_types(scope: &mut Scope) {
        scope.register_type::<usize>(StorageType::Scalar(ElemType::UInt(UIntKind::U32)));
        scope.register_type::<u32>(StorageType::Scalar(ElemType::UInt(UIntKind::U32)));
    }

    fn make_test_context() -> TtContext {
        use cubecl_common::profile::TimingMethod;
        use cubecl_core::ir::{DeviceProperties, HardwareProperties, VectorSize};
        use cubecl_cpp::shared::{Architecture, register_wmma_features};
        use cubecl_cpp::tt_metal::TtArchitecture;

        let arch = TtArchitecture::Wormhole;
        let warp_size = arch.warp_size();
        let topology = HardwareProperties {
            load_width: 128,
            plane_size_min: warp_size,
            plane_size_max: warp_size,
            max_bindings: crate::runtime::TT_MAX_BINDINGS,
            max_shared_memory_size: 1_500_000,
            max_cube_count: (1, 1, 1),
            max_units_per_cube: warp_size * 32,
            max_cube_dim: (u32::MAX, 1, 1),
            num_streaming_multiprocessors: None,
            num_tensor_cores: None,
            min_tensor_cores_dim: None,
            num_cpu_cores: None,
            max_vector_size: VectorSize::MAX,
        };
        let mut device_props = DeviceProperties::new(
            Default::default(),
            tt_memory_properties(),
            topology,
            TimingMethod::System,
        );
        cubecl_cpp::register_supported_types(&mut device_props);
        register_wmma_features(Vec::new(), &mut device_props);
        device_props.features.memory_reinterpret = true;
        device_props.features.alignment = true;

        TtContext::new(
            cubecl_cpp::shared::CompilationOptions::default(),
            device_props,
        )
    }
}

// ── Old crate-level helpers removed ─────────────────────────────────────
