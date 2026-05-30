use std::collections::HashSet;
use std::fmt::{self, Display, Write};

use crate::shared::{
    self, BarrierOps, Component, ComputeKernel, FmtLeft, IndexInstruction, Instruction, Item,
    Variable, WarpInstruction,
};

use super::compile::TtKernelAnalysis;
use super::dialect::TtMetalDialect;
use super::kernel::{TtBinaryComputeOp, TtUnaryComputeOp};

/// Generate the C++ source for a TT-Metal writer (dataflow) kernel.
///
/// Uses `TensorAccessorArgs` with `noc_async_write_tile` for correct
/// interleaved DRAM tile addressing. Each output buffer needs 2 compile-time
/// args: [`ArgConfig`] flags, [`AlignedPageSize`].
pub fn generate_writer_source(num_outputs: u32) -> String {
    let mut src = String::new();
    src.push_str("#include <cstdint>\n\n");
    src.push_str("void kernel_main() {\n");

    if num_outputs == 0 {
        src.push_str("    (void)get_arg_val<uint32_t>(0);\n");
        src.push_str("}\n");
        return src;
    }

    // Runtime args: dst_addr_0, ..., dst_addr_N, num_tiles
    for i in 0..num_outputs {
        src.push_str(&format!(
            "    uint32_t dst{}_addr = get_arg_val<uint32_t>({});\n",
            i, i
        ));
    }
    let num_tiles_idx = num_outputs;
    src.push_str(&format!(
        "    uint32_t num_tiles = get_arg_val<uint32_t>({});\n\n",
        num_tiles_idx
    ));

    // Output circular buffers start at index 16
    for i in 0..num_outputs {
        let cb_idx = 16 + i;
        src.push_str(&format!(
            "    constexpr uint32_t cb_out{} = {};\n",
            i, cb_idx
        ));
    }

    // Declare TensorAccessor for each output buffer.
    src.push('\n');
    for i in 0..num_outputs {
        if i == 0 {
            src.push_str(&format!(
                "    constexpr auto c{}_args = TensorAccessorArgs<0>();\n",
                i
            ));
        } else {
            src.push_str(&format!(
                "    constexpr auto c{}_args = TensorAccessorArgs<c{}_args.next_compile_time_args_offset()>();\n",
                i, i - 1
            ));
        }
        src.push_str(&format!(
            "    const auto c{} = TensorAccessor(c{}_args, dst{}_addr);\n",
            i, i, i
        ));
    }

    src.push_str("\n    for (uint32_t i = 0; i < num_tiles; i++) {\n");
    for idx in 0..num_outputs {
        src.push_str(&format!("        cb_wait_front(cb_out{idx}, 1);\n"));
        src.push_str(&format!(
            "        uint32_t l1_addr_out{idx} = get_read_ptr(cb_out{idx});\n"
        ));
        src.push_str(&format!(
            "        noc_async_write_tile(i, c{idx}, l1_addr_out{idx});\n"
        ));
    }
    src.push_str("        noc_async_write_barrier();\n");
    for idx in 0..num_outputs {
        src.push_str(&format!("        cb_pop_front(cb_out{idx}, 1);\n"));
    }
    src.push_str("    }\n");
    src.push_str("}\n");

    src
}

/// Generate a single-kernel dram_loopback-style copy source.
///
/// Uses a circular buffer as scratch space for the DRAM→DRAM copy.
/// Reads tiles from input DRAM into the CB, then writes from the CB
/// to output DRAM. No compute kernel, no separate reader/writer.
/// This matches the TT-Metal `dram_loopback` example pattern adapted to use CBs.
pub fn generate_dram_loopback_source() -> String {
    r#"#include <cstdint>

void kernel_main() {
    uint32_t dram_buffer_src_addr = get_arg_val<uint32_t>(0);
    uint32_t dram_buffer_dst_addr = get_arg_val<uint32_t>(1);
    uint32_t num_tiles = get_arg_val<uint32_t>(2);

    constexpr uint32_t cb_scratch = tt::CBIndex::c_0;

    constexpr auto in0_args = TensorAccessorArgs<0>();
    const auto in0 = TensorAccessor(in0_args, dram_buffer_src_addr);

    constexpr auto out0_args = TensorAccessorArgs<in0_args.next_compile_time_args_offset()>();
    const auto out0 = TensorAccessor(out0_args, dram_buffer_dst_addr);

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_reserve_back(cb_scratch, 1);
        uint32_t l1_addr = get_write_ptr(cb_scratch);
        noc_async_read_tile(i, in0, l1_addr);
        noc_async_read_barrier();
        cb_push_back(cb_scratch, 1);

        cb_wait_front(cb_scratch, 1);
        l1_addr = get_read_ptr(cb_scratch);
        noc_async_write_tile(i, out0, l1_addr);
        noc_async_write_barrier();
        cb_pop_front(cb_scratch, 1);
    }
}
"#
    .to_string()
}

/// Generate the C++ source for a simple copy compute kernel.
pub fn generate_copy_compute_source() -> String {
    r#"#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t dst_reg = 0;

    binary_op_init_common(cb_in0, cb_in0, cb_out0);
    copy_tile_init(cb_in0);

    for (uint32_t i = 0; i < num_tiles; i++) {
        tile_regs_acquire();
        cb_wait_front(cb_in0, 1);
        copy_tile(cb_in0, 0, dst_reg);
        tile_regs_commit();
        tile_regs_wait();

        cb_pop_front(cb_in0, 1);

        cb_reserve_back(cb_out0, 1);
        pack_tile(dst_reg, cb_out0);
        cb_push_back(cb_out0, 1);

        tile_regs_release();
    }
}
"#
    .to_string()
}

fn binary_compute_init_and_op(op: TtBinaryComputeOp) -> Option<(&'static str, &'static str)> {
    match op {
        TtBinaryComputeOp::Add => Some(("add_tiles_init", "add_tiles")),
        TtBinaryComputeOp::Sub => Some(("sub_tiles_init", "sub_tiles")),
        TtBinaryComputeOp::Mul => Some(("mul_tiles_init", "mul_tiles")),
        TtBinaryComputeOp::Div => None,
    }
}

pub fn generate_binary_compute_source(op: TtBinaryComputeOp) -> String {
    if matches!(op, TtBinaryComputeOp::Div) {
        return r#"#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
#include "api/compute/eltwise_binary_sfpu.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t lhs_reg = 0;
    constexpr uint32_t rhs_reg = 1;
    constexpr uint32_t dst_reg = 0;

    unary_op_init_common(cb_in0, cb_out0);
    div_binary_tile_init();

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        cb_reserve_back(cb_out0, 1);

        tile_regs_acquire();
        copy_tile_to_dst_init_short_with_dt(cb_in1, cb_in0);
        copy_tile(cb_in0, 0, lhs_reg);
        copy_tile_to_dst_init_short_with_dt(cb_in0, cb_in1);
        copy_tile(cb_in1, 0, rhs_reg);
        div_binary_tile(lhs_reg, rhs_reg, dst_reg);
        tile_regs_commit();
        tile_regs_wait();

        pack_tile(dst_reg, cb_out0);
        tile_regs_release();

        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);
        cb_push_back(cb_out0, 1);
    }
}
"#
        .to_string();
    }

    let (init_fn, op_fn) = binary_compute_init_and_op(op).expect("non-div binary op expected");
    format!(
        r#"#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/compute_kernel_api.h"

void kernel_main() {{
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t dst_reg = 0;

    binary_op_init_common(cb_in0, cb_in1, cb_out0);
    {init_fn}(cb_in0, cb_in1);

    for (uint32_t i = 0; i < num_tiles; i++) {{
        tile_regs_acquire();
        cb_wait_front(cb_in0, 1);
        cb_wait_front(cb_in1, 1);
        {op_fn}(cb_in0, cb_in1, 0, 0, dst_reg);
        tile_regs_commit();
        tile_regs_wait();

        cb_pop_front(cb_in0, 1);
        cb_pop_front(cb_in1, 1);

        cb_reserve_back(cb_out0, 1);
        pack_tile(dst_reg, cb_out0);
        cb_push_back(cb_out0, 1);

        tile_regs_release();
    }}
}}
"#
    )
}
/// Generate the C++ source for an element-wise addition compute kernel.
pub fn generate_add_compute_source() -> String {
    generate_binary_compute_source(TtBinaryComputeOp::Add)
}

fn unary_compute_spec(op: TtUnaryComputeOp) -> (Option<&'static str>, &'static str, &'static str) {
    match op {
        TtUnaryComputeOp::Abs => (None, "abs_tile_init", "abs_tile"),
        TtUnaryComputeOp::Sqrt => (Some("sqrt"), "sqrt_tile_init", "sqrt_tile"),
        TtUnaryComputeOp::Rsqrt => (Some("rsqrt"), "rsqrt_tile_init", "rsqrt_tile"),
        TtUnaryComputeOp::Sin => (Some("trigonometry"), "sin_tile_init", "sin_tile"),
        TtUnaryComputeOp::Cos => (Some("trigonometry"), "cos_tile_init", "cos_tile"),
        TtUnaryComputeOp::Tan => (Some("trigonometry"), "tan_tile_init", "tan_tile"),
        TtUnaryComputeOp::Tanh => (Some("trigonometry"), "tanh_tile_init", "tanh_tile"),
        TtUnaryComputeOp::Exp => (Some("exp"), "exp_tile_init", "exp_tile"),
        TtUnaryComputeOp::Log => (None, "log_tile_init", "log_tile"),
    }
}

pub fn generate_unary_compute_source(op: TtUnaryComputeOp) -> String {
    let (header, init_fn, op_fn) = unary_compute_spec(op);
    let op_include = header
        .map(|header| {
            format!(
                r#"#include "api/compute/eltwise_unary/{header}.h"
"#
            )
        })
        .unwrap_or_default();
    format!(
        r#"#include "api/compute/common.h"
#include "api/compute/tile_move_copy.h"
#include "api/compute/eltwise_unary/eltwise_unary.h"
{op_include}#include "api/compute/compute_kernel_api.h"

void kernel_main() {{
    uint32_t num_tiles = get_arg_val<uint32_t>(0);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_out0 = tt::CBIndex::c_16;
    constexpr uint32_t dst_reg = 0;

    init_sfpu(cb_in0, cb_out0);
    copy_tile_init(cb_in0);
    {init_fn}();

    for (uint32_t i = 0; i < num_tiles; i++) {{
        cb_wait_front(cb_in0, 1);
        tile_regs_acquire();
        copy_tile(cb_in0, 0, dst_reg);
        {op_fn}(dst_reg);
        tile_regs_commit();
        tile_regs_wait();

        cb_reserve_back(cb_out0, 1);
        pack_tile(dst_reg, cb_out0);
        cb_pop_front(cb_in0, 1);
        tile_regs_release();
        cb_push_back(cb_out0, 1);
    }}
}}
"#
    )
}
pub fn generate_noop_compute_source() -> String {
    "void kernel_main() {}
"
    .to_string()
}

#[derive(Clone)]
enum TtWarpSource {
    Snapshot {
        snapshot_name: String,
        item: Item<TtMetalDialect>,
    },
    CompareConst {
        snapshot_name: String,
        item: Item<TtMetalDialect>,
        op: TtWarpComparison,
        constant: String,
    },
    UnitPos,
    UnitPosPlane,
    UnitPosLessThanConst {
        constant: String,
    },
}

#[derive(Clone, Copy)]
enum TtWarpComparison {
    Less,
    Greater,
}

#[derive(Clone)]
struct TtWarpSnapshot {
    snapshot_name: String,
    buffer_name: String,
    item: Item<TtMetalDialect>,
}

#[derive(Default)]
struct TtWarpLoweringState {
    aliases: Vec<(String, TtWarpSource)>,
    snapshots: Vec<TtWarpSnapshot>,
}

fn tt_variable_key(var: &Variable<TtMetalDialect>) -> String {
    format!("{var}")
}

fn tt_sanitize_name(name: &str) -> String {
    name.chars()
        .map(|ch| if ch.is_ascii_alphanumeric() { ch } else { '_' })
        .collect()
}

impl TtWarpLoweringState {
    fn source_for(&self, var: &Variable<TtMetalDialect>) -> Option<TtWarpSource> {
        let key = tt_variable_key(var);
        self.aliases
            .iter()
            .rev()
            .find(|(candidate, _)| *candidate == key)
            .map(|(_, source)| source.clone())
    }

    fn bind(&mut self, var: &Variable<TtMetalDialect>, source: TtWarpSource) {
        let key = tt_variable_key(var);
        if let Some((_, existing)) = self
            .aliases
            .iter_mut()
            .rev()
            .find(|(candidate, _)| *candidate == key)
        {
            *existing = source;
        } else {
            self.aliases.push((key, source));
        }
    }

    fn record_snapshot(
        &mut self,
        buffer: &Variable<TtMetalDialect>,
        item: Item<TtMetalDialect>,
    ) -> String {
        let buffer_name = format!("{buffer}");
        if let Some(existing) = self
            .snapshots
            .iter()
            .find(|snapshot| snapshot.buffer_name == buffer_name)
        {
            return existing.snapshot_name.clone();
        }
        let snapshot_name = format!("tt_plane_snapshot_{}", tt_sanitize_name(&buffer_name));
        self.snapshots.push(TtWarpSnapshot {
            snapshot_name: snapshot_name.clone(),
            buffer_name,
            item,
        });
        snapshot_name
    }

    fn observe_instruction(&mut self, instruction: &Instruction<TtMetalDialect>) {
        match instruction {
            Instruction::Index(IndexInstruction { list, out, .. }) => {
                if matches!(
                    list,
                    Variable::GlobalInputArray(_, _) | Variable::GlobalOutputArray(_, _)
                ) {
                    let snapshot_name = self.record_snapshot(list, out.item());
                    self.bind(
                        out,
                        TtWarpSource::Snapshot {
                            snapshot_name,
                            item: out.item(),
                        },
                    );
                }
            }
            Instruction::Assign(it) | Instruction::SpecialCast(it) => {
                let source = match it.input {
                    Variable::UnitPos => Some(TtWarpSource::UnitPos),
                    Variable::UnitPosPlane => Some(TtWarpSource::UnitPosPlane),
                    _ => self.source_for(&it.input),
                };
                if let Some(source) = source {
                    self.bind(&it.out, source);
                }
            }
            Instruction::Select { then, out, .. } => {
                if let Some(source) = self.source_for(then) {
                    self.bind(out, source);
                }
            }
            Instruction::Lower(it) | Instruction::LowerEqual(it) => {
                if let Some(TtWarpSource::Snapshot {
                    snapshot_name,
                    item,
                }) = self.source_for(&it.lhs)
                {
                    if let Variable::Constant(_, _) = it.rhs {
                        self.bind(
                            &it.out,
                            TtWarpSource::CompareConst {
                                snapshot_name,
                                item,
                                op: TtWarpComparison::Less,
                                constant: format!("{}", it.rhs),
                            },
                        );
                    }
                } else if matches!(it.lhs, Variable::UnitPos | Variable::UnitPosPlane)
                    || matches!(
                        self.source_for(&it.lhs),
                        Some(TtWarpSource::UnitPos) | Some(TtWarpSource::UnitPosPlane)
                    )
                {
                    if let Variable::Constant(_, _) = it.rhs {
                        self.bind(
                            &it.out,
                            TtWarpSource::UnitPosLessThanConst {
                                constant: format!("{}", it.rhs),
                            },
                        );
                    }
                }
            }
            Instruction::Greater(it) | Instruction::GreaterEqual(it) => {
                if let Some(TtWarpSource::Snapshot {
                    snapshot_name,
                    item,
                }) = self.source_for(&it.lhs)
                {
                    if let Variable::Constant(_, _) = it.rhs {
                        self.bind(
                            &it.out,
                            TtWarpSource::CompareConst {
                                snapshot_name,
                                item,
                                op: TtWarpComparison::Greater,
                                constant: format!("{}", it.rhs),
                            },
                        );
                    }
                }
            }
            _ => {}
        }
    }
}

fn tt_source_component_expr(source: &TtWarpSource, lane_expr: &str, comp: usize) -> Option<String> {
    match source {
        TtWarpSource::Snapshot {
            snapshot_name,
            item,
        } => {
            if item.vectorization > 1 {
                Some(format!("{snapshot_name}[{lane_expr}].i_{comp}"))
            } else {
                Some(format!("{snapshot_name}[{lane_expr}]"))
            }
        }
        TtWarpSource::CompareConst {
            snapshot_name,
            item,
            op,
            constant,
        } => {
            let lhs = if item.vectorization > 1 {
                format!("{snapshot_name}[{lane_expr}].i_{comp}")
            } else {
                format!("{snapshot_name}[{lane_expr}]")
            };
            let cmp = match op {
                TtWarpComparison::Less => "<",
                TtWarpComparison::Greater => ">",
            };
            Some(format!("(({lhs}) {cmp} ({constant}))"))
        }
        TtWarpSource::UnitPos => Some(format!("({lane_expr})")),
        TtWarpSource::UnitPosPlane => Some(format!("(({lane_expr}) - tt_plane_base)")),
        TtWarpSource::UnitPosLessThanConst { constant } => {
            Some(format!("(({lane_expr}) < ({constant}))"))
        }
    }
}

fn tt_render_vector_assignment(
    out: &Variable<TtMetalDialect>,
    components: impl Fn(usize) -> String,
) -> String {
    let mut rendered = String::new();
    let item = out.item();
    if item.vectorization == 1 {
        let _ = writeln!(rendered, "{} = {};", out.fmt_left(), components(0));
    } else {
        let _ = write!(rendered, "{} = {{ ", out.fmt_left());
        for comp in 0..item.vectorization {
            if comp > 0 {
                let _ = write!(rendered, ", ");
            }
            let _ = write!(rendered, "{}", components(comp));
        }
        let _ = writeln!(rendered, " }};");
    }
    rendered
}

fn tt_render_reduce(
    out: &Variable<TtMetalDialect>,
    source: &TtWarpSource,
    combine: &str,
    reduction_fn: Option<&str>,
) -> Option<String> {
    let item = out.item();
    let elem = format!("{}", item.elem());
    Some(tt_render_vector_assignment(out, |comp| {
        let init = tt_source_component_expr(source, "tt_plane_base", comp).unwrap();
        let step = tt_source_component_expr(source, "tt_plane_base + tt_lane", comp).unwrap();
        match reduction_fn {
            Some(fn_name) => format!(
                "([&]() -> {elem} {{ {elem} acc = {init}; for (uint32_t tt_lane = 1; tt_lane < plane_dim_checked; ++tt_lane) {{ acc = {fn_name}(acc, {step}); }} return acc; }})()"
            ),
            None => format!(
                "([&]() -> {elem} {{ {elem} acc = {init}; for (uint32_t tt_lane = 1; tt_lane < plane_dim_checked; ++tt_lane) {{ acc {combine} {step}; }} return acc; }})()"
            ),
        }
    }))
}

fn tt_render_prefix_reduce(
    out: &Variable<TtMetalDialect>,
    source: &TtWarpSource,
    combine: &str,
    init_value: &str,
    inclusive: bool,
) -> Option<String> {
    let item = out.item();
    let elem = format!("{}", item.elem());
    Some(tt_render_vector_assignment(out, |comp| {
        let body = tt_source_component_expr(source, "tt_plane_base + tt_lane", comp).unwrap();
        let cmp = if inclusive {
            "<= unit_pos_plane"
        } else {
            "< unit_pos_plane"
        };
        format!(
            "([&]() -> {elem} {{ {elem} acc = {init_value}; for (uint32_t tt_lane = 0; tt_lane < plane_dim_checked; ++tt_lane) {{ if (tt_lane {cmp}) acc {combine} {body}; }} return acc; }})()"
        )
    }))
}

fn tt_render_warp_instruction(
    instruction: &WarpInstruction<TtMetalDialect>,
    state: &TtWarpLoweringState,
) -> Option<String> {
    match instruction {
        WarpInstruction::ReduceSum { input, out } => {
            state.source_for(input).and_then(|source| tt_render_reduce(out, &source, "+=", None))
        }
        WarpInstruction::ReduceProd { input, out } => {
            state.source_for(input).and_then(|source| tt_render_reduce(out, &source, "*=", None))
        }
        WarpInstruction::ReduceMax { input, out } => state
            .source_for(input)
            .and_then(|source| tt_render_reduce(out, &source, "", Some("max"))),
        WarpInstruction::ReduceMin { input, out } => state
            .source_for(input)
            .and_then(|source| tt_render_reduce(out, &source, "", Some("min"))),
        WarpInstruction::InclusiveSum { input, out } => state
            .source_for(input)
            .and_then(|source| tt_render_prefix_reduce(out, &source, "+=", "0", true)),
        WarpInstruction::ExclusiveSum { input, out } => state
            .source_for(input)
            .and_then(|source| tt_render_prefix_reduce(out, &source, "+=", "0", false)),
        WarpInstruction::InclusiveProd { input, out } => state
            .source_for(input)
            .and_then(|source| tt_render_prefix_reduce(out, &source, "*=", "1", true)),
        WarpInstruction::ExclusiveProd { input, out } => state
            .source_for(input)
            .and_then(|source| tt_render_prefix_reduce(out, &source, "*=", "1", false)),
        WarpInstruction::Broadcast { input, id, out } => state.source_for(input).and_then(|source| {
            Some(tt_render_vector_assignment(out, |comp| {
                let src = format!("((uint32_t)({id}) < plane_dim_checked ? uint32_t({id}) : unit_pos_plane)");
                tt_source_component_expr(&source, &format!("tt_plane_base + {src}"), comp).unwrap()
            }))
        }),
        WarpInstruction::Shuffle { input, src_lane, out } => state.source_for(input).and_then(|source| {
            Some(tt_render_vector_assignment(out, |comp| {
                let src = format!("((uint32_t)({src_lane}) < plane_dim_checked ? uint32_t({src_lane}) : unit_pos_plane)");
                tt_source_component_expr(&source, &format!("tt_plane_base + {src}"), comp).unwrap()
            }))
        }),
        WarpInstruction::ShuffleXor { input, mask, out } => state.source_for(input).and_then(|source| {
            Some(tt_render_vector_assignment(out, |comp| {
                format!("([&]() -> {} {{ uint32_t tt_src = unit_pos_plane ^ uint32_t({mask}); if (tt_src < plane_dim_checked) return {}; return {}; }})()", out.item().elem(), tt_source_component_expr(&source, "tt_plane_base + tt_src", comp).unwrap(), tt_source_component_expr(&source, "tt_plane_base + unit_pos_plane", comp).unwrap())
            }))
        }),
        WarpInstruction::ShuffleUp { input, delta, out } => state.source_for(input).and_then(|source| {
            Some(tt_render_vector_assignment(out, |comp| {
                format!("([&]() -> {} {{ uint32_t tt_delta = uint32_t({delta}); if (unit_pos_plane >= tt_delta) return {}; return {}; }})()", out.item().elem(), tt_source_component_expr(&source, "tt_plane_base + unit_pos_plane - tt_delta", comp).unwrap(), tt_source_component_expr(&source, "tt_plane_base + unit_pos_plane", comp).unwrap())
            }))
        }),
        WarpInstruction::ShuffleDown { input, delta, out } => state.source_for(input).and_then(|source| {
            Some(tt_render_vector_assignment(out, |comp| {
                format!("([&]() -> {} {{ uint32_t tt_delta = uint32_t({delta}); uint32_t tt_src = unit_pos_plane + tt_delta; if (tt_src < plane_dim_checked) return {}; return {}; }})()", out.item().elem(), tt_source_component_expr(&source, "tt_plane_base + tt_src", comp).unwrap(), tt_source_component_expr(&source, "tt_plane_base + unit_pos_plane", comp).unwrap())
            }))
        }),
        WarpInstruction::All { input, out } => state.source_for(input).map(|source| {
            let pred = tt_source_component_expr(&source, "tt_plane_base + tt_lane", 0).unwrap();
            format!("{} = ([&]() -> bool {{ bool tt_all = true; for (uint32_t tt_lane = 0; tt_lane < plane_dim_checked; ++tt_lane) {{ tt_all = tt_all && {}; }} return tt_all; }})();
", out.fmt_left(), pred)
        }),
        WarpInstruction::Any { input, out } => state.source_for(input).map(|source| {
            let pred = tt_source_component_expr(&source, "tt_plane_base + tt_lane", 0).unwrap();
            format!("{} = ([&]() -> bool {{ bool tt_any = false; for (uint32_t tt_lane = 0; tt_lane < plane_dim_checked; ++tt_lane) {{ tt_any = tt_any || {}; }} return tt_any; }})();
", out.fmt_left(), pred)
        }),
        WarpInstruction::Ballot { input, out } => state.source_for(input).map(|source| {
            let pred = tt_source_component_expr(&source, "tt_lane", 0).unwrap();
            format!("uint32_t tt_ballot = 0; for (uint32_t tt_lane = 0; tt_lane < plane_dim_checked && tt_lane < 32; ++tt_lane) {{ if ({}) tt_ballot |= (1u << tt_lane); }} {} = {{ tt_ballot, 0u, 0u, 0u }};
", pred, out.fmt_left())
        }),
        WarpInstruction::Elect { out } | WarpInstruction::ElectFallback { out } => {
            Some(format!("{} = (unit_pos_plane == 0);
", out.fmt_left()))
        }
    }
}

fn analyze_tt_warp_lowering(
    instructions: &[Instruction<TtMetalDialect>],
) -> (TtWarpLoweringState, Vec<String>) {
    if !instructions
        .iter()
        .any(|instruction| matches!(instruction, Instruction::Warp(_)))
    {
        return (
            TtWarpLoweringState::default(),
            instructions.iter().map(render_tt_instruction).collect(),
        );
    }

    let mut state = TtWarpLoweringState::default();
    let mut rendered = Vec::with_capacity(instructions.len());
    for instruction in instructions {
        match instruction {
            Instruction::Warp(warp) => {
                if let Some(custom) = tt_render_warp_instruction(warp, &state) {
                    rendered.push(custom);
                } else {
                    rendered.push(render_tt_instruction(instruction));
                }
            }
            Instruction::If { .. }
            | Instruction::IfElse { .. }
            | Instruction::Switch { .. }
            | Instruction::RangeLoop { .. }
            | Instruction::Loop { .. } => {
                rendered.push(render_tt_instruction(instruction));
            }
            _ => {
                state.observe_instruction(instruction);
                rendered.push(render_tt_instruction(instruction));
            }
        }
    }
    (state, rendered)
}

pub(crate) fn generate_scalar_writer_source(
    repr: &ComputeKernel<TtMetalDialect>,
    analysis: &TtKernelAnalysis,
) -> String {
    let tile_units = (analysis.tile_size_bytes as usize / analysis.unit_item.size()).max(1) as u32;
    let mut src = String::new();
    let num_tiles_idx = analysis.num_outputs;
    let num_units_idx = num_tiles_idx + 1;
    let cube_count_x_idx = num_tiles_idx + 2;
    let cube_count_y_idx = num_tiles_idx + 3;
    let cube_count_z_idx = num_tiles_idx + 4;
    let static_arg_offset_words = num_tiles_idx + 5;
    let dynamic_meta_offset_words =
        (repr.info.dynamic_meta_offset / core::mem::size_of::<u32>()) as u32;

    let _ = writeln!(src, "#include <cstdint>\n#include <algorithm>\n");
    let _ = writeln!(src, "using std::max;\nusing std::min;\n");
    let _ = writeln!(
        src,
        "{}",
        ScalarWriterTypeDefinitions {
            items: &repr.items,
            scalars: &repr.scalars,
            info: &repr.info,
            address_type: repr.flags.address_type,
        }
    );
    let _ = writeln!(src, "void kernel_main() {{");

    for output_idx in 0..analysis.num_outputs {
        let _ = writeln!(
            src,
            "    uint32_t dst{output_idx}_addr = get_arg_val<uint32_t>({output_idx});"
        );
    }
    let _ = writeln!(
        src,
        "    uint32_t num_tiles = get_arg_val<uint32_t>({num_tiles_idx});"
    );
    let _ = writeln!(
        src,
        "    uint32_t num_units = get_arg_val<uint32_t>({num_units_idx});"
    );
    let _ = writeln!(
        src,
        "    uint32_t cube_count_x = get_arg_val<uint32_t>({cube_count_x_idx});"
    );
    let _ = writeln!(
        src,
        "    uint32_t cube_count_y = get_arg_val<uint32_t>({cube_count_y_idx});"
    );
    let _ = writeln!(
        src,
        "    uint32_t cube_count_z = get_arg_val<uint32_t>({cube_count_z_idx});"
    );
    let _ = writeln!(src, "    constexpr uint32_t tile_units = {tile_units};");
    let _ = writeln!(
        src,
        "    constexpr uint32_t cube_dim_x = {};",
        repr.cube_dim.x
    );
    let _ = writeln!(
        src,
        "    constexpr uint32_t cube_dim_y = {};",
        repr.cube_dim.y
    );
    let _ = writeln!(
        src,
        "    constexpr uint32_t cube_dim_z = {};",
        repr.cube_dim.z
    );
    let _ = writeln!(
        src,
        "    uint32_t cube_units = cube_dim_x * cube_dim_y * cube_dim_z;"
    );

    for input_idx in 0..analysis.num_inputs {
        let _ = writeln!(
            src,
            "    constexpr uint32_t cb_in{input_idx} = {input_idx};"
        );
    }
    for output_idx in 0..analysis.num_outputs {
        let cb = 16 + output_idx;
        let _ = writeln!(src, "    constexpr uint32_t cb_out{output_idx} = {cb};");
    }

    src.push('\n');
    for output_idx in 0..analysis.num_outputs {
        if output_idx == 0 {
            let _ = writeln!(
                src,
                "    constexpr auto c{output_idx}_args = TensorAccessorArgs<0>();"
            );
        } else {
            let prev = output_idx - 1;
            let _ = writeln!(
                src,
                "    constexpr auto c{output_idx}_args = TensorAccessorArgs<c{prev}_args.next_compile_time_args_offset()>();"
            );
        }
        let _ = writeln!(
            src,
            "    const auto c{output_idx} = TensorAccessor(c{output_idx}_args, dst{output_idx}_addr);"
        );
    }

    if repr.body.info_static_len > 0 || !repr.info.scalars.is_empty() {
        let _ = writeln!(src, "    info_st info = {{}};");
        let _ = writeln!(
            src,
            "    constexpr uint32_t info_arg_offset_words = {static_arg_offset_words};"
        );
        let _ = writeln!(
            src,
            "    const uint8_t* runtime_info_bytes = reinterpret_cast<const uint8_t*>(get_arg_addr(info_arg_offset_words));",
        );

        for (field, (ty, _)) in repr.info.scalars.iter().zip(repr.scalars.iter()) {
            let _ = writeln!(
                src,
                "    __builtin_memcpy(info.scalars_{ty}, runtime_info_bytes + {}, sizeof(info.scalars_{ty}));",
                field.offset,
            );
        }

        if let Some(static_meta) = repr.info.sized_meta {
            let _ = writeln!(
                src,
                "    __builtin_memcpy(info.static_meta, runtime_info_bytes + {}, sizeof(info.static_meta));",
                static_meta.offset,
            );
        }
    }

    if repr.body.has_dynamic_meta {
        let _ = writeln!(
            src,
            "    constexpr uint32_t dynamic_meta_arg_offset_words = {};",
            static_arg_offset_words + dynamic_meta_offset_words,
        );
        let _ = writeln!(
            src,
            "    const tt_l1_ptr {}* dynamic_meta = reinterpret_cast<tt_l1_ptr const {}*>(get_arg_addr(dynamic_meta_arg_offset_words));",
            repr.body.address_type, repr.body.address_type,
        );
    }

    for const_array in &repr.body.const_arrays {
        let _ = write!(
            src,
            "    const {} arrays_{}[{}] = {{",
            const_array.item, const_array.index, const_array.size,
        );
        let item = const_array.item;
        for value in const_array.values.iter().copied() {
            let value = match value {
                shared::Variable::Constant(value, _) => shared::Variable::Constant(value, item),
                _ => unreachable!("constant arrays only contain constant values"),
            };
            let _ = write!(src, "{value},");
        }
        let _ = writeln!(src, "}};");
    }

    let _ = writeln!(
        src,
        "\n    for (uint32_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {{"
    );
    let _ = writeln!(
        src,
        "        uint32_t tile_start_unit = tile_idx * tile_units;"
    );
    let _ = writeln!(
        src,
        "        uint32_t tile_unit_count = num_units > tile_start_unit ? std::min(tile_units, num_units - tile_start_unit) : 0;"
    );
    for input_idx in 0..analysis.num_inputs {
        let _ = writeln!(src, "        cb_wait_front(cb_in{input_idx}, 1);");
    }

    let has_terminate_return = repr
        .body
        .instructions
        .iter()
        .any(|instruction| instruction.to_string().contains("return;"));
    if has_terminate_return {
        let _ = writeln!(src, "    bool terminate_kernel = false;");
    }
    for output_idx in 0..analysis.num_outputs {
        let _ = writeln!(src, "        cb_reserve_back(cb_out{output_idx}, 1);");
    }
    for output_idx in 0..analysis.num_outputs {
        let _ = writeln!(
            src,
            "        uint32_t l1_addr_out{output_idx}_prefill = get_write_ptr(cb_out{output_idx});"
        );
        let _ = writeln!(
            src,
            "        uint32_t page_bytes_out{output_idx}_prefill = get_tile_size(cb_out{output_idx});"
        );
        let _ = writeln!(
            src,
            "        uint64_t dst_noc_addr_out{output_idx}_prefill = c{output_idx}.get_noc_addr(tile_idx);"
        );
        let _ = writeln!(
            src,
            "        noc_async_read(dst_noc_addr_out{output_idx}_prefill, l1_addr_out{output_idx}_prefill, page_bytes_out{output_idx}_prefill);"
        );
    }
    if analysis.num_outputs > 0 {
        let _ = writeln!(src, "        noc_async_read_barrier();");
    }

    let output_binding_indices = analysis
        .output_binding_indices
        .iter()
        .copied()
        .collect::<HashSet<_>>();
    let mut read_idx = 0u32;
    let mut write_idx = 0u32;
    for (binding_index, binding) in repr.buffers.iter().enumerate() {
        if output_binding_indices.contains(&binding_index) {
            let _ = writeln!(
                src,
                "        {}* buffer_{} = reinterpret_cast<{}*>(get_write_ptr(cb_out{})) - tile_idx * tile_units;",
                binding.item, binding.id, binding.item, write_idx,
            );
            write_idx += 1;
        } else {
            let _ = writeln!(
                src,
                "        const {}* buffer_{} = reinterpret_cast<const {}*>(get_read_ptr(cb_in{})) - tile_idx * tile_units;",
                binding.item, binding.id, binding.item, read_idx,
            );
            read_idx += 1;
        }
    }

    let (warp_lowering, rendered_instructions) = analyze_tt_warp_lowering(&repr.body.instructions);
    emit_shared_memory_declarations(&mut src, repr);
    for snapshot in &warp_lowering.snapshots {
        let _ = writeln!(
            src,
            "        {} {}[tile_units];",
            snapshot.item, snapshot.snapshot_name
        );
        let _ = writeln!(
            src,
            "        for (uint32_t tt_snapshot_i = 0; tt_snapshot_i < tile_unit_count; ++tt_snapshot_i) {{"
        );
        let _ = writeln!(
            src,
            "            {}[tt_snapshot_i] = {}[tt_snapshot_i];",
            snapshot.snapshot_name, snapshot.buffer_name
        );
        let _ = writeln!(src, "        }}");
    }
    let _ = writeln!(
        src,
        "\n        for (uint32_t i = 0; i < tile_units; ++i) {{"
    );
    let _ = writeln!(
        src,
        "            uint32_t unit_idx = tile_idx * tile_units + i;"
    );
    let _ = writeln!(src, "            if (unit_idx >= num_units) break;");
    emit_builtin_locals(&mut src, &repr.flags.indexes);
    emit_local_array_declarations(&mut src, repr);
    if has_terminate_return {
        for rendered in &rendered_instructions {
            let rendered = rendered.replace(
                "return;",
                "terminate_kernel = true; goto tt_writer_finish_tile;",
            );
            let _ = write!(src, "            {rendered}");
        }
    } else {
        for rendered in &rendered_instructions {
            let _ = write!(src, "            {rendered}");
        }
    }
    let _ = writeln!(src, "        }}");
    if has_terminate_return {
        let _ = writeln!(src, "tt_writer_finish_tile:");
    }

    for output_idx in 0..analysis.num_outputs {
        let _ = writeln!(src, "        cb_push_back(cb_out{output_idx}, 1);");
        let _ = writeln!(src, "        cb_wait_front(cb_out{output_idx}, 1);");
        let _ = writeln!(
            src,
            "        uint32_t l1_addr_out{output_idx} = get_read_ptr(cb_out{output_idx});"
        );
        let _ = writeln!(
            src,
            "        uint32_t page_bytes_out{output_idx} = get_tile_size(cb_out{output_idx});"
        );
        let _ = writeln!(
            src,
            "        uint64_t dst_noc_addr_out{output_idx} = c{output_idx}.get_noc_addr(tile_idx);"
        );
        let _ = writeln!(
            src,
            "        noc_async_write(l1_addr_out{output_idx}, dst_noc_addr_out{output_idx}, page_bytes_out{output_idx});"
        );
    }
    let _ = writeln!(src, "        noc_async_write_barrier();");
    for output_idx in 0..analysis.num_outputs {
        let _ = writeln!(src, "        cb_pop_front(cb_out{output_idx}, 1);");
    }
    for input_idx in 0..analysis.num_inputs {
        let _ = writeln!(src, "        cb_pop_front(cb_in{input_idx}, 1);");
    }
    if has_terminate_return {
        let _ = writeln!(src, "        if (terminate_kernel) break;");
    }
    let _ = writeln!(src, "    }}");
    let _ = writeln!(src, "}}");

    src
}

fn render_tt_instruction(instruction: &shared::Instruction<TtMetalDialect>) -> String {
    match instruction {
        shared::Instruction::Barrier(barrier_op) => render_tt_barrier_op(barrier_op),
        _ => instruction.to_string(),
    }
}

fn render_tt_barrier_op(barrier_op: &BarrierOps<TtMetalDialect>) -> String {
    match barrier_op {
        BarrierOps::Declare { level, .. } => match level {
            cubecl_core::ir::BarrierLevel::Unit => {
                "// TT unit barrier declare elided in generic writer\n".to_string()
            }
            cubecl_core::ir::BarrierLevel::Cube => {
                "// TT cube barrier declare elided; generic writer serializes unit execution over shared scratch\n".to_string()
            }
        },
        BarrierOps::Init { level, .. } => match level {
            cubecl_core::ir::BarrierLevel::Unit => {
                "// TT unit barrier init elided; single-unit kernels are already synchronized\n".to_string()
            }
            cubecl_core::ir::BarrierLevel::Cube => {
                "// TT cube barrier init elided; generic writer serializes unit execution over shared scratch\n".to_string()
            }
        },
        BarrierOps::ArriveAndWait { level, .. } => match level {
            cubecl_core::ir::BarrierLevel::Unit => {
                "// TT unit barrier arrive_and_wait elided; single-unit kernels are already synchronized\n".to_string()
            }
            cubecl_core::ir::BarrierLevel::Cube => {
                "// TT cube barrier arrive_and_wait elided; generic writer serializes unit execution over shared scratch\n".to_string()
            }
        },
        BarrierOps::MemCopyAsync {
            source,
            destination,
            source_length,
            offset_source,
            offset_out,
            cooperative: false,
            ..
        } => format!(
            "for (uint32_t tt_barrier_copy_i = 0; tt_barrier_copy_i < {source_length}; ++tt_barrier_copy_i) {{\n                {destination}[{offset_out} + tt_barrier_copy_i] = {source}[{offset_source} + tt_barrier_copy_i];\n            }}\n"
        ),
        _ => panic!("unsupported TT barrier op reached writer: {barrier_op:?}"),
    }
}

fn emit_local_array_declarations(src: &mut String, repr: &ComputeKernel<TtMetalDialect>) {
    for array in &repr.body.local_arrays {
        let _ = writeln!(
            src,
            "            {} l_arr_{}[{}];",
            array.item, array.index, array.size
        );
    }
}

fn emit_shared_memory_declarations(src: &mut String, repr: &ComputeKernel<TtMetalDialect>) {
    for shared in &repr.body.shared_memories {
        match shared {
            shared::SharedMemory::Array {
                index,
                item,
                length,
                align,
                ..
            } => {
                let size_bytes = length * item.size();
                let align = (*align).max(1);
                let _ = writeln!(
                    src,
                    "        // TT shared scratch array size: {length}, {size_bytes} bytes"
                );
                let _ = writeln!(
                    src,
                    "        alignas({align}) {item} shared_memory_{index}[{length}];"
                );
            }
            shared::SharedMemory::Value {
                index, item, align, ..
            } => {
                let size_bytes = item.size();
                let align = (*align).max(1);
                let _ = writeln!(
                    src,
                    "        // TT shared scratch value size: {size_bytes} bytes"
                );
                let _ = writeln!(
                    src,
                    "        alignas({align}) {item} shared_memory_{index};"
                );
            }
        }
    }
}

fn emit_builtin_locals(src: &mut String, indexes: &crate::shared::CubeIndexFlags) {
    let needs_cube_pos_components = indexes.cube_pos || indexes.cube_pos_tuple;
    let needs_unit_pos_components = indexes.unit_pos_tuple;
    let needs_flattened_positions = needs_cube_pos_components
        || needs_unit_pos_components
        || indexes.unit_pos
        || indexes.absolute_pos
        || indexes.unit_pos_plane
        || indexes.plane_dim_checked
        || indexes.plane_pos;

    if needs_flattened_positions {
        let _ = writeln!(
            src,
            "            uint32_t cube_units = cube_dim_x * cube_dim_y * cube_dim_z;"
        );
        let _ = writeln!(
            src,
            "            uint32_t flat_cube_pos = cube_units == 0 ? 0 : (unit_idx / cube_units);"
        );
        let _ = writeln!(
            src,
            "            uint32_t unit_pos = cube_units == 0 ? 0 : (unit_idx % cube_units);"
        );
    }

    if needs_cube_pos_components || needs_unit_pos_components {
        let _ = writeln!(
            src,
            "            uint32_t cube_pos_x = cube_count_x == 0 ? 0 : (flat_cube_pos % cube_count_x);"
        );
        let _ = writeln!(
            src,
            "            uint32_t cube_pos_y = (cube_count_x == 0 || cube_count_y == 0) ? 0 : ((flat_cube_pos / cube_count_x) % cube_count_y);"
        );
        let _ = writeln!(
            src,
            "            uint32_t cube_pos_z = (cube_count_x == 0 || cube_count_y == 0) ? 0 : (flat_cube_pos / (cube_count_x * cube_count_y));"
        );
        let _ = writeln!(
            src,
            "            uint32_t unit_pos_x = cube_dim_x == 0 ? 0 : (unit_pos % cube_dim_x);"
        );
        let _ = writeln!(
            src,
            "            uint32_t unit_pos_y = (cube_dim_x == 0 || cube_dim_y == 0) ? 0 : ((unit_pos / cube_dim_x) % cube_dim_y);"
        );
        let _ = writeln!(
            src,
            "            uint32_t unit_pos_z = (cube_dim_x == 0 || cube_dim_y == 0) ? 0 : (unit_pos / (cube_dim_x * cube_dim_y));"
        );
    }
    if indexes.absolute_pos {
        let _ = writeln!(src, "            uint32_t absolute_pos = unit_idx;");
    }
    if indexes.cube_dim {
        let _ = writeln!(
            src,
            "            uint32_t cube_dim = cube_dim_x * cube_dim_y * cube_dim_z;"
        );
    }
    if indexes.cube_count {
        let _ = writeln!(
            src,
            "            uint32_t cube_count = cube_count_x * cube_count_y * cube_count_z;"
        );
    }
    if indexes.cube_pos {
        let _ = writeln!(src, "            uint32_t cube_pos = cube_pos_x;");
    }
    if indexes.plane_dim {
        let _ = writeln!(src, "            uint32_t plane_dim = 32;");
    }
    if indexes.unit_pos_plane || indexes.plane_dim_checked || indexes.plane_pos {
        let _ = writeln!(src, "            uint32_t unit_pos_plane = unit_pos % 32;");
        let _ = writeln!(
            src,
            "            uint32_t tt_plane_base = i - unit_pos_plane;"
        );
        let _ = writeln!(
            src,
            "            uint32_t tt_plane_remaining = tile_unit_count > tt_plane_base ? (tile_unit_count - tt_plane_base) : 0;"
        );
    }
    if indexes.plane_dim_checked {
        let _ = writeln!(
            src,
            "            uint32_t plane_dim_checked = tt_plane_remaining < 32 ? tt_plane_remaining : 32;"
        );
    }
    if indexes.plane_pos {
        let _ = writeln!(src, "            uint32_t plane_pos = unit_pos_plane;");
    }
}

struct ScalarWriterTypeDefinitions<'a> {
    items: &'a HashSet<Item<TtMetalDialect>>,
    scalars: &'a [(crate::shared::Elem<TtMetalDialect>, usize)],
    info: &'a cubecl_core::Info,
    address_type: Item<TtMetalDialect>,
}

impl Display for ScalarWriterTypeDefinitions<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        scalar_writer_vectorized_definitions(f, self.items)?;
        shared::type_info_definition_sized::<TtMetalDialect>(
            f,
            self.info,
            self.scalars,
            self.address_type,
        )
    }
}

fn scalar_writer_vectorized_definitions(
    f: &mut fmt::Formatter<'_>,
    items: &HashSet<Item<TtMetalDialect>>,
) -> fmt::Result {
    let mut emitted = HashSet::new();
    for item in items.iter() {
        let elem = item.elem;
        let size = item.vectorization;
        let alignment = elem.size() * size;
        if size > 1 {
            let item_name = item.to_string();
            if !emitted.insert(item_name.clone()) {
                continue;
            }
            write!(
                f,
                "
struct alignas({alignment}) {item_name} {{"
            )?;
            for i in 0..size {
                write!(
                    f,
                    "
    {elem} i_{i};"
                )?;
            }
            f.write_str(
                "
};",
            )?;
        }
    }
    Ok(())
}
