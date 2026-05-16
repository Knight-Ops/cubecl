use cubecl_core::ir::{Operation, Scope};
use cubecl_core::prelude::KernelDefinition;

/// The type of TT-Metal compute operation needed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TtOpKind {
    /// 1 input → 1 output, no arithmetic (uses `copy_tile`)
    Copy,
    /// 2 inputs → 1 output, eltwise binary add (uses `add_tiles`)
    EltwiseBinaryAdd,
    /// Operation not recognized — will fail compilation
    Unknown,
}

/// Detect the operation type by walking the `CubeCL` IR.
///
/// Currently detects `Copy` (no arithmetic) and `EltwiseBinaryAdd` (`Add`/`Mul` arithmetic).
pub fn detect_op_kind(kernel: &KernelDefinition) -> TtOpKind {
    let op = find_operation(&kernel.body);

    match op {
        Some(Operation::Arithmetic(
            cubecl_core::ir::Arithmetic::Add(_) | cubecl_core::ir::Arithmetic::Mul(_),
        )) => TtOpKind::EltwiseBinaryAdd,
        Some(Operation::Arithmetic(_)) => TtOpKind::Unknown,
        Some(Operation::Copy(_)) => TtOpKind::Copy,
        None => TtOpKind::Copy,
        _ => TtOpKind::Unknown,
    }
}

fn find_operation(scope: &Scope) -> Option<Operation> {
    for instruction in scope.instructions.iter() {
        if let op @ (Operation::Arithmetic(_) | Operation::Copy(_)) = &instruction.operation {
            return Some(op.clone());
        }
    }
    None
}
