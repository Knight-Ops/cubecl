use cubecl_cpp::{metal::MslDialect, shared::Instruction};
use cubecl_ir::StorageType;
use cubecl_runtime::{
    compiler::{CompilationError, Compiler},
    kernel::{KernelArg, KernelDefinition},
    server::ExecutionMode,
};
use std::fmt::Display;

#[derive(Clone, Debug)]
pub struct MetaliumCompilationOptions {
    /// Emit the host-side C++ program scaffold alongside kernel snippets.
    pub include_host_stub: bool,
}

impl Default for MetaliumCompilationOptions {
    fn default() -> Self {
        Self {
            include_host_stub: true,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct MetaliumCompiler;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetaliumKernelSection {
    pub name: &'static str,
    pub source: String,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct MetaliumSourceBundle {
    pub kernel_name: String,
    pub notes: Vec<String>,
    pub sections: Vec<MetaliumKernelSection>,
}

pub type MetaliumExecutable = MetaliumSourceBundle;

impl MetaliumSourceBundle {
    pub fn host_source(&self) -> Option<&str> {
        self.section("host").map(|it| it.source.as_str())
    }

    pub fn reader_source(&self) -> Option<&str> {
        self.section("reader").map(|it| it.source.as_str())
    }

    pub fn compute_source(&self) -> Option<&str> {
        self.section("compute").map(|it| it.source.as_str())
    }

    pub fn writer_source(&self) -> Option<&str> {
        self.section("writer").map(|it| it.source.as_str())
    }

    pub fn section(&self, name: &str) -> Option<&MetaliumKernelSection> {
        self.sections.iter().find(|section| section.name == name)
    }
}

impl Display for MetaliumSourceBundle {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "// TT-Metalium source bundle for `{}`", self.kernel_name)?;

        if !self.notes.is_empty() {
            writeln!(f, "// Notes:")?;
            for note in &self.notes {
                writeln!(f, "// - {note}")?;
            }
        }

        for section in &self.sections {
            writeln!(f)?;
            writeln!(f, "// ===== {} =====", section.name)?;
            writeln!(f, "{}", section.source)?;
        }

        Ok(())
    }
}

#[derive(Clone, Debug)]
struct SectionFunction {
    includes: Vec<&'static str>,
    prelude: Vec<String>,
    signature: String,
    body: Vec<Instruction<MslDialect>>,
    epilogue: Vec<String>,
}

impl Display for SectionFunction {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for include in &self.includes {
            writeln!(f, "#include {include}")?;
        }

        if !self.includes.is_empty() && (!self.prelude.is_empty() || !self.body.is_empty()) {
            writeln!(f)?;
        }

        for line in &self.prelude {
            writeln!(f, "{line}")?;
        }

        if !self.prelude.is_empty() {
            writeln!(f)?;
        }

        writeln!(f, "{} {{", self.signature)?;
        for instruction in &self.body {
            let rendered = instruction.to_string();
            for line in rendered.lines() {
                writeln!(f, "    {line}")?;
            }
        }
        writeln!(f, "}}")?;

        for line in &self.epilogue {
            writeln!(f, "{line}")?;
        }

        Ok(())
    }
}

impl Compiler for MetaliumCompiler {
    type Representation = MetaliumSourceBundle;
    type CompilationOptions = MetaliumCompilationOptions;

    fn compile(
        &mut self,
        kernel: KernelDefinition,
        compilation_options: &Self::CompilationOptions,
        mode: ExecutionMode,
        addr_type: StorageType,
    ) -> Result<Self::Representation, CompilationError> {
        let kernel_name = kernel.options.kernel_name.clone();
        let notes = build_notes(&kernel, mode, addr_type);
        let mut sections = vec![
            MetaliumKernelSection {
                name: "reader",
                source: build_reader_section(&kernel).to_string(),
            },
            MetaliumKernelSection {
                name: "compute",
                source: build_compute_section(&kernel).to_string(),
            },
            MetaliumKernelSection {
                name: "writer",
                source: build_writer_section(&kernel).to_string(),
            },
        ];

        if compilation_options.include_host_stub {
            sections.insert(
                0,
                MetaliumKernelSection {
                    name: "host",
                    source: build_host_section(&kernel).to_string(),
                },
            );
        }

        Ok(MetaliumSourceBundle {
            kernel_name,
            notes,
            sections,
        })
    }

    fn elem_size(&self, elem: cubecl_ir::ElemType) -> usize {
        elem.size()
    }

    fn extension(&self) -> &'static str {
        "ttmetal.cpp"
    }
}

fn build_notes(
    kernel: &KernelDefinition,
    mode: ExecutionMode,
    addr_type: StorageType,
) -> Vec<String> {
    let mut notes = vec![
        "CubeCL global-buffer IR still needs an explicit lowering pass into TT-Metalium host/dataflow/compute kernels."
            .to_string(),
        "This bundle is a scaffold intended for backend bring-up, inspection, and iterative lowering."
            .to_string(),
        format!(
            "Kernel uses {} buffer bindings, {} tensor-map bindings, and {} scalar groups.",
            kernel.buffers.len(),
            kernel.tensor_maps.len(),
            kernel.scalars.len()
        ),
        format!(
            "Cube dimensions are ({}, {}, {}); TT core mapping must be chosen by the runtime/program builder.",
            kernel.cube_dim.x, kernel.cube_dim.y, kernel.cube_dim.z
        ),
        format!("Execution mode during compilation: {mode:?}."),
        format!("Kernel address type: {addr_type}."),
    ];

    if !kernel.tensor_maps.is_empty() {
        notes.push(
            "Tensor-map bindings are preserved in metadata only for now; no TT tensor accessor lowering is implemented yet."
                .to_string(),
        );
    }

    notes
}

fn build_host_section(kernel: &KernelDefinition) -> SectionFunction {
    let mut body = vec![
        comment("Create the unit mesh device and command queue."),
        comment("MeshDevice / MeshCommandQueue setup is still performed by the future cxx bridge."),
        comment("CreateProgram() and CoreCoord selection belong here."),
    ];
    body.extend(binding_comments("Buffer binding", &kernel.buffers));
    body.push(comment(format!(
        "TODO: Allocate MeshBuffer objects for the {} buffer bindings.",
        kernel.buffers.len()
    )));
    body.push(comment(
        "TODO: Choose page sizes / layouts and create circular buffers for the lowered pipeline.",
    ));
    body.push(comment(
        "TODO: Create reader / compute / writer kernels from the generated source sections.",
    ));
    body.push(comment(
        "TODO: Set runtime args for DRAM/L1 addresses, scalar values, and work partitioning.",
    ));
    body.push(comment(
        "TODO: Add the program to MeshWorkload and enqueue it on the mesh command queue.",
    ));
    body.push(comment("return program;"));

    SectionFunction {
        includes: vec![
            "<tt-metalium/host_api.hpp>",
            "<tt-metalium/device.hpp>",
            "<tt-metalium/distributed.hpp>",
        ],
        prelude: vec![
            "using namespace tt;".to_string(),
            "using namespace tt::tt_metal;".to_string(),
        ],
        signature: format!("Program build_{}()", kernel.options.kernel_name),
        body,
        epilogue: vec![],
    }
}

fn build_reader_section(kernel: &KernelDefinition) -> SectionFunction {
    let mut body = vec![
        comment(format!(
            "Reader scaffold for `{}`.",
            kernel.options.kernel_name
        )),
        comment(
            "TODO: Lower CubeCL buffer/tensor-map reads into TT-Metalium DRAM -> L1 or circular-buffer transfers.",
        ),
    ];
    body.extend(binding_comments("Buffer", &kernel.buffers));
    body.extend(tensor_map_comments(&kernel.tensor_maps));
    body.push(comment(
        "Suggested next step: group contiguous reads into page- or tile-sized transfers and push them into circular buffers.",
    ));

    SectionFunction {
        includes: vec!["<cstdint>", "\"api/dataflow/dataflow_api.h\""],
        prelude: vec![],
        signature: "void kernel_main()".to_string(),
        body,
        epilogue: vec![],
    }
}

fn build_compute_section(kernel: &KernelDefinition) -> SectionFunction {
    let mut body = vec![
        comment(format!(
            "Compute scaffold for `{}`.",
            kernel.options.kernel_name
        )),
        comment(format!(
            "Cube dims: ({}, {}, {}).",
            kernel.cube_dim.x, kernel.cube_dim.y, kernel.cube_dim.z
        )),
        comment("TODO: Lower CubeCL arithmetic/control flow into either:"),
        comment("1. a page-oriented Baby RISC-V kernel for correctness-first execution, or"),
        comment("2. a tiled TT compute kernel plus circular-buffer orchestration for performance."),
    ];

    if kernel.scalars.is_empty() {
        body.push(comment("No scalar groups were captured for this kernel."));
    } else {
        for (index, scalar) in kernel.scalars.iter().enumerate() {
            body.push(comment(format!(
                "Scalar group {index}: type={}, count={}",
                scalar.ty, scalar.count
            )));
        }
    }

    body.push(comment(
        "Kernel body lowering is intentionally left as a TODO until buffer movement and core mapping are explicit.",
    ));

    SectionFunction {
        includes: vec!["<cstdint>", "\"api/compute/compute_kernel_api.h\""],
        prelude: vec![],
        signature: "void kernel_main()".to_string(),
        body,
        epilogue: vec![],
    }
}

fn build_writer_section(kernel: &KernelDefinition) -> SectionFunction {
    let mut body = vec![
        comment(format!(
            "Writer scaffold for `{}`.",
            kernel.options.kernel_name
        )),
        comment(
            "TODO: Lower CubeCL writeback into TT-Metalium L1/circular-buffer -> DRAM transfers.",
        ),
    ];
    for (index, arg) in kernel.buffers.iter().enumerate() {
        if matches!(
            arg.visibility,
            cubecl_runtime::kernel::Visibility::ReadWrite
        ) {
            body.push(comment(format!(
                "Writable buffer {index}: id={}, type={}",
                arg.id, arg.ty
            )));
        }
    }
    body.push(comment(
        "Suggested next step: assign output circular buffers and emit matching SetRuntimeArgs host wiring.",
    ));

    SectionFunction {
        includes: vec!["<cstdint>", "\"api/dataflow/dataflow_api.h\""],
        prelude: vec![],
        signature: "void kernel_main()".to_string(),
        body,
        epilogue: vec![],
    }
}

fn visibility_name(arg: &KernelArg) -> &'static str {
    match arg.visibility {
        cubecl_runtime::kernel::Visibility::Read => "read",
        cubecl_runtime::kernel::Visibility::ReadWrite => "read_write",
    }
}

fn comment(content: impl Into<String>) -> Instruction<MslDialect> {
    Instruction::Comment {
        content: content.into(),
    }
}

fn binding_comments(prefix: &str, bindings: &[KernelArg]) -> Vec<Instruction<MslDialect>> {
    bindings
        .iter()
        .enumerate()
        .map(|(index, arg)| {
            comment(format!(
                "{prefix} {index}: id={}, visibility={}, type={}",
                arg.id,
                visibility_name(arg),
                arg.ty
            ))
        })
        .collect()
}

fn tensor_map_comments(bindings: &[KernelArg]) -> Vec<Instruction<MslDialect>> {
    bindings
        .iter()
        .enumerate()
        .map(|(index, arg)| {
            comment(format!(
                "Tensor map {index}: id={}, visibility={}, type={}",
                arg.id,
                visibility_name(arg),
                arg.ty
            ))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use cubecl_ir::{ElemType, FloatKind, Scope, StorageType, Type, UIntKind};
    use cubecl_runtime::kernel::{KernelArg, KernelOptions, ScalarKernelArg, Visibility};
    use cubecl_runtime::server::CubeDim;

    fn sample_kernel() -> KernelDefinition {
        KernelDefinition {
            buffers: vec![
                KernelArg {
                    id: 0,
                    visibility: Visibility::Read,
                    ty: Type::scalar(ElemType::Float(FloatKind::BF16)),
                    size: None,
                    has_extended_meta: false,
                },
                KernelArg {
                    id: 1,
                    visibility: Visibility::ReadWrite,
                    ty: Type::scalar(ElemType::Float(FloatKind::BF16)),
                    size: None,
                    has_extended_meta: false,
                },
            ],
            tensor_maps: vec![],
            scalars: vec![ScalarKernelArg {
                ty: StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
                count: 2,
            }],
            cube_dim: CubeDim::new_2d(8, 4),
            body: Scope::root(false),
            options: KernelOptions {
                kernel_name: "axpy_like".to_string(),
                debug_symbols: false,
                cluster_dim: None,
            },
        }
    }

    #[test]
    fn emits_bundle_sections() {
        let mut compiler = MetaliumCompiler;
        let bundle = compiler
            .compile(
                sample_kernel(),
                &MetaliumCompilationOptions {
                    include_host_stub: true,
                },
                ExecutionMode::Checked,
                StorageType::Scalar(ElemType::UInt(UIntKind::U32)),
            )
            .unwrap();

        assert!(bundle.host_source().unwrap().contains("CreateProgram"));
        assert!(bundle.reader_source().unwrap().contains("Buffer 0"));
        assert!(bundle.compute_source().unwrap().contains("Scalar group 0"));
        assert!(
            bundle
                .writer_source()
                .unwrap()
                .contains("Writable buffer 1")
        );
        assert!(bundle.to_string().contains("TT-Metalium source bundle"));
    }
}
