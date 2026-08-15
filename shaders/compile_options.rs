// Single source of truth for how this project's Slang shaders are compiled.
//
// `include!`d by **both** consumers rather than shared as a module, because
// `build.rs` runs before the crate exists and so cannot `use` anything from it:
//
// - `build.rs` — the shaders baked into the binary via `OUT_DIR`.
// - `src/shader_compiler/compiler.rs` — the runtime `ShaderCompiler`.
//
// These two used to carry independent literals and had silently drifted apart on
// every axis: the runtime path compiled **row-major** matrices against the build
// script's column-major, at `High` instead of `Maximal`, targeting SPIR-V 1.5
// instead of 1.6 — while `build.rs` claimed the bytes were interchangeable. A
// row/column mismatch transposes every matrix, which is silent wrong output, not
// a crash. Keeping one definition makes that class of drift unrepresentable.
//
// The includer must have `shader_slang as slang` in scope.

/// SPIR-V target profile. 1.6 is required by the `spvDescriptorHeapEXT` lowering
/// (untyped pointers + `OpBufferPointerEXT`).
const SPIRV_PROFILE: &str = "spirv_1_6";

/// Build the `CompilerOptions` every Slang compile in this project must use.
///
/// `descriptor_heap_cap` is the caller's resolved `spvDescriptorHeapEXT`
/// capability — both callers already check it for `is_unknown()` and report their
/// own error, so it is taken as a parameter rather than resolved here.
///
/// `debug` adds maximal SPIR-V debug info so Aftermath/RenderDoc/Nsight can resolve
/// shader source. Optimization deliberately stays `Maximal` even then: dropping it
/// changes register allocation enough to hide the bugs being chased.
fn slang_compiler_options(descriptor_heap_cap: slang::CapabilityID, debug: bool) -> slang::CompilerOptions {
    let opts = slang::CompilerOptions::default()
        .optimization(slang::OptimizationLevel::Maximal)
        // Column-major matches nalgebra's column-major `Matrix4` on the CPU (and
        // GLSL's default), so `mul(view_inverse, v)` in the RT shaders agrees with
        // what `set_matrices` in `resource_manager.rs` uploads.
        .matrix_layout_row(false)
        .capability(descriptor_heap_cap);

    if debug {
        opts.debug_information(slang::DebugInfoLevel::Maximal)
    } else {
        opts
    }
}
