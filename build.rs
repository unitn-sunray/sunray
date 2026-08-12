use std::{ffi::CString, fs::File, io::Write};

use shader_slang as slang;
use shader_slang::Downcast;

/*
* This build script compiles shaders in the shaders/ directory into .spirv files under $OUT_DIR.
* Slang shaders are compiled with `shader-slang` and emit SPIR-V with the
* `spvDescriptorHeapEXT` capability enabled (matches the runtime compiler in
* src/shader_compiler), so they plug straight into a VK_EXT_descriptor_heap pipeline.
 */

fn output_file_prefix(name: &str) -> String {
    format!("{}/{}", std::env::var("OUT_DIR").unwrap(), name)
}
fn input_file_prefix(name: &str) -> String {
    format!("{}/{}", std::env::var("CARGO_MANIFEST_DIR").unwrap(), name)
}

fn shader_debug_enabled() -> bool {
    // Force-enable Slang debug info (and disable optimization)
    // so a GPU debugger (Aftermath, RenderDoc, RGP, Nsight) can resolve shader
    // source locations. Tied to a build-time env var because shader compilation
    // is a build-time step — runtime DiagnosticTool selection alone can't
    // change what's already baked into the SPIR-V.
    println!("cargo::rerun-if-env-changed=SUNRAY_SHADER_DEBUG");
    matches!(
        std::env::var("SUNRAY_SHADER_DEBUG").as_deref(),
        Ok("1") | Ok("true") | Ok("TRUE")
    )
}

/// Compile a Slang module to SPIR-V at build time. `module_name` is the file stem under
/// `shaders/` (no `.slang`). Mirrors the runtime compiler in `src/shader_compiler/compiler.rs`,
/// so the bytes the two paths produce are interchangeable.
fn compile_slang_shader(module_name: &str, entry_point: &str, out_file_name: &str) {
    let global_session =
        slang::GlobalSession::new().expect("Failed to create Slang GlobalSession (is the Slang runtime DLL on PATH?)");

    let descriptor_heap_cap = global_session.find_capability("spvDescriptorHeapEXT");
    if descriptor_heap_cap.is_unknown() {
        panic!(
            "Slang does not know the `spvDescriptorHeapEXT` capability — \
             the installed Slang predates PR #10177 (Feb 2026). Update the Slang runtime."
        );
    }

    let debug = shader_debug_enabled();

    // Column-major matrix storage matches nalgebra's column-major Matrix4 on the
    // CPU (and GLSL's default), so reading `matrices.view_inverse * v` in the
    // Slang RT shaders produces the same result as the original GLSL.
    //
    // When SUNRAY_SHADER_DEBUG=1, drop optimization to None and ask Slang to
    // emit maximal SPIR-V debug info so Aftermath/RenderDoc resolve source.
    let session_options = if debug {
        slang::CompilerOptions::default()
            .optimization(slang::OptimizationLevel::Maximal) //TODO level of opt
            .debug_information(slang::DebugInfoLevel::Maximal)
            .matrix_layout_row(false)
            .capability(descriptor_heap_cap)
    } else {
        slang::CompilerOptions::default()
            .optimization(slang::OptimizationLevel::Maximal)
            .matrix_layout_row(false)
            .capability(descriptor_heap_cap)
    };

    let target_desc = slang::TargetDesc::default()
        .format(slang::CompileTarget::Spirv)
        .profile(global_session.find_profile("spirv_1_6"));

    let targets = [target_desc];

    let shaders_dir = input_file_prefix("shaders");
    let search_path =
        CString::new(shaders_dir.clone()).unwrap_or_else(|e| panic!("shaders dir '{shaders_dir}' contains nul byte: {e}"));
    let search_paths = [search_path.as_ptr()];

    let session_desc = slang::SessionDesc::default()
        .targets(&targets)
        .search_paths(&search_paths)
        .options(&session_options);

    let session = global_session
        .create_session(&session_desc)
        .expect("Slang create_session returned null");

    let module = session
        .load_module(module_name)
        .unwrap_or_else(|e| panic!("Slang load_module(\"{module_name}\") failed: {e}"));

    let entry = module
        .find_entry_point_by_name(entry_point)
        .unwrap_or_else(|| panic!("entry point \"{entry_point}\" not found in module \"{module_name}\""));

    let program = session
        .create_composite_component_type(&[module.downcast().clone(), entry.downcast().clone()])
        .unwrap_or_else(|e| panic!("Slang create_composite_component_type failed: {e}"));

    let linked = program
        .link()
        .unwrap_or_else(|e| panic!("Slang link failed for \"{module_name}::{entry_point}\": {e}"));

    let spirv_blob = linked
        .entry_point_code(0, 0)
        .unwrap_or_else(|e| panic!("Slang entry_point_code failed for \"{module_name}::{entry_point}\": {e}"));

    let mut out_file = File::create(output_file_prefix(out_file_name))
        .unwrap_or_else(|e| panic!("While opening/creating shader spirv file '{out_file_name}' for write: {e}"));
    out_file
        .write_all(spirv_blob.as_slice())
        .unwrap_or_else(|e| panic!("While writing to shader spirv file '{out_file_name}': {e}"));
}

fn main() {
    println!("cargo::rerun-if-changed=shaders/");

    // Raytracing pipeline (heap mode). One Slang module per stage; the entry
    // point matches the [shader("…")] attribute inside each file.
    compile_slang_shader("ray_miss", "ray_miss", "ray_miss.spirv");
    compile_slang_shader("any_hit", "any_hit", "any_hit.spirv");
    compile_slang_shader("closest_hit", "closest_hit", "closest_hit.spirv");
    compile_slang_shader("ray_gen_ris", "ray_gen_ris", "ray_gen_ris.spirv");
    compile_slang_shader("ray_gen_final", "ray_gen_final", "ray_gen_final.spirv");
    compile_slang_shader("postprocess", "main", "postprocess.spirv");
    compile_slang_shader("denoise", "main", "denoise.spirv");
    compile_slang_shader("temporal_accumulation", "main", "temporal_accumulation.spirv");

    // egui overlay (Bevy integration). One module, two stages; each entry point is
    // emitted as a SPIR-V "main" (matches how the RT stages are handled).
    compile_slang_shader("egui", "vertex_main", "egui_vert.spirv");
    compile_slang_shader("egui", "fragment_main", "egui_frag.spirv");
}
