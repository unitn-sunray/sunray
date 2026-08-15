use std::ffi::CString;
use std::path::PathBuf;

use shader_slang as slang;
use shader_slang::Downcast;

use crate::error::{SrError, SrResult};

// Shared with `build.rs`; see that file's header for why this is an `include!`
// and not a module.
include!("../../shaders/compile_options.rs");

/// Slang compiler bound to a single shaders directory. Cheap to keep alive — the
/// expensive object is the `GlobalSession`, which we hold for the renderer's lifetime.
/// Each `compile()` call spins up a fresh `Session` so options stay independent
/// per-compile (good enough for the first iteration; per-stage caching can come later).
pub struct ShaderCompiler {
    global_session: slang::GlobalSession,
    descriptor_heap_cap: slang::CapabilityID,
    /// CString-stored to keep the `*const i8` we hand to `SessionDesc::search_paths` valid.
    search_path: CString,
}

impl ShaderCompiler {
    pub fn new(shaders_dir: PathBuf) -> SrResult<Self> {
        let global_session = slang::GlobalSession::new().ok_or_else(|| {
            SrError::new_custom("Failed to create Slang GlobalSession (is the Slang runtime DLL on PATH?)".into())
        })?;

        let descriptor_heap_cap = global_session.find_capability("spvDescriptorHeapEXT");
        if descriptor_heap_cap.is_unknown() {
            return Err(SrError::new_custom(
                "Slang does not know the `spvDescriptorHeapEXT` capability — \
                 the installed Slang predates PR #10177 (Feb 2026). Update the Slang runtime."
                    .into(),
            ));
        }

        let dir_str = shaders_dir
            .to_str()
            .ok_or_else(|| SrError::new_custom(format!("non-utf8 shaders dir: {shaders_dir:?}")))?;
        let search_path =
            CString::new(dir_str).map_err(|e| SrError::new_custom(format!("shaders dir contains nul byte: {e}")))?;

        Ok(Self {
            global_session,
            descriptor_heap_cap,
            search_path,
        })
    }

    /// Compiles a Slang module + entry point to SPIR-V bytes ready for `vk::ShaderModuleCreateInfo`.
    /// `module_name` is the file stem under the shaders dir (no `.slang`); the entry point is
    /// looked up by name on that module.
    pub fn compile(&self, module_name: &str, entry_point: &str) -> SrResult<Vec<u8>> {
        // Same options the build script bakes in — see `shaders/compile_options.rs`.
        // Runtime compiles never want debug info; `SUNRAY_SHADER_DEBUG` is a
        // build-time knob for the baked shaders.
        let session_options = slang_compiler_options(self.descriptor_heap_cap, false);

        let target_desc = slang::TargetDesc::default()
            .format(slang::CompileTarget::Spirv)
            .profile(self.global_session.find_profile(SPIRV_PROFILE));

        let targets = [target_desc];
        let search_paths = [self.search_path.as_ptr()];

        let session_desc = slang::SessionDesc::default()
            .targets(&targets)
            .search_paths(&search_paths)
            .options(&session_options);

        let session = self
            .global_session
            .create_session(&session_desc)
            .ok_or_else(|| SrError::new_custom("Slang create_session returned null".into()))?;

        let module = session
            .load_module(module_name)
            .map_err(|e| SrError::new_custom(format!("Slang load_module(\"{module_name}\") failed: {e}")))?;

        let entry = module
            .find_entry_point_by_name(entry_point)
            .ok_or_else(|| SrError::new_custom(format!("entry point \"{entry_point}\" not found in module \"{module_name}\"")))?;

        let program = session
            .create_composite_component_type(&[module.downcast().clone(), entry.downcast().clone()])
            .map_err(|e| SrError::new_custom(format!("Slang create_composite_component_type failed: {e}")))?;

        let linked = program
            .link()
            .map_err(|e| SrError::new_custom(format!("Slang link failed: {e}")))?;

        let spirv_blob = linked.entry_point_code(0, 0).map_err(|e| {
            SrError::new_custom(format!(
                "Slang entry_point_code failed for \"{module_name}::{entry_point}\": {e}"
            ))
        })?;

        Ok(spirv_blob.as_slice().to_vec())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Compiles every heap-mode Slang compute shader the renderer loads at
    /// runtime and asserts each yields non-empty, u32-aligned SPIR-V. This is a
    /// GPU-free smoke test of shader *syntax* (the Slang runtime DLL must be on
    /// PATH; if it isn't, `ShaderCompiler::new` fails with a clear message).
    #[test]
    fn heap_slang_shaders_compile() {
        let shaders_dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("shaders");
        let compiler = ShaderCompiler::new(shaders_dir).expect("ShaderCompiler::new failed");
        for module in ["temporal_accumulation", "denoise", "postprocess"] {
            let spirv = compiler
                .compile(module, "main")
                .unwrap_or_else(|e| panic!("compiling shaders/{module}.slang failed: {e}"));
            assert!(!spirv.is_empty(), "{module} produced empty SPIR-V");
            assert_eq!(spirv.len() % 4, 0, "{module} SPIR-V byte length not u32-aligned");
        }
    }

    /// Byte offsets of the push-constant block's members, in member order, parsed
    /// straight out of a SPIR-V module.
    ///
    /// Hand-rolled rather than pulling in a SPIR-V crate: this needs four opcodes
    /// and the whole walk is a couple of dozen lines.
    ///
    /// The block is identified via the `OpVariable`, not by scanning for a
    /// `PushConstant` pointer type — Slang emits one of those per member type it
    /// access-chains into (`_ptr_PushConstant_uint`, `_ptr_PushConstant_ulong`, …),
    /// so only the variable's own type points at the block struct. Decorations also
    /// precede types in the module, hence collect-then-filter rather than one pass.
    fn push_constant_offsets(spirv: &[u8]) -> Vec<u32> {
        const OP_TYPE_POINTER: u32 = 32;
        const OP_VARIABLE: u32 = 59;
        const OP_MEMBER_DECORATE: u32 = 72;
        const STORAGE_CLASS_PUSH_CONSTANT: u32 = 9;
        const DECORATION_OFFSET: u32 = 35;
        const HEADER_WORDS: usize = 5;

        assert_eq!(spirv.len() % 4, 0, "SPIR-V byte length not u32-aligned");
        let words: Vec<u32> = spirv
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        assert!(words.len() > HEADER_WORDS, "SPIR-V too short to hold anything");
        assert_eq!(words[0], 0x0723_0203, "not a little-endian SPIR-V module");

        // pointer type id -> pointee type id, for PushConstant pointers only
        let mut pc_pointees: std::collections::HashMap<u32, u32> = std::collections::HashMap::new();
        // result type id of the one PushConstant OpVariable
        let mut var_ptr_type: Option<u32> = None;
        // (struct type id, member index, byte offset)
        let mut decorated: Vec<(u32, u32, u32)> = Vec::new();

        let mut i = HEADER_WORDS;
        while i < words.len() {
            let opcode = words[i] & 0xFFFF;
            let len = (words[i] >> 16) as usize;
            assert!(len > 0, "malformed SPIR-V: zero-length instruction at word {i}");
            assert!(i + len <= words.len(), "malformed SPIR-V: instruction runs past end");

            if opcode == OP_TYPE_POINTER && len >= 4 && words[i + 2] == STORAGE_CLASS_PUSH_CONSTANT {
                pc_pointees.insert(words[i + 1], words[i + 3]);
            } else if opcode == OP_VARIABLE && len >= 4 && words[i + 3] == STORAGE_CLASS_PUSH_CONSTANT {
                let ty = words[i + 1];
                if let Some(prev) = var_ptr_type {
                    assert_eq!(prev, ty, "more than one push-constant variable");
                }
                var_ptr_type = Some(ty);
            } else if opcode == OP_MEMBER_DECORATE && len >= 5 && words[i + 3] == DECORATION_OFFSET {
                decorated.push((words[i + 1], words[i + 2], words[i + 4]));
            }
            i += len;
        }

        let var_ptr_type = var_ptr_type.expect("module declares no PushConstant variable");
        let block_type = *pc_pointees
            .get(&var_ptr_type)
            .expect("push-constant variable's pointer type was never declared");
        let mut members: Vec<(u32, u32)> = decorated
            .into_iter()
            .filter(|(ty, _, _)| *ty == block_type)
            .map(|(_, member, offset)| (member, offset))
            .collect();
        members.sort_unstable();
        members.into_iter().map(|(_, offset)| offset).collect()
    }

    /// The push-constant blocks are fed by `vkCmdPushDataEXT` as raw bytes against
    /// a `#[repr(C)]` Rust struct — nothing validates that the two agree, so a
    /// field added on one side and not the other is silent GPU garbage rather than
    /// an error. Pin every block's offsets to its Rust mirror.
    ///
    /// Reads the SPIR-V the build script baked into `OUT_DIR`, i.e. exactly the
    /// bytes the renderer ships.
    #[test]
    fn push_constant_layouts_match_rust() {
        use crate::vulkan_abstraction::{
            DenoiseHeapPushConstant, PostprocessPushConstant, RaytracingHeapPushConstant,
            TemporalAccumulationHeapPushConstant,
        };
        use std::mem::offset_of;

        let cases: [(&str, &[u8], Vec<u32>); 4] = [
            (
                "ray_gen_final",
                include_bytes!(concat!(env!("OUT_DIR"), "/ray_gen_final.spirv")),
                vec![
                    offset_of!(RaytracingHeapPushConstant, tlas) as u32,
                    offset_of!(RaytracingHeapPushConstant, raw_color) as u32,
                    offset_of!(RaytracingHeapPushConstant, depth_img) as u32,
                    offset_of!(RaytracingHeapPushConstant, normal_img) as u32,
                    offset_of!(RaytracingHeapPushConstant, diffuse_img) as u32,
                    offset_of!(RaytracingHeapPushConstant, motion_vec_img) as u32,
                    offset_of!(RaytracingHeapPushConstant, matrices) as u32,
                    offset_of!(RaytracingHeapPushConstant, meshes_info) as u32,
                    offset_of!(RaytracingHeapPushConstant, emissive_triangles) as u32,
                    offset_of!(RaytracingHeapPushConstant, emissive_indirection) as u32,
                    offset_of!(RaytracingHeapPushConstant, entity_transforms) as u32,
                    offset_of!(RaytracingHeapPushConstant, blue_noise_tex) as u32,
                    offset_of!(RaytracingHeapPushConstant, blue_noise_sampler) as u32,
                    offset_of!(RaytracingHeapPushConstant, reservoirs) as u32,
                    offset_of!(RaytracingHeapPushConstant, reservoirs_gi) as u32,
                    offset_of!(RaytracingHeapPushConstant, frame_count) as u32,
                    offset_of!(RaytracingHeapPushConstant, use_srgb) as u32,
                ],
            ),
            (
                "denoise",
                include_bytes!(concat!(env!("OUT_DIR"), "/denoise.spirv")),
                vec![
                    offset_of!(DenoiseHeapPushConstant, temporal_result) as u32,
                    offset_of!(DenoiseHeapPushConstant, depth) as u32,
                    offset_of!(DenoiseHeapPushConstant, normal) as u32,
                    offset_of!(DenoiseHeapPushConstant, diffuse) as u32,
                    offset_of!(DenoiseHeapPushConstant, spatial_output) as u32,
                    offset_of!(DenoiseHeapPushConstant, frame_count) as u32,
                    offset_of!(DenoiseHeapPushConstant, step_width) as u32,
                    offset_of!(DenoiseHeapPushConstant, width) as u32,
                    offset_of!(DenoiseHeapPushConstant, height) as u32,
                ],
            ),
            (
                "temporal_accumulation",
                include_bytes!(concat!(env!("OUT_DIR"), "/temporal_accumulation.spirv")),
                vec![
                    offset_of!(TemporalAccumulationHeapPushConstant, raw_rt_color) as u32,
                    offset_of!(TemporalAccumulationHeapPushConstant, motion_vector) as u32,
                    offset_of!(TemporalAccumulationHeapPushConstant, history) as u32,
                    offset_of!(TemporalAccumulationHeapPushConstant, accum_output) as u32,
                    offset_of!(TemporalAccumulationHeapPushConstant, frame_count) as u32,
                    offset_of!(TemporalAccumulationHeapPushConstant, width) as u32,
                    offset_of!(TemporalAccumulationHeapPushConstant, height) as u32,
                ],
            ),
            (
                "postprocess",
                include_bytes!(concat!(env!("OUT_DIR"), "/postprocess.spirv")),
                vec![
                    offset_of!(PostprocessPushConstant, input_idx) as u32,
                    offset_of!(PostprocessPushConstant, output_idx) as u32,
                    offset_of!(PostprocessPushConstant, exposure) as u32,
                ],
            ),
        ];

        for (name, spirv, expected) in cases {
            assert_eq!(
                push_constant_offsets(spirv),
                expected,
                "{name}.slang push-constant offsets diverged from its Rust mirror"
            );
        }

        // egui rides the `bevy` feature, so its struct only exists there.
        #[cfg(feature = "bevy")]
        {
            use crate::bevy_integration::egui_paint::EguiPushConstant;
            assert_eq!(
                push_constant_offsets(include_bytes!(concat!(env!("OUT_DIR"), "/egui_vert.spirv"))),
                vec![
                    offset_of!(EguiPushConstant, screen_size_points) as u32,
                    offset_of!(EguiPushConstant, tex) as u32,
                    offset_of!(EguiPushConstant, samp) as u32,
                ],
                "egui.slang push-constant offsets diverged from its Rust mirror"
            );
        }
    }
}
