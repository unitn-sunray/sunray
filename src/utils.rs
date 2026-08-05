#![macro_use]

use ash::vk;
use std::path::PathBuf;

// ---------------------------------------------------------------------------
// Environment variables
//
// Every runtime knob the library reads lives here, so the list is greppable in
// one place. All are debug/diagnostic toggles — anything that changes rendering
// behaviour belongs on the `Renderer` API instead. Documented in README.md.
//
// `SUNRAY_SHADER_DEBUG` is *build-time* only and lives in `build.rs`; it can't
// share these constants because the build script is a separate crate.
// ---------------------------------------------------------------------------

/// Vulkan validation layer. Defaults to on in debug builds, off in release.
pub(crate) const ENABLE_VALIDATION_LAYER: &str = "SUNRAY_ENABLE_VALIDATION_LAYER";
/// GPU-assisted validation. Defaults to off; does nothing unless the validation layer is on.
pub(crate) const ENABLE_GPUAV: &str = "SUNRAY_ENABLE_GPUAV";
/// Nsight Graphics capture aid: debug-utils labels + object names. Defaults to off.
/// Takes precedence over [`ENABLE_NVIDIA_AFTERMATH`].
pub(crate) const ENABLE_NSIGHT: &str = "SUNRAY_ENABLE_NSIGHT";
/// NVIDIA Aftermath crash dumps. Defaults to off.
pub(crate) const ENABLE_NVIDIA_AFTERMATH: &str = "SUNRAY_ENABLE_NVIDIA_AFTERMATH";
/// Whole-frame serialization. Defaults to **on**; set to 0 to opt into frame
/// overlap (has a known async-UAF driver crash — see `Renderer::render`).
pub(crate) const SERIALIZE_FRAMES: &str = "SUNRAY_SERIALIZE_FRAMES";
/// Per-frame render-graph dump (DOT + text). Unset = off. A boolean-true value
/// (`1`/`true`/`on`) dumps into [`DEFAULT_GRAPH_DUMP_DIR`]; anything else is
/// taken as the destination directory. See [`graph_dump_dir`].
pub(crate) const GRAPH_DUMP_DIR: &str = "SUNRAY_GRAPH_DUMP_DIR";
/// Transient memory aliasing algorithm: `slot` (default) or `bucket`. Changes how
/// much VRAM the render graph's transient resources take, never what is rendered.
/// Parsed by [`AliasStrategy::from_env`](crate::render_graph::alias::AliasStrategy::from_env).
pub(crate) const ALIAS_STRATEGY: &str = "SUNRAY_ALIAS_STRATEGY";
/// Diagnostic bisect: keep only the first N stages of the unified graph, so a frame
/// is still acquired, blitted and presented but does progressively less GPU work.
/// Unset = the whole pipeline. See [`strip_stages`] for the ladder.
///
/// ponytail: TEMPORARY — scaffolding for the NVIDIA driver crash only, delete once
///           the driver bug in `docs/NVIDIA_BUG_REPORT.md` is fixed. Removal is
///           `STRIP_STAGES`, [`strip_stages`], the `stop_after!` macro and its six
///           call sites in `Renderer::build_unified_graph`, and the README row.
///           Nothing in the renderer depends on it: unset means "build everything".
pub(crate) const STRIP_STAGES: &str = "SUNRAY_STRIP";
/// Swapchain present mode override: `fifo`, `fifo_relaxed`, `mailbox`, `immediate`.
/// Unset = the lowest-latency supported mode. Diagnostic knob — the NVIDIA driver
/// crash in `docs/NVIDIA_DRIVER_CRASH_REPORT.md` involves the present worker thread,
/// and the documented soak-stable runs were on `fifo`. See [`present_mode`].
pub(crate) const PRESENT_MODE: &str = "SUNRAY_PRESENT_MODE";

/// Where graph dumps land when the var is on but names no directory: `<crate>/debug`
/// (git-ignored).
pub(crate) const DEFAULT_GRAPH_DUMP_DIR: &str = "debug";

pub(crate) const IS_DEBUG_BUILD: bool = cfg!(debug_assertions);

/// Parse an env var as a boolean. Accepts `1`/`0`, `true`/`false`, `on`/`off`
/// (case-insensitive). `None` if unset, empty, or unrecognized — callers then
/// use their own default rather than silently reading an off.
pub(crate) fn env_var_as_bool(name: &str) -> Option<bool> {
    match std::env::var(name).ok()?.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "on" | "yes" => Some(true),
        "0" | "false" | "off" | "no" => Some(false),
        other => {
            log::warn!("{name}: unrecognized value {other:?} (expected 1/0, true/false, on/off) — ignoring");
            None
        }
    }
}

/// Destination directory for per-frame graph dumps, or `None` when dumping is off.
///
/// `SUNRAY_GRAPH_DUMP_DIR=1` (or `true`/`on`) dumps into `<crate>/debug`; any
/// other non-empty value is used verbatim as a path.
pub(crate) fn graph_dump_dir() -> Option<PathBuf> {
    let raw = std::env::var(GRAPH_DUMP_DIR).ok()?;
    match raw.trim().to_ascii_lowercase().as_str() {
        "" | "0" | "false" | "off" | "no" => None,
        "1" | "true" | "on" | "yes" => Some(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(DEFAULT_GRAPH_DUMP_DIR)),
        _ => Some(PathBuf::from(raw)),
    }
}

/// How many stages of the unified graph to build, from the front:
///
/// | value | graph contains |
/// |---|---|
/// | 0 | nothing — acquire, blit the (unwritten) output, present |
/// | 1 | + staging copies and the BLAS / TLAS builds |
/// | 2 | + the RIS ray-tracing pass |
/// | 3 | + the final-shading ray-tracing pass |
/// | 4 | + temporal accumulation |
/// | 5 | + the a-trous denoise passes |
/// | unset / ≥6 | + postprocess, i.e. the real frame |
///
/// A ladder rather than an on/off flag because the point is to find the lowest rung
/// that still reproduces the driver crash in `docs/NVIDIA_DRIVER_CRASH_REPORT.md`.
/// Levels below 6 render garbage by construction — the blit source is never written,
/// which is also why they trip `VUID-VkImageMemoryBarrier2-newLayout-01198`. Only
/// rung 6 (the real frame) is validation-clean.
///
/// ponytail: TEMPORARY — see [`STRIP_STAGES`] for what to delete when the driver
///           bug is fixed.
pub(crate) fn strip_stages() -> usize {
    let Ok(raw) = std::env::var(STRIP_STAGES) else {
        return usize::MAX;
    };
    match raw.trim().parse::<usize>() {
        Ok(n) => n,
        Err(_) if raw.trim().is_empty() => usize::MAX,
        Err(_) => {
            log::warn!("{STRIP_STAGES}: expected a non-negative integer, got {raw:?} — building the full graph");
            usize::MAX
        }
    }
}

/// [`PRESENT_MODE`] override, or `None` to let the swapchain pick.
pub(crate) fn present_mode() -> Option<vk::PresentModeKHR> {
    let raw = std::env::var(PRESENT_MODE).ok()?;
    match raw.trim().to_ascii_lowercase().as_str() {
        "" => None,
        "fifo" => Some(vk::PresentModeKHR::FIFO),
        "fifo_relaxed" => Some(vk::PresentModeKHR::FIFO_RELAXED),
        "mailbox" => Some(vk::PresentModeKHR::MAILBOX),
        "immediate" => Some(vk::PresentModeKHR::IMMEDIATE),
        other => {
            log::warn!("{PRESENT_MODE}: unrecognized value {other:?} (expected fifo/fifo_relaxed/mailbox/immediate) — ignoring");
            None
        }
    }
}

pub(crate) fn tuple_to_extent2d((width, height): (u32, u32)) -> ash::vk::Extent2D {
    ash::vk::Extent2D { width, height }
}

pub(crate) fn tuple_to_extent3d(tuple: (u32, u32)) -> ash::vk::Extent3D {
    tuple_to_extent2d(tuple).into()
}

pub(crate) fn realign_data(bytes: &[u8], starting_alignment: usize, target_alignment: usize) -> Vec<u8> {
    let mut i = 0;
    let mut ret = Vec::new();

    while bytes.len() >= (i + 1) * starting_alignment {
        for j in 0..starting_alignment.min(target_alignment) {
            ret.push(bytes[i * starting_alignment + j]);
        }
        ret.extend(std::iter::repeat_n(0x00, target_alignment.saturating_sub(starting_alignment)));

        i += 1;
    }

    ret
}

#[repr(C)] // guarantee 'bytes' comes after '_align'
pub struct AlignedAs<Align, Bytes: ?Sized> {
    pub _align: [Align; 0],
    pub bytes: Bytes,
}

#[macro_export]
macro_rules! include_bytes_align_as {
    ($align_ty:ty, $path:expr) => {{
        // const block expression to encapsulate the static
        use $crate::utils::AlignedAs;

        // this assignment is made possible by CoerceUnsized
        static ALIGNED: &AlignedAs<$align_ty, [u8]> = &AlignedAs {
            _align: [],
            bytes: *include_bytes!($path),
        };

        &ALIGNED.bytes
    }};
}

pub fn na_mat4_to_vk_transform(m: nalgebra::Matrix4<f32>) -> vk::TransformMatrixKHR {
    // VkTransformMatrixKHR is a row-major 3x4 affine, flattened to [f32; 12].
    vk::TransformMatrixKHR {
        matrix: [
            m.m11, m.m12, m.m13, m.m14, m.m21, m.m22, m.m23, m.m24, m.m31, m.m32, m.m33, m.m34,
        ],
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bool_and_dump_dir_parsing() {
        let set = |v: &str| unsafe { std::env::set_var("SUNRAY_TEST_VAR", v) };

        for on in ["1", "true", "TRUE", "On", " yes "] {
            set(on);
            assert_eq!(env_var_as_bool("SUNRAY_TEST_VAR"), Some(true), "{on}");
        }
        for off in ["0", "false", "OFF", "no"] {
            set(off);
            assert_eq!(env_var_as_bool("SUNRAY_TEST_VAR"), Some(false), "{off}");
        }
        set("banana");
        assert_eq!(env_var_as_bool("SUNRAY_TEST_VAR"), None);
        assert_eq!(env_var_as_bool("SUNRAY_DEFINITELY_UNSET_VAR"), None);

        // A bool-true value means "the default directory", anything else is a path.
        let dump = |v: &str| unsafe { std::env::set_var(GRAPH_DUMP_DIR, v) };
        dump("1");
        assert!(graph_dump_dir().unwrap().ends_with(DEFAULT_GRAPH_DUMP_DIR));
        dump("some/where");
        assert_eq!(graph_dump_dir(), Some(PathBuf::from("some/where")));
        dump("0");
        assert_eq!(graph_dump_dir(), None);
        unsafe { std::env::remove_var(GRAPH_DUMP_DIR) };
        assert_eq!(graph_dump_dir(), None);
    }
}
