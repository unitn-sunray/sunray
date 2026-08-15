use crate::vulkan_abstraction::Material;

/// Per-BLAS data uploaded to GPU and read by shaders. Stored in the meshes-info
/// arena buffer; the slot index is what every instance of that BLAS passes as
/// `gl_InstanceCustomIndexEXT`, so instances sharing a BLAS share one entry.
///
/// Mirrors `shaders/rt_types.slang::MeshInfo` byte for byte — 128 bytes, with
/// `material` at offset 16. `_pad` is load-bearing: `Material` opens with a
/// `float4`, which std430 puts on a 16-byte boundary, so the two slots cannot be
/// followed directly by it.
#[derive(Clone, Copy)]
#[repr(C, packed)]
pub(crate) struct EntityGpuData {
    /// Descriptor-heap slot of the BLAS vertex buffer, read in the shader as a
    /// `StructuredBuffer<VertexAttributes>`. A heap slot rather than a device
    /// address because Slang tags every load through a `T*` (buffer-device-address)
    /// pointer `Aligned 4`, which blocks the driver from widening a `float3`/`float4`
    /// fetch into a single 128-bit load — measurably worse in closest-hit/any-hit.
    pub(crate) vertex_buffer: u32,
    /// Heap slot of the BLAS index buffer, read as a `StructuredBuffer<uint>`.
    pub(crate) index_buffer: u32,
    pub(crate) _pad: [u32; 2],
    pub(crate) material: Material,
}

// The shader reads this through a heap `StructuredBuffer<MeshInfo>` whose array
// stride must be 128; nothing at runtime validates that, so pin it here.
const _: () = assert!(
    size_of::<EntityGpuData>() == 128,
    "EntityGpuData must stay in lockstep with MeshInfo in shaders/rt_types.slang"
);
const _: () = assert!(
    std::mem::offset_of!(EntityGpuData, material) == 16,
    "Material must sit at offset 16 — std430 puts its leading float4 on a 16-byte boundary"
);
