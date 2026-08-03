use std::sync::Arc;

use ash::vk;

use crate::render_graph::resource::ResourceDesc;
use crate::vulkan_abstraction::descriptor_heap::DescriptorSlot;
use crate::{error::SrResult, vulkan_abstraction};

/// Plain-data copy of the sampler parameters. Under the descriptor-heap model the sampler
/// is materialised by `vkWriteSamplerDescriptorsEXT` from a `SamplerCreateInfo` directly,
/// so there is no `vkCreateSampler` / `vkDestroySampler` round-trip and no `vk::Sampler`
/// handle to track — these params are the entire sampler.
#[derive(Copy, Clone)]
struct SamplerParams {
    min_filter: vk::Filter,
    mag_filter: vk::Filter,
    address_mode_u: vk::SamplerAddressMode,
    address_mode_v: vk::SamplerAddressMode,
    address_mode_w: vk::SamplerAddressMode,
    mipmap_mode: vk::SamplerMipmapMode,
    max_anisotropy: f32,
}

pub struct Sampler {
    core: Arc<vulkan_abstraction::Core>,
    params: SamplerParams,
    slot: DescriptorSlot,
}

impl Sampler {
    /// Construct a sampler from a render-graph descriptor.
    pub fn new_from_desc(core: Arc<vulkan_abstraction::Core>, desc: &SamplerDesc) -> SrResult<Self> {
        Self::new(
            core,
            desc.min_filter,
            desc.mag_filter,
            desc.address_mode_u,
            desc.address_mode_v,
            desc.address_mode_w,
            desc.mipmap_mode,
        )
    }

    pub fn new(
        core: Arc<vulkan_abstraction::Core>,
        min_filter: vk::Filter,
        mag_filter: vk::Filter,
        address_mode_u: vk::SamplerAddressMode,
        address_mode_v: vk::SamplerAddressMode,
        address_mode_w: vk::SamplerAddressMode,
        mipmap_mode: vk::SamplerMipmapMode,
    ) -> SrResult<Self> {
        let params = SamplerParams {
            min_filter,
            mag_filter,
            address_mode_u,
            address_mode_v,
            address_mode_w,
            mipmap_mode,
            max_anisotropy: core.device().properties().limits.max_sampler_anisotropy,
        };
        let slot = core.descriptor_heap_mut().alloc_sampler_slot();
        core.descriptor_heap_mut()
            .write_sampler(slot, &params.create_info())
            .expect("descriptor heap write_sampler failed");

        Ok(Self { core, params, slot })
    }

    /// Heap slot for this sampler in the sampler heap. Allocated and written on first call.
    pub fn slot(&self) -> u32 {
        self.slot.shader_index()
    }
}

impl SamplerParams {
    fn create_info(&self) -> vk::SamplerCreateInfo<'static> {
        vk::SamplerCreateInfo::default()
            .flags(vk::SamplerCreateFlags::empty())
            .min_filter(self.min_filter)
            .mag_filter(self.mag_filter)
            .address_mode_u(self.address_mode_u)
            .address_mode_v(self.address_mode_v)
            .address_mode_w(self.address_mode_w)
            .anisotropy_enable(true)
            .max_anisotropy(self.max_anisotropy)
            .unnormalized_coordinates(false)
            .compare_enable(false)
            .compare_op(vk::CompareOp::ALWAYS)
            .mipmap_mode(self.mipmap_mode)
            .mip_lod_bias(0.0)
            .min_lod(0.0)
            .max_lod(0.0)
    }
}

impl Drop for Sampler {
    fn drop(&mut self) {
        self.core.descriptor_heap_mut().free(self.slot);
    }
}

impl crate::render_graph::graph::Resource for Sampler {
    type Desc = SamplerDesc;

    fn borrow_resource(res: &crate::render_graph::resource::AnyRenderResource) -> &Self {
        match res {
            crate::render_graph::resource::AnyRenderResource::OwnedSampler(s) => s,
            crate::render_graph::resource::AnyRenderResource::ImportedSampler(arc) => arc.as_ref(),
            _ => panic!("borrow_resource::<Sampler> called on non-sampler AnyRenderResource variant"),
        }
    }
}

impl crate::render_graph::graph::RgImportable<SamplerDesc> for std::sync::Arc<Sampler> {
    fn import(&self) -> SamplerDesc {
        SamplerDesc {
            min_filter: self.params.min_filter,
            mag_filter: self.params.mag_filter,
            address_mode_u: self.params.address_mode_u,
            address_mode_v: self.params.address_mode_v,
            address_mode_w: self.params.address_mode_w,
            mipmap_mode: self.params.mipmap_mode,
        }
    }
}

impl From<std::sync::Arc<Sampler>> for crate::render_graph::graph::GraphResourceImportInfo {
    fn from(val: std::sync::Arc<Sampler>) -> Self {
        crate::render_graph::graph::GraphResourceImportInfo::Sampler { resource: val }
    }
}

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct SamplerDesc {
    pub min_filter: vk::Filter,
    pub mag_filter: vk::Filter,
    pub address_mode_u: vk::SamplerAddressMode,
    pub address_mode_v: vk::SamplerAddressMode,
    pub address_mode_w: vk::SamplerAddressMode,
    pub mipmap_mode: vk::SamplerMipmapMode,
}

impl ResourceDesc for SamplerDesc {
    type Resource = Sampler;
}
