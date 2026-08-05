use ash::vk;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum HeapKind {
    //TODO could be removed, we know where they go based on what they are [ResourceDescriptorKind]
    Resource,
    Sampler,
}

/// What kind of resource descriptor a slot holds. The heap splits the resource
/// area into three contiguous sections by [`ResourceSection`]; each kind routes
/// to exactly one section.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ResourceDescriptorKind {
    SampledImage,
    StorageImage,
    UniformTexelBuffer,
    StorageTexelBuffer,
    UniformBuffer,
    StorageBuffer,
    AccelerationStructure,
}

impl ResourceDescriptorKind {
    pub fn descriptor_type(self) -> vk::DescriptorType {
        match self {
            Self::SampledImage => vk::DescriptorType::SAMPLED_IMAGE,
            Self::StorageImage => vk::DescriptorType::STORAGE_IMAGE,
            Self::UniformTexelBuffer => vk::DescriptorType::UNIFORM_TEXEL_BUFFER,
            Self::StorageTexelBuffer => vk::DescriptorType::STORAGE_TEXEL_BUFFER,
            Self::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
            Self::StorageBuffer => vk::DescriptorType::STORAGE_BUFFER,
            Self::AccelerationStructure => vk::DescriptorType::ACCELERATION_STRUCTURE_KHR,
        }
    }

    /// Which section of the resource heap this kind lives in.
    pub fn section(self) -> ResourceSection {
        match self {
            Self::SampledImage | Self::StorageImage => ResourceSection::Image,
            Self::UniformTexelBuffer | Self::StorageTexelBuffer => ResourceSection::TexelBuffer,
            Self::UniformBuffer | Self::StorageBuffer | Self::AccelerationStructure => ResourceSection::Buffer,
        }
    }
}

/// The three contiguous sections of the resource heap. Images and texel buffers
/// share the image descriptor stride (texel buffer descriptors are the same
/// size as image descriptors on this extension); buffers use the buffer stride.
/// Splitting into fixed sections removes the per-descriptor page lookup the GPU
/// had to do with the old paged layout — the shader just adds a static base
/// index for its section.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum ResourceSection {
    Image,
    TexelBuffer,
    Buffer,
}

/// Index of a single descriptor inside a heap. Not RAII — the owning resource
/// must call [`super::DescriptorHeap::free`] in its Drop impl.
#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub struct DescriptorSlot {
    pub kind: HeapKind,
    /// `ResourceDescriptorHeap[index]` / `SamplerDescriptorHeap[index]` in the shader.
    /// For resource slots this is `byte_offset / type_descriptor_size`, i.e.
    /// `section_base_index + slot_in_section`.
    pub index: u32,
    /// Section this slot belongs to; needed by `free` to return the local index
    /// to the right allocator. Ignored for samplers.
    pub section: ResourceSection,
}

impl DescriptorSlot {
    pub fn shader_index(self) -> u32 {
        self.index
    }
}

/// Uniform-stride bump + free-list allocator. One instance per resource section,
/// plus one for the sampler heap.
#[derive(Debug)]
pub(crate) struct SlotAllocator {
    capacity: u32,
    high_water: u32,
    free_list: Vec<u32>,
}

impl SlotAllocator {
    pub fn new(capacity: u32) -> Self {
        Self {
            capacity,
            high_water: 0,
            free_list: Vec::new(),
        }
    }

    pub fn alloc(&mut self) -> Option<u32> {
        if let Some(i) = self.free_list.pop() {
            return Some(i);
        }
        if self.high_water >= self.capacity {
            return None;
        }
        let i = self.high_water;
        self.high_water += 1;
        Some(i)
    }

    pub fn free(&mut self, index: u32) {
        debug_assert!(index < self.high_water);
        // A double free hands the same index to two live resources, whose descriptors
        // then clobber each other in the heap — silent, and a nightmare to trace back
        // from the garbage the shader reads. O(n) scan, debug builds only.
        debug_assert!(
            !self.free_list.contains(&index),
            "descriptor slot {index} freed twice: it is already on the free list"
        );
        self.free_list.push(index);
    }
}
