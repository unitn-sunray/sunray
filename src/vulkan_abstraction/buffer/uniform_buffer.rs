use crate::error::SrResult;
use crate::vulkan_abstraction::Buffer;
use crate::vulkan_abstraction::{HostAccessibleBuffer, RawBuffer};
use crate::{impl_buffer_trait, vulkan_abstraction};
use ash::vk;
use std::marker::PhantomData;
use std::sync::Arc;

pub struct UniformBuffer<T> {
    raw: RawBuffer,
    // `fn() -> T` rather than `T`: this is a type tag for a byte buffer, and the
    // plain form would make the buffer non-`Send` for any non-`Send` `T`.
    _marker: PhantomData<fn() -> T>,
}
impl_buffer_trait!(UniformBuffer<T>);

impl<T> UniformBuffer<T> {
    pub fn new(core: Arc<vulkan_abstraction::Core>, len: vk::DeviceSize) -> SrResult<Self> {
        let byte_size = len * std::mem::size_of::<T>() as vk::DeviceSize;
        let raw = RawBuffer::new_aligned(
            core,
            byte_size,
            1,
            gpu_allocator::MemoryLocation::CpuToGpu,
            // SHADER_DEVICE_ADDRESS so the Slang RT shaders can read matrices via
            // a `Matrices*` buffer-device-address pointer — needed because Slang's
            // heap-descriptor lowering emits invalid OpAccessChain storage classes
            // when reading struct members of a `StructuredBuffer<T>` element on
            // this Slang version (see `Matrices` doc in `shaders/rt_types.slang`).
            // STORAGE_BUFFER is kept for backwards-compatibility with any legacy
            // descriptor-set path.
            vk::BufferUsageFlags::UNIFORM_BUFFER
                | vk::BufferUsageFlags::STORAGE_BUFFER
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
            "uniform buffer",
        )?;
        Ok(Self {
            raw,
            _marker: PhantomData,
        })
    }
}

impl<T> HostAccessibleBuffer<T> for UniformBuffer<T> {
    fn map_mut(&mut self) -> SrResult<&mut [T]> {
        self.raw.map_mut::<T>()
    }

    fn map(&self) -> SrResult<&[T]> {
        self.raw.map::<T>()
    }

    fn len(&self) -> usize {
        (self.raw.byte_size as usize) / std::mem::size_of::<T>()
    }
}
