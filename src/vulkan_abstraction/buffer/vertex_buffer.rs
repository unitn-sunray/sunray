use std::{ops::Deref, sync::Arc};

use ash::vk;

use crate::{error::*, vulkan_abstraction};

pub struct VertexBuffer {
    buffer: vulkan_abstraction::GpuOnlyBuffer,
    len: usize,
    stride: usize,
}

impl VertexBuffer {
    //build a vertex buffer with flags for usage in a blas
    pub fn new_for_blas_from_data<T: Copy>(core: Arc<vulkan_abstraction::Core>, data: &[T]) -> SrResult<Self> {
        let usage_flags = vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;

        Ok(Self {
            buffer: vulkan_abstraction::GpuOnlyBuffer::new_from_data(core, data, usage_flags, "vertex buffer for BLAS usage")?,
            len: data.len(),
            stride: std::mem::size_of::<T>(),
        })
    }

    //build a vertex buffer with flags for usage in a blas
    pub fn new_for_blas<T>(core: Arc<vulkan_abstraction::Core>, len: vk::DeviceSize) -> SrResult<Self> {
        let usage_flags = vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::VERTEX_BUFFER
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;

        Ok(Self {
            buffer: vulkan_abstraction::GpuOnlyBuffer::new::<T>(core, len, usage_flags, "vertex buffer for BLAS usage")?,
            len: len as usize,
            stride: std::mem::size_of::<T>(),
        })
    }

    #[allow(dead_code)]
    pub fn buffer(&self) -> &vulkan_abstraction::GpuOnlyBuffer {
        &self.buffer
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn stride(&self) -> usize {
        self.stride
    }
}
impl Deref for VertexBuffer {
    type Target = vulkan_abstraction::GpuOnlyBuffer;
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}
