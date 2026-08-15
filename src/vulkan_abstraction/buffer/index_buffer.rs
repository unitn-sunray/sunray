use std::{any::TypeId, ops::Deref, sync::Arc};

use ash::vk;

use crate::vulkan_abstraction::GpuOnlyBuffer;
use crate::{error::*, vulkan_abstraction};

//TODO reworked into a gpubuffer method with custom flags basically since the type can be inferred from the generic
pub struct IndexBuffer {
    buffer: GpuOnlyBuffer,
    len: usize,
    idx_type: vk::IndexType,
}
impl IndexBuffer {
    //build an index buffer with flags for usage in a blas
    pub fn new_for_blas_from_data<T>(core: Arc<vulkan_abstraction::Core>, data: &[T]) -> SrResult<Self>
    where
        T: 'static + Copy,
    {
        let usage_flags = vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;

        let idx_type = match get_index_type::<T>() {
            Some(idx_type) => idx_type,
            None => {
                return Err(SrError::new_custom(
                    "attempting to construct IndexBuffer from invalid type".to_string(),
                ));
            }
        };

        let buffer = GpuOnlyBuffer::new_from_data(core, data, usage_flags, "index buffer for BLAS usage")?;

        Ok(Self {
            buffer,
            len: data.len(),
            idx_type,
        })
    }
    pub fn new_for_blas<T>(core: Arc<vulkan_abstraction::Core>, len: vk::DeviceSize) -> SrResult<Self>
    where
        T: 'static,
    {
        let usage_flags = vk::BufferUsageFlags::TRANSFER_DST
            | vk::BufferUsageFlags::INDEX_BUFFER
            | vk::BufferUsageFlags::STORAGE_BUFFER
            | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
            | vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR;

        let idx_type = match get_index_type::<T>() {
            Some(idx_type) => idx_type,
            None => {
                return Err(SrError::new_custom(
                    "attempting to construct IndexBuffer from invalid type".to_string(),
                ));
            }
        };

        let buffer = GpuOnlyBuffer::new::<T>(core, len, usage_flags, "index buffer for BLAS usage")?;

        Ok(Self {
            buffer,
            len: len as usize,
            idx_type,
        })
    }
    #[allow(dead_code)]
    pub fn buffer(&self) -> &GpuOnlyBuffer {
        &self.buffer
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    pub fn index_type(&self) -> vk::IndexType {
        self.idx_type
    }
}
impl Deref for IndexBuffer {
    type Target = GpuOnlyBuffer;
    fn deref(&self) -> &Self::Target {
        &self.buffer
    }
}

fn get_index_type<T: 'static>() -> Option<vk::IndexType> {
    let idx_type = if TypeId::of::<T>() == TypeId::of::<u32>() {
        vk::IndexType::UINT32
    } else if TypeId::of::<T>() == TypeId::of::<u16>() {
        vk::IndexType::UINT16
    } else if TypeId::of::<T>() == TypeId::of::<u8>() {
        assert_eq!(vk::IndexType::UINT8_KHR, vk::IndexType::UINT8_EXT);
        vk::IndexType::UINT8_KHR
    } else {
        return None;
    };

    Some(idx_type)
}
