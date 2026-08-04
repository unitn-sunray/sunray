use crate::error::{SrError, SrResult};
use crate::vulkan_abstraction::Buffer;
use crate::vulkan_abstraction::{GpuOnlyBuffer, HostAccessibleBuffer, RawBuffer, infer_read_masks_from_usage};
use crate::{impl_buffer_trait, vulkan_abstraction};
use ash::vk;
use ash::vk::DeviceSize;
use std::marker::PhantomData;
use std::sync::Arc;

pub struct StagingBuffer<T> {
    raw: RawBuffer,
    // `fn() -> T` rather than `T`: this is a type tag for a byte buffer, and the
    // plain form would make the buffer non-`Send` for any non-`Send` `T`.
    _marker: PhantomData<fn() -> T>,
}

impl_buffer_trait!(StagingBuffer<T>);

impl<T> StagingBuffer<T> {
    pub fn new_temp(core: Arc<vulkan_abstraction::Core>, len: vk::DeviceSize) -> SrResult<Self> {
        //TODO this gets used for new from data and it has no flags
        let byte_size = (len * std::mem::size_of::<T>() as vk::DeviceSize) as vk::DeviceSize;
        let raw = RawBuffer::new_aligned(
            core,
            byte_size,
            1,
            gpu_allocator::MemoryLocation::CpuToGpu,
            vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            "staging buffer",
        )?;
        Ok(Self {
            raw,
            _marker: PhantomData,
        })
    }

    pub fn new(
        core: Arc<vulkan_abstraction::Core>,
        len: vk::DeviceSize,
        buffer_usage_flags: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        Self::new_aligned(core, len, 1, buffer_usage_flags, name)
    }

    /// Like `new`, but guarantees the buffer's device address is a multiple of
    /// `alignment` (e.g. `shaderGroupBaseAlignment` for SBT buffers).
    pub fn new_aligned(
        core: Arc<vulkan_abstraction::Core>,
        len: vk::DeviceSize,
        alignment: u64,
        buffer_usage_flags: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        let byte_size = len * std::mem::size_of::<T>() as vk::DeviceSize;
        let raw = RawBuffer::new_aligned(
            core,
            byte_size,
            alignment,
            gpu_allocator::MemoryLocation::CpuToGpu,
            buffer_usage_flags,
            name,
        )?;
        Ok(Self {
            raw,
            _marker: PhantomData,
        })
    }

    pub fn new_temp_from_data(core: Arc<vulkan_abstraction::Core>, data: &[T]) -> SrResult<Self>
    where
        T: Copy,
    {
        if data.is_empty() {
            return Ok(Self::new_null(core));
        }

        let mut staging_buffer = Self::new_temp(core, data.len() as vk::DeviceSize)?;

        let mapped_memory = staging_buffer.map_mut()?;
        mapped_memory[0..data.len()].copy_from_slice(data);

        Ok(staging_buffer)
    }

    pub fn new_from_data(
        core: Arc<vulkan_abstraction::Core>,
        data: &[T],
        buffer_usage_flags: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self>
    where
        T: Copy,
    {
        if data.is_empty() {
            return Ok(Self::new_null(core));
        }

        let mut staging_buffer = Self::new(core, data.len() as vk::DeviceSize, buffer_usage_flags, name)?;

        let mapped_memory = staging_buffer.map_mut()?;
        mapped_memory[0..data.len()].copy_from_slice(data);

        Ok(staging_buffer)
    }

    pub fn new_from_data_with_custom_length(
        core: Arc<vulkan_abstraction::Core>,
        data: &[T],
        len: vk::DeviceSize,
        buffer_usage_flags: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self>
    where
        T: Copy,
    {
        if len < (data.len() as vk::DeviceSize) {
            return Err(SrError::new_custom(format!(
                "attempted to create an insufficiently sized buffer, src : {} bytes , dst : {len} bytes",
                data.len()
            )));
        }

        let mut staging_buffer = Self::new(core, len, buffer_usage_flags, name)?;

        let mapped_memory = staging_buffer.map_mut()?;
        mapped_memory[0..data.len()].copy_from_slice(data);

        Ok(staging_buffer)
    }

    pub fn new_cloned_to_gpu_only_buffer(&self, usage: vk::BufferUsageFlags, name: &'static str) -> SrResult<GpuOnlyBuffer> {
        let mut dst = GpuOnlyBuffer::new::<T>(self.raw.core.clone(), self.len() as vk::DeviceSize, usage, name)?;
        self.clone_section_into_gpu_only_buffer(0, self.byte_size(), &mut dst)?;
        Ok(dst)
    }

    pub fn clone_section_into_gpu_only_buffer(
        &self,
        src_offset: DeviceSize,
        src_section_length: DeviceSize,
        dst: &mut GpuOnlyBuffer,
    ) -> SrResult<()> {
        if self.is_null() {
            return Ok(());
        }
        if dst.is_null() {
            return Err(SrError::new_custom(
                "attempted to clone from a non-null buffer to a null buffer".to_string(),
            ));
        }

        if self.byte_size() < src_section_length + src_offset {
            return Err(SrError::new_custom(format!(
                "attempted to clone from outside the src buffer, src : {src_section_length} bytes , dst : {} bytes",
                dst.byte_size()
            )));
        }
        if dst.byte_size() < src_section_length {
            return Err(SrError::new_custom(format!(
                "attempted to clone into an insufficiently sized buffer, src : {src_section_length} bytes , dst : {} bytes",
                dst.byte_size()
            )));
        }

        let device = self.raw.core.device().inner();
        let (dst_stage_mask, dst_access_mask) = infer_read_masks_from_usage(dst.usage());

        // ponytail: upload on the universal queue — no dedicated transfer queue, so no
        // queue-family ownership transfer. Move back to the transfer queue only if
        // uploads measurably contend with graphics work.
        let cmd = vulkan_abstraction::cmd_buffer::new_command_buffer(self.raw.core.graphics_cmd_pool(), device)?;

        let begin_info = vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
        unsafe { device.begin_command_buffer(cmd, &begin_info) }?;

        let regions = [vk::BufferCopy::default()
            .size(src_section_length)
            .src_offset(src_offset)
            .dst_offset(0)];
        unsafe { device.cmd_copy_buffer(cmd, self.inner(), dst.inner(), &regions) };

        let barrier = vk::BufferMemoryBarrier2::default()
            .src_stage_mask(vk::PipelineStageFlags2::TRANSFER)
            .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
            .dst_stage_mask(dst_stage_mask)
            .dst_access_mask(dst_access_mask)
            .src_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .dst_queue_family_index(vk::QUEUE_FAMILY_IGNORED)
            .buffer(dst.inner())
            .offset(0)
            .size(src_section_length);
        let dep = vk::DependencyInfo::default().buffer_memory_barriers(std::slice::from_ref(&barrier));
        unsafe { device.cmd_pipeline_barrier2(cmd, &dep) };

        unsafe { device.end_command_buffer(cmd) }?;
        self.raw.core.graphics_queue().submit_sync(cmd)?;
        unsafe { device.free_command_buffers(self.raw.core.graphics_cmd_pool().inner(), &[cmd]) };

        Ok(())
    }

    pub fn clone_into_gpu_only_buffer(&self, dst: &mut GpuOnlyBuffer) -> SrResult<()> {
        self.clone_section_into_gpu_only_buffer(0, self.byte_size(), dst)?;

        Ok(())
    }
}

impl<T> HostAccessibleBuffer<T> for StagingBuffer<T> {
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
