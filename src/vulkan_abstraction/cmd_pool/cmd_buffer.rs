use std::sync::Arc;

use crate::{error::*, vulkan_abstraction};
use ash::vk;

// Device::free_command_buffers must be called on vk::CommandBuffer before it is dropped
pub fn new_command_buffer_vec(
    cmd_pool: &vulkan_abstraction::CmdPool,
    device: &ash::Device,
    len: usize,
) -> SrResult<Vec<vk::CommandBuffer>> {
    new_command_buffer_vec_impl(cmd_pool, device, vk::CommandBufferLevel::PRIMARY, len)
}
pub fn new_command_buffer_vec_secondary(
    cmd_pool: &vulkan_abstraction::CmdPool,
    device: &ash::Device,
    len: usize,
) -> SrResult<Vec<vk::CommandBuffer>> {
    new_command_buffer_vec_impl(cmd_pool, device, vk::CommandBufferLevel::SECONDARY, len)
}
pub fn new_command_buffer(cmd_pool: &vulkan_abstraction::CmdPool, device: &ash::Device) -> SrResult<vk::CommandBuffer> {
    let v = new_command_buffer_vec_impl(cmd_pool, device, vk::CommandBufferLevel::PRIMARY, 1)?;
    v.into_iter()
        .next()
        .ok_or_else(|| SrError::new_custom("Error in new_command_buffer".to_string()))
}
pub fn new_command_buffer_secondary(cmd_pool: &vulkan_abstraction::CmdPool, device: &ash::Device) -> SrResult<vk::CommandBuffer> {
    let v = new_command_buffer_vec_impl(cmd_pool, device, vk::CommandBufferLevel::SECONDARY, 1)?;
    v.into_iter()
        .next()
        .ok_or_else(|| SrError::new_custom("Error in new_command_buffer_secondary".to_string()))
}

fn new_command_buffer_vec_impl(
    cmd_pool: &vulkan_abstraction::CmdPool,
    device: &ash::Device,
    level: vk::CommandBufferLevel,
    len: usize,
) -> SrResult<Vec<vk::CommandBuffer>> {
    let info = vk::CommandBufferAllocateInfo::default()
        .command_pool(cmd_pool.inner())
        .level(level)
        .command_buffer_count(len as u32);

    let ret = unsafe { device.allocate_command_buffers(&info) }?;

    Ok(ret)
}

// this is for now assumed to be a non-ONE_TIME_SUBMIT command buffer
pub struct CmdBuffer {
    handle: vk::CommandBuffer,
    fence: vulkan_abstraction::Fence,
    core: Arc<vulkan_abstraction::Core>,
}

impl CmdBuffer {
    pub fn new(core: Arc<vulkan_abstraction::Core>) -> SrResult<Self> {
        let handle = vulkan_abstraction::cmd_buffer::new_command_buffer(core.graphics_cmd_pool(), core.device().inner())?;
        let fence = vulkan_abstraction::Fence::new_signaled(Arc::clone(core.device()))?;
        Ok(Self { core, handle, fence })
    }
    pub fn inner(&self) -> vk::CommandBuffer {
        self.handle
    }

    pub fn fence(&self) -> &vulkan_abstraction::Fence {
        &self.fence
    }
    pub fn fence_mut(&mut self) -> &mut vulkan_abstraction::Fence {
        &mut self.fence
    }
}

// Set once we've dumped diagnostic state for a DEVICE_LOST, so we don't spam
// the log on every fence/cmd-buffer drop during shutdown.
static DEVICE_LOST_REPORTED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

impl Drop for CmdBuffer {
    fn drop(&mut self) {
        unsafe {
            match self.fence.wait() {
                Ok(()) => {}
                Err(e) => match e.get_source() {
                    ErrorSource::Vulkan(vk_err) => {
                        log::warn!("VkWaitForFences returned {vk_err:?} in CmdBuffer::drop");
                        if *vk_err == ash::vk::Result::ERROR_DEVICE_LOST
                            && !DEVICE_LOST_REPORTED.swap(true, std::sync::atomic::Ordering::SeqCst)
                        {
                            self.core.log_graphics_queue_checkpoints();
                        }
                    }
                    _ => log::error!("VkWaitForFences returned {e} in CmdBuffer::drop"),
                },
            }
            self.core
                .device()
                .inner()
                .free_command_buffers(self.core.graphics_cmd_pool().inner(), &[self.handle]);
        }
    }
}
