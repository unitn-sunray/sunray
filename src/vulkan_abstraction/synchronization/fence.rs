use crate::{
    error::{ErrorSource, SrResult},
    vulkan_abstraction,
};
use ash::vk;
use std::sync::Arc;

pub fn wait_fence(device: &vulkan_abstraction::Device, fence: vk::Fence) -> SrResult<()> {
    if fence != vk::Fence::null() {
        unsafe { device.inner().wait_for_fences(&[fence], true, u64::MAX) }?;
    }
    Ok(())
}

pub struct Fence {
    device: Arc<vulkan_abstraction::Device>,
    handle: vk::Fence,
    fence_waited: bool,
}

impl Fence {
    pub fn new_signaled(device: Arc<vulkan_abstraction::Device>) -> SrResult<Self> {
        Self::new(device, vk::FenceCreateFlags::SIGNALED)
    }
    pub fn new_unsignaled(device: Arc<vulkan_abstraction::Device>) -> SrResult<Self> {
        Self::new(device, vk::FenceCreateFlags::empty())
    }
    pub fn new(device: Arc<vulkan_abstraction::Device>, flags: vk::FenceCreateFlags) -> SrResult<Self> {
        let fence_info = vk::FenceCreateInfo::default().flags(flags);

        let handle = unsafe { device.inner().create_fence(&fence_info, None) }?;

        Ok(Self {
            device,
            handle,
            fence_waited: true,
        })
    }
    pub fn reset(&mut self) -> SrResult<()> {
        self.fence_waited = true;
        unsafe { self.device.inner().reset_fences(&[self.handle]) }?;

        Ok(())
    }
    pub fn submit(&mut self) -> SrResult<vk::Fence> {
        self.wait()?;
        self.fence_waited = false;
        Ok(self.handle)
    }
    pub fn get_fence_for_wait(&mut self) -> SrResult<vk::Fence> {
        self.fence_waited = true;

        Ok(self.handle)
    }
    pub fn wait(&mut self) -> SrResult<()> {
        if !self.fence_waited {
            wait_fence(&self.device, self.handle)?;
            self.fence_waited = true;
        }

        self.reset()?;

        Ok(())
    }
    /// # Safety
    /// The caller must not signal, reset or destroy the returned handle behind
    /// this `Fence`'s back — it tracks its own signalled state for `Drop`.
    pub unsafe fn inner(&self) -> vk::Fence {
        self.handle
    }
}

impl Drop for Fence {
    fn drop(&mut self) {
        // don't panic in drop, if possible
        match self.wait() {
            Ok(()) => {}
            Err(e) => match e.get_source() {
                ErrorSource::Vulkan(e) => {
                    log::warn!("VkWaitForFences returned {e:?} in Fence::drop")
                }
                _ => log::error!("VkWaitForFences returned {e} in Fence::drop"),
            },
        }
        unsafe { self.device.inner().destroy_fence(self.handle, None) };
    }
}
