use std::sync::Arc;

use ash::vk;
use ash::vk::TaggedStructure;

use crate::{error::SrResult, vulkan_abstraction};

pub struct Semaphore {
    core: Arc<vulkan_abstraction::Core>,
    handle: vk::Semaphore,
}

impl Semaphore {
    pub fn new(core: Arc<vulkan_abstraction::Core>) -> SrResult<Self> {
        let handle = unsafe {
            core.device().inner().create_semaphore(
                &vk::SemaphoreCreateInfo::default()
                    // there are no fields in info besides flags and flags has (currently) no valid values besides empty
                    .flags(vk::SemaphoreCreateFlags::empty()),
                None,
            )
        }?;

        Ok(Self { core, handle })
    }

    pub fn inner(&self) -> vk::Semaphore {
        self.handle
    }
}

impl Drop for Semaphore {
    fn drop(&mut self) {
        unsafe {
            self.core.device().inner().destroy_semaphore(self.handle, None);
        }
    }
}

/// Timeline semaphore (Vulkan 1.2 `SemaphoreType::TIMELINE`). The renderer
/// signals it with the absolute frame count when a frame's GPU work completes,
/// so "wait for frame N" is `wait(N)` and the counter value is the last
/// finished frame — no per-frame fences to track or recycle.
pub struct TimelineSemaphore {
    core: Arc<vulkan_abstraction::Core>,
    handle: vk::Semaphore,
}

impl TimelineSemaphore {
    pub fn new(core: Arc<vulkan_abstraction::Core>, initial_value: u64) -> SrResult<Self> {
        let mut type_info = vk::SemaphoreTypeCreateInfo::default()
            .semaphore_type(vk::SemaphoreType::TIMELINE)
            .initial_value(initial_value);
        let create_info = vk::SemaphoreCreateInfo::default().push(&mut type_info);

        let handle = unsafe { core.device().inner().create_semaphore(&create_info, None) }?;

        Ok(Self { core, handle })
    }

    pub fn inner(&self) -> vk::Semaphore {
        self.handle
    }

    /// The last value the timeline reached (i.e. the last completed frame).
    pub fn counter_value(&self) -> SrResult<u64> {
        Ok(unsafe { self.core.device().inner().get_semaphore_counter_value(self.handle) }?)
    }

    /// Block until the timeline reaches `value`. Returns immediately if it
    /// already has (including `value` ≤ the initial value).
    pub fn wait(&self, value: u64) -> SrResult<()> {
        let semaphores = [self.handle];
        let values = [value];
        let wait_info = vk::SemaphoreWaitInfo::default().semaphores(&semaphores).values(&values);
        unsafe { self.core.device().inner().wait_semaphores(&wait_info, u64::MAX) }?;
        Ok(())
    }
}

impl Drop for TimelineSemaphore {
    fn drop(&mut self) {
        unsafe {
            self.core.device().inner().destroy_semaphore(self.handle, None);
        }
    }
}
