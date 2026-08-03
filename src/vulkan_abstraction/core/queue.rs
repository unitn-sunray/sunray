use std::sync::Arc;

use crate::{error::*, vulkan_abstraction};
use ash::vk;

pub struct Queue {
    queue: vk::Queue,
    queue_family_index: u32,
    queue_index: u32,

    device: Arc<vulkan_abstraction::Device>,
}
impl Queue {
    pub fn new(device: Arc<vulkan_abstraction::Device>, queue_index: u32, queue_family_index: u32) -> SrResult<Self> {
        let queue = unsafe { device.inner().get_device_queue(queue_family_index, queue_index) };
        Ok(Self {
            queue,
            queue_family_index,
            queue_index,
            device,
        })
    }

    pub fn wait_idle(&self) -> SrResult<()> {
        unsafe { self.device.inner().queue_wait_idle(self.queue) }?;
        Ok(())
    }

    pub fn submit_async(
        &self,
        command_buffer: vk::CommandBuffer,
        wait_semaphores: &[vk::Semaphore],
        wait_dst_stages: &[vk::PipelineStageFlags],
        signal_semaphores: &[vk::Semaphore],
        signal_fence: vk::Fence,
    ) -> SrResult<()> {
        if cfg!(debug_assertions) && wait_semaphores.len() != wait_dst_stages.len() {
            return Err(SrError::new_custom(
                "Incorrect parameters to Queue::submit_async: wait_semaphores.len() != wait_dst_stages.len()".to_string(),
            ));
        }

        let wait_semaphore_infos: Vec<vk::SemaphoreSubmitInfo> = wait_semaphores
            .iter()
            .zip(wait_dst_stages.iter())
            .map(|(sem, stage)| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(*sem)
                    // PipelineStageFlags bit values are compatible with the matching subset of PipelineStageFlags2.
                    .stage_mask(vk::PipelineStageFlags2::from_raw(stage.as_raw() as u64))
            })
            .collect();

        let signal_semaphore_infos: Vec<vk::SemaphoreSubmitInfo> = signal_semaphores
            .iter()
            .map(|sem| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(*sem)
                    .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
            })
            .collect();

        let cmd_buf_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)];

        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_semaphore_infos)
            .command_buffer_infos(&cmd_buf_infos)
            .signal_semaphore_infos(&signal_semaphore_infos);

        unsafe { self.device.inner().queue_submit2(self.queue, &[submit_info], signal_fence) }?;

        Ok(())
    }

    /// Like [`Self::submit_async`], but additionally signals a timeline
    /// semaphore with a specific value when the submission completes (used by
    /// the renderer to mark the absolute frame count on its frame timeline).
    pub fn submit_async_with_timeline(
        &self,
        command_buffer: vk::CommandBuffer,
        wait_semaphores: &[vk::Semaphore],
        wait_dst_stages: &[vk::PipelineStageFlags],
        signal_timeline: (vk::Semaphore, u64),
        signal_fence: vk::Fence,
    ) -> SrResult<()> {
        if cfg!(debug_assertions) && wait_semaphores.len() != wait_dst_stages.len() {
            return Err(SrError::new_custom(
                "Incorrect parameters to Queue::submit_async_with_timeline: wait_semaphores.len() != wait_dst_stages.len()"
                    .to_string(),
            ));
        }

        let wait_semaphore_infos: Vec<vk::SemaphoreSubmitInfo> = wait_semaphores
            .iter()
            .zip(wait_dst_stages.iter())
            .map(|(sem, stage)| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(*sem)
                    // PipelineStageFlags bit values are compatible with the matching subset of PipelineStageFlags2.
                    .stage_mask(vk::PipelineStageFlags2::from_raw(stage.as_raw() as u64))
            })
            .collect();

        let signal_semaphore_infos = [vk::SemaphoreSubmitInfo::default()
            .semaphore(signal_timeline.0)
            .value(signal_timeline.1)
            .stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)];

        let cmd_buf_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)];

        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_semaphore_infos)
            .command_buffer_infos(&cmd_buf_infos)
            .signal_semaphore_infos(&signal_semaphore_infos);

        unsafe { self.device.inner().queue_submit2(self.queue, &[submit_info], signal_fence) }?;

        Ok(())
    }

    /// General timeline-aware submit: each wait/signal is `(semaphore, value,
    /// stage2)`. `value` is used for timeline semaphores and ignored for binary
    /// ones, so this covers both. Used by the render graph (waits the previous
    /// frame's graph timeline + signals this frame's) and the blit (waits the
    /// graph timeline + signals the frame timeline).
    pub fn submit_async_timelines(
        &self,
        command_buffer: vk::CommandBuffer,
        waits: &[(vk::Semaphore, u64, vk::PipelineStageFlags2)],
        signals: &[(vk::Semaphore, u64, vk::PipelineStageFlags2)],
        signal_fence: vk::Fence,
    ) -> SrResult<()> {
        let wait_infos: Vec<vk::SemaphoreSubmitInfo> = waits
            .iter()
            .map(|(sem, value, stage)| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(*sem)
                    .value(*value)
                    .stage_mask(*stage)
            })
            .collect();
        let signal_infos: Vec<vk::SemaphoreSubmitInfo> = signals
            .iter()
            .map(|(sem, value, stage)| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(*sem)
                    .value(*value)
                    .stage_mask(*stage)
            })
            .collect();
        let cmd_buf_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)];
        let submit_info = vk::SubmitInfo2::default()
            .wait_semaphore_infos(&wait_infos)
            .command_buffer_infos(&cmd_buf_infos)
            .signal_semaphore_infos(&signal_infos);

        unsafe { self.device.inner().queue_submit2(self.queue, &[submit_info], signal_fence) }?;

        Ok(())
    }

    pub fn submit_sync(&self, command_buffer: vk::CommandBuffer) -> SrResult<()> {
        let cmd_buf_infos = [vk::CommandBufferSubmitInfo::default().command_buffer(command_buffer)];
        let submit_info = vk::SubmitInfo2::default().command_buffer_infos(&cmd_buf_infos);

        let mut fence = vulkan_abstraction::Fence::new_unsignaled(Arc::clone(&self.device))?;

        unsafe { self.device.inner().queue_submit2(self.queue, &[submit_info], fence.submit()?) }?;
        fence.wait()?;

        Ok(())
    }

    pub fn queue_family_index(&self) -> u32 {
        self.queue_family_index
    }

    pub fn queue_index(&self) -> u32 {
        self.queue_index
    }

    #[allow(dead_code)]
    pub fn inner(&self) -> vk::Queue {
        self.queue
    }
}
