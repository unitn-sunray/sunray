//! # What a memory barrier does:
//! - Wait `src_pipeline_stage` to complete\
//! - Make **available** the writes perfomed in the `src_pipeline_stage` + `src_access_mask combination`\
//! - Make memory **visible** to the `dst_pipeline_stage` + `dst_access_mask` combination\
//! - Start `dst_pipeline_stage`
//!
//! # Theory:
//! Execution order and memory order are two different things and we have to manage them individually.\
//! Also GPUs have different caches that we need to make coherent to avoid errors.
//!
//! ## Memory concepts:
//! - available (flush caches) - the top level cache contains the most up-to-date data
//! - visible (invalidate caches) - the memory is **available** and is **visible** to the pipeline stage + access mask combination
//!
//! ### References:
//! - <https://themaister.net/blog/2019/08/14/yet-another-blog-explaining-vulkan-synchronization/>
//! - <https://www.sctheblog.com/blog/vulkan-synchronization/>

use ash::vk;

use crate::vulkan_abstraction;

pub mod fence;
pub mod semaphore;
pub use fence::*;
pub use semaphore::*;

/// # Creates a memory barrier (sync2)
///
/// # Safety
/// `cmd_buf` must be a live command buffer in the recording state, owned by a
/// pool this thread has exclusive access to, and the barriers must name only
/// resources still alive for the whole submission.
pub unsafe fn cmd_memory_barrier(
    core: &vulkan_abstraction::Core,
    cmd_buf: vk::CommandBuffer,
    memory_barriers: &[vk::MemoryBarrier2],
    buffer_memory_barriers: &[vk::BufferMemoryBarrier2],
    image_memory_barriers: &[vk::ImageMemoryBarrier2],
) {
    let dependency_info = vk::DependencyInfo::default()
        .memory_barriers(memory_barriers)
        .buffer_memory_barriers(buffer_memory_barriers)
        .image_memory_barriers(image_memory_barriers);

    unsafe {
        core.device().inner().cmd_pipeline_barrier2(cmd_buf, &dependency_info);
    }
}

/// # Creates an image memory barrier (sync2)
///
/// # Safety
/// Same contract as [`cmd_memory_barrier`], plus `image` must outlive the
/// submission and `old_layout` must match the layout the image is actually in.
// The parameter list mirrors `VkImageMemoryBarrier2` field for field; grouping it
// into a struct would just re-create the ash builder that already exists.
#[allow(clippy::too_many_arguments)]
pub unsafe fn cmd_image_memory_barrier(
    core: &vulkan_abstraction::Core,
    cmd_buf: vk::CommandBuffer,
    image: vk::Image,
    src_stage_mask: vk::PipelineStageFlags2,
    dst_stage_mask: vk::PipelineStageFlags2,
    src_access_mask: vk::AccessFlags2,
    dst_access_mask: vk::AccessFlags2,
    old_layout: vk::ImageLayout,
    new_layout: vk::ImageLayout,
) {
    let image_memory_barrier = vk::ImageMemoryBarrier2::default()
        .src_stage_mask(src_stage_mask)
        .dst_stage_mask(dst_stage_mask)
        .src_access_mask(src_access_mask)
        .dst_access_mask(dst_access_mask)
        .old_layout(old_layout)
        .new_layout(new_layout)
        .image(image)
        .subresource_range(
            vk::ImageSubresourceRange::default()
                .aspect_mask(vk::ImageAspectFlags::COLOR)
                .base_mip_level(0)
                .level_count(1)
                .base_array_layer(0)
                .layer_count(1),
        );

    unsafe {
        cmd_memory_barrier(core, cmd_buf, &[], &[], &[image_memory_barrier]);
    };
}
