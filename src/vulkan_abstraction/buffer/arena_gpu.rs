use std::collections::VecDeque;
use std::rc::Rc;
use std::sync::Arc;

use ash::vk;

use crate::MAX_FRAMES_IN_FLIGHT;
use crate::error::*;
use crate::render_graph::Handle;
use crate::vulkan_abstraction;
use crate::vulkan_abstraction::RawBuffer;

use super::{Buffer, GpuOnlyBuffer, HostAccessibleBuffer, StagingBuffer};

/// Base trait for all arena-style slot buffers (both direct-indexed and keyed).
pub trait ArenaBuffer: Buffer {
    /// Number of element slots the arena can hold.
    fn capacity(&self) -> vk::DeviceSize;
    /// Reclaim slots whose deferred-free delay has elapsed.
    fn process_pending_frees(&mut self);
}

/// Direct-indexed arena buffer (like a `Vec` with stable slot indices).
/// Keeps a ring-buffered staging buffer for per-frame writes and a GPU-only
/// buffer for shader access. Slots are allocated from a free-list and freed
/// with deferred deallocation.
pub struct ArenaGpuBuffer<T: Copy> {
    staging: StagingBuffer<T>,
    gpu_only: Arc<GpuOnlyBuffer>,
    capacity: vk::DeviceSize,
    free_slots: Vec<usize>,
    pending_free_slots: VecDeque<(u64, usize)>,
    core: Rc<vulkan_abstraction::Core>,
    handle: Option<Handle<RawBuffer>>,
}

impl<T: Copy> Buffer for ArenaGpuBuffer<T> {
    fn inner(&self) -> vk::Buffer {
        self.gpu_only.inner()
    }

    fn usage(&self) -> vk::BufferUsageFlags {
        self.gpu_only.usage()
    }

    fn raw(&self) -> &RawBuffer {
        self.gpu_only.raw()
    }

    fn raw_mut(&mut self) -> &mut RawBuffer {
        Arc::get_mut(&mut self.gpu_only)
            .expect("Cannot get mutable reference to gpu_only, Arc has multiple owners")
            .raw_mut()
    }

    fn byte_size(&self) -> vk::DeviceSize {
        self.gpu_only.byte_size()
    }

    fn is_null(&self) -> bool {
        self.gpu_only.is_null()
    }

    fn get_device_address(&self) -> vk::DeviceAddress {
        self.gpu_only.get_device_address()
    }

    fn new_null(core: Rc<vulkan_abstraction::Core>) -> Self {
        Self {
            staging: StagingBuffer::new_null(core.clone()),
            gpu_only: Arc::new(GpuOnlyBuffer::new_null(core.clone())),
            capacity: 0,
            free_slots: vec![],
            pending_free_slots: VecDeque::new(),
            core,
            handle: None,
        }
    }
}

impl<T: Copy> ArenaBuffer for ArenaGpuBuffer<T> {
    fn capacity(&self) -> vk::DeviceSize {
        self.capacity
    }

    fn process_pending_frees(&mut self) {
        let current_frame = *self.core.absolute_frame_count.borrow() as u64;
        while let Some(&(frame_freed, slot)) = self.pending_free_slots.front() {
            if current_frame >= frame_freed + MAX_FRAMES_IN_FLIGHT as u64 {
                self.free_slots.push(slot);
                self.pending_free_slots.pop_front();
            } else {
                break;
            }
        }
    }
}

impl<T: Copy> ArenaGpuBuffer<T> {
    pub fn new(
        core: Rc<vulkan_abstraction::Core>,
        capacity: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        let staging = StagingBuffer::new(
            core.clone(),
            capacity * MAX_FRAMES_IN_FLIGHT as vk::DeviceSize,
            usage | vk::BufferUsageFlags::TRANSFER_SRC,
            name,
        )?;

        let gpu_only = GpuOnlyBuffer::new::<T>(
            core.clone(),
            capacity,
            usage | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            name,
        )?;

        let free_slots = (0..capacity as usize).rev().collect();

        Ok(Self {
            staging,
            gpu_only: Arc::new(gpu_only),
            capacity,
            free_slots,
            pending_free_slots: VecDeque::new(),
            core,
            handle: None,
        })
    }

    #[allow(dead_code)]
    pub(crate) fn new_from_data(
        core: Rc<vulkan_abstraction::Core>,
        data: &[T],
        usage: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        let capacity = data.len() as vk::DeviceSize;

        if capacity == 0 {
            return Ok(Self::new_null(core));
        }

        let staging = StagingBuffer::new_from_data_with_custom_length(
            core.clone(),
            data,
            capacity * MAX_FRAMES_IN_FLIGHT as vk::DeviceSize,
            usage | vk::BufferUsageFlags::TRANSFER_SRC,
            name,
        )?;

        let mut gpu_only = GpuOnlyBuffer::new::<T>(
            core.clone(),
            capacity,
            usage | vk::BufferUsageFlags::TRANSFER_SRC | vk::BufferUsageFlags::TRANSFER_DST,
            name,
        )?;

        staging.clone_section_into_gpu_only_buffer(0, capacity * std::mem::size_of::<T>() as vk::DeviceSize, &mut gpu_only)?;

        Ok(Self {
            staging,
            gpu_only: Arc::new(gpu_only),
            capacity,
            free_slots: vec![],
            pending_free_slots: VecDeque::new(),
            core,
            handle: None,
        })
    }

    /// Write data to a specific slot in the ring-buffered staging area.
    /// Returns the slot and a `BufferCopy` region to submit on a command buffer.
    pub fn write_to_slot(&mut self, slot: usize, data: &T) -> SrResult<(usize, vk::BufferCopy)> {
        let frame_module = *self.core.absolute_frame_count.borrow() % MAX_FRAMES_IN_FLIGHT;
        let staging_index = slot + (self.capacity as usize * frame_module);

        let mapped = self.staging.map_mut()?;
        mapped[staging_index] = *data;

        let size = std::mem::size_of::<T>() as vk::DeviceSize;
        let dst_offset = (slot as vk::DeviceSize) * size;
        let src_offset = (staging_index as vk::DeviceSize) * size;

        Ok((
            slot,
            vk::BufferCopy::default()
                .src_offset(src_offset)
                .dst_offset(dst_offset)
                .size(size),
        ))
    }

    /// Pop a free slot from the stack.
    pub fn allocate_slot(&mut self) -> SrResult<usize> {
        self.free_slots
            .pop()
            .ok_or_else(|| SrError::new_custom("Arena out of capacity!".to_string()))
    }

    /// Allocates a slot for new data. Returns the assigned index and the
    /// `BufferCopy` region that needs to be submitted on a command buffer.
    pub fn allocate_and_update(&mut self, data: &T) -> SrResult<(usize, vk::BufferCopy)> {
        let slot = self.allocate_slot()?;
        self.write_to_slot(slot, data)
    }

    /// Write new data to an existing slot. Returns the `BufferCopy` region
    /// that needs to be submitted on a command buffer.
    pub fn update(&mut self, slot: usize, data: &T) -> SrResult<vk::BufferCopy> {
        let (_, copy) = self.write_to_slot(slot, data)?;
        Ok(copy)
    }

    /// Frees an index so it can be reused by future allocations.
    pub fn free_index(&mut self, index: usize) {
        let current_frame = *self.core.absolute_frame_count.borrow() as u64;
        self.pending_free_slots.push_back((current_frame, index));
    }

    /// This frame's graph handle, or `None` before [`Self::import_into`] ran for
    /// the current graph build.
    pub fn handle(&self) -> Option<&Handle<RawBuffer>> {
        self.handle.as_ref()
    }

    /// Import the GPU-side buffer into `rg` and cache the resulting handle.
    /// A handle is only valid for one graph build (`RenderGraph::reset` clears
    /// the virtual resources and restarts the id counter), so this must be
    /// called on *every* rebuild, after `reset`.
    pub fn import_into(&mut self, rg: &mut crate::render_graph::RenderGraph) -> Handle<RawBuffer> {
        let handle = rg.import(self.gpu_only.clone());
        self.handle = Some(handle.clone());
        handle
    }

    pub fn inner_gpu(&self) -> vk::Buffer {
        self.gpu_only.inner()
    }

    /// The staging (host-visible) side of the arena — the *source* of a
    /// staging→GPU copy. Returned as the buffer itself rather than a bare
    /// `vk::Buffer` so callers can bounds-check a copy region against its
    /// `byte_size` (see `TransferPassBuilder::copy_from_raw`).
    pub fn staging(&self) -> &StagingBuffer<T> {
        &self.staging
    }

    pub fn gpu_only(&self) -> &GpuOnlyBuffer {
        &self.gpu_only
    }
}
