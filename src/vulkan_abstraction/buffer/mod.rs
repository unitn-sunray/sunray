pub mod arena_gpu;
pub mod gpu_only_buffer;
pub mod index_buffer;
pub mod staging_buffer;
pub mod uniform_buffer;
pub mod vertex_buffer;

pub use arena_gpu::*;
pub use gpu_only_buffer::*;
pub use index_buffer::*;
pub use staging_buffer::*;
pub use uniform_buffer::*;
pub use vertex_buffer::*;

use crate::render_graph::resource::ResourceDesc;
use crate::vulkan_abstraction::descriptor_heap::{DescriptorSlot, ResourceDescriptorKind};
use crate::{error::*, vulkan_abstraction};
use ash::vk;
use ash::vk::{BufferUsageFlags, DeviceAddress, DeviceSize, Handle};
use std::cell::Cell;
use std::fmt::{Debug, Formatter};
use std::rc::Rc;
use std::sync::Arc;
use crate::vulkan_abstraction::Core;

//TODO revert capacity as vk::device length some methods signatures
//TODO should gpu only buffer have a generic and some methods can be moved inside the buffer trait like new,new with data ecc.. with a default impl
pub fn get_memory_type_index(
    core: &vulkan_abstraction::Core,
    mem_prop_flags: vk::MemoryPropertyFlags,
    mem_requirements: &vk::MemoryRequirements,
) -> SrResult<u32> {
    type BitsType = u32;
    let bits: BitsType = mem_requirements.memory_type_bits;
    assert_ne!(bits, 0);

    let mem_types = core.device().memory_properties().memory_types;
    let mut idx = -1;
    for i in 0..BitsType::BITS as usize {
        let mem_type_is_supported = bits & (1 << i) != 0;
        if mem_type_is_supported && mem_types[i].property_flags & mem_prop_flags == mem_prop_flags {
            idx = i as isize;
            break;
        }
    }
    if idx < 0 {
        return Err(SrError::new_custom("Vertex Buffer Memory Type not supported!".to_string()));
    }

    Ok(idx as u32)
}
pub trait Buffer {
    fn inner(&self) -> vk::Buffer;

    fn usage(&self) -> BufferUsageFlags;

    fn raw(&self) -> &vulkan_abstraction::RawBuffer;

    fn raw_mut(&mut self) -> &mut vulkan_abstraction::RawBuffer;

    fn byte_size(&self) -> vk::DeviceSize;
    fn is_null(&self) -> bool;
    fn get_device_address(&self) -> vk::DeviceAddress;
    fn new_null(core: Rc<vulkan_abstraction::Core>) -> Self
    where
        Self: Sized;
}

/// Exclusive trait for host-visible buffers (CpuToGpu or GpuToCpu) that can be mapped.
pub trait HostAccessibleBuffer<T>: Buffer {
    fn map_mut(&mut self) -> SrResult<&mut [T]>;

    fn map(&self) -> SrResult<&[T]>;

    fn get(&self) -> SrResult<&T> {
        self.map().map(|s| &s[0])
    }

    fn get_mut(&mut self) -> SrResult<&mut T> {
        self.map_mut().map(|s| &mut s[0])
    }

    fn len(&self) -> usize;
}
pub struct RawBuffer {
    core: Rc<vulkan_abstraction::Core>,
    buffer: vk::Buffer,
    allocation: gpu_allocator::vulkan::Allocation,
    byte_size: u64,
    usage: BufferUsageFlags,
    /// Lazily-allocated heap slot for `UNIFORM_BUFFER` descriptors.
    uniform_slot: Cell<Option<DescriptorSlot>>,
    /// Lazily-allocated heap slot for `STORAGE_BUFFER` descriptors.
    storage_slot: Cell<Option<DescriptorSlot>>,
    /// True when this RawBuffer holds its own `Allocation`. False when memory is
    /// owned elsewhere (e.g. by a transient alias slot in `TransientResources`);
    /// `Drop` still destroys the `vk::Buffer` but skips `Allocator::free`.
    owns_memory: bool,
}
impl Buffer for RawBuffer {
    fn inner(&self) -> vk::Buffer {
        self.buffer
    }

    fn usage(&self) -> BufferUsageFlags {
        self.usage
    }

    fn raw(&self) -> &RawBuffer {
        self
    }

    fn raw_mut(&mut self) -> &mut RawBuffer {
        self
    }

    fn byte_size(&self) -> DeviceSize {
       self.byte_size
    }

    fn is_null(&self) -> bool {
       self.buffer.is_null()
    }

    fn get_device_address(&self) -> DeviceAddress {
        self.device_address()
    }

    fn new_null(core: Rc<vulkan_abstraction::Core>) -> Self {
        Self {
            core,
            buffer: vk::Buffer::null(),
            allocation: gpu_allocator::vulkan::Allocation::default(),
            byte_size: 0,
            usage: BufferUsageFlags::empty(),
            uniform_slot: Cell::new(None),
            storage_slot: Cell::new(None),
            // Null buffer: the existing `Drop` early-returns on null anyway, but mark
            // ownership consistent with `new_aligned`.
            owns_memory: true,
        }
    }
}


// `RawBuffer` can't `#[derive(Debug)]`: its `core: Rc<Core>` field doesn't
// implement `Debug` (and `Allocation` / the `Cell` slots aren't worth printing).
// A hand-written impl lets `Debug` types that hold a `RawBuffer` (e.g.
// `Arc<RawBuffer>` inside `TlasBuildDesc` / `ASDesc`) derive `Debug` normally.
impl Debug for RawBuffer {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("RawBuffer")
            .field("buffer", &self.buffer)
            .field("byte_size", &self.byte_size)
            .field("usage", &self.usage)
            .field("owns_memory", &self.owns_memory)
            .finish_non_exhaustive()
    }
}

impl RawBuffer {
    /// Construct a buffer from a render-graph descriptor.
    pub fn new_from_desc(core: Rc<vulkan_abstraction::Core>, desc: &BufferDesc) -> SrResult<Self> {
        Self::new_aligned(
            core,
            desc.byte_size,
            desc.alignment,
            desc.memory_location,
            desc.usage,
            desc.name,
        )
    }

    pub fn new_aligned(
        core: Rc<vulkan_abstraction::Core>,
        byte_size: vk::DeviceSize,
        alignment: u64,
        memory_location: gpu_allocator::MemoryLocation,
        buffer_usage_flags: vk::BufferUsageFlags,
        name: &'static str,
    ) -> SrResult<Self> {
        if byte_size == 0 {
            return Ok(Self::new_null(core));
        }

        let _queue_family_indices = [
            core.graphics_queue().queue_family_index(),
            core.transfer_queue().queue_family_index(),
        ];

        let device = core.device().inner();

        let buffer = {
            let buf_info = vk::BufferCreateInfo::default()
                .size(byte_size)
                .usage(buffer_usage_flags)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            unsafe { device.create_buffer(&buf_info, None) }?
        };

        let mem_requirements = unsafe { device.get_buffer_memory_requirements(buffer) };
        let mem_requirements = mem_requirements.alignment(mem_requirements.alignment.max(alignment));

        let allocation = core.allocator_mut().allocate(&gpu_allocator::vulkan::AllocationCreateDesc {
            name,
            requirements: mem_requirements,
            location: memory_location,
            linear: true,
            allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
        })?;

        unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset()) }?;

        Ok(Self {
            core,
            buffer,
            allocation,
            byte_size,
            usage: buffer_usage_flags,
            uniform_slot: Cell::new(None),
            storage_slot: Cell::new(None),
            owns_memory: true,
        })
    }

   

    /// Create a bare `vk::Buffer` handle and report its memory requirements without
    /// allocating or binding any memory. Counterpart to `Image::create_unbound`;
    /// used by the render-graph transient allocator.
    pub(crate) fn create_unbound(
        core: &vulkan_abstraction::Core,
        byte_size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
    ) -> SrResult<(vk::Buffer, vk::MemoryRequirements)> {
        let buf_info = vk::BufferCreateInfo::default()
            .size(byte_size)
            .usage(usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);
        let buffer = unsafe { core.device().inner().create_buffer(&buf_info, None) }?;
        let reqs = unsafe { core.device().inner().get_buffer_memory_requirements(buffer) };
        Ok((buffer, reqs))
    }

    /// Wrap an already-bound `vk::Buffer` whose memory is owned elsewhere. `Drop`
    /// will destroy the handle but NOT free the memory.
    pub(crate) fn from_aliased(
        core: Rc<vulkan_abstraction::Core>,
        buffer: vk::Buffer,
        byte_size: u64,
        usage: vk::BufferUsageFlags,
    ) -> SrResult<Self> {
        Ok(Self {
            core,
            buffer,
            allocation: gpu_allocator::vulkan::Allocation::default(),
            byte_size,
            usage,
            uniform_slot: Cell::new(None),
            storage_slot: Cell::new(None),
            owns_memory: false,
        })
    }
    

    /// GPU device address of this buffer. Requires the buffer to have been
    /// created with `SHADER_DEVICE_ADDRESS` usage; returns 0 for a null buffer.
    /// Used to feed buffer-device-address fields of heap-mode push constants
    /// (e.g. the ReSTIR reservoir buffers) when the buffer is also imported into
    /// the render graph as an `Arc<RawBuffer>` for hazard tracking.
    pub fn device_address(&self) -> vk::DeviceAddress {
        if self.buffer == vk::Buffer::null() {
            return 0;
        }
        let info = vk::BufferDeviceAddressInfo::default().buffer(self.buffer);
        unsafe { self.core.device().inner().get_buffer_device_address(&info) }
    }

    /// Heap slot for `UNIFORM_BUFFER`. Lazily allocates on first call.
    pub fn uniform_slot(&self) -> u32 {
        self.descriptor_slot(&self.uniform_slot, ResourceDescriptorKind::UniformBuffer)
    }

    /// Heap slot for `STORAGE_BUFFER`. Lazily allocates on first call.
    pub fn storage_slot(&self) -> u32 {
        self.descriptor_slot(&self.storage_slot, ResourceDescriptorKind::StorageBuffer)
    }

    fn descriptor_slot(&self, cell: &Cell<Option<DescriptorSlot>>, kind: ResourceDescriptorKind) -> u32 {
        if let Some(s) = cell.get() {
            return s.shader_index();
        }
        debug_assert!(!self.buffer.is_null(), "cannot allocate descriptor slot for null buffer");
        let info = vk::BufferDeviceAddressInfo::default().buffer(self.buffer);
        let address = unsafe { self.core.device().inner().get_buffer_device_address(&info) };
        let mut heap = self.core.descriptor_heap_mut();
        let slot = heap.alloc_resource_slot(kind);
        heap.write_buffer(slot, address, self.byte_size, kind)
            .expect("descriptor heap write_buffer failed");
        cell.set(Some(slot));
        slot.shader_index()
    }

    pub fn map<V: Sized>(&self) -> SrResult<&[V]> {
        if !self.buffer.is_null() {
            let slice = self.allocation.mapped_slice().unwrap();
            let ret = unsafe { std::slice::from_raw_parts_mut(slice.as_ptr() as *mut V, slice.len() / std::mem::size_of::<V>()) };
            Ok(ret)
        } else {
            Ok(&[])
        }
    }

    pub fn map_mut<V: Sized>(&mut self) -> SrResult<&mut [V]> {
        if !self.buffer.is_null() {
            let slice = self.allocation.mapped_slice_mut().unwrap();
            let ret = unsafe { std::slice::from_raw_parts_mut(slice.as_ptr() as *mut V, slice.len() / std::mem::size_of::<V>()) };
            Ok(ret)
        } else {
            Ok(&mut [])
        }
    }
}

impl Drop for RawBuffer {
    fn drop(&mut self) {
        if self.buffer != vk::Buffer::null() {
            {
                let mut heap = self.core.descriptor_heap_mut();
                if let Some(s) = self.uniform_slot.get() {
                    heap.free(s);
                }
                if let Some(s) = self.storage_slot.get() {
                    heap.free(s);
                }
            }

            let device = self.core.device().inner();
            unsafe {
                device.destroy_buffer(self.buffer, None);
            }
            // Aliased buffers don't own their memory (held by a transient slot in
            // TransientResources); skip free in that case.
            if self.owns_memory {
                let allocation = std::mem::take(&mut self.allocation);
                if let Err(e) = self.core.allocator_mut().free(allocation) {
                    log::error!("Allocator::free returned {e} in RawBuffer::drop");
                }
            }
        }
    }
}
#[macro_export]
macro_rules! impl_buffer_trait {
    ($type:ident < $first_gen:ident $(, $rest_gens:ident)* >) => {
        impl < $first_gen $(, $rest_gens)* > Buffer for $type < $first_gen $(, $rest_gens)* > {
            fn inner(&self) -> vk::Buffer {
                self.raw.buffer
            }

            fn raw(&self) -> &vulkan_abstraction::RawBuffer {
                &self.raw
            }

            fn usage(&self) -> vk::BufferUsageFlags {
                self.raw().usage.clone()
            }

             fn raw_mut(&mut self) -> &mut vulkan_abstraction::RawBuffer {
                &mut self.raw
            }


            fn byte_size(&self) -> vk::DeviceSize {
                self.raw.byte_size
            }

            fn is_null(&self) -> bool {
                self.raw.buffer == vk::Buffer::null()
            }

            fn get_device_address(&self) -> vk::DeviceAddress {
                if self.is_null() {
                    return 0;
                }
                let info = vk::BufferDeviceAddressInfo::default().buffer(self.raw.buffer);
                unsafe { self.raw.core.device().inner().get_buffer_device_address(&info) }
            }

            fn new_null(core: Rc<vulkan_abstraction::Core>) -> Self {
                Self {
                    raw: RawBuffer::new_null(core),
                    _marker: Default::default(),
                }
            }
        }
    };

    ($type:ident) => {
        impl Buffer for $type {
            fn inner(&self) -> vk::Buffer {
                self.raw.buffer
            }

            fn usage(&self) -> vk::BufferUsageFlags {
                self.raw().usage.clone()
            }

             fn raw(&self) -> &vulkan_abstraction::RawBuffer {
                &self.raw
            }

             fn raw_mut(&mut self) -> &mut vulkan_abstraction::RawBuffer {
                &mut self.raw
            }


            fn byte_size(&self) -> vk::DeviceSize {
                self.raw.byte_size
            }

            fn is_null(&self) -> bool {
                self.raw.buffer == vk::Buffer::null()
            }

            fn get_device_address(&self) -> vk::DeviceAddress {
                if self.is_null() {
                    return 0;
                }
                let info = vk::BufferDeviceAddressInfo::default().buffer(self.raw.buffer);
                unsafe { self.raw.core.device().inner().get_buffer_device_address(&info) }
            }

            fn new_null(core: Rc<vulkan_abstraction::Core>) -> Self {
                Self {
                    raw: RawBuffer::new_null(core),
                }
            }
        }
    };
}

/// Derives the appropriate pipeline stages and access flags for reading a buffer
/// based on the usage flags it was created with.
pub fn infer_read_masks_from_usage(usage: vk::BufferUsageFlags) -> (vk::PipelineStageFlags2, vk::AccessFlags2) {
    let mut stage_mask = vk::PipelineStageFlags2::empty();
    let mut access_mask = vk::AccessFlags2::empty();

    if usage.contains(vk::BufferUsageFlags::VERTEX_BUFFER) {
        stage_mask |= vk::PipelineStageFlags2::VERTEX_INPUT;
        access_mask |= vk::AccessFlags2::VERTEX_ATTRIBUTE_READ;
    }

    if usage.contains(vk::BufferUsageFlags::INDEX_BUFFER) {
        stage_mask |= vk::PipelineStageFlags2::VERTEX_INPUT;
        access_mask |= vk::AccessFlags2::INDEX_READ;
    }

    if usage.contains(vk::BufferUsageFlags::UNIFORM_BUFFER) {
        // UBOs can be read in almost any shader stage
        stage_mask |= vk::PipelineStageFlags2::ALL_GRAPHICS | vk::PipelineStageFlags2::COMPUTE_SHADER;
        access_mask |= vk::AccessFlags2::UNIFORM_READ;
    }

    if usage.contains(vk::BufferUsageFlags::STORAGE_BUFFER) {
        // SSBOs can be read in compute or graphics
        stage_mask |= vk::PipelineStageFlags2::ALL_GRAPHICS | vk::PipelineStageFlags2::COMPUTE_SHADER;
        access_mask |= vk::AccessFlags2::SHADER_READ;
    }

    if usage.contains(vk::BufferUsageFlags::INDIRECT_BUFFER) {
        stage_mask |= vk::PipelineStageFlags2::DRAW_INDIRECT;
        access_mask |= vk::AccessFlags2::INDIRECT_COMMAND_READ;
    }

    // Raytracing specific flags
    if usage.contains(vk::BufferUsageFlags::ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_KHR) {
        stage_mask |= vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR;
        access_mask |= vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR;
    }
    if usage.contains(vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR) {
        stage_mask |= vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR;
        access_mask |= vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR;
    }

    // Fallback if somehow it's none of the above
    if stage_mask.is_empty() {
        stage_mask = vk::PipelineStageFlags2::ALL_COMMANDS;
        access_mask = vk::AccessFlags2::MEMORY_READ;
    }

    (stage_mask, access_mask)
}

impl crate::render_graph::graph::Resource for RawBuffer {
    type Desc = BufferDesc;

    fn borrow_resource(res: &crate::render_graph::resource::AnyRenderResource) -> &Self {
        match res {
            crate::render_graph::resource::AnyRenderResource::OwnedBuffer(buf) => buf,
            _ => panic!("borrow_resource::<RawBuffer> called on non-buffer AnyRenderResource variant"),
        }
    }
}

impl crate::render_graph::graph::RgImportable<BufferDesc> for Arc<RawBuffer> {
    fn import(&self) -> BufferDesc {
        BufferDesc {
            byte_size: self.byte_size,
            alignment: 1,
            memory_location: gpu_allocator::MemoryLocation::GpuOnly,
            usage: self.usage,
            name: "imported",
        }
    }
}


impl From<Arc<RawBuffer>> for crate::render_graph::graph::GraphResourceImportInfo {
    fn from(val: Arc<RawBuffer>) -> Self {
        crate::render_graph::graph::GraphResourceImportInfo::Buffer {
            resource: val,
            //TODO let the caller supply the initial access state instead of defaulting to Nothing
            access_type: vk_sync_fork::AccessType::Nothing,
        }
    }
}



#[derive(Clone, Debug)]
pub struct BufferDesc {
    pub byte_size: vk::DeviceSize,
    pub alignment: u64,
    pub memory_location: gpu_allocator::MemoryLocation,
    pub usage: vk::BufferUsageFlags,
    pub name: &'static str,
}

impl ResourceDesc for BufferDesc {
    type Resource = RawBuffer;
}
