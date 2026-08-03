use std::ffi::c_void;
use std::ptr::NonNull;

use crate::error::{SrError, SrResult};
use crate::vulkan_abstraction::descriptor_heap::slot::{
    DescriptorSlot, HeapKind, ResourceDescriptorKind, ResourceSection, SlotAllocator,
};
use crate::vulkan_abstraction::error::HeapError;
use ash::vk::TaggedStructure;
use ash::{ext, vk};
use gpu_allocator::vulkan::{Allocation, AllocationCreateDesc, AllocationScheme, Allocator};
use log::info;

//TODO this should handle growing and shrinking by doubling the section that filled up.

/// Default descriptor counts per resource section. Splitting the resource heap
/// into three fixed-stride sections (image / texel / buffer) means each section
/// is now sized independently. These defaults aim to match the ~4k-per-type
/// budget of the previous paged layout.
pub const DEFAULT_IMAGE_CAPACITY: u32 = 4_096;
pub const DEFAULT_TEXEL_BUFFER_CAPACITY: u32 = 1_024;
pub const DEFAULT_BUFFER_CAPACITY: u32 = 4_096;
pub const DEFAULT_SAMPLER_CAPACITY: u32 = 2_048;

#[derive(Debug)]
struct ResourceSubHeap {
    buffer: vk::Buffer,
    allocation: Allocation,
    device_address: vk::DeviceAddress,
    mapped: NonNull<u8>,
    /// Total heap byte size (app area + driver-reserved tail).
    byte_size: u64,
    /// Bytes the application owns; descriptors live in `[0, app_byte_size)`. The driver
    /// reserved range sits in `[app_byte_size, byte_size)` and must not be touched.
    app_byte_size: u64,

    image_descriptor_size: u64,
    buffer_descriptor_size: u64,

    /// Pre-computed shader-index base for each section:
    /// `index = section_byte_offset / type_descriptor_size + slot_in_section`.
    /// The byte offsets themselves are derived at construction time and are
    /// not needed at runtime — only this folded form is.
    image_section_base_index: u32,
    texel_section_base_index: u32,
    buffer_section_base_index: u32,

    image_alloc: SlotAllocator,
    texel_alloc: SlotAllocator,
    buffer_alloc: SlotAllocator,
}

#[derive(Debug)]
struct SamplerSubHeap {
    buffer: vk::Buffer,
    allocation: Allocation,
    device_address: vk::DeviceAddress,
    ///non nullable pointer to a memory region in bytes for better pointer arithmetics returned by mapping cpu mem to gpu
    mapped: NonNull<u8>,
    /// Total heap byte size (app area + driver-reserved tail).
    byte_size: u64,
    /// Bytes the application owns; samplers live in `[0, app_byte_size)`.
    app_byte_size: u64,
    descriptor_size: u64,
    stride: u64,
    alloc: SlotAllocator,
}

unsafe impl Send for ResourceSubHeap {}
unsafe impl Send for SamplerSubHeap {}

pub struct DescriptorHeap {
    resource: ResourceSubHeap,
    sampler: SamplerSubHeap,
    /// Device-reported minimum reserved range that must be passed in `BindHeapInfoEXT`.
    /// We currently set the reserved range = the full heap (offset 0, size = heap size),
    /// so these are kept around mainly for the assertion at bind time.
    min_resource_reserved: u64,
    min_sampler_reserved: u64,
    ext: ext::descriptor_heap::Device,
    device: ash::Device,
}

impl DescriptorHeap {
    // The four capacities and the two ash loaders are all independent knobs with
    // no natural grouping; a params struct here would have exactly one caller.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &ash::Device,
        ext: &ext::descriptor_heap::Device,
        allocator: &mut Allocator,
        props: &vk::PhysicalDeviceDescriptorHeapPropertiesEXT,
        image_capacity: u32,
        texel_capacity: u32,
        buffer_capacity: u32,
        sampler_capacity: u32,
        gpuav_enabled: bool,
    ) -> SrResult<Self> {
        let image_size = props.image_descriptor_size.max(1);
        let buffer_size = props.buffer_descriptor_size.max(1);
        let sampler_size = props.sampler_descriptor_size.max(1);

        let image_alignment = props.image_descriptor_alignment.max(1);
        let buffer_alignment = props.buffer_descriptor_alignment.max(1);

        // VK_EXT_descriptor_heap: `BindHeapInfoEXT::reservedRange*` marks a sub-range
        // the *implementation* uses for its own internal descriptors (embedded samplers,
        // fixed ops, etc.); the application must NOT write descriptors there. We park
        // that reserved range at the tail of each heap so the app area stays at
        // `[0, app_byte_size)` and shader byte-offset addressing (`index * descriptor_size`)
        // lines up with our slot indices unchanged. The tail still has to be backed by
        // real memory, so we size the buffer = app area + reserved range.
        //
        // GPU-AV's shader instrumentation bumps `minResourceHeapReservedRange` at
        // runtime (observed warning: "Setting minResourceHeapReservedRange to 96928
        // (reserving 160 bytes)" on RTX 3060 Ti). The bump happens AFTER we've read
        // properties and sized the buffer, so without padding the driver/validation
        // spills 160 bytes past the buffer's end into the last few app descriptor
        // slots and clobbers them. Only added when GPU-AV is on — production builds
        // don't waste the allocation. 4 KiB comfortably covers the ~160-byte bump
        // and any future validation/instrumentation growth.
        let gpuav_slack: u64 = if gpuav_enabled { 4096 } else { 0 };
        let min_resource_reserved = props.min_resource_heap_reserved_range.max(1) + gpuav_slack;
        let min_sampler_reserved = props.min_sampler_heap_reserved_range.max(1) + gpuav_slack;

        // Layout: [ images | texel buffers | buffers | reserved tail ].
        //
        // Each section's byte offset must satisfy:
        //   1. It is a multiple of that section's descriptor size — so shader
        //      indexing `byte_offset / descriptor_size` produces an integer.
        //   2. It is a multiple of the driver-reported descriptor alignment.
        //
        // Images start at 0 trivially. Texel uses the image stride (texel
        // descriptor size == image descriptor size on this extension), so its
        // base is image-aligned and follows the image section directly. The
        // buffer section is then bumped up to buffer-size alignment.
        let image_section_byte_offset: u64 = 0;
        let image_section_bytes = (image_capacity as u64) * image_size;

        let texel_align = lcm(image_size, image_alignment);
        let texel_section_byte_offset = align_up(image_section_byte_offset + image_section_bytes, texel_align);
        let texel_section_bytes = (texel_capacity as u64) * image_size;

        let buffer_align = lcm(buffer_size, buffer_alignment);
        let buffer_section_byte_offset = align_up(texel_section_byte_offset + texel_section_bytes, buffer_align);
        let buffer_section_bytes = (buffer_capacity as u64) * buffer_size;

        let app_resource_byte_size = buffer_section_byte_offset + buffer_section_bytes;

        // Shader index = byte_offset / descriptor_size + slot_in_section. The
        // base is constant per section so we cache it once.
        debug_assert!(image_section_byte_offset.is_multiple_of(image_size));
        debug_assert!(texel_section_byte_offset.is_multiple_of(image_size));
        debug_assert!(buffer_section_byte_offset.is_multiple_of(buffer_size));

        let image_section_base_index = (image_section_byte_offset / image_size) as u32;
        let texel_section_base_index = (texel_section_byte_offset / image_size) as u32;
        let buffer_section_base_index = (buffer_section_byte_offset / buffer_size) as u32;

        let resource_byte_size = app_resource_byte_size + min_resource_reserved;
        let sampler_stride = align_up(sampler_size, props.sampler_descriptor_alignment.max(1));
        let app_sampler_byte_size = (sampler_capacity as u64) * sampler_stride;
        let sampler_byte_size = app_sampler_byte_size + min_sampler_reserved;

        if app_resource_byte_size + min_resource_reserved > props.max_resource_heap_size {
            return Err(SrError::new(
                HeapError::OutOfMemory.into(),
                format!(
                    "Cannot allocate resource buffer of size: {} with reserved : {} ,It surpassed allowed size {} ",
                    app_resource_byte_size, min_resource_reserved, props.max_resource_heap_size
                ),
            ));
        }

        if sampler_byte_size + min_sampler_reserved > props.max_sampler_heap_size {
            return Err(SrError::new(
                HeapError::OutOfMemory.into(),
                format!(
                    "Cannot allocate sampler buffer of size: {} with reserved : {} ,It surpassed allowed size {} ",
                    sampler_byte_size, min_sampler_reserved, props.max_sampler_heap_size
                ),
            ));
        }

        let resource = ResourceSubHeap::new(
            device,
            allocator,
            resource_byte_size,
            app_resource_byte_size,
            props.resource_heap_alignment.max(1),
            image_size,
            buffer_size,
            image_section_base_index,
            texel_section_base_index,
            buffer_section_base_index,
            image_capacity,
            texel_capacity,
            buffer_capacity,
            "sunray_resource_descriptor_heap",
        )?;

        let sampler = SamplerSubHeap::new(
            device,
            allocator,
            sampler_byte_size,
            app_sampler_byte_size,
            props.sampler_heap_alignment.max(1),
            sampler_size,
            sampler_stride,
            sampler_capacity,
            "sunray_sampler_descriptor_heap",
        )?;
        info!("sampler heap : {sampler:?}, resource heap :  {resource:?}");
        Ok(Self {
            resource,
            sampler,
            min_resource_reserved,
            min_sampler_reserved,
            ext: ext.clone(),
            device: device.clone(),
        })
    }

    /// Free the heap's GPU allocations. Must be called before the gpu-allocator is dropped,
    /// otherwise the allocations leak (and gpu-allocator will warn). After this call the
    /// heap is in an invalid state and must not be used.
    pub fn shutdown(&mut self, allocator: &mut Allocator) {
        unsafe {
            if self.resource.buffer != vk::Buffer::null() {
                self.device.destroy_buffer(self.resource.buffer, None);
                self.resource.buffer = vk::Buffer::null();
            }
            if self.sampler.buffer != vk::Buffer::null() {
                self.device.destroy_buffer(self.sampler.buffer, None);
                self.sampler.buffer = vk::Buffer::null();
            }
        }
        let res_alloc = std::mem::take(&mut self.resource.allocation);
        let samp_alloc = std::mem::take(&mut self.sampler.allocation);
        let _ = allocator.free(res_alloc);
        let _ = allocator.free(samp_alloc);
    }

    pub fn alloc_resource_slot(&mut self, kind: ResourceDescriptorKind) -> DescriptorSlot {
        let section = kind.section();
        let local = self
            .resource
            .alloc_mut(section)
            .alloc()
            .expect("descriptor resource heap section exhausted");
        let index = self.resource.base_index(section) + local;
        DescriptorSlot {
            kind: HeapKind::Resource,
            index,
            section,
        }
    }

    pub fn alloc_sampler_slot(&mut self) -> DescriptorSlot {
        let index = self.sampler.alloc.alloc().expect("descriptor sampler heap exhausted");
        DescriptorSlot {
            kind: HeapKind::Sampler,
            index,
            section: ResourceSection::Buffer, // unused for samplers
        }
    }

    pub fn free(&mut self, slot: DescriptorSlot) {
        match slot.kind {
            HeapKind::Resource => {
                let local = slot.index - self.resource.base_index(slot.section);
                self.resource.alloc_mut(slot.section).free(local);
            }
            HeapKind::Sampler => self.sampler.alloc.free(slot.index),
        }
    }

    pub fn resource_device_address(&self) -> vk::DeviceAddress {
        self.resource.device_address
    }

    pub fn sampler_device_address(&self) -> vk::DeviceAddress {
        self.sampler.device_address
    }

    pub fn resource_size(&self) -> u64 {
        self.resource.byte_size
    }

    pub fn sampler_size(&self) -> u64 {
        self.sampler.byte_size
    }

    /// Shader-index base of a resource section. Useful for shaders that want to
    /// emit `section_base + local_index` at draw time (e.g. via a push constant)
    /// rather than receiving fully-resolved indices.
    pub fn resource_section_base_index(&self, section: ResourceSection) -> u32 {
        self.resource.base_index(section)
    }

    /// Write an image descriptor into the resource heap at `slot`. The view-create-info
    /// is used by the driver to construct the on-heap descriptor; it does not need to
    /// outlive this call.
    pub fn write_image(
        &mut self,
        slot: DescriptorSlot,
        view_info: &vk::ImageViewCreateInfo<'_>,
        layout: vk::ImageLayout,
        kind: ResourceDescriptorKind,
    ) -> SrResult<()> {
        debug_assert_eq!(slot.kind, HeapKind::Resource);
        debug_assert_eq!(slot.section, ResourceSection::Image);
        let image_info = vk::ImageDescriptorInfoEXT::default().view(view_info).layout(layout);
        let resource_info = vk::ResourceDescriptorInfoEXT::default()
            .ty(kind.descriptor_type())
            .data(vk::ResourceDescriptorDataEXT { p_image: &image_info });
        self.write_resource(slot, resource_info, self.resource.image_descriptor_size)
    }

    /// Write a texel buffer descriptor. Texel buffers live in the texel section
    /// and use the same descriptor stride as images, so the destination byte
    /// is computed with `image_descriptor_size`.
    pub fn write_texel_buffer(
        &mut self,
        slot: DescriptorSlot,
        address: vk::DeviceAddress,
        size: vk::DeviceSize,
        format: vk::Format,
        kind: ResourceDescriptorKind,
    ) -> SrResult<()> {
        debug_assert_eq!(slot.kind, HeapKind::Resource);
        debug_assert_eq!(slot.section, ResourceSection::TexelBuffer);
        let texel_info = vk::TexelBufferDescriptorInfoEXT::default()
            .format(format)
            .address_range(vk::DeviceAddressRangeEXT { address, size });
        let resource_info =
            vk::ResourceDescriptorInfoEXT::default()
                .ty(kind.descriptor_type())
                .data(vk::ResourceDescriptorDataEXT {
                    p_texel_buffer: &texel_info,
                });
        self.write_resource(slot, resource_info, self.resource.image_descriptor_size)
    }

    pub fn write_buffer(
        &mut self,
        slot: DescriptorSlot,
        address: vk::DeviceAddress,
        size: vk::DeviceSize,
        kind: ResourceDescriptorKind,
    ) -> SrResult<()> {
        debug_assert_eq!(slot.kind, HeapKind::Resource);
        debug_assert_eq!(slot.section, ResourceSection::Buffer);
        let range = vk::DeviceAddressRangeEXT { address, size };
        let resource_info = vk::ResourceDescriptorInfoEXT::default()
            .ty(kind.descriptor_type())
            .data(vk::ResourceDescriptorDataEXT { p_address_range: &range });
        self.write_resource(slot, resource_info, self.resource.buffer_descriptor_size)
    }

    pub fn write_acceleration_structure(&mut self, slot: DescriptorSlot, address: vk::DeviceAddress) -> SrResult<()> {
        debug_assert_eq!(slot.kind, HeapKind::Resource);
        debug_assert_eq!(slot.section, ResourceSection::Buffer);
        let range = vk::DeviceAddressRangeEXT {
            address,
            size: 0, // ignored for AS descriptors
        };
        let resource_info = vk::ResourceDescriptorInfoEXT::default()
            .ty(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .data(vk::ResourceDescriptorDataEXT { p_address_range: &range });
        // AS descriptors are sized like buffer descriptors on this extension.
        self.write_resource(slot, resource_info, self.resource.buffer_descriptor_size)
    }

    pub fn write_sampler(&mut self, slot: DescriptorSlot, info: &vk::SamplerCreateInfo<'_>) -> SrResult<()> {
        debug_assert_eq!(slot.kind, HeapKind::Sampler);
        let dst = self.sampler_dst(slot.index);
        let dst_range = [vk::HostAddressRangeEXT {
            address: dst.as_ptr() as *mut c_void,
            size: self.sampler.descriptor_size as usize,
            _marker: std::marker::PhantomData,
        }];
        unsafe { self.ext.write_sampler_descriptors(std::slice::from_ref(info), &dst_range) }?;
        Ok(())
    }

    /// Bind both heaps to a command buffer. Must be called before any draw/dispatch
    /// that references descriptors via the heap.
    pub fn cmd_bind(&self, cmd_buf: vk::CommandBuffer) {
        // The driver-reserved range sits at the tail of each heap (see `new` for why).
        // App descriptors live in `[0, app_byte_size)`; the driver owns
        // `[app_byte_size, byte_size)`.
        let resource_bind = vk::BindHeapInfoEXT::default()
            .heap_range(vk::DeviceAddressRangeEXT {
                address: self.resource.device_address,
                size: self.resource.byte_size,
            })
            .reserved_range_offset(self.resource.app_byte_size)
            .reserved_range_size(self.min_resource_reserved);
        let sampler_bind = vk::BindHeapInfoEXT::default()
            .heap_range(vk::DeviceAddressRangeEXT {
                address: self.sampler.device_address,
                size: self.sampler.byte_size,
            })
            .reserved_range_offset(self.sampler.app_byte_size)
            .reserved_range_size(self.min_sampler_reserved);
        unsafe {
            self.ext.cmd_bind_resource_heap(cmd_buf, &resource_bind);
            self.ext.cmd_bind_sampler_heap(cmd_buf, &sampler_bind);
        }
    }

    fn write_resource(
        &mut self,
        slot: DescriptorSlot,
        info: vk::ResourceDescriptorInfoEXT<'_>,
        descriptor_size: u64,
    ) -> SrResult<()> {
        let dst = self.resource_dst(slot.index, descriptor_size);
        let dst_range = [vk::HostAddressRangeEXT {
            address: dst.as_ptr() as *mut c_void,
            size: descriptor_size as usize,
            _marker: std::marker::PhantomData,
        }];
        unsafe { self.ext.write_resource_descriptors(std::slice::from_ref(&info), &dst_range) }?;
        Ok(())
    }

    fn resource_dst(&self, shader_index: u32, descriptor_size: u64) -> NonNull<u8> {
        let offset = (shader_index as u64) * descriptor_size;
        debug_assert!(
            offset + descriptor_size <= self.resource.app_byte_size,
            "descriptor write past the app range; would land in driver-reserved memory"
        );
        unsafe { NonNull::new_unchecked(self.resource.mapped.as_ptr().add(offset as usize)) }
    }

    fn sampler_dst(&self, index: u32) -> NonNull<u8> {
        let offset = (index as u64) * self.sampler.stride;
        debug_assert!(
            offset + self.sampler.stride <= self.sampler.app_byte_size,
            "sampler write past the app range; would land in driver-reserved memory"
        );
        unsafe { NonNull::new_unchecked(self.sampler.mapped.as_ptr().add(offset as usize)) }
    }
}

impl Drop for DescriptorHeap {
    fn drop(&mut self) {
        // Resources outlive the heap in practice (Core owns them in field-declaration order),
        // so explicit shutdown is via Core::Drop calling shutdown(). This is a backstop only.
        unsafe {
            if self.resource.buffer != vk::Buffer::null() {
                self.device.destroy_buffer(self.resource.buffer, None);
            }
            if self.sampler.buffer != vk::Buffer::null() {
                self.device.destroy_buffer(self.sampler.buffer, None);
            }
        }
    }
}

impl ResourceSubHeap {
    // Private, one call site, and every argument is a distinct heap-layout figure
    // already computed by that caller.
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        byte_size: u64,
        app_byte_size: u64,
        heap_alignment: u64,
        image_descriptor_size: u64,
        buffer_descriptor_size: u64,
        image_section_base_index: u32,
        texel_section_base_index: u32,
        buffer_section_base_index: u32,
        image_capacity: u32,
        texel_capacity: u32,
        buffer_capacity: u32,
        name: &'static str,
    ) -> SrResult<Self> {
        let (buffer, allocation, device_address, mapped) =
            create_heap_buffer(device, allocator, byte_size, heap_alignment, name)?;
        Ok(Self {
            buffer,
            allocation,
            device_address,
            mapped,
            byte_size,
            app_byte_size,
            image_descriptor_size,
            buffer_descriptor_size,
            image_section_base_index,
            texel_section_base_index,
            buffer_section_base_index,
            image_alloc: SlotAllocator::new(image_capacity),
            texel_alloc: SlotAllocator::new(texel_capacity),
            buffer_alloc: SlotAllocator::new(buffer_capacity),
        })
    }

    fn alloc_mut(&mut self, section: ResourceSection) -> &mut SlotAllocator {
        match section {
            ResourceSection::Image => &mut self.image_alloc,
            ResourceSection::TexelBuffer => &mut self.texel_alloc,
            ResourceSection::Buffer => &mut self.buffer_alloc,
        }
    }

    fn base_index(&self, section: ResourceSection) -> u32 {
        match section {
            ResourceSection::Image => self.image_section_base_index,
            ResourceSection::TexelBuffer => self.texel_section_base_index,
            ResourceSection::Buffer => self.buffer_section_base_index,
        }
    }
}

impl SamplerSubHeap {
    // See `ResourceSubHeap::new`.
    #[allow(clippy::too_many_arguments)]
    fn new(
        device: &ash::Device,
        allocator: &mut Allocator,
        byte_size: u64,
        app_byte_size: u64,
        heap_alignment: u64,
        descriptor_size: u64,
        stride: u64,
        capacity: u32,
        name: &'static str,
    ) -> SrResult<Self> {
        let (buffer, allocation, device_address, mapped) =
            create_heap_buffer(device, allocator, byte_size, heap_alignment, name)?;
        Ok(Self {
            buffer,
            allocation,
            device_address,
            mapped,
            byte_size,
            app_byte_size,
            descriptor_size,
            stride,
            alloc: SlotAllocator::new(capacity),
        })
    }
}

fn create_heap_buffer(
    device: &ash::Device,
    allocator: &mut Allocator,
    byte_size: u64,
    // Heap addresses must be multiples of resourceHeapAlignment/samplerHeapAlignment,
    // which can exceed the buffer's own memory-requirements alignment on some drivers.
    heap_alignment: u64,
    name: &'static str,
) -> SrResult<(vk::Buffer, Allocation, vk::DeviceAddress, NonNull<u8>)> {
    let mut usage2 = vk::BufferUsageFlags2CreateInfo::default()
        .usage(vk::BufferUsageFlags2::DESCRIPTOR_HEAP_EXT | vk::BufferUsageFlags2::SHADER_DEVICE_ADDRESS);
    let buf_info = vk::BufferCreateInfo::default()
        .size(byte_size)
        .usage(vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
        .sharing_mode(vk::SharingMode::EXCLUSIVE)
        .push(&mut usage2);

    let buffer = unsafe { device.create_buffer(&buf_info, None) }?;
    let mem_reqs = unsafe { device.get_buffer_memory_requirements(buffer) };
    let mem_reqs = mem_reqs.alignment(mem_reqs.alignment.max(heap_alignment));

    let mut allocation = allocator.allocate(&AllocationCreateDesc {
        name,
        requirements: mem_reqs,
        location: gpu_allocator::MemoryLocation::CpuToGpu,
        linear: true,
        allocation_scheme: AllocationScheme::GpuAllocatorManaged,
    })?;

    unsafe { device.bind_buffer_memory(buffer, allocation.memory(), allocation.offset())? };

    let device_address = unsafe { device.get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer)) };

    let mapped_slice = allocation
        .mapped_slice_mut()
        .expect("descriptor heap buffer must be host-visible and mapped");
    let mapped = NonNull::new(mapped_slice.as_mut_ptr()).expect("non-null mapped pointer");

    Ok((buffer, allocation, device_address, mapped))
}

///aligns v up to the nearest multiple of a without floating point for faster calc with the use of integer rounding
fn align_up(v: u64, a: u64) -> u64 {
    v.div_ceil(a) * a
}

fn lcm(a: u64, b: u64) -> u64 {
    if a == 0 || b == 0 { 0 } else { a / gcd(a, b) * b }
}

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}
