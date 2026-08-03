pub mod device;
pub mod instance;
pub mod queue;
pub mod queues;

pub use device::*;
use gpu_allocator::vulkan::{Allocator, AllocatorCreateDesc};
pub use instance::*;
pub use queues::*;

use crate::vulkan_abstraction;
use crate::vulkan_abstraction::Queue;
use crate::vulkan_abstraction::diagnostics::DiagnosticTool;
use crate::{CreateSurfaceFn, error::*};
use ash::{ext, khr, vk};
use parking_lot::RawMutex;
use parking_lot::lock_api::MutexGuard;
use std::cell::{Ref, RefCell, RefMut};
use std::ffi::CStr;
use std::rc::Rc;

#[rustfmt::skip]
pub struct Core {
    //TODO core is completely single thread
    //TODO core gets distributed way too often when only the device is needed most of the time
    
    // Note: do not reorder the fields in this struct: they will be dropped in the same order they are declared
    pub absolute_frame_count: RefCell<usize>,

    acceleration_structure_device: khr::acceleration_structure::Device,
    ray_tracing_pipeline_device: khr::ray_tracing_pipeline::Device,
    descriptor_heap_device: ext::descriptor_heap::Device, //TODO don't know where to put these params as the almost seem more fit into the descriptor heap and this whole thing could even go in resource manager
    descriptor_heap_instance: ext::descriptor_heap::Instance,
    descriptor_heap: RefCell<vulkan_abstraction::DescriptorHeap>,

    queues: vulkan_abstraction::Queues,

    allocator: RefCell<Allocator>,

    device: Rc<vulkan_abstraction::Device>,
    instance: vulkan_abstraction::Instance,
    entry: ash::Entry,
}

impl Core {
    pub fn new(with_validation_layer: bool, with_gpuav: bool, image_format: vk::Format) -> SrResult<Self> {
        Ok(Self::new_with_surface(
            with_validation_layer,
            with_gpuav,
            DiagnosticTool::None,
            image_format,
            &[],
            None,
        )?
        .0)
    }

    // It is necessary to pass a function to create the surface, because surface depends on instance,
    // device depends on surface (if present), and both device and instance are created and owned inside
    // Core so this seems to be the best approach to allow the user to build its own surface.
    pub fn new_with_surface(
        with_validation_layer: bool,
        with_gpuav: bool,
        diagnostics: DiagnosticTool,
        image_format: vk::Format,
        required_instance_extensions: &[*const i8],
        create_surface: Option<&CreateSurfaceFn>,
    ) -> SrResult<(Self, Option<vk::SurfaceKHR>)> {
        let entry = ash::Entry::linked();

        let mut instance = vulkan_abstraction::Instance::new(
            &entry,
            required_instance_extensions,
            with_validation_layer,
            with_gpuav,
            diagnostics,
        )?;

        let surface_support = match create_surface.as_ref() {
            Some(f) => Some((
                f(&entry, instance.inner())?,
                khr::surface::Instance::load(&entry, instance.inner()),
            )),
            None => None,
        };

        let raytracing_device_extensions = [
            khr::ray_tracing_pipeline::NAME,
            khr::acceleration_structure::NAME,
            khr::deferred_host_operations::NAME,
            ext::descriptor_heap::NAME,
            // Required by SPV_KHR_untyped_pointers, which SPV_EXT_descriptor_heap depends on.
            // The Slang heap-mode codegen emits OpUntyped* ops, so without this the SPIR-V
            // module is rejected at vkCreateShaderModule.
            vk::KHR_SHADER_UNTYPED_POINTERS_NAME,
        ]
        .map(CStr::as_ptr);

        let mut device_extensions = raytracing_device_extensions.to_vec();

        if surface_support.is_some() {
            device_extensions.push(khr::swapchain::NAME.as_ptr());
        }

        // Diagnostic-tool device extensions (e.g. NV_device_diagnostics_config +
        // NV_device_diagnostic_checkpoints when Aftermath is selected).
        for ext_name in diagnostics.device_extensions() {
            device_extensions.push(ext_name.as_ptr());
        }

        let device = Rc::new(Device::new(
            &instance,
            &device_extensions,
            diagnostics,
            image_format,
            &surface_support,
        )?);

        // Load device-side checkpoint + debug-utils function pointers (no-op when
        // diagnostics == None and debug_utils is off).
        {
            let inst_handle = instance.inner().clone();
            let dev_handle = device.inner().clone();
            let debug_utils_available = instance.debug_utils_enabled();
            instance
                .diagnostics_mut()
                .load_device(&inst_handle, &dev_handle, debug_utils_available);
        }

        let mut allocator = Allocator::new(&AllocatorCreateDesc {
            instance: instance.inner().clone(),
            device: device.inner().clone(),
            physical_device: device.physical_device(),
            debug_settings: Default::default(),
            // NOTE: Ideally, check the BufferDeviceAddressFeatures struct.
            buffer_device_address: true,
            allocation_sizes: Default::default(),
        })?;

        let acceleration_structure_device = khr::acceleration_structure::Device::load(instance.inner(), device.inner());
        let ray_tracing_pipeline_device = khr::ray_tracing_pipeline::Device::load(instance.inner(), device.inner());
        let descriptor_heap_device = ext::descriptor_heap::Device::load(instance.inner(), device.inner());
        let descriptor_heap_instance = ext::descriptor_heap::Instance::load(&entry, instance.inner());

        let descriptor_heap = vulkan_abstraction::DescriptorHeap::new(
            device.inner(),
            &descriptor_heap_device,
            &mut allocator,
            device.descriptor_heap_properties(),
            vulkan_abstraction::DEFAULT_IMAGE_CAPACITY,
            vulkan_abstraction::DEFAULT_TEXEL_BUFFER_CAPACITY,
            vulkan_abstraction::DEFAULT_BUFFER_CAPACITY,
            vulkan_abstraction::DEFAULT_SAMPLER_CAPACITY,
            with_gpuav,
        )?;

        // Queues are deduplicated by family inside `Queues::new`: dedicated transfer /
        // async-compute families get their own VkQueue + command pool, and any role the
        // GPU doesn't expose aliases the graphics queue (sharing its mutex and pool).
        let queues = vulkan_abstraction::Queues::new(&device)?;

        Ok((
            Self {
                absolute_frame_count: RefCell::new(0),
                entry,
                instance,
                device,
                allocator: RefCell::new(allocator),
                acceleration_structure_device,
                ray_tracing_pipeline_device,
                descriptor_heap_device,
                descriptor_heap_instance,
                descriptor_heap: RefCell::new(descriptor_heap),
                queues,
            },
            surface_support.map(|(s, _)| s),
        ))
    }

    #[allow(unused)]
    pub fn entry(&self) -> &ash::Entry {
        &self.entry
    }

    #[allow(unused)]
    pub fn instance(&self) -> &ash::Instance {
        self.instance.inner()
    }

    pub fn device(&self) -> &Rc<vulkan_abstraction::Device> {
        &self.device
    }

    pub fn clone_device(&self) -> Rc<vulkan_abstraction::Device> {
        self.device.clone()
    }

    pub fn acceleration_structure_device(&self) -> &khr::acceleration_structure::Device {
        &self.acceleration_structure_device
    }
    pub fn rt_pipeline_device(&self) -> &khr::ray_tracing_pipeline::Device {
        &self.ray_tracing_pipeline_device
    }
    pub fn queues(&self) -> &vulkan_abstraction::Queues {
        &self.queues
    }

    pub fn graphics_queue(&self) -> MutexGuard<'_, RawMutex, Queue> {
        self.queues.graphics()
    }

    pub fn transfer_queue(&self) -> MutexGuard<'_, RawMutex, Queue> {
        self.queues.transfer()
    }

    pub fn async_compute_queue(&self) -> MutexGuard<'_, RawMutex, Queue> {
        self.queues.async_compute()
    }

    pub fn allocator(&self) -> Ref<'_, Allocator> {
        self.allocator.borrow()
    }
    pub fn allocator_mut(&self) -> RefMut<'_, Allocator> {
        self.allocator.borrow_mut()
    }

    pub fn graphics_cmd_pool(&self) -> &vulkan_abstraction::CmdPool {
        self.queues.graphics_pool()
    }

    pub fn descriptor_heap(&self) -> Ref<'_, vulkan_abstraction::DescriptorHeap> {
        self.descriptor_heap.borrow()
    }

    pub fn descriptor_heap_mut(&self) -> RefMut<'_, vulkan_abstraction::DescriptorHeap> {
        self.descriptor_heap.borrow_mut()
    }

    pub fn descriptor_heap_device(&self) -> &ext::descriptor_heap::Device {
        &self.descriptor_heap_device
    }

    pub fn descriptor_heap_instance(&self) -> &ext::descriptor_heap::Instance {
        &self.descriptor_heap_instance
    }

    pub fn transfer_cmd_pool(&self) -> &vulkan_abstraction::CmdPool {
        self.queues.transfer_pool()
    }

    pub fn async_compute_cmd_pool(&self) -> &vulkan_abstraction::CmdPool {
        self.queues.async_compute_pool()
    }

    /// Insert a named checkpoint into the command stream — no-op unless a
    /// crash-analysis tool (e.g. NVIDIA Aftermath) is active. After a
    /// `VK_ERROR_DEVICE_LOST`, the driver reports which checkpoints had
    /// completed, narrowing down which dispatch faulted.
    pub fn cmd_set_checkpoint(&self, cmd: vk::CommandBuffer, label: &'static std::ffi::CStr) {
        self.instance.diagnostics().cmd_set_checkpoint(cmd, label);
    }

    /// Log every checkpoint that completed on the graphics queue before the
    /// last fault — call from a DEVICE_LOST handler to find the faulting
    /// dispatch. Cheap to call even when no diagnostic tool is active.
    pub fn log_graphics_queue_checkpoints(&self) {
        let queue = self.queues.graphics().inner();
        self.instance.diagnostics().log_queue_checkpoints(queue);
    }

    pub fn diagnostic_tool(&self) -> vulkan_abstraction::DiagnosticTool {
        self.instance.diagnostics().tool()
    }

    /// Open a labeled command-buffer region for GPU captures (Nsight Graphics /
    /// RenderDoc). No-op without `VK_EXT_debug_utils`. Balance with
    /// [`Self::cmd_end_debug_label`].
    pub fn cmd_begin_debug_label(&self, cmd: vk::CommandBuffer, label: &std::ffi::CStr) {
        self.instance.diagnostics().cmd_begin_label(cmd, label);
    }

    /// Close the most recent [`Self::cmd_begin_debug_label`] region.
    pub fn cmd_end_debug_label(&self, cmd: vk::CommandBuffer) {
        self.instance.diagnostics().cmd_end_label(cmd);
    }

    /// Name a Vulkan object for GPU captures. No-op without `VK_EXT_debug_utils`.
    /// The object type is derived from the handle's `vk::Handle::TYPE`.
    pub fn set_debug_object_name<H: vk::Handle>(&self, handle: H, name: &std::ffi::CStr) {
        self.instance.diagnostics().set_object_name(handle, name);
    }

    /// Whether debug-utils labels/naming are active (a capture tool or
    /// validation enabled them). Cheap gate for label-emitting code paths.
    pub fn debug_labels_enabled(&self) -> bool {
        self.instance.diagnostics().labels_enabled()
    }
}

impl Drop for Core {
    fn drop(&mut self) {
        // Free descriptor-heap GPU allocations explicitly while the allocator is still alive.
        // Without this gpu-allocator panics on its own drop because of unfreed allocations.
        let mut heap = self.descriptor_heap.borrow_mut();
        let mut allocator = self.allocator.borrow_mut();
        heap.shutdown(&mut allocator);
    }
}
