use parking_lot::{RwLock, RwLockReadGuard};
use std::{collections::HashSet, ffi::CStr};

use crate::vulkan_abstraction::diagnostics::{self, DiagnosticTool};
use crate::{error::*, vulkan_abstraction};
use ash::vk::TaggedStructure;
use ash::{
    khr,
    vk::{self, FormatFeatureFlags},
};

pub struct Device {
    device: ash::Device,
    physical_device_memory_properties: vk::PhysicalDeviceMemoryProperties,
    physical_device: vk::PhysicalDevice,
    physical_device_properties: vk::PhysicalDeviceProperties,
    physical_device_rt_pipeline_properties: vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static>,
    physical_device_acceleration_structure_properties: vk::PhysicalDeviceAccelerationStructurePropertiesKHR<'static>,
    physical_device_descriptor_heap_properties: vk::PhysicalDeviceDescriptorHeapPropertiesEXT<'static>,
    graphics_queue_family_index: u32,
    transfer_queue_family_index: Option<u32>,
    async_compute_queue_family_index: Option<u32>,
    surface_support_details: Option<RwLock<SurfaceSupportDetails>>,
}

impl Device {
    pub fn new(
        instance: &vulkan_abstraction::Instance,
        device_extensions: &[*const i8],
        diagnostics: DiagnosticTool,
        image_format: vk::Format,
        surface_to_support: &Option<(vk::SurfaceKHR, khr::surface::Instance)>,
    ) -> SrResult<Self> {
        let instance = instance.inner();
        let physical_devices = unsafe { instance.enumerate_physical_devices() }?;

        let (
            physical_device,
            surface_support_details,
            graphics_queue_family_index,
            transfer_queue_family_index,
            async_compute_queue_family_index,
        ) = physical_devices
            .into_iter()
            //only allow devices which support all required extensions
            .filter(|physical_device| {
                Self::check_device_extension_support(instance, *physical_device, device_extensions).unwrap_or(false)
            })
            //check for blit support
            .filter(|physical_device| {
                let mut format_properties2 = vk::FormatProperties2::default();
                unsafe {
                    instance.get_physical_device_format_properties2(*physical_device, image_format, &mut format_properties2)
                };
                let format_properties = format_properties2.format_properties;

                format_properties
                    .optimal_tiling_features
                    .contains(FormatFeatureFlags::BLIT_SRC)
                    && format_properties
                        .linear_tiling_features
                        .contains(FormatFeatureFlags::BLIT_DST)
            })
            // filter out devices without swapchain support if necessary
            .filter_map(|physical_device| {
                if let Some((surface, surface_instance)) = &surface_to_support {
                    let surface_support_details =
                        SurfaceSupportDetails::new(*surface, surface_instance, physical_device).unwrap();
                    if surface_support_details.check_swapchain_support() {
                        Some((physical_device, Some(RwLock::new(surface_support_details))))
                    } else {
                        None
                    }
                } else {
                    Some((physical_device, None))
                }
            })
            //choose a suitable queue family, and filter out devices without one
            .filter_map(|(physical_device, surface_support_details)| {
                Some((
                    physical_device,
                    surface_support_details,
                    Self::select_graphics_queue_family(instance, physical_device)?,
                    Self::select_dedicated_transfer_queue(instance, physical_device),
                    Self::select_async_compute_queue(instance, physical_device),
                ))
            })
            // try to get a discrete or at least integrated gpu
            .max_by_key(
                |(
                    physical_device,
                    _surface_support_details,
                    _graphics_queue_family_index,
                    _transfer_queue_family_index,
                    _async_compute_queue_family_index,
                )| {
                    let mut props2 = vk::PhysicalDeviceProperties2::default();
                    unsafe { instance.get_physical_device_properties2(*physical_device, &mut props2) };
                    let device_type = props2.properties.device_type;

                    match device_type {
                        vk::PhysicalDeviceType::DISCRETE_GPU => 2,
                        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                        _ => 0,
                    }
                },
            )
            .ok_or(SrError::new_custom("No suitable GPU found!".to_string()))?;

        let device = {
            let graphics_priorities = [1.0];
            let transfer_priorities = [0.5];
            let async_compute_priorities = [0.5];

            let mut queue_create_infos = Vec::new();

            queue_create_infos.push(
                vk::DeviceQueueCreateInfo::default()
                    .queue_family_index(graphics_queue_family_index)
                    .queue_priorities(&graphics_priorities),
            );

            if let Some(actual_transfer_queue_family_index) = transfer_queue_family_index
                && graphics_queue_family_index != actual_transfer_queue_family_index
            {
                queue_create_infos.push(
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(actual_transfer_queue_family_index)
                        .queue_priorities(&transfer_priorities),
                );
            }

            if let Some(actual_async_compute_family_index) = async_compute_queue_family_index
                && graphics_queue_family_index != actual_async_compute_family_index
            {
                queue_create_infos.push(
                    vk::DeviceQueueCreateInfo::default()
                        .queue_family_index(actual_async_compute_family_index)
                        .queue_priorities(&async_compute_priorities),
                );
            }

            // maintenance9 would let the staging upload skip the transfer->graphics queue
            // ownership transfer (BestPractices-PipelineBarrier-unneeded-QFOT hint), but
            // enabling it crashes during init on this driver/loader stack (host
            // STATUS_ACCESS_VIOLATION right after descriptor-heap creation). Left disabled;
            // the QFOT in staging_buffer is therefore still required and correct.
            let mut maintenance9_features = vk::PhysicalDeviceMaintenance9FeaturesKHR::default().maintenance9(false);
            let mut vk14_features = vk::PhysicalDeviceVulkan14Features::default().maintenance5(true);
            // dynamic_rendering: the Bevy egui overlay paints into the swapchain image via
            // VK_KHR_dynamic_rendering (cmd_begin_rendering, no VkRenderPass/framebuffer).
            let mut vk13_features = vk::PhysicalDeviceVulkan13Features::default()
                .synchronization2(true)
                .dynamic_rendering(true);
            // enable some device features necessary for ray-tracing
            let mut vk12_features = vk::PhysicalDeviceVulkan12Features::default()
                .buffer_device_address(true) // necessary for ray-tracing
                .timeline_semaphore(true)
                .vulkan_memory_model(true)
                .vulkan_memory_model_device_scope(true)
                .storage_buffer8_bit_access(true)
                .shader_float16(true);
            let mut physical_device_rt_pipeline_features =
                vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::default().ray_tracing_pipeline(true);
            let mut physical_device_acceleration_structure_features =
                vk::PhysicalDeviceAccelerationStructureFeaturesKHR::default().acceleration_structure(true);
            let mut physical_device_descriptor_heap_features =
                vk::PhysicalDeviceDescriptorHeapFeaturesEXT::default().descriptor_heap(true);
            // VK_KHR_shader_untyped_pointers — required by SPV_KHR_untyped_pointers, which
            // SPV_EXT_descriptor_heap depends on. Slang emits OpUntyped* ops in heap-mode SPIR-V
            // so without this feature `vkCreateShaderModule` fails validation.
            let mut physical_device_shader_untyped_pointers_features =
                vk::PhysicalDeviceShaderUntypedPointersFeaturesKHR::default().shader_untyped_pointers(true);

            // shader_storage_image_*_without_format: Slang lowers `RWTexture2D<float4>` to
            // `OpTypeImage ... Unknown` and the resulting SPIR-V advertises the matching
            // StorageImage{Read,Write}WithoutFormat capabilities. Without these features
            // enabled, the postprocess dispatch silently produces zeros.
            //
            // shader_*_array_dynamic_indexing: temporal_accumulation.glsl indexes
            // `accumulation_images[accum_idx]` (storage image array) and
            // `history_samplers[history_idx]` (sampler array) with push-constant indices.
            // Without these features dynamic indexing of those arrays is undefined
            // behavior — observed symptom was the temporal pass producing near-zero
            // output, which then made denoise/postprocess look black.
            let mut physical_device_features = vk::PhysicalDeviceFeatures2::default().features(
                vk::PhysicalDeviceFeatures::default()
                    .sampler_anisotropy(true)
                    .shader_storage_image_read_without_format(true)
                    .shader_storage_image_write_without_format(true)
                    // r11f_g11f_b10f / rg16f are storage formats from the extended set;
                    // without this feature the SPIR-V capability `StorageImageExtendedFormats`
                    // is unmet and operations like `imageSize` return zero.
                    .shader_storage_image_extended_formats(true)
                    .shader_uniform_buffer_array_dynamic_indexing(true)
                    .shader_sampled_image_array_dynamic_indexing(true)
                    .shader_storage_buffer_array_dynamic_indexing(true)
                    .shader_storage_image_array_dynamic_indexing(true)
                    .shader_int16(true)
                    // RT shaders use 64-bit buffer-device-addresses (tlas / matrices /
                    // reservoirs in RaytracingPC), so their SPIR-V declares the Int64
                    // capability; enabling shaderInt64 satisfies VUID-VkShaderModuleCreateInfo-pCode-08740.
                    .shader_int64(true),
            );

            // Optional diagnostics p_next (currently only NVIDIA Aftermath). The struct
            // outlives the create call because it lives in this binding.
            let mut diagnostics_config = diagnostics::device_diagnostics_p_next(diagnostics);

            let mut device_create_info = vk::DeviceCreateInfo::default()
                .enabled_extension_names(device_extensions)
                .push(&mut vk12_features)
                .push(&mut vk13_features)
                .push(&mut physical_device_rt_pipeline_features)
                .push(&mut physical_device_acceleration_structure_features)
                .push(&mut physical_device_descriptor_heap_features)
                .push(&mut physical_device_shader_untyped_pointers_features)
                .push(&mut physical_device_features)
                .push(&mut maintenance9_features)
                .push(&mut vk14_features)
                .queue_create_infos(&queue_create_infos);

            if let Some(cfg) = diagnostics_config.as_mut() {
                device_create_info = device_create_info.push(cfg);
            }

            unsafe { instance.create_device(physical_device, &device_create_info, None) }?
        };
        //necessary for memory allocations
        let physical_device_memory_properties = {
            let mut mem_props2 = vk::PhysicalDeviceMemoryProperties2::default();
            unsafe { instance.get_physical_device_memory_properties2(physical_device, &mut mem_props2) };
            mem_props2.memory_properties
        };

        let (
            physical_device_properties,
            physical_device_rt_pipeline_properties,
            physical_device_acceleration_structure_properties,
            physical_device_descriptor_heap_properties,
        ) = {
            let mut physical_device_rt_pipeline_properties = vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
            let mut physical_device_acceleration_structure_properties =
                vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
            let mut physical_device_descriptor_heap_properties = vk::PhysicalDeviceDescriptorHeapPropertiesEXT::default();

            let mut physical_device_properties = vk::PhysicalDeviceProperties2::default()
                .push(&mut physical_device_rt_pipeline_properties)
                .push(&mut physical_device_acceleration_structure_properties)
                .push(&mut physical_device_descriptor_heap_properties);

            unsafe { instance.get_physical_device_properties2(physical_device, &mut physical_device_properties) };

            (
                physical_device_properties.properties,
                physical_device_rt_pipeline_properties,
                physical_device_acceleration_structure_properties,
                physical_device_descriptor_heap_properties,
            )
        };

        Ok(Self {
            device,
            physical_device,
            physical_device_properties,
            physical_device_memory_properties,
            physical_device_rt_pipeline_properties,
            physical_device_acceleration_structure_properties,
            physical_device_descriptor_heap_properties,
            graphics_queue_family_index,
            transfer_queue_family_index,
            async_compute_queue_family_index,
            surface_support_details,
        })
    }

    fn queue_family_properties2(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Vec<vk::QueueFamilyProperties> {
        let count = unsafe { instance.get_physical_device_queue_family_properties2_len(physical_device) };
        let mut props2 = vec![vk::QueueFamilyProperties2::default(); count];
        unsafe { instance.get_physical_device_queue_family_properties2(physical_device, &mut props2) };
        props2.into_iter().map(|p| p.queue_family_properties).collect()
    }

    fn select_graphics_queue_family(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Option<u32> {
        Self::queue_family_properties2(instance, physical_device)
            .into_iter()
            .enumerate()
            .filter(|(_queue_family_index, queue_family_props)| queue_family_props.queue_flags.contains(vk::QueueFlags::GRAPHICS))
            .map(|(queue_family_index, _)| queue_family_index as u32)
            .next()
    }

    fn select_dedicated_transfer_queue(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Option<u32> {
        Self::queue_family_properties2(instance, physical_device)
            .into_iter()
            .enumerate()
            .filter(|(_queue_family_index, queue_family_props)| {
                queue_family_props.queue_flags.contains(vk::QueueFlags::TRANSFER)
                    && !queue_family_props.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && !queue_family_props.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            })
            .map(|(queue_family_index, _)| queue_family_index as u32)
            .next()
    }
    fn select_async_compute_queue(instance: &ash::Instance, physical_device: vk::PhysicalDevice) -> Option<u32> {
        Self::queue_family_properties2(instance, physical_device)
            .into_iter()
            .enumerate()
            .filter(|(_queue_family_index, queue_family_props)| {
                queue_family_props.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    && !queue_family_props.queue_flags.contains(vk::QueueFlags::GRAPHICS)
            })
            .map(|(queue_family_index, _)| queue_family_index as u32)
            .next()
    }

    fn check_device_extension_support(
        instance: &ash::Instance,
        physical_device: vk::PhysicalDevice,
        required_device_extensions: &[*const i8],
    ) -> SrResult<bool> {
        let required_exts_set: HashSet<&CStr> = required_device_extensions
            .iter()
            .map(|p| unsafe { CStr::from_ptr(*p) })
            .collect();

        let available_exts = unsafe { instance.enumerate_device_extension_properties(physical_device) }?;

        let available_exts_set: HashSet<&CStr> = available_exts
            .iter()
            .map(|props| props.extension_name_as_c_str())
            .collect::<Result<_, _>>()
            .map_err(|e| {
                SrError::new_custom(format!(
                    "Error while checking device extension support. Could not convert extension name to CStr with message: {e}"
                ))
            })?;

        let all_exts_are_available = required_exts_set.is_subset(&available_exts_set);

        Ok(all_exts_are_available)
    }

    pub fn inner(&self) -> &ash::Device {
        &self.device
    }
    pub fn physical_device(&self) -> vk::PhysicalDevice {
        self.physical_device
    }
    pub fn properties(&self) -> &vk::PhysicalDeviceProperties {
        &self.physical_device_properties
    }
    pub fn rt_pipeline_properties(&self) -> &vk::PhysicalDeviceRayTracingPipelinePropertiesKHR<'static> {
        &self.physical_device_rt_pipeline_properties
    }

    pub fn acceleration_structure_properties(&self) -> &vk::PhysicalDeviceAccelerationStructurePropertiesKHR<'static> {
        &self.physical_device_acceleration_structure_properties
    }

    pub fn descriptor_heap_properties(&self) -> &vk::PhysicalDeviceDescriptorHeapPropertiesEXT<'static> {
        &self.physical_device_descriptor_heap_properties
    }

    pub fn memory_properties(&self) -> &vk::PhysicalDeviceMemoryProperties {
        &self.physical_device_memory_properties
    }
    pub fn graphics_queue_family_index(&self) -> u32 {
        self.graphics_queue_family_index
    }
    pub fn transfer_queue_family_index(&self) -> Option<u32> {
        self.transfer_queue_family_index
    }
    pub fn async_compute_queue_family_index(&self) -> Option<u32> {
        self.async_compute_queue_family_index
    }

    /// `read_recursive`, not `read`: callers routinely hold two of these guards at
    /// once (e.g. `build_swapchain` reads `surface_formats` and `surface_capabilities`
    /// in the same scope). Plain `read` can deadlock on the second one if a writer
    /// has queued in between; this used to be a `RefCell`, where overlapping shared
    /// borrows were always legal.
    pub fn surface_support_details(&self) -> RwLockReadGuard<'_, SurfaceSupportDetails> {
        self.surface_support_details.as_ref().unwrap().read_recursive()
    }
    pub fn update_surface_support_details(&self, surface: vk::SurfaceKHR, surface_instance: &khr::surface::Instance) {
        *self.surface_support_details.as_ref().unwrap().write() =
            SurfaceSupportDetails::new(surface, surface_instance, self.physical_device).unwrap();
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_device(None);
        }
    }
}

pub struct SurfaceSupportDetails {
    pub surface_capabilities: vk::SurfaceCapabilitiesKHR,
    pub surface_formats: Vec<vk::SurfaceFormatKHR>,
    pub surface_present_modes: Vec<vk::PresentModeKHR>,
}

impl SurfaceSupportDetails {
    /*
    TODO: bad error handling.
    many different phases can cause vulkan errors in choosing a physical device.
    we don't keep track of which errors occur because if no suitable device is found we'd need to tell the user what error makes each device unsuitable
    */
    fn new(
        surface: vk::SurfaceKHR,
        surface_instance: &khr::surface::Instance,
        physical_device: vk::PhysicalDevice,
    ) -> SrResult<Self> {
        let surface_capabilities =
            unsafe { surface_instance.get_physical_device_surface_capabilities(physical_device, surface) }?;

        let surface_formats = unsafe { surface_instance.get_physical_device_surface_formats(physical_device, surface) }?;

        let surface_present_modes =
            unsafe { surface_instance.get_physical_device_surface_present_modes(physical_device, surface) }?;

        Ok(Self {
            surface_capabilities,
            surface_formats,
            surface_present_modes,
        })
    }

    fn check_swapchain_support(&self) -> bool {
        !self.surface_formats.is_empty() && !self.surface_present_modes.is_empty()
    }
}
