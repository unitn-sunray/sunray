use std::sync::Arc;

use crate::error::*;
use crate::vulkan_abstraction;
use crate::vulkan_abstraction::Buffer;
use ash::vk;

/// Owned build inputs for a single acceleration-structure build. Owns the
/// realized geometry/range arrays (rather than borrowing) so a deferred build
/// closure can be `'static` — the render graph holds the recording closure
/// across the frame, past the scope that produced the geometry. This is sound
/// because a triangle/instances geometry stores only **device addresses**, never
/// borrowed CPU memory, so `<'static>` here loses no real lifetime. Holds no
/// policy beyond `ty`/`flags`.
pub struct AsBuildInputs {
    pub ty: vk::AccelerationStructureTypeKHR,
    pub flags: vk::BuildAccelerationStructureFlagsKHR,
    pub geometries: Vec<vk::AccelerationStructureGeometryKHR<'static>>,
    pub ranges: Vec<vk::AccelerationStructureBuildRangeInfoKHR>,
}

/// A recorded-but-not-yet-submitted acceleration-structure build/update. The
/// destination handle already exists (so [`AccelerationStructure::device_address`]
/// is valid the moment the job is produced), but the
/// `vkCmdBuildAccelerationStructures` call is deferred into `record` so a caller
/// — the render graph — can batch it into its own command buffer instead of the
/// producing scope owning the submit.
///
/// The caller allocates a scratch buffer of at least `scratch_size` bytes that
/// satisfies `scratch_alignment`, then calls [`AsBuildJob::record`] exactly once.
/// Both the scratch buffer and the geometry buffers the build reads must stay
/// alive until that command buffer's submission completes.
/// Records one deferred acceleration-structure build into a command buffer,
/// given the scratch buffer the caller allocated for it.
type AsRecordFn = dyn FnOnce(vk::CommandBuffer, &vulkan_abstraction::GpuOnlyBuffer) + Send;

pub struct AsBuildJob {
    /// Minimum size, in bytes, of the scratch buffer passed to [`Self::record`].
    pub scratch_size: vk::DeviceSize,
    /// Alignment the scratch buffer's device address must satisfy.
    pub scratch_alignment: u64,
    record: Box<AsRecordFn>,
}

impl AsBuildJob {
    /// Record the deferred build into `cmd_buf`, using `scratch` as the build
    /// scratch. `scratch` must be at least [`Self::scratch_size`] bytes and
    /// satisfy [`Self::scratch_alignment`]. Consumes the job — a build is
    /// recorded exactly once.
    pub fn record(self, cmd_buf: vk::CommandBuffer, scratch: &vulkan_abstraction::GpuOnlyBuffer) {
        debug_assert!(
            scratch.byte_size() >= self.scratch_size,
            "scratch buffer too small: {} < required {}",
            scratch.byte_size(),
            self.scratch_size,
        );
        (self.record)(cmd_buf, scratch);
    }
}

/// A bare GPU acceleration-structure **resource**: just the handle, its backing
/// buffer, and the device address. No build description and no rebuild policy —
/// those belong to the owning wrapper ([`vulkan_abstraction::Blas`] /
/// [`vulkan_abstraction::Tlas`]). A future cluster BLAS, which has no
/// `vk::AccelerationStructureKHR` handle (only an address), is expected to be a
/// *sibling* resource type exposing the same `device_address()`.
pub struct AccelerationStructure {
    core: Arc<vulkan_abstraction::Core>,
    handle: vk::AccelerationStructureKHR,
    #[allow(dead_code)]
    buffer: vulkan_abstraction::GpuOnlyBuffer,
    device_address: vk::DeviceAddress,
}

impl AccelerationStructure {
    /// Allocate the backing buffer and create the (unbuilt) handle **now**, but
    /// **defer** the `vkCmdBuildAccelerationStructures` recording into the
    /// returned [`AsBuildJob`]. The device address is valid as soon as this
    /// returns (the handle exists on a bound buffer), so callers can bake it into
    /// instance data / push constants before the build has executed.
    ///
    /// The caller allocates a scratch buffer per the job's `scratch_size` /
    /// `scratch_alignment`, then calls [`AsBuildJob::record`] with a command
    /// buffer and that scratch. This is the seam the render graph runs the build
    /// through — the recording closure is `'static` (it owns `inputs`).
    pub fn build(core: Arc<vulkan_abstraction::Core>, inputs: AsBuildInputs) -> SrResult<(Self, AsBuildJob)> {
        assert_eq!(inputs.geometries.len(), inputs.ranges.len());
        let AsBuildInputs {
            ty,
            flags,
            geometries,
            ranges,
        } = inputs;

        // Sizing pass: the same geometry info (minus dst handle + scratch address)
        // that the deferred recording rebuilds once the scratch is known.
        let size_info = unsafe {
            let incomplete_build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .geometries(&geometries)
                .flags(flags)
                .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                .ty(ty);
            let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
            let primitive_counts = ranges.iter().map(|i| i.primitive_count).collect::<Vec<_>>();
            core.acceleration_structure_device().get_acceleration_structure_build_sizes(
                vk::AccelerationStructureBuildTypeKHR::DEVICE,
                &incomplete_build_geometry_info,
                Some(primitive_counts.as_slice()),
                &mut size_info,
            );
            size_info
        };

        let (handle, buffer, device_address) = Self::create_backed(&core, ty, size_info.acceleration_structure_size)?;
        let scratch_alignment = core
            .device()
            .acceleration_structure_properties()
            .min_acceleration_structure_scratch_offset_alignment as u64;

        // The recording closure owns `geometries` / `ranges` (both `'static`,
        // holding only device addresses) so it satisfies the graph's `'static`
        // render-closure bound. `handle` is `Copy` — the same handle lives both
        // in `Self` and here.
        let record_core = Arc::clone(&core);
        let record = Box::new(
            move |cmd_buf: vk::CommandBuffer, scratch: &vulkan_abstraction::GpuOnlyBuffer| {
                let build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                    .geometries(&geometries)
                    .flags(flags)
                    .mode(vk::BuildAccelerationStructureModeKHR::BUILD)
                    .ty(ty)
                    .dst_acceleration_structure(handle)
                    .scratch_data(vk::DeviceOrHostAddressKHR {
                        device_address: scratch.get_device_address(),
                    });

                unsafe {
                    record_core.acceleration_structure_device().cmd_build_acceleration_structures(
                        cmd_buf,
                        &[build_geometry_info],
                        &[Some(ranges.as_slice())],
                    );
                }
            },
        );

        Ok((
            Self {
                core,
                handle,
                buffer,
                device_address,
            },
            AsBuildJob {
                scratch_size: size_info.build_scratch_size,
                scratch_alignment,
                record,
            },
        ))
    }

    /// Convenience: build synchronously on the graphics queue (one-shot command
    /// buffer: allocate scratch, record + submit + wait + free). Behaviorally
    /// identical to the old `new_sync`; used by the classic path. Internally it
    /// just runs the deferred [`Self::build`] job on a throwaway command buffer.
    pub fn build_sync(core: Arc<vulkan_abstraction::Core>, inputs: AsBuildInputs) -> SrResult<Self> {
        let (accel, job) = Self::build(Arc::clone(&core), inputs)?;

        // The scratch must outlive the submit — it does, dropped at end of scope
        // after `submit_sync` has waited for the build to finish.
        let scratch = vulkan_abstraction::GpuOnlyBuffer::new_aligned::<u8>(
            Arc::clone(&core),
            job.scratch_size,
            job.scratch_alignment,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            "acceleration structure build scratch buffer",
        )?;

        let cmd_buf = vulkan_abstraction::cmd_buffer::new_command_buffer(core.graphics_cmd_pool(), core.device().inner())?;
        unsafe {
            core.device().inner().begin_command_buffer(
                cmd_buf,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }

        job.record(cmd_buf, &scratch);

        unsafe { core.device().inner().end_command_buffer(cmd_buf)? }

        // NOTE: synchronous submit — bad for throughput when many builds happen
        // back to back. The deferred `build` API exists precisely so callers can
        // batch into the render graph.
        core.graphics_queue().submit_sync(cmd_buf)?;

        unsafe {
            core.device()
                .inner()
                .free_command_buffers(core.graphics_cmd_pool().inner(), &[cmd_buf]);
        }

        Ok(accel)
    }

    /// Prepare an in-place UPDATE (src = dst = this handle) of `inputs`, deferring
    /// the `vkCmdBuildAccelerationStructures` recording into the returned
    /// [`AsBuildJob`] exactly like [`Self::build`]. `inputs.flags` must contain
    /// `ALLOW_UPDATE` and the geometry layout must match the original build (see
    /// the Vulkan spec restrictions). The handle/address is unchanged (an update
    /// mutates in place), so nothing about `self` needs to be swapped afterwards.
    ///
    /// Takes `&self` (not `&mut self`): an in-place UPDATE mutates the GPU-side
    /// structure but touches none of this wrapper's Rust fields, so it stays
    /// callable through a shared `Arc<AccelerationStructure>` (the graph holds one
    /// while the update job is in flight).
    #[allow(dead_code)]
    pub fn update(&self, inputs: AsBuildInputs) -> SrResult<AsBuildJob> {
        assert_eq!(inputs.geometries.len(), inputs.ranges.len());
        let AsBuildInputs {
            ty,
            flags,
            geometries,
            ranges,
        } = inputs;

        let size_info = unsafe {
            let incomplete_build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                .geometries(&geometries)
                .flags(flags)
                .mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
                .ty(ty);
            let mut size_info = vk::AccelerationStructureBuildSizesInfoKHR::default();
            let primitive_counts = ranges.iter().map(|i| i.primitive_count).collect::<Vec<_>>();
            self.core
                .acceleration_structure_device()
                .get_acceleration_structure_build_sizes(
                    vk::AccelerationStructureBuildTypeKHR::DEVICE,
                    &incomplete_build_geometry_info,
                    Some(primitive_counts.as_slice()),
                    &mut size_info,
                );
            size_info
        };

        let scratch_alignment = self
            .core
            .device()
            .acceleration_structure_properties()
            .min_acceleration_structure_scratch_offset_alignment as u64;

        let handle = self.handle;
        let record_core = Arc::clone(&self.core);
        let record = Box::new(
            move |cmd_buf: vk::CommandBuffer, scratch: &vulkan_abstraction::GpuOnlyBuffer| {
                let build_geometry_info = vk::AccelerationStructureBuildGeometryInfoKHR::default()
                    .geometries(&geometries)
                    .flags(flags)
                    .mode(vk::BuildAccelerationStructureModeKHR::UPDATE)
                    .ty(ty)
                    .src_acceleration_structure(handle)
                    .dst_acceleration_structure(handle)
                    .scratch_data(vk::DeviceOrHostAddressKHR {
                        device_address: scratch.get_device_address(),
                    });

                unsafe {
                    record_core.acceleration_structure_device().cmd_build_acceleration_structures(
                        cmd_buf,
                        &[build_geometry_info],
                        &[Some(ranges.as_slice())],
                    );
                }
            },
        );

        Ok(AsBuildJob {
            scratch_size: size_info.build_scratch_size,
            scratch_alignment,
            record,
        })
    }

    /// Convenience synchronous in-place UPDATE on the graphics queue. Runs the
    /// deferred [`Self::update`] job on a throwaway command buffer. `&self` for the
    /// same reason as [`Self::update`].
    #[allow(dead_code)]
    pub fn update_sync(&self, inputs: AsBuildInputs) -> SrResult<()> {
        let job = self.update(inputs)?;

        let scratch = vulkan_abstraction::GpuOnlyBuffer::new_aligned::<u8>(
            Arc::clone(&self.core),
            job.scratch_size,
            job.scratch_alignment,
            vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS | vk::BufferUsageFlags::STORAGE_BUFFER,
            "acceleration structure update scratch buffer",
        )?;

        let cmd_buf =
            vulkan_abstraction::cmd_buffer::new_command_buffer(self.core.graphics_cmd_pool(), self.core.device().inner())?;
        unsafe {
            self.core.device().inner().begin_command_buffer(
                cmd_buf,
                &vk::CommandBufferBeginInfo::default().flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
        }

        job.record(cmd_buf, &scratch);

        unsafe { self.core.device().inner().end_command_buffer(cmd_buf)? }
        self.core.graphics_queue().submit_sync(cmd_buf)?;
        unsafe {
            self.core
                .device()
                .inner()
                .free_command_buffers(self.core.graphics_cmd_pool().inner(), &[cmd_buf]);
        }

        Ok(())
    }

    /// Record a COMPACT copy of this structure into a freshly-allocated,
    /// minimum-sized buffer, returning the new (compacted) resource. `cmd_buf`
    /// is only *recorded* — the caller submits it (any queue, any time) and must
    /// keep `self` alive until that submission completes (the copy reads it).
    ///
    /// `ty` must match this structure's type (the resource layer no longer
    /// stores it — the owning wrapper knows it). `compacted_size` must come from
    /// a prior compacted-size query (see [`vulkan_abstraction::CompactionQueryPool`]),
    /// and this structure must have been built with `ALLOW_COMPACTION`.
    pub fn record_compact_copy(
        &self,
        cmd_buf: vk::CommandBuffer,
        ty: vk::AccelerationStructureTypeKHR,
        compacted_size: vk::DeviceSize,
    ) -> SrResult<Self> {
        let (handle, buffer, device_address) = Self::create_backed(&self.core, ty, compacted_size)?;

        let copy_info = vk::CopyAccelerationStructureInfoKHR::default()
            .src(self.handle)
            .dst(handle)
            .mode(vk::CopyAccelerationStructureModeKHR::COMPACT);

        unsafe {
            self.core
                .acceleration_structure_device()
                .cmd_copy_acceleration_structure(cmd_buf, &copy_info);
        }

        Ok(Self {
            core: Arc::clone(&self.core),
            handle,
            buffer,
            device_address,
        })
    }

    /// Allocate a backing buffer of `size`, create an AS handle of type `ty` on
    /// it, and read back its device address. Shared by build and compaction.
    fn create_backed(
        core: &Arc<vulkan_abstraction::Core>,
        ty: vk::AccelerationStructureTypeKHR,
        size: vk::DeviceSize,
    ) -> SrResult<(
        vk::AccelerationStructureKHR,
        vulkan_abstraction::GpuOnlyBuffer,
        vk::DeviceAddress,
    )> {
        let buffer = vulkan_abstraction::GpuOnlyBuffer::new::<u8>(
            Arc::clone(core),
            size,
            vk::BufferUsageFlags::ACCELERATION_STRUCTURE_STORAGE_KHR
                | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                | vk::BufferUsageFlags::STORAGE_BUFFER,
            Self::buffer_name(ty),
        )?;

        let create_info = vk::AccelerationStructureCreateInfoKHR::default()
            .ty(ty)
            .size(size)
            .buffer(buffer.inner())
            .offset(0)
            .create_flags(vk::AccelerationStructureCreateFlagsKHR::empty());

        let handle = unsafe {
            core.acceleration_structure_device()
                .create_acceleration_structure(&create_info, None)
        }?;

        // Valid as soon as the handle exists (on a bound buffer), independent of
        // whether the build has been submitted yet.
        let device_address = unsafe {
            core.acceleration_structure_device()
                .get_acceleration_structure_device_address(
                    &vk::AccelerationStructureDeviceAddressInfoKHR::default().acceleration_structure(handle),
                )
        };

        Ok((handle, buffer, device_address))
    }

    fn buffer_name(ty: vk::AccelerationStructureTypeKHR) -> &'static str {
        match ty {
            vk::AccelerationStructureTypeKHR::TOP_LEVEL => "TLAS buffer",
            vk::AccelerationStructureTypeKHR::BOTTOM_LEVEL => "BLAS buffer",
            vk::AccelerationStructureTypeKHR::GENERIC => "generic acceleration structure buffer",
            _ => "(unknown AS type) acceleration structure buffer",
        }
    }

    /// `vkGetAccelerationStructureDeviceAddressKHR`, cached at creation.
    pub fn device_address(&self) -> vk::DeviceAddress {
        self.device_address
    }

    pub fn inner(&self) -> vk::AccelerationStructureKHR {
        self.handle
    }

    pub fn core(&self) -> &Arc<vulkan_abstraction::Core> {
        &self.core
    }
}

impl Drop for AccelerationStructure {
    fn drop(&mut self) {
        unsafe {
            self.core
                .acceleration_structure_device()
                .destroy_acceleration_structure(self.handle, None);
        }
    }
}
