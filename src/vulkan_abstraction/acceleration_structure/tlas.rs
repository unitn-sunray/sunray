use crate::error::*;
use crate::vulkan_abstraction;
use crate::vulkan_abstraction::acceleration_structure::{AsState, BuildType, OpType};
use crate::vulkan_abstraction::descriptor_heap::{DescriptorSlot, ResourceDescriptorKind};
use crate::vulkan_abstraction::{AccelerationStructure, AsBuildInputs, AsBuildJob, Buffer};
use ash::vk;
use std::sync::Arc;
// Resources:
// - https://github.com/adrien-ben/vulkan-examples-rs
// - https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/

pub struct Tlas {
    /// The live structure, held behind an `Arc` so the render graph can import it
    /// (as a synchronization resource) while a build/update job writes it — see
    /// [`Self::queue_build`]. A rebuild swaps in a fresh `Arc`; an in-place update
    /// keeps the same one.
    accel: Arc<AccelerationStructure>,
    slot: DescriptorSlot,
    build_type: BuildType,
    /// Instance count the current structure was last built/updated for. An
    /// in-place UPDATE requires the count to be unchanged, so a change forces a
    /// rebuild (see [`Self::queue_build`]).
    last_count: u32,
    /// Shared rebuild-vs-update heuristic state (see [`AsState`]).
    state: AsState,
    /// The build op currently in flight (recorded but not yet observed complete),
    /// folded back into `state` by [`Self::mark_built`]. `None` == nothing pending.
    op: Option<OpType>,
}

/// Plain-data description of a TLAS build (instances buffer address + count). No
/// handles, no lifetimes — carried only as the phantom `Desc` on a
/// `Handle<AccelerationStructure>` (see [`super::ASDesc`]).
/// TODO this desc is lacking information like min size and actual size
#[derive(Debug, Clone)]
pub struct TlasBuildDesc {
    pub instances_address: vk::DeviceAddress,
    pub instance_count: u32,
}

impl Tlas {
    /// Build a TLAS over the `instance_count` instances already written into
    /// `instances_buffer`
    pub fn new(
        core: Arc<vulkan_abstraction::Core>,
        instances_buffer: &impl Buffer,
        instance_count: u32,
        build_type: BuildType,
    ) -> SrResult<Self> {
        let accel = Arc::new(AccelerationStructure::build_sync(
            Arc::clone(&core),
            Self::make_inputs(instances_buffer, instance_count, build_type),
        )?);

        let slot = {
            let mut heap = core.descriptor_heap_mut();
            let slot = heap.alloc_resource_slot(ResourceDescriptorKind::AccelerationStructure);
            heap.write_acceleration_structure(slot, accel.device_address())?;
            slot
        };

        Ok(Self {
            accel,
            slot,
            build_type,
            last_count: instance_count,
            // Built synchronously here — no op is in flight for `mark_built` to observe.
            state: AsState::initial(build_type),
            op: None,
        })
    }

    /// Fold the operation currently in flight back into the heuristic state and
    /// clear it. Run once per frame from an end-of-frame closure after the
    /// build/update job has completed on the GPU; `None` (an idle frame) advances
    /// the settle counter. Mirrors [`Blas::mark_built`]. See [`AsState::mark_built`].
    #[allow(dead_code)]
    pub fn mark_built(&mut self) {
        self.state.mark_built(self.op.take());
    }

    /// The operation the heuristic wants recorded this frame given whether the
    /// instances changed (`None` = nothing to do), stored into `self.op` so the
    /// matching [`Self::mark_built`] can fold it back in on completion.
    #[allow(dead_code)]
    pub fn plan_op(&mut self, inputs_changed: bool) -> Option<OpType> {
        self.op = self.state.next_op(inputs_changed);
        self.op
    }

    /// Rebuild the TLAS from instances already written into `instances_buffer`
    /// (the renderer's frame-local buffer). Synchronous. Replaces the underlying
    /// structure and re-points the heap slot at the new structure's address.
    ///
    /// The old structure is dropped immediately; the renderer waits for device
    /// idle before this, so no in-flight frame still references it.
    pub fn rebuild_from_buffer(&mut self, instance_count: u32, instances_buffer: &impl Buffer) -> SrResult<()> {
        let accel = AccelerationStructure::build_sync(
            Arc::clone(self.accel.core()),
            Self::make_inputs(instances_buffer, instance_count, self.build_type),
        )?;

        self.accel = Arc::new(accel);
        self.last_count = instance_count;
        // A rebuild yields a new handle/address — re-point the heap slot so it
        // never goes stale (harmless to the live RT path, which reads
        // `device_address()` fresh as a push constant, but correct for any
        // heap-descriptor consumer).
        self.write_slot()?;

        log::debug!("TOP_LEVEL acceleration structure rebuilt");
        Ok(())
    }

    /// In-place SYNCHRONOUS UPDATE of the TLAS from instances already written into
    /// `instances_buffer` — same instance count / layout, new contents
    /// (transforms, BLAS references). Requires the TLAS was built with a
    /// [`BuildType`] that sets `ALLOW_UPDATE`. Mirrors `Blas::update`: cheaper
    /// than a full rebuild and, since an UPDATE keeps the same handle/address, the
    /// heap slot stays valid so no re-point is needed.
    #[allow(unused)]
    pub fn update(&mut self, instance_count: u32, instances_buffer: &impl Buffer) -> SrResult<()> {
        if !Self::build_flags(self.build_type).contains(vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE) {
            return Err(SrError::new_custom("The structure is not updatable".to_string()));
        }

        self.accel
            .update_sync(Self::make_inputs(instances_buffer, instance_count, self.build_type))?;

        log::debug!("TOP_LEVEL acceleration structure updated in place");
        Ok(())
    }

    /// **Deferred** build/update for the render graph — the entry point
    /// `ResourceManager::queue_tlas_build` records into the graph this frame.
    ///
    /// Picks the operation with the shared heuristic ([`AsState::next_op`]): the
    /// per-frame instances buffer is always freshly written, so the inputs are
    /// treated as changed, and the choice is between an in-place UPDATE and a full
    /// (fast) rebuild. A change in `instance_count` (or a non-`ALLOW_UPDATE` build
    /// type) forces a rebuild, since an UPDATE requires the same instance layout.
    ///
    /// On UPDATE the same structure is kept (handle/address unchanged); on a
    /// rebuild a fresh structure is swapped into `self.accel` **now** (its address
    /// is valid immediately, so it can be baked into this frame's push constants)
    /// and the heap slot is re-pointed. Returns:
    ///   - an `Arc` clone of the live structure for the graph to import as the
    ///     build pass's write target (and the RT pass's read),
    ///   - its device address (for the RT push constant), and
    ///   - the [`AsBuildJob`] the graph records into its command buffer.
    ///
    /// `self.op` is set to the operation actually chosen; the caller folds it back
    /// in with [`Self::mark_built`] from an end-of-frame closure once the job's
    /// submission has completed.
    pub fn queue_build(
        &mut self,
        instance_count: u32,
        instances_buffer: &impl Buffer,
    ) -> SrResult<(Arc<AccelerationStructure>, vk::DeviceAddress, AsBuildJob)> {
        let updatable = Self::build_flags(self.build_type).contains(vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE);
        let can_update = updatable && instance_count == self.last_count;

        // `inputs_changed = true`: a fresh instances buffer every frame. With that,
        // `next_op` never returns `SlowBuild`/`None`, so this resolves to Update
        // (when permitted) or FastBuild (heuristic churn cap, or a forced rebuild).
        let op = match self.state.next_op(true) {
            Some(OpType::Update) if can_update => OpType::Update,
            Some(OpType::Update) => OpType::FastBuild,
            Some(other) => other,
            None if can_update => OpType::Update,
            None => OpType::FastBuild,
        };

        let inputs = Self::make_inputs(instances_buffer, instance_count, self.build_type);
        let job = match op {
            OpType::Update => self.accel.update(inputs)?,
            OpType::FastBuild | OpType::SlowBuild => {
                let (new_accel, job) = AccelerationStructure::build(Arc::clone(self.accel.core()), inputs)?;
                // Previous-frame's graph import of the old structure was dropped by
                // `RenderGraph::reset`, and its submission has completed, so swapping
                // (and dropping the old `Arc`) here is safe.
                self.accel = Arc::new(new_accel);
                self.last_count = instance_count;
                self.write_slot()?;
                job
            }
        };

        self.op = Some(op);
        Ok((Arc::clone(&self.accel), self.accel.device_address(), job))
    }

    /// Map a [`BuildType`] to its Vulkan build flags. `SometimesChanges`
    /// reproduces the pre-rework `allow_update = true, fast_build = false` path.
    fn build_flags(build_type: BuildType) -> vk::BuildAccelerationStructureFlagsKHR {
        match build_type {
            BuildType::RapidlyChanging => {
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_BUILD | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE
            }
            BuildType::SometimesChanges => {
                vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE | vk::BuildAccelerationStructureFlagsKHR::ALLOW_UPDATE
            }
            BuildType::Static => vk::BuildAccelerationStructureFlagsKHR::PREFER_FAST_TRACE,
        }
    }

    /// (Re)write the current structure's device address into the heap slot.
    fn write_slot(&self) -> SrResult<()> {
        self.accel
            .core()
            .descriptor_heap_mut()
            .write_acceleration_structure(self.slot, self.accel.device_address())
    }

    /// Realize the owned build inputs for a TLAS over `instance_count` instances
    /// in `instances_buffer`. The geometry stores only the buffer's device
    /// address, so the `'static` geometry struct borrows nothing.
    fn make_inputs(instances_buffer: &impl Buffer, instance_count: u32, build_type: BuildType) -> AsBuildInputs {
        AsBuildInputs {
            ty: vk::AccelerationStructureTypeKHR::TOP_LEVEL,
            flags: Self::build_flags(build_type),
            geometries: vec![Self::make_geometry(instances_buffer)],
            ranges: vec![Self::make_build_range_info(instance_count)],
        }
    }

    fn make_geometry(instances_buffer: &impl Buffer) -> vk::AccelerationStructureGeometryKHR<'static> {
        vk::AccelerationStructureGeometryKHR::default()
            .geometry_type(vk::GeometryTypeKHR::INSTANCES)
            .flags(vk::GeometryFlagsKHR::empty())
            .geometry(vk::AccelerationStructureGeometryDataKHR {
                instances: vk::AccelerationStructureGeometryInstancesDataKHR::default()
                    .array_of_pointers(false)
                    .data(vk::DeviceOrHostAddressConstKHR {
                        device_address: instances_buffer.get_device_address(),
                    }),
            })
    }

    fn make_build_range_info(primitive_count: u32) -> vk::AccelerationStructureBuildRangeInfoKHR {
        vk::AccelerationStructureBuildRangeInfoKHR::default()
            .primitive_count(primitive_count)
            .primitive_offset(0)
            .first_vertex(0)
            .transform_offset(0)
    }

    pub fn inner(&self) -> vk::AccelerationStructureKHR {
        self.accel.inner()
    }

    /// `vkGetAccelerationStructureDeviceAddressKHR` of the underlying TLAS
    /// (cached). Used by the heap-mode RT pipelines because Slang's
    /// `DescriptorHandle<RaytracingAccelerationStructure>` codegen is broken on
    /// `spvDescriptorHeapEXT` (Slang issue #10671) — the shader does the
    /// uint64→AS convert via inline SPIR-V instead.
    pub fn device_address(&self) -> vk::DeviceAddress {
        self.accel.device_address()
    }

    pub fn slot(&self) -> u32 {
        self.slot.shader_index()
    }
}

impl Drop for Tlas {
    fn drop(&mut self) {
        self.accel.core().descriptor_heap_mut().free(self.slot);
    }
}
