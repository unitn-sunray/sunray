use std::sync::Arc;

use crate::error::*;
use crate::vulkan_abstraction;
use crate::vulkan_abstraction::AccelerationStructure;
use ash::vk;

/// A pool of `ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR` queries used to drive
/// acceleration-structure compaction.
///
/// Compaction is intentionally a **three-step**, caller-driven flow so it can run
/// at a different moment / on a different queue from the build:
/// 1. Build the structure with `ALLOW_COMPACTION` set in its build flags.
/// 2. After the build has completed (insert a barrier so the query reads finished
///    data), record [`Self::cmd_reset_and_query`] into a command buffer, submit
///    it, then read the size back with [`Self::read_size`].
/// 3. Record the COMPACT copy with
///    [`AccelerationStructure::record_compact_copy`] / [`vulkan_abstraction::Blas::record_compaction`],
///    submit it, and drop the pre-compaction structure once that submission
///    completes.
pub struct CompactionQueryPool {
    core: Arc<vulkan_abstraction::Core>,
    pool: vk::QueryPool,
    capacity: u32,
}

impl CompactionQueryPool {
    pub fn new(core: Arc<vulkan_abstraction::Core>, capacity: u32) -> SrResult<Self> {
        let create_info = vk::QueryPoolCreateInfo::default()
            .query_type(vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR)
            .query_count(capacity);

        let pool = unsafe { core.device().inner().create_query_pool(&create_info, None) }?;

        Ok(Self { core, pool, capacity })
    }

    /// Record (into `cmd_buf`) a reset of the whole pool followed by a
    /// compacted-size query for each structure in `structures` (query `i` ←
    /// `structures[i]`). The builds of those structures MUST have completed —
    /// with an appropriate `ACCELERATION_STRUCTURE_BUILD` → `…READ` barrier —
    /// before `cmd_buf` executes this.
    pub fn cmd_reset_and_query(&self, cmd_buf: vk::CommandBuffer, structures: &[&AccelerationStructure]) {
        debug_assert!(structures.len() as u32 <= self.capacity);
        let handles: Vec<vk::AccelerationStructureKHR> = structures.iter().map(|s| s.inner()).collect();
        unsafe {
            self.core
                .device()
                .inner()
                .cmd_reset_query_pool(cmd_buf, self.pool, 0, self.capacity);
            self.core
                .acceleration_structure_device()
                .cmd_write_acceleration_structures_properties(
                    cmd_buf,
                    &handles,
                    vk::QueryType::ACCELERATION_STRUCTURE_COMPACTED_SIZE_KHR,
                    self.pool,
                    0,
                );
        }
    }

    /// Read back the compacted size written into query `index`. Blocks (WAIT)
    /// until the result is available, so the command buffer that recorded the
    /// query must already have been submitted.
    pub fn read_size(&self, index: u32) -> SrResult<vk::DeviceSize> {
        let mut result: [vk::DeviceSize; 1] = [0];
        unsafe {
            self.core.device().inner().get_query_pool_results(
                self.pool,
                index,
                &mut result,
                vk::QueryResultFlags::WAIT | vk::QueryResultFlags::TYPE_64,
            )?;
        }
        Ok(result[0])
    }
}

impl Drop for CompactionQueryPool {
    fn drop(&mut self) {
        unsafe {
            self.core.device().inner().destroy_query_pool(self.pool, None);
        }
    }
}
