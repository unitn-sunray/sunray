use std::sync::Arc;

use ash::vk;
use parking_lot::lock_api::MutexGuard;
use parking_lot::{Mutex, RawMutex};

use crate::error::*;
use crate::vulkan_abstraction::{CmdPool, Device, Queue};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueueRole {
    Graphics,
    Transfer,
    AsyncCompute,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QueuesConf {
    GraphicsOnly,
    GraphicsAndTransfer,
    GraphicsAndAsyncCompute,
    GraphicsAsyncComputeAndTransfer,
}

/// Owns the queues used by [`Core`](crate::vulkan_abstraction::Core),
/// There exists 3 queues at most and 1 at least
/// A universal (graphics)
/// A transfer
/// A async compute
///
/// Note: Always ask for the [`Config`](QueuesConf) to make sure you are not asking for the same mutex lock twice
pub struct Queues {
    // one entry per distinct queue family actually used
    queues: Vec<Mutex<Queue>>,
    pools: Vec<CmdPool>,
    // role -> index into `queues`/`pools`; aliased roles share an index
    graphics: usize,
    transfer: usize,
    async_compute: usize,
}

impl Queues {
    pub fn new(device: &Arc<Device>) -> SrResult<Self> {
        let graphics_family = device.graphics_queue_family_index();

        let mut families: Vec<u32> = Vec::new();
        let mut queues: Vec<Mutex<Queue>> = Vec::new();
        let mut pools: Vec<CmdPool> = Vec::new();

        // Returns the index for `family`, creating the queue + command pool the first
        // time that family is seen. A distinct family yields a distinct `VkQueue`.
        let mut resolve = |family: u32| -> SrResult<usize> {
            if let Some(index) = families.iter().position(|&f| f == family) {
                return Ok(index);
            }
            queues.push(Mutex::new(Queue::new(Arc::clone(device), 0, family)?));
            pools.push(CmdPool::new(
                Arc::clone(device),
                family,
                vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER,
            )?);
            families.push(family);
            Ok(families.len() - 1)
        };

        // Graphics is always present; every absent role falls back to it.
        let graphics = resolve(graphics_family)?;
        let transfer = match device.transfer_queue_family_index() {
            Some(family) => resolve(family)?,
            None => graphics,
        };
        let async_compute = match device.async_compute_queue_family_index() {
            Some(family) => resolve(family)?,
            None => graphics,
        };

        Ok(Self {
            queues,
            pools,
            graphics,
            transfer,
            async_compute,
        })
    }

    fn index(&self, role: QueueRole) -> usize {
        match role {
            QueueRole::Graphics => self.graphics,
            QueueRole::Transfer => self.transfer,
            QueueRole::AsyncCompute => self.async_compute,
        }
    }

    /// Which queues are backed by their own family rather than aliasing graphics.
    /// Use this before locking two roles to avoid taking the same mutex twice.
    pub fn config(&self) -> QueuesConf {
        let has_transfer = self.transfer != self.graphics;
        let has_async_compute = self.async_compute != self.graphics;
        match (has_transfer, has_async_compute) {
            (false, false) => QueuesConf::GraphicsOnly,
            (true, false) => QueuesConf::GraphicsAndTransfer,
            (false, true) => QueuesConf::GraphicsAndAsyncCompute,
            (true, true) => QueuesConf::GraphicsAsyncComputeAndTransfer,
        }
    }

    /// Lock the queue serving `role`. Aliased roles return a guard on the same
    /// underlying mutex, so concurrent submits to a shared `VkQueue` are serialized.
    fn lock(&self, role: QueueRole) -> MutexGuard<'_, RawMutex, Queue> {
        self.queues[self.index(role)].lock()
    }

    /// The command pool matching `role`'s queue family.
    fn pool(&self, role: QueueRole) -> &CmdPool {
        &self.pools[self.index(role)]
    }

    pub fn graphics(&self) -> MutexGuard<'_, RawMutex, Queue> {
        self.lock(QueueRole::Graphics)
    }
    pub fn transfer(&self) -> MutexGuard<'_, RawMutex, Queue> {
        self.lock(QueueRole::Transfer)
    }
    pub fn async_compute(&self) -> MutexGuard<'_, RawMutex, Queue> {
        self.lock(QueueRole::AsyncCompute)
    }

    pub fn graphics_pool(&self) -> &CmdPool {
        self.pool(QueueRole::Graphics)
    }
    pub fn transfer_pool(&self) -> &CmdPool {
        self.pool(QueueRole::Transfer)
    }
    pub fn async_compute_pool(&self) -> &CmdPool {
        self.pool(QueueRole::AsyncCompute)
    }
}
