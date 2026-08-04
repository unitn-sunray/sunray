//! Seeded random render graphs, for validating and benchmarking the aliasing
//! policy on datasets far bigger and messier than a hand-written fixture.
//!
//! The generator produces *pass declarations* — `(read, write)` lists — and then
//! runs the real [`analyze_passes`] over them, so lifetimes, hazard edges and
//! components all come out of the same code path `compile` uses. Nothing here
//! reimplements graph logic; it only makes up plausible input for it.
//!
//! Everything is a pure function of `(seed, params)`: same inputs, same graph, on
//! every machine and every run.

use crate::render_graph::alias::{AliasResource, Placement};
use crate::render_graph::graph::{
    PassAnalysis, PassResourceAccessSyncType, PassResourceAccessType, RenderGraph, analyze_passes, kahn_toposort,
};
use crate::render_graph::resource::{GraphResourceDesc, GraphResourceInfo, ResourceRef};
use crate::vulkan_abstraction::buffer::BufferDesc;
use crate::vulkan_abstraction::image::ImageDesc;
use ash::vk;
use gpu_allocator::MemoryLocation;
use rand::{RngExt, SeedableRng, rngs::StdRng};
use std::collections::HashMap;
use vk_sync_fork::AccessType;

/// Shape of the generated graph. The defaults are roughly a heavy deferred +
/// ray-tracing frame: a few hundred resources over a few dozen passes.
#[derive(Clone, Copy, Debug)]
pub struct GenParams {
    pub resources: usize,
    pub passes: usize,
    /// Upper bound on how many passes a resource stays live past its producer.
    /// Small relative to `passes` is what makes aliasing possible at all.
    pub max_lifetime: usize,
    /// Reads a pass declares on resources produced earlier.
    pub reads_per_pass: usize,
    /// Chance (0..100) that a pass also re-writes an already-produced resource,
    /// which is what generates WAR/WAW hazards and multi-epoch resources.
    pub rewrite_percent: u32,
}

impl Default for GenParams {
    fn default() -> Self {
        Self {
            resources: 256,
            passes: 64,
            max_lifetime: 8,
            reads_per_pass: 4,
            rewrite_percent: 20,
        }
    }
}

const WRITES: [AccessType; 4] = [
    AccessType::ComputeShaderWrite,
    AccessType::AnyShaderWrite,
    AccessType::TransferWrite,
    AccessType::ColorAttachmentWrite,
];

const READS: [AccessType; 5] = [
    AccessType::ComputeShaderReadOther,
    AccessType::ComputeShaderReadSampledImageOrUniformTexelBuffer,
    AccessType::RayTracingShaderReadOther,
    AccessType::AnyShaderReadOther,
    AccessType::TransferRead,
];

/// A generated graph, plus everything derived from it that the aliasing policy and
/// the barrier planner need.
pub struct GraphFixture {
    pub(crate) virtual_resources: Vec<GraphResourceInfo>,
    pub(crate) declarations: Vec<(Vec<ResourceRef>, Vec<ResourceRef>)>,
    pub(crate) analysis: PassAnalysis,
    /// Pass id → position in the topological order.
    pub(crate) schedule_pos: Vec<usize>,
    pub(crate) alias_resources: Vec<AliasResource>,
    pub(crate) alias_components: Vec<Vec<u32>>,
}

impl GraphFixture {
    pub fn pass_count(&self) -> usize {
        self.declarations.len()
    }

    pub fn resource_count(&self) -> usize {
        self.alias_resources.len()
    }

    /// Input for [`alias::plan`](crate::render_graph::alias::plan).
    pub fn alias_input(&self) -> (&[AliasResource], &[Vec<u32>]) {
        (&self.alias_resources, &self.alias_components)
    }

    /// Re-run the hazard scan + component pass. Returns the component count so the
    /// benchmark harness cannot optimize the call away.
    pub fn run_analyze(&self) -> usize {
        analyze_passes(self.declarations.iter().map(|(r, w)| (r.as_slice(), w.as_slice())))
            .components
            .len()
    }

    /// Re-run the topological sort. Returns the number of passes ordered.
    pub fn run_toposort(&self) -> usize {
        kahn_toposort(&self.analysis.dep_graph)
            .expect("generated graphs are acyclic")
            .len()
    }

    /// Re-run the barrier planner against a placement. Returns the barrier count.
    pub fn run_plan_barriers(&self, placements: &HashMap<u32, Placement>) -> usize {
        let (barriers, _) = RenderGraph::plan_barriers(
            &self.virtual_resources,
            &self.analysis.resource_usages,
            &self.schedule_pos,
            placements,
        );
        barriers.values().map(Vec::len).sum()
    }
}

/// Just the aliasing input, for tests and benchmarks that don't care about
/// barriers. Lifetimes come from the real hazard scan, not from a shortcut.
pub fn gen_resources(seed: u64, params: &GenParams) -> (Vec<AliasResource>, Vec<Vec<u32>>) {
    let fixture = gen_graph(seed, params);
    (fixture.alias_resources, fixture.alias_components)
}

/// Build a random graph and run the real analysis over it.
pub fn gen_graph(seed: u64, params: &GenParams) -> GraphFixture {
    let mut rng = StdRng::seed_from_u64(seed);
    let n = params.resources.max(1);
    let passes = params.passes.max(1);

    // ── Resource descriptions ───────────────────────────────────────────────
    // Sizes are log-uniform (1 KiB .. 16 MiB): a handful of large render targets
    // dominating a long tail of small buffers, which is the distribution offset
    // packing is supposed to exploit and uniform sizes would hide.
    let mut virtual_resources = Vec::with_capacity(n);
    let mut alias_resources = Vec::with_capacity(n);
    for id in 0..n as u32 {
        let size = 1u64 << rng.random_range(10..=24);
        let size = size + rng.random_range(0..size); // jitter off the exact power of two
        let alignment = 1u64 << rng.random_range(8..=16);
        // Most resources accept any memory type; a minority are restricted, which
        // is what forces the placer to open buckets it would rather have reused.
        let memory_type_bits = match rng.random_range(0..10) {
            0 => 0b0011,
            1 => 0b1100,
            _ => 0b1111,
        };
        let location = if rng.random_range(0..10) == 0 {
            MemoryLocation::CpuToGpu
        } else {
            MemoryLocation::GpuOnly
        };
        let is_image = rng.random_range(0..2) == 0;

        virtual_resources.push(GraphResourceInfo::Created(if is_image {
            GraphResourceDesc::Image(ImageDesc {
                extent: vk::Extent3D {
                    width: 64,
                    height: 64,
                    depth: 1,
                },
                format: vk::Format::R8G8B8A8_UNORM,
                tiling: vk::ImageTiling::OPTIMAL,
                location,
                usage: vk::ImageUsageFlags::STORAGE | vk::ImageUsageFlags::TRANSFER_SRC,
                name: "gen_image",
            })
        } else {
            GraphResourceDesc::Buffer(BufferDesc {
                byte_size: size,
                alignment,
                memory_location: location,
                usage: vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::TRANSFER_SRC,
                name: "gen_buffer",
            })
        }));

        // first_pass / last_pass are filled in from the hazard scan below.
        alias_resources.push(AliasResource {
            id,
            size,
            alignment,
            memory_type_bits,
            location,
            first_pass: 0,
            last_pass: 0,
        });
    }

    // ── Pass declarations ───────────────────────────────────────────────────
    // Each resource has exactly one producer, spread evenly across the schedule.
    // Consumers only ever read resources produced earlier, so every hazard edge
    // points from a lower pass id to a higher one and the graph is acyclic by
    // construction — `kahn_toposort` is being benchmarked, not stress-tested for
    // cycle detection.
    let producer_of: Vec<usize> = (0..n).map(|r| r * passes / n).collect();
    let mut produced_by_pass: Vec<Vec<u32>> = vec![Vec::new(); passes];
    for (r, p) in producer_of.iter().enumerate() {
        produced_by_pass[*p].push(r as u32);
    }

    let mut declarations: Vec<(Vec<ResourceRef>, Vec<ResourceRef>)> = Vec::with_capacity(passes);
    // Resources available to read at pass p, in production order.
    let mut available: Vec<u32> = Vec::with_capacity(n);

    for produced in &produced_by_pass {
        let mut reads: Vec<ResourceRef> = Vec::new();
        let mut writes: Vec<ResourceRef> = Vec::new();

        // Reads are biased towards recent production: a resource read long after it
        // was written would keep a bucket busy for the whole frame, which is not
        // what real graphs look like.
        if !available.is_empty() {
            let window = available.len().min(n * params.max_lifetime / passes.max(1) + 1);
            for _ in 0..rng.random_range(0..=params.reads_per_pass) {
                let idx = available.len() - 1 - rng.random_range(0..window);
                let res_id = available[idx];
                if reads.iter().any(|r| r.id == res_id) {
                    continue;
                }
                reads.push(res_ref(res_id, READS[rng.random_range(0..READS.len())]));
            }

            if rng.random_range(0..100) < params.rewrite_percent {
                let idx = available.len() - 1 - rng.random_range(0..window);
                let res_id = available[idx];
                writes.push(res_ref(res_id, WRITES[rng.random_range(0..WRITES.len())]));
            }
        }

        for res_id in produced {
            writes.push(res_ref(*res_id, WRITES[rng.random_range(0..WRITES.len())]));
        }

        declarations.push((reads, writes));
        available.extend_from_slice(produced);
    }

    // ── The real analysis ───────────────────────────────────────────────────
    let analysis = analyze_passes(declarations.iter().map(|(r, w)| (r.as_slice(), w.as_slice())));

    let topo = kahn_toposort(&analysis.dep_graph).expect("generated graph is acyclic by construction");
    let mut schedule_pos = vec![0usize; passes];
    for (pos, pass_id) in topo.iter().enumerate() {
        schedule_pos[*pass_id] = pos;
    }

    // Lifetimes come from the scan. A resource the scan never saw cannot be placed,
    // so drop it from the aliasing input rather than inventing a lifetime for it.
    alias_resources.retain_mut(|r| match analysis.resource_usages.get(&r.id) {
        Some(usage) => {
            r.first_pass = usage.first_pass;
            r.last_pass = usage.last_pass;
            true
        }
        None => false,
    });
    let alias_components: Vec<Vec<u32>> = analysis
        .components
        .iter()
        .map(|c| {
            let mut ids: Vec<u32> = c.resources.clone();
            ids.sort_unstable();
            ids
        })
        .collect();

    GraphFixture {
        virtual_resources,
        declarations,
        analysis,
        schedule_pos,
        alias_resources,
        alias_components,
    }
}

fn res_ref(id: u32, access_type: AccessType) -> ResourceRef {
    ResourceRef {
        id,
        access: PassResourceAccessType {
            access_type,
            sync_type: PassResourceAccessSyncType::AlwaysSync,
        },
    }
}
