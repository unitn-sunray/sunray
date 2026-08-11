//! Transient memory aliasing policy: which resources share memory, and where.
//!
//! This module is deliberately Vulkan-free — plain data in, plain data out — for
//! the same reason [`RenderGraph::plan_barriers`](super::graph) is: the policy is
//! the part worth testing, and testing it should not need a GPU. `populate` in
//! [`super::transient_resources`] turns `vk::MemoryRequirements` into
//! [`AliasResource`]s, calls [`plan`], and binds the result.
//!
//! Three strategies, selected by `SUNRAY_ALIAS_STRATEGY`:
//!
//! * [`AliasStrategy::Off`] — no aliasing at all: one bucket per resource. The
//!   control both others are measured against, and the bisect knob for "is this
//!   corruption the aliasing?".
//! * [`AliasStrategy::Slot`] — greedy interval-graph coloring. Every member of a
//!   bucket sits at offset 0 and the bucket is `max(member sizes)`, so two members
//!   never co-occupy. Simple and fast, but a 4 MiB scratch buffer folded into a
//!   bucket opened by a 64 MiB G-buffer strands the other 60 MiB.
//! * [`AliasStrategy::Bucket`] — the PathFinder algorithm ("GPU Memory Aliasing").
//!   Buckets are sized by their largest member and smaller resources are packed
//!   *at offsets* into the byte regions left free by lifetime-disjoint occupants.
//!   Recovers that waste; costs O(N² log N) instead of O(N·buckets).
//!
//! Both emit the same [`Placement`] shape, so everything downstream — binding,
//! and the aliasing barriers in `RenderGraph::alias_predecessors` — is shared.

use gpu_allocator::MemoryLocation;
use std::collections::HashMap;

/// One transient resource as the placer sees it: a size, a memory-compatibility
/// mask, and an inclusive lifetime in pass ids.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AliasResource {
    pub id: u32,
    pub size: u64,
    /// `vk::MemoryRequirements::alignment`. Always a power of two.
    pub alignment: u64,
    pub memory_type_bits: u32,
    pub location: MemoryLocation,
    /// Inclusive: the resource must be live from `first_pass` through `last_pass`.
    pub first_pass: usize,
    pub last_pass: usize,
}

/// Where a resource ended up: byte range `[offset, offset + size)` inside `bucket`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Placement {
    pub bucket: u32,
    pub offset: u64,
    pub size: u64,
}

impl Placement {
    /// Do these two placements share any byte? Only meaningful within one bucket,
    /// which this checks. This is the predicate that decides whether two resources
    /// need an aliasing barrier between them.
    pub fn overlaps(&self, other: &Placement) -> bool {
        self.bucket == other.bucket && self.offset < other.offset + other.size && other.offset < self.offset + self.size
    }
}

/// Aggregated memory requirements for one bucket — what `gpu_allocator` is asked
/// for. `memory_type_bits` is the AND across members; an empty intersection means
/// the placer folded together two resources that cannot share memory.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BucketReqs {
    pub size: u64,
    pub alignment: u64,
    pub memory_type_bits: u32,
    pub location: MemoryLocation,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AliasStrategy {
    /// One resource per bucket at a time; bucket size is the largest member.
    #[default]
    Slot,
    /// Offset packing: lifetime-disjoint resources share a bucket's byte range.
    Bucket,
    /// No aliasing: one bucket per resource. Wastes memory on purpose — it is the
    /// bisect knob for "is this corruption the aliasing?". No two resources share a
    /// byte, so `alias_predecessors` finds nothing and no aliasing barrier is
    /// emitted; if a bug survives this, it is not an aliasing bug.
    Off,
}

impl AliasStrategy {
    /// Parse `SUNRAY_ALIAS_STRATEGY`. Unset or unrecognized → [`Self::Slot`].
    pub fn from_env() -> Self {
        let Ok(raw) = std::env::var(crate::utils::ALIAS_STRATEGY) else {
            return Self::Slot;
        };
        match raw.trim().to_ascii_lowercase().as_str() {
            "" | "slot" => Self::Slot,
            "bucket" => Self::Bucket,
            "off" | "none" => Self::Off,
            other => {
                log::warn!(
                    "{}: unrecognized value {other:?} (expected `slot`, `bucket` or `off`) — using `slot`",
                    crate::utils::ALIAS_STRATEGY
                );
                Self::Slot
            }
        }
    }
}

/// Assign every resource in `resources` a bucket and an offset.
///
/// `components` groups resource ids that may alias each other — one entry per
/// weakly-connected component of the pass graph. Aliasing is computed
/// independently per group; bucket ids are global, so `buckets[p.bucket]` indexes
/// the returned `Vec` regardless of which component a resource came from. Ids not
/// present in `resources` are ignored, and a resource in no component is not
/// placed.
///
/// `granularity` is `VkPhysicalDeviceLimits::bufferImageGranularity`; pass 1 to
/// disable the padding. It only bites the bucket strategy — under `Slot` members
/// never co-occupy.
///
/// Output is deterministic: same input, same placement, every time. The barrier
/// planner reads these placements to decide where aliasing barriers go, so it has
/// to be.
pub fn plan(
    strategy: AliasStrategy,
    resources: &[AliasResource],
    components: &[Vec<u32>],
    granularity: u64,
) -> (HashMap<u32, Placement>, Vec<BucketReqs>) {
    debug_assert!(
        resources.iter().all(|r| r.alignment.is_power_of_two()),
        "Vulkan guarantees power-of-two alignments; offset math here relies on max == lcm"
    );
    let granularity = granularity.max(1);
    let by_id: HashMap<u32, &AliasResource> = resources.iter().map(|r| (r.id, r)).collect();

    let mut placements: HashMap<u32, Placement> = HashMap::with_capacity(resources.len());
    let mut buckets: Vec<BucketReqs> = Vec::new();

    for component in components {
        let mut members: Vec<&AliasResource> = component.iter().filter_map(|id| by_id.get(id).copied()).collect();
        match strategy {
            AliasStrategy::Slot => plan_slots(&mut members, &mut placements, &mut buckets),
            AliasStrategy::Bucket => plan_buckets(&mut members, granularity, &mut placements, &mut buckets),
            AliasStrategy::Off => plan_off(&members, &mut placements, &mut buckets),
        }
    }

    (placements, buckets)
}

/// One bucket per resource — aliasing disabled. Order-independent, so no sort.
fn plan_off(members: &[&AliasResource], placements: &mut HashMap<u32, Placement>, buckets: &mut Vec<BucketReqs>) {
    for r in members {
        placements.insert(
            r.id,
            Placement {
                bucket: buckets.len() as u32,
                offset: 0,
                size: r.size,
            },
        );
        buckets.push(BucketReqs {
            size: r.size,
            alignment: r.alignment,
            memory_type_bits: r.memory_type_bits,
            location: r.location,
        });
    }
}

/// Greedy interval-graph coloring, earliest-start first, first fit. This is the
/// original `populate` policy, preserved as the control the bucket strategy is
/// measured against.
fn plan_slots(members: &mut [&AliasResource], placements: &mut HashMap<u32, Placement>, buckets: &mut Vec<BucketReqs>) {
    members.sort_unstable_by_key(|r| (r.first_pass, r.id));

    // (last_pass of the current occupant, bucket id). Buckets are never retired
    // from this list, so it grows to the component's bucket count.
    let mut active: Vec<(usize, u32)> = Vec::new();

    for r in members.iter().copied() {
        // Strictly-before: lifetimes are inclusive, so a bucket freed on the very
        // pass this resource starts is still busy.
        let candidate = active
            .iter()
            .position(|(last_pass, bucket)| *last_pass < r.first_pass && compatible(&buckets[*bucket as usize], r));

        let bucket = match candidate {
            Some(idx) => {
                let bucket = active[idx].1;
                fold(&mut buckets[bucket as usize], r);
                buckets[bucket as usize].size = buckets[bucket as usize].size.max(r.size);
                active[idx].0 = r.last_pass;
                bucket
            }
            None => {
                let bucket = buckets.len() as u32;
                buckets.push(BucketReqs {
                    size: r.size,
                    alignment: r.alignment,
                    memory_type_bits: r.memory_type_bits,
                    location: r.location,
                });
                active.push((r.last_pass, bucket));
                bucket
            }
        };

        placements.insert(
            r.id,
            Placement {
                bucket,
                offset: 0,
                size: r.size,
            },
        );
    }
}

/// The PathFinder bucket algorithm.
///
/// Largest-first: a bucket's capacity is fixed by the resource that seeds it and
/// never grows, so seeding with the largest remaining resource is what makes room
/// for everything else. Each subsequent resource is dropped into the *smallest*
/// free region that fits, which keeps the big regions intact for the big
/// resources still to come.
fn plan_buckets(
    members: &mut [&AliasResource],
    granularity: u64,
    placements: &mut HashMap<u32, Placement>,
    buckets: &mut Vec<BucketReqs>,
) {
    // Descending size; id breaks ties so the output is reproducible.
    members.sort_unstable_by_key(|r| (std::cmp::Reverse(r.size), r.id));

    let mut unplaced: Vec<Option<&AliasResource>> = members.iter().copied().map(Some).collect();
    let mut remaining = unplaced.len();

    while remaining > 0 {
        // Seed a fresh bucket with the largest resource still unplaced. Everything
        // before `seed_idx` is already placed, so the inner scan starts after it.
        let seed_idx = unplaced.iter().position(Option::is_some).expect("remaining > 0");
        let seed = unplaced[seed_idx].take().expect("just found a Some");
        remaining -= 1;

        let bucket = buckets.len() as u32;
        buckets.push(BucketReqs {
            size: seed.size,
            alignment: seed.alignment,
            memory_type_bits: seed.memory_type_bits,
            location: seed.location,
        });
        // (resource, offset) for this bucket — the occupancy map `fit` reads.
        let mut occupants: Vec<(&AliasResource, u64)> = vec![(seed, 0)];
        placements.insert(
            seed.id,
            Placement {
                bucket,
                offset: 0,
                size: seed.size,
            },
        );

        for entry in unplaced.iter_mut().skip(seed_idx + 1) {
            let Some(r) = *entry else { continue };
            if !compatible(&buckets[bucket as usize], r) {
                continue;
            }
            let Some(offset) = fit(r, &occupants, buckets[bucket as usize].size, granularity) else {
                continue;
            };

            fold(&mut buckets[bucket as usize], r);
            occupants.push((r, offset));
            placements.insert(
                r.id,
                Placement {
                    bucket,
                    offset,
                    size: r.size,
                },
            );
            *entry = None;
            remaining -= 1;
        }
    }
}

/// May `r` join this bucket at all? Memory type and heap must agree before any
/// offset arithmetic is worth doing.
fn compatible(bucket: &BucketReqs, r: &AliasResource) -> bool {
    bucket.location == r.location && (bucket.memory_type_bits & r.memory_type_bits) != 0
}

/// Widen a bucket's requirements to admit `r`. Size is *not* touched — the two
/// strategies disagree on it (slot grows to `max`, bucket is fixed by its seed).
fn fold(bucket: &mut BucketReqs, r: &AliasResource) {
    bucket.alignment = bucket.alignment.max(r.alignment);
    bucket.memory_type_bits &= r.memory_type_bits;
}

/// The smallest free region in `[0, capacity)` that can hold `r` without touching
/// an occupant whose lifetime overlaps `r`'s, or `None` if it fits nowhere.
///
/// Occupants that *don't* overlap `r` in time are free real estate — that is the
/// whole trick. The blocking intervals are sorted and swept; the gaps between them
/// are the aliasable regions. (The article builds the same set with a Start/End
/// marker counter; sweeping a sorted interval list is the same computation in
/// fewer lines.)
fn fit(r: &AliasResource, occupants: &[(&AliasResource, u64)], capacity: u64, granularity: u64) -> Option<u64> {
    let mut blocked: Vec<(u64, u64)> = occupants
        .iter()
        .filter(|(o, _)| lifetimes_overlap(o, r))
        .map(|(o, off)| (*off, off.saturating_add(o.size.next_multiple_of(granularity))))
        .collect();
    blocked.sort_unstable();

    // ponytail: granularity is applied to every placement rather than only where a
    // linear and a non-linear neighbour actually meet. bufferImageGranularity is 1
    // on most desktop GPUs, so this usually costs nothing; if it ever shows up in
    // the quality report, track each occupant's linearity and pad only at the seam.
    let align = r.alignment.max(granularity);
    let need = r.size.next_multiple_of(granularity);

    let mut best: Option<(u64, u64)> = None; // (region length, chosen offset)
    let mut cursor = 0u64;

    // The sentinel closes the tail region `[cursor, capacity)`.
    for (start, end) in blocked.into_iter().chain(std::iter::once((capacity, capacity))) {
        if start > cursor {
            let offset = cursor.next_multiple_of(align);
            if offset.saturating_add(need) <= start {
                let len = start - cursor;
                // Smallest fitting region wins; lowest offset breaks ties because
                // `blocked` is sorted ascending and `<` keeps the first.
                if best.is_none_or(|(best_len, _)| len < best_len) {
                    best = Some((len, offset));
                }
            }
        }
        cursor = cursor.max(end);
    }

    best.map(|(_, offset)| offset)
}

/// Inclusive-interval overlap. Two resources whose lifetimes overlap must never
/// share a byte.
fn lifetimes_overlap(a: &AliasResource, b: &AliasResource) -> bool {
    a.first_pass <= b.last_pass && b.first_pass <= a.last_pass
}

/// Total bytes the placement asks the allocator for.
pub fn total_bytes(buckets: &[BucketReqs]) -> u64 {
    buckets.iter().map(|b| b.size).sum()
}

/// The best any aliasing scheme could do: the largest total size of resources
/// simultaneously live at any single pass. A placement can only ever meet or
/// exceed this, so `total_bytes / lower_bound` is a strategy's efficiency.
///
/// Ignores alignment and memory-type incompatibility, so it is a lower bound and
/// not necessarily achievable.
pub fn live_bytes_lower_bound(resources: &[AliasResource]) -> u64 {
    let Some(last) = resources.iter().map(|r| r.last_pass).max() else {
        return 0;
    };
    (0..=last)
        .map(|pass| {
            resources
                .iter()
                .filter(|r| r.first_pass <= pass && pass <= r.last_pass)
                .map(|r| r.size)
                .sum::<u64>()
        })
        .max()
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::render_graph::bench_support::{GenParams, gen_graph, gen_resources};

    const GPU: MemoryLocation = MemoryLocation::GpuOnly;

    fn res(id: u32, size: u64, first: usize, last: usize) -> AliasResource {
        AliasResource {
            id,
            size,
            alignment: 256,
            memory_type_bits: 0b111,
            location: GPU,
            first_pass: first,
            last_pass: last,
        }
    }

    fn one_component(resources: &[AliasResource]) -> Vec<Vec<u32>> {
        vec![resources.iter().map(|r| r.id).collect()]
    }

    /// Every property the placement must hold for the memory to actually be safe
    /// and for the allocator not to reject it. Run against both strategies.
    fn check_invariants(
        resources: &[AliasResource],
        placements: &HashMap<u32, Placement>,
        buckets: &[BucketReqs],
        granularity: u64,
        what: &str,
    ) {
        let granularity = granularity.max(1);
        assert_eq!(placements.len(), resources.len(), "{what}: every resource must be placed");

        for r in resources {
            let p = placements
                .get(&r.id)
                .unwrap_or_else(|| panic!("{what}: res {} unplaced", r.id));
            let b = &buckets[p.bucket as usize];

            assert_eq!(p.size, r.size, "{what}: res {} placement size", r.id);
            assert!(
                p.offset + p.size <= b.size,
                "{what}: res {} runs past the end of bucket {} ({}+{} > {})",
                r.id,
                p.bucket,
                p.offset,
                p.size,
                b.size
            );
            // The bucket's base is aligned to `b.alignment` >= r.alignment, and
            // both are powers of two, so an offset aligned to r.alignment lands
            // the resource on a correctly aligned address.
            let align = r.alignment.max(granularity);
            assert_eq!(
                p.offset % align,
                0,
                "{what}: res {} offset {} not {align}-aligned",
                r.id,
                p.offset
            );
            assert!(
                b.alignment >= r.alignment,
                "{what}: bucket {} under-aligned for res {}",
                p.bucket,
                r.id
            );
            assert_eq!(b.location, r.location, "{what}: bucket {} heap mismatch", p.bucket);
            assert_ne!(
                b.memory_type_bits & r.memory_type_bits,
                0,
                "{what}: res {} cannot live in bucket {}",
                r.id,
                p.bucket
            );
        }

        // The one that matters: concurrently-live resources must never share a byte.
        for (i, a) in resources.iter().enumerate() {
            for b in &resources[i + 1..] {
                if !lifetimes_overlap(a, b) {
                    continue;
                }
                let (pa, pb) = (placements[&a.id], placements[&b.id]);
                assert!(
                    !pa.overlaps(&pb),
                    "{what}: res {} [{}..={}] @ bucket {} {}..{} aliases live res {} [{}..={}] @ {}..{}",
                    a.id,
                    a.first_pass,
                    a.last_pass,
                    pa.bucket,
                    pa.offset,
                    pa.offset + pa.size,
                    b.id,
                    b.first_pass,
                    b.last_pass,
                    pb.offset,
                    pb.offset + pb.size
                );
            }
        }

        for (i, b) in buckets.iter().enumerate() {
            assert_ne!(b.memory_type_bits, 0, "{what}: bucket {i} has no usable memory type");
            assert!(b.alignment.is_power_of_two(), "{what}: bucket {i} alignment {}", b.alignment);
            assert!(
                placements.values().any(|p| p.bucket == i as u32),
                "{what}: bucket {i} is empty"
            );
        }

        assert!(
            total_bytes(buckets) >= live_bytes_lower_bound(resources),
            "{what}: total {} below the concurrently-live lower bound {}",
            total_bytes(buckets),
            live_bytes_lower_bound(resources)
        );
    }

    #[test]
    fn disjoint_lifetimes_share_one_bucket_side_by_side() {
        // One big resource, then two smalls that both outlive it — under Slot they
        // each need their own bucket-worth of the big size; under Bucket they sit
        // side by side inside the big one's footprint.
        let r = vec![res(0, 4096, 0, 1), res(1, 1024, 2, 5), res(2, 1024, 2, 5)];
        let c = one_component(&r);

        let (p, b) = plan(AliasStrategy::Bucket, &r, &c, 1);
        check_invariants(&r, &p, &b, 1, "bucket");
        assert_eq!(b.len(), 1, "all three fit in the 4096 bucket");
        assert_eq!(b[0].size, 4096);
        assert!(!p[&1].overlaps(&p[&2]), "the two live-at-once smalls must not share bytes");

        let (p, b) = plan(AliasStrategy::Slot, &r, &c, 1);
        check_invariants(&r, &p, &b, 1, "slot");
        assert_eq!(b.len(), 2, "slot cannot co-locate res 1 and res 2");
        assert!(total_bytes(&b) > 4096);
    }

    #[test]
    fn a_resource_that_fits_nowhere_opens_a_new_bucket() {
        // Two 4096s alive at the same time: the second cannot go anywhere in the
        // first's bucket, so it seeds its own.
        let r = vec![res(0, 4096, 0, 3), res(1, 4096, 1, 4)];
        let (p, b) = plan(AliasStrategy::Bucket, &r, &one_component(&r), 1);
        check_invariants(&r, &p, &b, 1, "bucket");
        assert_eq!(b.len(), 2);
    }

    #[test]
    fn incompatible_memory_types_are_never_folded_together() {
        let mut a = res(0, 4096, 0, 1);
        a.memory_type_bits = 0b0011;
        let mut b = res(1, 1024, 2, 3);
        b.memory_type_bits = 0b1100; // no overlap with `a`
        let r = vec![a, b];

        for strategy in [AliasStrategy::Slot, AliasStrategy::Bucket] {
            let (p, buckets) = plan(strategy, &r, &one_component(&r), 1);
            check_invariants(&r, &p, &buckets, 1, "incompat");
            assert_eq!(buckets.len(), 2, "{strategy:?}: disjoint memory types cannot share");
        }
    }

    #[test]
    fn components_never_alias_across_each_other() {
        let r = vec![res(0, 4096, 0, 1), res(1, 1024, 5, 6)];
        // Same lifetimes as `disjoint_lifetimes_...` would happily merge, but the
        // two live in separate components.
        let c = vec![vec![0], vec![1]];
        for strategy in [AliasStrategy::Slot, AliasStrategy::Bucket] {
            let (p, buckets) = plan(strategy, &r, &c, 1);
            check_invariants(&r, &p, &buckets, 1, "components");
            assert_eq!(buckets.len(), 2, "{strategy:?}");
            assert_ne!(p[&0].bucket, p[&1].bucket);
        }
    }

    #[test]
    fn granularity_pads_placements_apart() {
        let r = vec![res(0, 4096, 0, 1), res(1, 100, 2, 5), res(2, 100, 2, 5)];
        let (p, b) = plan(AliasStrategy::Bucket, &r, &one_component(&r), 1024);
        check_invariants(&r, &p, &b, 1024, "granularity");
        // 100-byte resources must still be pushed a full granule apart.
        assert!(p[&1].offset.abs_diff(p[&2].offset) >= 1024);
    }

    #[test]
    fn placement_is_deterministic() {
        let (r, c) = gen_resources(0xC0FFEE, &GenParams::default());
        for strategy in [AliasStrategy::Slot, AliasStrategy::Bucket] {
            let a = plan(strategy, &r, &c, 64);
            let b = plan(strategy, &r, &c, 64);
            assert_eq!(a.0, b.0, "{strategy:?}: placements differ between runs");
            assert_eq!(a.1, b.1, "{strategy:?}: buckets differ between runs");
        }
    }

    /// The seeded random sweep. Both strategies, 32 seeds, checked against every
    /// invariant — this is what says the bucket implementation is actually correct
    /// rather than merely smaller.
    #[test]
    fn random_placements_hold_every_invariant() {
        let params = GenParams::default();
        for seed in 0..32u64 {
            let (resources, components) = gen_resources(seed, &params);
            for granularity in [1, 64, 1024] {
                for strategy in [AliasStrategy::Slot, AliasStrategy::Bucket, AliasStrategy::Off] {
                    let (placements, buckets) = plan(strategy, &resources, &components, granularity);
                    check_invariants(
                        &resources,
                        &placements,
                        &buckets,
                        granularity,
                        &format!("seed {seed} granularity {granularity} {strategy:?}"),
                    );
                }
            }
        }
    }

    /// Bucket packing should beat slot coloring on realistic size distributions.
    /// Not a hard guarantee of the algorithm, but if it ever regresses the whole
    /// exercise was pointless, so it is worth failing on.
    #[test]
    fn bucket_beats_slot_on_average() {
        let params = GenParams::default();
        let (mut slot_total, mut bucket_total) = (0u64, 0u64);
        for seed in 0..32u64 {
            let (r, c) = gen_resources(seed, &params);
            slot_total += total_bytes(&plan(AliasStrategy::Slot, &r, &c, 64).1);
            bucket_total += total_bytes(&plan(AliasStrategy::Bucket, &r, &c, 64).1);
        }
        assert!(
            bucket_total < slot_total,
            "bucket {bucket_total} did not beat slot {slot_total}"
        );
    }

    /// Memory-quality comparison. Criterion measures time; this measures the two
    /// things that actually decide whether the bucket algorithm is worth its
    /// complexity: bytes allocated against the concurrently-live lower bound, and
    /// the barrier count the resulting placement costs (denser packing means more
    /// sequential byte reuse, and every reuse needs an aliasing barrier).
    ///
    /// `cargo test alias_quality_report -- --ignored --nocapture`
    #[test]
    #[ignore = "reporting only — run with --ignored --nocapture"]
    fn alias_quality_report() {
        let gib = |b: u64| b as f64 / (1024.0 * 1024.0 * 1024.0);
        println!();
        println!(
            "{:<8} {:>6} {:>8} {:>10} {:>10} {:>8} {:>9}",
            "strategy", "res", "buckets", "GiB", "optimum", "ratio", "barriers"
        );
        for resources_n in [64usize, 256, 1024, 4096] {
            let params = GenParams {
                resources: resources_n,
                passes: resources_n / 4,
                ..GenParams::default()
            };
            let fixture = gen_graph(7, &params);
            let (r, c) = fixture.alias_input();
            let optimum = live_bytes_lower_bound(r);
            for strategy in [AliasStrategy::Off, AliasStrategy::Slot, AliasStrategy::Bucket] {
                let (placements, buckets) = plan(strategy, r, c, 64);
                let total = total_bytes(&buckets);
                println!(
                    "{:<8} {:>6} {:>8} {:>10.2} {:>10.2} {:>7.0}% {:>9}",
                    format!("{strategy:?}").to_lowercase(),
                    resources_n,
                    buckets.len(),
                    gib(total),
                    gib(optimum),
                    100.0 * total as f64 / optimum.max(1) as f64,
                    fixture.run_plan_barriers(&placements),
                );
            }
        }
        println!();
    }
}
