//! Wall-clock scaling of the transient aliasing policy and the render graph
//! analysis it feeds, on seeded random graphs.
//!
//! `Slot` is O(R · buckets); `Bucket` is O(R² log R). This is where that shows up.
//! For the metric that actually decides which strategy to ship — bytes allocated
//! against the concurrently-live lower bound — run the in-crate report instead:
//!
//! ```text
//! cargo test alias_quality_report -- --ignored --nocapture
//! ```

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use std::hint::black_box;
use sunray::render_graph::alias::{self, AliasStrategy};
use sunray::render_graph::bench_support::{GenParams, gen_graph};

const SEED: u64 = 0x5EED;
const SIZES: [usize; 4] = [64, 256, 1024, 4096];

/// Resource counts scale with pass counts the way real graphs do — a graph with
/// 4096 resources and 64 passes is not a bigger frame, it is an unrealistic one.
fn params(resources: usize) -> GenParams {
    GenParams {
        resources,
        passes: (resources / 4).max(8),
        ..GenParams::default()
    }
}

fn bench_plan(c: &mut Criterion) {
    let mut group = c.benchmark_group("alias/plan");
    for n in SIZES {
        let fixture = gen_graph(SEED, &params(n));
        let (resources, components) = fixture.alias_input();
        for strategy in [AliasStrategy::Slot, AliasStrategy::Bucket] {
            let id = BenchmarkId::new(format!("{strategy:?}").to_lowercase(), n);
            group.bench_function(id, |b| {
                b.iter(|| alias::plan(strategy, black_box(resources), black_box(components), 64))
            });
        }
    }
    group.finish();
}

fn bench_graph(c: &mut Criterion) {
    let mut group = c.benchmark_group("graph");
    for n in SIZES {
        let fixture = gen_graph(SEED, &params(n));
        let (resources, components) = fixture.alias_input();

        group.bench_function(BenchmarkId::new("analyze", n), |b| b.iter(|| fixture.run_analyze()));
        group.bench_function(BenchmarkId::new("toposort", n), |b| b.iter(|| fixture.run_toposort()));

        // Barrier planning reads the placement, so it is benchmarked under both —
        // `Bucket` puts more resources in each bucket, which widens the alias
        // predecessor sets the planner has to union.
        for strategy in [AliasStrategy::Slot, AliasStrategy::Bucket] {
            let (placements, _) = alias::plan(strategy, resources, components, 64);
            let id = BenchmarkId::new(format!("plan_barriers/{strategy:?}").to_lowercase(), n);
            group.bench_function(id, |b| b.iter(|| fixture.run_plan_barriers(black_box(&placements))));
        }
    }
    group.finish();
}

criterion_group!(benches, bench_plan, bench_graph);
criterion_main!(benches);
