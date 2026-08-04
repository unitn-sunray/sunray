//! Structured dump of a compiled render-graph frame for offline visualization.
//!
//! Emitted once per frame into `$SUNRAY_GRAPH_DUMP_DIR` (set it to a directory,
//! or to `1` to use `<crate>/debug`; unset = zero cost). Two files per frame:
//!   - `graph_frame_<n>.dot` — Graphviz: passes as nodes, dependency edges
//!     labeled with the barriers they carry, plus a `FRAME_ENTRY` node holding
//!     the init / cross-frame barriers. Render with `dot -Tsvg`.
//!   - `graph_frame_<n>.txt` — the resource table (kind / size / `bucket@offset` /
//!     **imported cross-frame access**) and the transient aliasing report.
//!
//! The cross-frame access column shows the access each imported resource is
//! declared to carry into the frame by the `__imports` node (see
//! `graph::build_internal_passes`), which is what the hazard scan orders its
//! first consumer against. A row marked `<== enters UNDEFINED` carries
//! `Nothing`: it still gets a barrier, but from UNDEFINED, so its previous
//! contents are discarded. That is correct for a resource written first every
//! frame and a bug for one meant to carry data across the frame boundary — such
//! a resource needs its end state threaded back in via
//! `RenderGraph::import_with_usage`.

use std::fmt::Write as _;

use vk_sync_fork as vk_sync;

use crate::render_graph::alias::Placement;
use crate::render_graph::graph::ResourceBarrier;

/// One row of the per-frame resource table.
pub(crate) struct ResourceDumpInfo {
    pub id: u32,
    /// e.g. "created-image", "imported-buffer", "imported-as".
    pub kind: &'static str,
    /// Size / extent / name detail for the row.
    pub detail: String,
    /// Where this resource binds in aliased memory (transient only).
    pub placement: Option<Placement>,
    /// For imported resources: every access the previous frame left it in, which
    /// `__imports` declares and the first consumer is ordered against. `None` for
    /// created ones.
    pub import_access: Option<Vec<vk_sync::AccessType>>,
}

/// Everything needed to render one frame's graph, gathered by `compile`.
pub(crate) struct GraphDump<'a> {
    pub frame: u64,
    pub pass_names: Vec<String>,
    /// (src_pass, dst_pass, resources whose hazards forced the ordering).
    /// Edges express ordering only — barriers belong to a schedule position, not
    /// to an edge, so they live in `barriers_at`.
    pub edges: Vec<(usize, usize, &'a [u32])>,
    pub resources: Vec<ResourceDumpInfo>,
    /// Barriers issued immediately before each pass, in the order recorded.
    pub barriers_at: &'a [(usize, Vec<ResourceBarrier>)],
    /// The `TransientResources` aliasing/barrier report (`Debug` output).
    pub aliasing_report: String,
}

impl GraphDump<'_> {
    fn res_label(&self, id: u32) -> String {
        match self.resources.iter().find(|r| r.id == id) {
            Some(r) => format!("res{} {}", id, r.detail),
            None => format!("res{id}"),
        }
    }

    fn barrier_label(&self, b: &ResourceBarrier) -> String {
        format!(
            "{}: {:?}->{:?}{}",
            self.res_label(b.resource_id),
            b.prev,
            b.next,
            if b.discard { " [discard]" } else { "" }
        )
    }

    /// Graphviz DOT of passes + dependency edges + a frame-entry node.
    pub(crate) fn to_dot(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "digraph render_graph_frame_{} {{", self.frame);
        let _ = writeln!(s, "  rankdir=LR;");
        let _ = writeln!(s, "  node [shape=box, style=rounded, fontname=\"monospace\"];");
        let _ = writeln!(s, "  edge [fontname=\"monospace\", fontsize=9];");

        for (i, name) in self.pass_names.iter().enumerate() {
            let _ = writeln!(s, "  pass_{i} [label=\"pass {i}\\n{}\"];", escape(name));
        }

        // Barriers hang off the pass they precede, not off an edge — one note node
        // per barrier point, drawn into its pass.
        for (pass, barriers) in self.barriers_at {
            if barriers.is_empty() {
                continue;
            }
            let mut lbl = format!("barriers before pass {pass}");
            for b in barriers {
                let _ = write!(lbl, "\\n{}", escape(&self.barrier_label(b)));
            }
            let _ = writeln!(
                s,
                "  barrier_{pass} [shape=note, style=filled, fillcolor=\"#ffe8b3\", label=\"{lbl}\"];"
            );
            let _ = writeln!(s, "  barrier_{pass} -> pass_{pass} [style=dashed, color=\"#b38f00\"];");
        }

        for (src, dst, resources) in &self.edges {
            let label = resources.iter().map(|r| self.res_label(*r)).collect::<Vec<_>>().join("\\n");
            let _ = writeln!(s, "  pass_{src} -> pass_{dst} [label=\"{}\"];", escape(&label));
        }
        let _ = writeln!(s, "}}");
        s
    }

    /// Human-readable resource table + aliasing report.
    pub(crate) fn to_text(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "=== Render graph frame {} ===", self.frame);
        let _ = writeln!(s, "\nPasses:");
        for (i, name) in self.pass_names.iter().enumerate() {
            let _ = writeln!(s, "  pass {i}: {name}");
        }

        let _ = writeln!(s, "\nResources (id | kind | detail | bucket@offset | cross-frame access):");
        for r in &self.resources {
            // `bucket@offset` — under SUNRAY_ALIAS_STRATEGY=bucket several resources
            // share a bucket at once, so the offset is what distinguishes them.
            let slot = r
                .placement
                .map(|p| format!("{}@{}", p.bucket, p.offset))
                .unwrap_or_else(|| "-".into());
            match &r.import_access {
                Some(accesses) => {
                    let flag = if accesses.iter().all(|a| *a == vk_sync::AccessType::Nothing) {
                        "   <== enters UNDEFINED (contents discarded)"
                    } else {
                        ""
                    };
                    let _ = writeln!(
                        s,
                        "  {:>3} | {:<16} | {:<28} | {:<16} | {:?}{}",
                        r.id, r.kind, r.detail, slot, accesses, flag
                    );
                }
                None => {
                    let _ = writeln!(s, "  {:>3} | {:<16} | {:<28} | {:<16} | -", r.id, r.kind, r.detail, slot);
                }
            }
        }

        let _ = writeln!(s, "\nDependency edges — ordering only (src -> dst : resources):");
        for (src, dst, resources) in &self.edges {
            let _ = writeln!(s, "  pass {src} -> pass {dst}");
            for r in *resources {
                let _ = writeln!(s, "      {}", self.res_label(*r));
            }
        }

        let total: usize = self.barriers_at.iter().map(|(_, b)| b.len()).sum();
        let _ = writeln!(
            s,
            "\nBarriers ({total} transitions at {} point(s), in schedule order):",
            self.barriers_at.len()
        );
        if self.barriers_at.is_empty() {
            let _ = writeln!(s, "  (none)");
        }
        for (pass, barriers) in self.barriers_at {
            let _ = writeln!(s, "  before pass {pass}:");
            for b in barriers {
                let _ = writeln!(s, "      {}", self.barrier_label(b));
            }
        }

        let _ = writeln!(s, "\n{}", self.aliasing_report);
        s
    }

    /// Write both files into `dir`, creating it if needed. Errors are logged,
    /// not propagated — dumping is a diagnostic aid and must never fail a frame.
    pub(crate) fn write_to(&self, dir: &std::path::Path) {
        if let Err(e) = std::fs::create_dir_all(dir) {
            log::warn!("graph dump: failed to create {}: {e}", dir.display());
            return;
        }
        let base = dir.join(format!("graph_frame_{:05}", self.frame));
        let base = base.display();
        if let Err(e) = std::fs::write(format!("{base}.dot"), self.to_dot()) {
            log::warn!("graph dump: failed to write {base}.dot: {e}");
        }
        if let Err(e) = std::fs::write(format!("{base}.txt"), self.to_text()) {
            log::warn!("graph dump: failed to write {base}.txt: {e}");
        }
    }
}

/// Escape a string for a Graphviz double-quoted label.
fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}
