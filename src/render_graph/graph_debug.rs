//! Structured dump of a compiled render-graph frame for offline visualization.
//!
//! Emitted once per frame into `$SUNRAY_GRAPH_DUMP_DIR` (set the env var to a
//! directory to enable; unset = zero cost). Two files per frame:
//!   - `graph_frame_<n>.dot` — Graphviz: passes as nodes, dependency edges
//!     labeled with the barriers they carry, plus a `FRAME_ENTRY` node holding
//!     the init / cross-frame barriers. Render with `dot -Tsvg`.
//!   - `graph_frame_<n>.txt` — the resource table (kind / size / aliasing slot /
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

use crate::render_graph::graph::ResourceBarrier;

/// One row of the per-frame resource table.
pub(crate) struct ResourceDumpInfo {
    pub id: u32,
    /// e.g. "created-image", "imported-buffer", "imported-as".
    pub kind: &'static str,
    /// Size / extent / name detail for the row.
    pub detail: String,
    /// Aliasing slot this resource binds to (transient only).
    pub slot: Option<u32>,
    /// For imported resources: the access the previous frame left it in, which
    /// drives whether a cross-frame barrier is emitted. `None` for created ones.
    pub import_access: Option<vk_sync::AccessType>,
}

/// Everything needed to render one frame's graph, gathered by `compile`.
pub(crate) struct GraphDump<'a> {
    pub frame: u64,
    pub pass_names: Vec<String>,
    /// (src_pass, dst_pass, barriers-on-this-edge).
    pub edges: Vec<(usize, usize, &'a [ResourceBarrier])>,
    pub resources: Vec<ResourceDumpInfo>,
    /// Init + cross-frame barriers issued before any pass (frame entry).
    pub init_barriers: &'a [ResourceBarrier],
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
        format!("{}: {:?}->{:?}", self.res_label(b.resource_id), b.prev_access, b.next_access)
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

        // Frame-entry node: init + cross-frame barriers.
        if !self.init_barriers.is_empty() {
            let mut lbl = String::from("FRAME ENTRY\\n(init + cross-frame barriers)");
            for b in self.init_barriers {
                let _ = write!(lbl, "\\n{}", escape(&self.barrier_label(b)));
            }
            let _ = writeln!(
                s,
                "  frame_entry [shape=note, style=filled, fillcolor=\"#ffe8b3\", label=\"{lbl}\"];"
            );
        }

        for (src, dst, barriers) in &self.edges {
            let label = barriers.iter().map(|b| self.barrier_label(b)).collect::<Vec<_>>().join("\\n");
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

        let _ = writeln!(s, "\nResources (id | kind | detail | slot | cross-frame access):");
        for r in &self.resources {
            let slot = r.slot.map(|s| s.to_string()).unwrap_or_else(|| "-".into());
            match r.import_access {
                Some(access) => {
                    let flag = if access == vk_sync::AccessType::Nothing {
                        "   <== enters UNDEFINED (contents discarded)"
                    } else {
                        ""
                    };
                    let _ = writeln!(
                        s,
                        "  {:>3} | {:<16} | {:<28} | slot {:<3} | {:?}{}",
                        r.id, r.kind, r.detail, slot, access, flag
                    );
                }
                None => {
                    let _ = writeln!(s, "  {:>3} | {:<16} | {:<28} | slot {:<3} | -", r.id, r.kind, r.detail, slot);
                }
            }
        }

        let _ = writeln!(s, "\nDependency edges (src -> dst : barriers):");
        for (src, dst, barriers) in &self.edges {
            let _ = writeln!(s, "  pass {src} -> pass {dst}");
            for b in *barriers {
                let _ = writeln!(s, "      {}", self.barrier_label(b));
            }
        }

        let _ = writeln!(s, "\nInit / cross-frame barriers (frame entry):");
        if self.init_barriers.is_empty() {
            let _ = writeln!(s, "  (none)");
        } else {
            for b in self.init_barriers {
                let _ = writeln!(s, "  {}", self.barrier_label(b));
            }
        }

        let _ = writeln!(s, "\n{}", self.aliasing_report);
        s
    }

    /// Write both files into `dir`. Errors are logged, not propagated — dumping
    /// is a diagnostic aid and must never fail a frame.
    pub(crate) fn write_to(&self, dir: &str) {
        let base = format!("{dir}/graph_frame_{:05}", self.frame);
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
