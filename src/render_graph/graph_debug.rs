//! Structured dump of a compiled render-graph frame for offline visualization.
//!
//! Emitted once per frame into `$SUNRAY_GRAPH_DUMP_DIR` (set it to a directory,
//! or to `1` to use `<crate>/debug`; unset = zero cost). Three files per frame:
//!   - `graph_frame_<n>.dot` — Graphviz: passes as nodes listing every resource
//!     they read/write, dependency edges, and a note node per barrier point.
//!   - `graph_frame_<n>.svg` — the above rendered, if Graphviz's `dot` is on
//!     `PATH`. No `dot`, no `.svg`; nothing else changes.
//!   - `graph_frame_<n>.txt` — the resource table (kind / size / live pass range /
//!     `bucket@offset` / **imported cross-frame access**) and the aliasing report.
//!
//! Resources read as `res<id>` when the graph owns them (transient — freed at
//! frame end, memory may be aliased to another transient) and `ext<id>` when they
//! are imported (external — outlives the frame). A pass's usage line is tagged
//! `<-- first use` / `<-- last use` where that pass bounds the resource's live
//! range, which for a transient is exactly the window the aliaser packs against.
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

use std::collections::HashMap;
use std::fmt::Write as _;

use vk_sync_fork as vk_sync;

use crate::render_graph::alias::Placement;
use crate::render_graph::graph::ResourceBarrier;
use crate::render_graph::resource::ResourceRef;

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
    /// `(read, write)` declarations of each pass, parallel to `pass_names`.
    pub pass_uses: Vec<(&'a [ResourceRef], &'a [ResourceRef])>,
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
    /// `res<id>` for a graph-owned transient, `ext<id>` for an imported external —
    /// the prefix answers "does this outlive the frame?" without a table lookup.
    fn res_label(&self, id: u32) -> String {
        match self.resources.iter().find(|r| r.id == id) {
            Some(r) if r.kind.starts_with("imported") => format!("ext{} {}", id, r.detail),
            Some(r) => format!("res{} {}", id, r.detail),
            None => format!("res{id}"),
        }
    }

    /// First and last pass touching each resource, by id. Passes are visited in
    /// schedule order, so the last write wins for the end of the span.
    fn lifetimes(&self) -> HashMap<u32, (usize, usize)> {
        let mut spans: HashMap<u32, (usize, usize)> = HashMap::new();
        for (pass, (read, write)) in self.pass_uses.iter().enumerate() {
            for r in read.iter().chain(write.iter()) {
                spans.entry(r.id).and_modify(|s| s.1 = pass).or_insert((pass, pass));
            }
        }
        spans
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

    /// `R`/`W` + resource + declared access, one entry per resource a pass touches,
    /// tagged where this pass is the resource's first or last use in the frame —
    /// for a transient that span is exactly the window its memory must stay live.
    fn use_labels(&self, pass: usize, spans: &HashMap<u32, (usize, usize)>) -> Vec<String> {
        let (read, write) = self.pass_uses[pass];
        read.iter()
            .map(|r| ('R', r))
            .chain(write.iter().map(|w| ('W', w)))
            .map(|(rw, r)| {
                let span = match spans.get(&r.id) {
                    Some(&(f, l)) if f == l => " <-- only use",
                    Some(&(f, _)) if f == pass => " <-- first use",
                    Some(&(_, l)) if l == pass => " <-- last use",
                    _ => "",
                };
                format!("{rw} {} [{:?}]{span}", self.res_label(r.id), r.access.access_type)
            })
            .collect()
    }

    /// Graphviz DOT of passes + dependency edges + a frame-entry node.
    pub(crate) fn to_dot(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "digraph render_graph_frame_{} {{", self.frame);
        // TB: a 30+ pass chain laid out LR is a mile-wide strip nothing can read.
        let _ = writeln!(s, "  rankdir=TB;");
        let _ = writeln!(s, "  node [shape=box, style=rounded, fontname=\"monospace\"];");
        let _ = writeln!(s, "  edge [fontname=\"monospace\", fontsize=9];");
        let _ = writeln!(
            s,
            "  legend [shape=plaintext, label=\"res<id> = transient (graph-owned, may alias)\\l\
             ext<id> = external (imported, outlives the frame)\\l\
             R / W   = declared read / write\\l\
             first/last use = the transient's live range\\l\"];"
        );

        let spans = self.lifetimes();
        for (i, name) in self.pass_names.iter().enumerate() {
            // `\l` = left-aligned line break, so the usage list reads as a column.
            let mut lbl = format!("pass {i}: {}\\l", escape(name));
            for u in self.use_labels(i, &spans) {
                let _ = write!(lbl, "{}\\l", escape(&u));
            }
            let _ = writeln!(s, "  pass_{i} [label=\"{lbl}\"];");
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

        // Edges are unlabeled: the resources that forced each ordering are listed
        // per pass in the node labels, and per edge in the .txt dump. Repeating
        // them on the edges made the layout unreadably wide.
        for (src, dst, _) in &self.edges {
            let _ = writeln!(s, "  pass_{src} -> pass_{dst};");
        }
        let _ = writeln!(s, "}}");
        s
    }

    /// Human-readable resource table + aliasing report.
    pub(crate) fn to_text(&self) -> String {
        let mut s = String::new();
        let _ = writeln!(s, "=== Render graph frame {} ===", self.frame);
        let spans = self.lifetimes();
        let _ = writeln!(s, "\nPasses (res<id> = transient, ext<id> = external/imported):");
        for (i, name) in self.pass_names.iter().enumerate() {
            let _ = writeln!(s, "  pass {i}: {name}");
            for u in self.use_labels(i, &spans) {
                let _ = writeln!(s, "      {u}");
            }
        }

        let _ = writeln!(
            s,
            "\nResources (id | kind | detail | live passes | bucket@offset | cross-frame access):"
        );
        for r in &self.resources {
            // `bucket@offset` — under SUNRAY_ALIAS_STRATEGY=bucket several resources
            // share a bucket at once, so the offset is what distinguishes them.
            let slot = r
                .placement
                .map(|p| format!("{}@{}", p.bucket, p.offset))
                .unwrap_or_else(|| "-".into());
            // For a transient this span is the window its memory must stay live,
            // i.e. what the aliaser packs against.
            let live = match spans.get(&r.id) {
                Some((f, l)) => format!("{f}..{l}"),
                None => "unused".into(),
            };
            match &r.import_access {
                Some(accesses) => {
                    let flag = if accesses.iter().all(|a| *a == vk_sync::AccessType::Nothing) {
                        "   <== enters UNDEFINED (contents discarded)"
                    } else {
                        ""
                    };
                    let _ = writeln!(
                        s,
                        "  {:>3} | {:<16} | {:<28} | {:<8} | {:<16} | {:?}{}",
                        r.id, r.kind, r.detail, live, slot, accesses, flag
                    );
                }
                None => {
                    let _ = writeln!(
                        s,
                        "  {:>3} | {:<16} | {:<28} | {:<8} | {:<16} | -",
                        r.id, r.kind, r.detail, live, slot
                    );
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
        // Render the .dot if Graphviz is installed; no dot on PATH just means no
        // .svg, which is why the failure is a debug line and not a warning.
        match std::process::Command::new("dot")
            .args(["-Tsvg", &format!("{base}.dot"), "-o", &format!("{base}.svg")])
            .status()
        {
            Ok(s) if s.success() => {}
            Ok(s) => log::warn!("graph dump: dot -Tsvg exited {s}"),
            Err(e) => log::debug!("graph dump: no .svg ({e})"),
        }
    }
}

/// Escape a string for a Graphviz double-quoted label.
fn escape(s: &str) -> String {
    s.replace('\\', "\\\\").replace('"', "\\\"")
}
