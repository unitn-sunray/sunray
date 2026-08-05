# NVIDIA driver crash: presents × CPU frame work on a busy queue (VK_EXT_descriptor_heap dev driver)

Investigation report for the access violation that crashes the `window` example when
the frame loop is pipelined. Bottom line: **a driver-internal NULL-pointer dereference
on an NVIDIA worker thread**, triggered when the renderer does CPU-side frame work
(preparation, recording, submits) while the graphics queue is still executing earlier
work *and* swapchain presents are in the pipe. It is not an API-usage error:
synchronization validation is silent, and the crash survives every structural
workaround attempted (semaphore wait stages, BDA caching, AS-build placement,
submission depth gating, dedicated command pools). Crash frequency scales with how
much the CPU overlaps the busy queue — seconds with full 2-frames-in-flight overlap,
~30 s even with conservative N−1 pacing; only the fully-serialized loop
(queue drained before each frame's CPU work) is soak-stable. The shipped workaround
drains the graphics queue at the start of each presenting frame; offscreen rendering
keeps the full frames-in-flight overlap.

## Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 Ti |
| Driver | 32.0.16.1047 (`nvoglv64.dll`, the **`VK_EXT_descriptor_heap` developer driver**) |
| OS | Windows 11 Pro 10.0.26200 |
| App | `cargo run --example window` (debug), glTF scene `Room.glb`, FIFO present |
| Renderer | sunray @ branch `fable_test` (2-frames-in-flight rework) |

## Symptom

`STATUS_ACCESS_VIOLATION` (0xc0000005) terminating the process within ~1–6 s of
rendering (a few hundred to ~750 frames; under the api_dump layer's slowdown it
reached frame ~749, so it scales with frame count, not wall time). Faulting module is
always the NVIDIA user-mode driver at the **same offset**:

```
Faulting module name: nvoglv64.dll, version: 32.0.16.1047
Exception code:       0xc0000005
Fault offset:         0x000000000015f942
```

(Windows Application event log, Event ID 1000, multiple instances 2026-06-12.)

## Crash dump analysis

Full-memory WER dumps analyzed with `minidump-stackwalk`
(`%LOCALAPPDATA%\CrashDumps\window.exe.*.dmp`):

- **Crashing thread is a driver-internal worker (Thread 12)** — its stack is entirely
  `nvoglv64.dll` + `ntdll.dll`, no application or loader frames.
- The faulting instruction is `mov r8, qword [rsi]` with `rsi = 0` — a **NULL-pointer
  read inside the driver**, deterministic at `nvoglv64.dll+0x15f942`.
- The **main thread** was meanwhile inside a WSI call (in `win32u.dll` syscall invoked
  from `nvoglv64.dll` — the present/acquire path).
- The api_dump trace confirms asynchrony: the last main-thread API call had already
  returned (the dump died mid-line while *printing* its result), i.e. the AV fired on
  another thread while the main thread was in benign query code.

## What triggers it (experiment matrix)

Configurations tested, ~40–180 s soaks; "overlap" = the CPU prepares and records frame
N+1 while frame N's GPU work is still executing (frame-timeline wait at N−2 instead of
N−1):

| # | Configuration | Result |
|---|--------------|--------|
| 1 | Window, full overlap (CPU runs MAX_FRAMES_IN_FLIGHT ahead), TLAS build at head of graph cmd buffer | **crash ≤ ~6 s** |
| 2 | Headless (no swapchain), full overlap, same renderer, same scene, 5000 frames | stable |
| 3 | #1 + acquire-semaphore waited at TRANSFER (spec fix, see below) | **crash** |
| 4 | #3 + `vkGetBufferDeviceAddress` cached (zero per-frame BDA calls) | **crash** |
| 5 | #4 + TLAS build moved to its own async submission (semaphore into graph) | **crash** |
| 6 | #5 + per-frame TLAS builds **disabled entirely** (static TLAS) | **crash** |
| 7 | #5 + submission gated on frame N−1 completion (CPU records ahead, GPU queue depth 1) | **crash** |
| 8 | #7 + dedicated `vkCommandPool` per re-recorded command buffer | **crash** |
| 9 | N−1 timeline pacing (no CPU run-ahead) + the async setup submission of #5 | **crash ≤ ~6 s** |
| 10 | N−1 timeline pacing + head-of-graph TLAS build + all fixes | **crash ~27–40 s** |
| 11 | Committed baseline (fully serialized: `vkDeviceWaitIdle` at frame start, fence wait after submit) | stable (180 s soak) |
| 12 | Rework + `vkQueueWaitIdle` at the start of each presenting frame (shipped) | stable (240 s soak) |

Reading of the matrix:

- The crash needs **presents** (headless full overlap is rock solid for thousands of
  frames) **and** CPU-side driver activity while the queue is busy, but is **not**
  caused by any specific call we make: it persists with no AS builds at all (#6), no
  per-frame BDA queries (#4), isolated command pools (#8), and a GPU queue depth of
  one (#7).
- It is probabilistic per frame and the probability tracks how much CPU driver work
  overlaps the busy queue: full run-ahead ⇒ seconds (#1), N−1 pacing (CPU work still
  overlaps the previous frame's presents/barrier) ⇒ tens of seconds (#10), queue
  drained before CPU work ⇒ stable (#11, #12).
- Splitting the frame into an extra small submission per frame (#5/#9) made it
  *worse* (fast reproduction even without run-ahead), pointing at per-submission
  bookkeeping consumed by the worker thread that also services presents.
- The crashing worker performs a hash-style lookup (splitmix64 mixing constant
  `0xff51afd7ed558ccd` in registers) and dereferences a NULL entry — consistent with
  an internal table being read by the worker while the application thread mutates it
  (insert/remove racing lookup).

## API-correctness findings (fixed, kept regardless)

Two real issues were found and fixed during the investigation — neither was the
trigger, but both are genuine:

1. **Acquire-semaphore wait stage.** The blit that writes the swapchain image waited
   the acquire semaphore at `ALL_GRAPHICS`, which does **not** include the TRANSFER
   stage the blit (and the image's `UNDEFINED` discard transition, which had
   `srcStage=NONE`) executes in — so the swapchain write was never actually ordered
   after the presentation engine released the image. Fixed: wait at `TRANSFER`, and
   chain the discard transition's `srcStage` from `TRANSFER`.
2. **Per-frame `vkGetBufferDeviceAddress` round-trips.** Now cached in `RawBuffer`
   (the address is immutable per buffer object).

## Workaround shipped

`Renderer::serialize_frames_workaround()` (`src/lib.rs`): when the renderer presents
through its internal swapchain, `render()` calls `vkQueueWaitIdle` on the graphics
queue at the **start of the frame**, draining all in-flight GPU work before the CPU
does any driver-side frame preparation, recording, or submission. This reproduces the
one soak-stable condition (no CPU driver activity while the queue is busy) while
keeping the entire structural rework intact: per-slot upload buffers, the
double-buffered TLAS built inside the frame's command buffer, rotated per-slot graph
command buffers, zero synchronous submits, and zero per-frame allocations are all
unchanged — only the run-ahead is removed for the presenting path. Offscreen rendering
(no swapchain) keeps the full `MAX_FRAMES_IN_FLIGHT` overlap and is unaffected.

Set **`SUNRAY_FULL_FRAMES_IN_FLIGHT=1`** to re-enable full overlap with a swapchain —
use this to re-test whenever a newer descriptor-heap driver lands. If it survives a
few minutes of the `window` example, the workaround can be removed.

## Update 2026-08-04 — still present on 610.88, same site

Driver 32.0.16.1088 (610.88), RTX 3060 Ti, branch `render-graph`. Every WER record is
`nvoglv64.dll+0x15f9a2` — the same code site as `+0x15f942` on 1047, shifted by the
build. Ruled out by direct experiment, all crashing at that one offset:

| Variable | Values tried | Result |
|---|---|---|
| `SUNRAY_ALIAS_STRATEGY` | `slot`, `bucket`, **`off`** (no aliasing at all, one allocation per resource, zero alias barriers) | crash |
| `SUNRAY_ENABLE_GPUAV` | `0`, `1` | crash |
| Build profile | release (~frame 560), debug (~frame 105) | crash |
| Implicit layers | all force-disabled via `VK_LOADER_LAYERS_DISABLE=~implicit~` (ReShade, OBS, GOG Galaxy, Steam, NV Optimus/present) | crash |
| Present mode | `MAILBOX` (today's auto-pick) and `FIFO` (what the soaks above ran on), via the new `SUNRAY_PRESENT_MODE` | crash |
| Whole-queue drain before any CPU frame work (row #12 restored verbatim at the head of `render_to_swapchain_with`) | on | **crash, ~4 s** — reverted, it does not carry its weight |

So it is not the transient-memory aliasing, not a third-party layer, not the present
mode, and — new since 1047 — **row #12's queue drain no longer helps**. Note that today's
`SUNRAY_SERIALIZE_FRAMES=1` is a weaker thing than row #12 was: it is
`wait_graph_timeline(frame_value)` at the *end* of `render`, which covers the graph
submission only, leaving presents and the egui overlay in flight. Neither form is stable
on release.

Release crashes at ~frame 100–560; **debug builds are stable** with serialization on.
That split is timing, not logic — it is the same probabilistic overlap sensitivity the
matrix above measured, just re-expressed as optimization level.

### Per-frame object churn: aggravator, not cause

`TransientResources::populate` calls `free_internal_state()` **every frame** — every
transient `VkImage`/`VkImageView`/`VkBuffer` destroyed, its allocation freed, its
descriptor-heap slot recycled, then all of it recreated. That is the shape the crash-dump
analysis implicates (application thread mutating a driver object table while a worker
reads it), and `SUNRAY_ALIAS_STRATEGY=off` does *not* test it — the churn is identical
under all three strategies.

`SUNRAY_LEAK_TRANSIENTS=1` tests it directly: handles, views and samplers are `forget`ed
instead of destroyed and `SlotAllocator::free` retires the index instead of recycling it,
so no descriptor index is ever handed out twice and no `vkDestroyImage` ever runs. (The
bucket allocations are still freed — leaking those too OOMs at ~frame 90, before the
crash window.) Six runs per arm, release, `SERIALIZE_FRAMES=1`, `ALIAS_STRATEGY=slot`,
`DESCRIPTOR_HEAP_SCALE=128`:

| arm | access violations | last frame per run |
|---|---|---|
| `LEAK_TRANSIENTS=0` | **6/6** | 96, 343, 292, 388, 494, 1075 |
| `LEAK_TRANSIENTS=1` | **3/5** | 97, 200, 91, 2392*, 10985* |

`*` = clean app-side exit (descriptor section exhausted), not a crash.

Same sweep with `SERIALIZE_FRAMES=0` (frame overlap — the historically fast trigger):

| arm | access violations | last frame per run |
|---|---|---|
| `LEAK_TRANSIENTS=0` | **6/6** | 105, 101, 104, 100, 107, 102 |
| `LEAK_TRANSIENTS=1` | **2/2** | 102, 100 |

Under overlap it is a 6/6 reproduction inside ~2 s with a spread of seven frames,
identical with leaking on. **This is the repro to hand a driver report** — far tighter
than the serialized case, which spans 91..1075 frames. (Don't read the collapsed fps in
the last heartbeat as a stall precursor: Aftermath's dump handler and WER take seconds to
run while the main thread is still looping.)

**Verdict: refuted as the cause** — it still faults at frame 91 with nothing ever freed
or reused. But it is a genuine aggravator: the surviving runs went an order of magnitude
past the no-leak best. Consistent with the original matrix, where crash probability
tracked how much CPU-side driver work overlapped a busy queue rather than any specific
call. Single runs cannot distinguish these arms — the crash frame spans 91..1075 within
one configuration, so anything below ~5 runs per arm is noise.

Descriptor-heap ceiling found while sizing this test: the resource descriptor buffer caps
at **32 MiB** and the sampler buffer at **128 KiB** (4080 usable slots). `LEAK_TRANSIENTS`
and the `DESCRIPTOR_HEAP_SCALE` knob it needed were both removed once the hypothesis died;
the tables above are the record.

### Update 2026-08-05 — bisected to the temporal-accumulation pass

Candidates ruled out since, without a single soak run each:

| Candidate | Verdict |
|---|---|
| NVIDIA Aftermath's crash handler (`SUNRAY_ENABLE_NVIDIA_AFTERMATH`, on for every run in the tables above) | crashes with it off too |
| The egui overlay / `finalize` submission outside the graph timeline | not on the crashing path at all — `examples/window` calls `render_to_swapchain`, i.e. `finalize: None`, so the graph itself signals the present semaphore (`FrameOutput::Present`) |
| Cross-thread Vulkan object destruction | there is none. The watcher thread (`lib.rs:385`) only calls `vkWaitSemaphores` and stores an atomic; every end-of-frame callback runs on the render thread inside `render` (`lib.rs:983`) |
| Machine-specific (VRAM, ReBAR, factory OC) | the same build runs clean on an RX 9060 XT. Doesn't separate "NVIDIA driver defect" from "app misuse only NVIDIA sees", since `VK_EXT_descriptor_heap` exists nowhere else |

**`SUNRAY_STRIP=N`** (added for this) builds only the first N stages of the unified graph
and compiles what it has, leaving acquire → blit → present untouched. It closes the gap
between the two known-stable extremes — headless full-overlap and a bare present loop.
Release, `SERIALIZE_FRAMES=0`, `ALIAS_STRATEGY=slot`, 3 runs per rung, 25 s cap:

| `SUNRAY_STRIP` | graph contains | AVs | last frame per run |
|---|---|---|---|
| 0 | nothing — acquire, blit, present | 0/1 | 54194 (survived) |
| 1 | + staging copies, BLAS / TLAS builds | 0/3 | 44010, 47090, 50827 (survived) |
| 2 | + RIS ray-tracing pass | 0/3 | 3131, 8783, 3183 (survived) |
| 3 | + final-shading ray-tracing pass | 0/3 | 1353, 1322, 1335 (survived) |
| **4** | **+ temporal accumulation** | **3/3** | 116, 110, 110 |
| 5 | + a-trous denoise | 3/3 | 95, 95, 98 |
| 6 | + postprocess (the real frame) | 3/3 | 103, 97, 92 |

Rung 3 sustains 1335 frames at the same ~53 fps that rung 4 dies at after 110 — a 12x
gap, not a sampling artifact. `add_temporal_pass` is the switch.

Five controls, each 3 runs, narrow what about that rung matters:

| Control | Result |
|---|---|
| Rung 4, `SUNRAY_ALIAS_STRATEGY=off` — no aliasing, one dedicated allocation per transient | **3/3 AV**, frames 110, 109, 109 |
| Rung 4, accumulation ping-pong replaced by two per-frame transients — **no cross-frame image anywhere** | **3/3 AV**, frames 115, 112, 109 |
| Rung 3 + a compute pass reading `rg_rt_raw_color` (transient, written by the RT passes) and writing the output import | 0/3, ~2490 frames in 45 s |
| Rung 3 + the same pass writing a **per-frame transient** instead | 0/3, ~1600 frames in 30 s |
| Rung 3 + the same pass reading `rg_motion_vec` (R16G16_SFLOAT) instead | 0/3, ~1590 frames in 30 s |

The rung-3 controls really ran: writing `postprocess_out` again makes the
`newLayout-01198` noise disappear (20 lines → 0), and each costs ~4 fps.

Ruled out at the rung where the crash appears, therefore:

* transient memory **aliasing** — `off` dies identically;
* the **cross-frame temporal image** — replacing the ping-pong with per-frame transients
  changes nothing, which also clears `set_import_access` threading, ping-pong parity and
  cross-frame layout carry;
* **a compute dispatch** as such;
* **reading a transient** an earlier pass wrote (both `raw_color` and `motion`);
* **writing a per-frame transient** from a compute pass.

The temporal machinery was also read end to end and is sound: `analyze_passes` emits
WAR edges (a write is ordered after every `readers_since_write`), `declare_previous_imports`
routes `Nothing` to the write list so the first user always gets a barrier, the write-back
loop threads per-copy, and `resize_internal_images` is `device_wait_idle`-gated and
recreates every backing.

A sixth control settles the last question. Running **one denoise pass** at rung 3
instead — a different shader, one storage read + three *sampled* reads + one storage
write, five heap image descriptors — crashes **3/3** (frames 536, 104, 104).

So it is not the temporal-accumulation shader, and it is not the sampled reads or their
layout transitions either (temporal accumulation has none — all four of its images are
GENERAL storage).

Three synthetic controls then pinned the actual variable. Building the payload out of
repeated postprocess dispatches makes every property independently adjustable:

| compute work in the frame | distinct images | dispatches | image reads | result |
|---|---|---|---|---|
| read `A` → write `B` | **2** | 1 | 1 | clean, ~2490 frames |
| read `A` → write `B`, twice (same in, same out) | **2** | 2 | 2 | clean, ~1615 frames |
| read `A` → write `B`; read `A` → write `D` | **3** | 2 | 2 | **3/3 AV**, 109, 106, 108 |
| read `A` → write `B`; read `C` → write `D` | **4** | 2 | 2 | **3/3 AV**, 110, 111, 111 |
| temporal accumulation | **4** | 1 | 3 | **3/3 AV**, 116, 110, 110 |
| denoise pass 0 | **5** | 1 | 4 | **3/3 AV**, 536, 104, 104 |

**The correlate is the number of distinct images the frame's compute work references: 2
is stable, 3 or more crashes.** Not the dispatch count (2 dispatches clean at 2 images),
not per-dispatch (two dispatches each identical to the clean control crash once they span
3 distinct images), not the read count (the 3-image crash reads one distinct image; the
clean 2-dispatch row reads two), not read-versus-write (adding one distinct *output* is
enough), not format, lifetime, or shader.

The ray-tracing asymmetry survives all of it: the RT passes reference five distinct images
through the same heap in the same frames and never fault. Whatever the driver mishandles
is on the compute path.

Unrelated robustness hole found while wiring that control: `transient_resources.rs:179`
indexes `usages[res_id]` and panics for a `Created` resource no pass ever touches. Real
graphs always consume what they create, so nothing hits it today.

Note on validation: rungs 0–5 emit `VUID-VkImageMemoryBarrier2-newLayout-01198`
(`newLayout` is `UNDEFINED`), which is an artifact of the knob itself — nothing writes
`postprocess_out`, so `run_present`'s restore barrier falls back to
`[AccessType::Nothing]` (`graph.rs:1791`) and vk_sync maps that to `UNDEFINED`. Rung 6,
the configuration that actually crashes, is validation-clean.

## Reproducing for a driver report

1. Driver 32.0.16.1047, any descriptor-heap-capable build presumably.
2. `SUNRAY_FULL_FRAMES_IN_FLIGHT=1 cargo run --example window` (validation on or off —
   it crashes either way and validation reports nothing beforehand).
3. Crashes in seconds; WER dump shows the worker-thread NULL deref at
   `nvoglv64.dll+0x15f942`.
