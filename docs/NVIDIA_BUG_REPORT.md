# Driver crash: NULL dereference on a driver worker thread when compute dispatches reference three or more distinct heap images while presenting

`VK_EXT_descriptor_heap` developer driver. Submittable summary; the full investigation
log, including everything that was ruled out and how, is in
`NVIDIA_DRIVER_CRASH_REPORT.md` alongside this file.

## Summary

A Vulkan application that presents to a swapchain and runs compute dispatches referencing
**three or more distinct images** through the descriptor heap crashes inside the NVIDIA
user-mode driver within ~110 frames. The faulting thread is driver-internal, not
application code. The identical application whose compute work touches **two** distinct
images runs indefinitely, as does the same frame with the compute dispatches removed.

The threshold is on the count of *distinct images* the frame's compute work references. It
does not depend on how those images are split across dispatches, whether they are read or
written, their formats, their lifetimes, or which shader is used — see the table below,
where two dispatches are stable at two images and crash at three.

Ray-tracing dispatches referencing *five* distinct images through the same descriptor heap,
in the same frames, are stable. The defect appears specific to the compute path.

## Environment

|             |                                                                                                             |
|-------------|-------------------------------------------------------------------------------------------------------------|
| GPU         | NVIDIA GeForce RTX 3060 Ti                                                                                  |
| Driver      | 32.0.16.1088 (610.88), the `VK_EXT_descriptor_heap` developer driver. Also reproduced on 32.0.16.1047       |
| OS          | Windows 11 Pro 10.0.26200                                                                                   |
| API         | Vulkan 1.4, `VK_EXT_descriptor_heap`, timeline semaphores, `vkCmdPipelineBarrier2`                          |
| Application | `sunray`, `cargo run --release --example window` , https://github.com/unitn-sunray/sunray/tree/render-graph |
| Cross-check | The same build runs correctly on an AMD RX 9060 XT — but see the limits noted below: that build cannot take the heap path |

## Symptom

`STATUS_ACCESS_VIOLATION` (0xc0000005) terminates the process. Every occurrence faults at
the same address in the user-mode driver:

```
Faulting module name: nvoglv64.dll, version: 32.0.16.1088
Exception code:       0xc0000005
Fault offset:         0x000000000015f9a2      (0x15f942 on 32.0.16.1047)
```

From full-memory WER dumps (`minidump-stackwalk`):

* The crashing thread is a **driver-internal worker**. Its stack is entirely `nvoglv64.dll`
  and `ntdll.dll` — no application, loader, or callback frames.
* The faulting instruction is `mov r8, qword [rsi]` with `rsi = 0`: a NULL-pointer read
  inside the driver.
* Registers at the fault hold the splitmix64 finalizer constant `0xff51afd7ed558ccd`,
  consistent with a hash-table lookup that returned a NULL entry.
* The application's main thread is meanwhile inside a WSI call (`win32u.dll` syscall
  entered from `nvoglv64.dll`), i.e. the acquire/present path. An api_dump trace confirms
  the last application-side call had already returned before the fault — the AV fires
  asynchronously on the driver worker.

The Vulkan validation layers, including synchronization validation, are **silent** in the
crashing configuration.

## Reproduction

Frame overlap makes it deterministic and fast: 3/3 runs, access violation at frame
~100–110, within ~2 seconds.

```
set SUNRAY_SERIALIZE_FRAMES=0
cargo run --release --example window
```

The application also has a stage-stripping switch, `SUNRAY_STRIP=N`, that builds only the
first N stages of its render graph while leaving acquire → blit → present untouched. It
localizes the trigger precisely:

| `SUNRAY_STRIP` | frame graph contains                              | runs  | access violations | last frame per run                          |
|----------------|---------------------------------------------------|-------|-------------------|---------------------------------------------|
| 0              | acquire, blit, present only                       | 1     | 0                 | 54194 (survived 25 s)                       |
| 1              | + buffer uploads, BLAS / TLAS builds              | 3     | 0                 | 44010, 47090, 50827                         |
| 2              | + ray-tracing pass (RIS)                          | 3     | 0                 | 3131, 8783, 3183                            |
| 3              | + second ray-tracing pass                         | 3     | 0                 | 1353, 1322, 1335 (3386, 3410, 3416 at 60 s) |
| **4**          | **+ compute dispatch touching 4 distinct images** | **3** | **3**             | 116, 110, 110                               |
| 5              | + 8 more compute dispatches                       | 3     | 3                 | 95, 95, 98                                  |
| 6              | + final compute dispatch                          | 3     | 3                 | 103, 97, 92                                 |

Rung 3 sustains 3400 frames at the same ~55 fps at which rung 4 dies after 110.

## What the threshold actually is

Each row below replaces rung 4's real work with a synthetic compute payload, holding the
rest of the frame identical, and varies exactly one property. `A`, `C` are images the
ray-tracing passes wrote; `B`, `D` are compute outputs. 3 runs each, release, frame
overlap on.

| compute work in the frame                                                 | distinct images | dispatches | image reads | result                     |
|---------------------------------------------------------------------------|-----------------|------------|-------------|----------------------------|
| read `A` → write `B`                                                      | **2**           | 1          | 1           | clean, ~2490 frames / 45 s |
| …writing a per-frame image instead of a persistent one                    | 2               | 1          | 1           | clean, ~1600 frames / 30 s |
| …reading `R16G16_SFLOAT` instead of `B10G11R11_UFLOAT_PACK32`             | 2               | 1          | 1           | clean, ~1590 frames / 30 s |
| read `A` → write `B`, **twice** (same input, same output)                 | **2**           | 2          | 2           | clean, ~1615 frames / 30 s |
| read `A` → write `B`; read `A` → write `D`                                | **3**           | 2          | 2           | **3/3 AV**, 109, 106, 108  |
| read `A` → write `B`; read `C` → write `D`                                | **4**           | 2          | 2           | **3/3 AV**, 110, 111, 111  |
| temporal accumulation: read `A`,`C`,`E` → write `B` (all GENERAL storage) | **4**           | 1          | 3           | **3/3 AV**, 116, 110, 110  |
| …with its two cross-frame images replaced by per-frame images             | 4               | 1          | 3           | **3/3 AV**, 115, 112, 109  |
| denoise, a different shader: 1 storage + 3 sampled reads → 1 write        | **5**           | 1          | 4           | **3/3 AV**, 536, 104, 104  |

Reading down the "distinct images" column is the whole result: **2 is stable, 3 or more
crashes**, and nothing else lines up.

* **Not the dispatch count.** Two dispatches are clean at two distinct images.
* **Not per-dispatch.** Two dispatches of one read and one write each — individually
  identical to the clean single-dispatch case — crash as soon as they touch three
  distinct images between them.
* **Not the number of reads.** The 3-image crash reads a *single* distinct image; the
  clean two-dispatch row reads two.
* **Not reads versus writes.** Adding one distinct *output* is enough (rows 4 → 5).
* **Not the shader, format, or resource lifetime.** Two unrelated shaders crash; one of
  them uses no sampled images and performs no layout transitions at all.

Meanwhile the two ray-tracing passes at rungs 2–3 reference five distinct storage images,
two storage buffers and a TLAS through the same descriptor heap, in the very same frames,
and are stable for 50000 frames.

## Why we believe the defect is in the driver

An application can corrupt driver state, so "the stack is all `nvoglv64.dll`" proves nothing
on its own. The argument is that every application-side mechanism capable of producing this
is either held constant across the clean/crashing boundary, or varies the wrong way.

**How images reach the shader.** There are no descriptor sets. Every image is addressed by a
`uint` index into the global heap, delivered in a push constant. A descriptor is written by
`vkWriteResourceDescriptorsEXT` into the host-mapped heap buffer, once per image object, at
first use. Transient images are recreated per frame and so cost one write per frame each
(two if used both as `STORAGE` and `SAMPLED`); persistent images are written once, ever.

**Our descriptor traffic does not order the results.** Per-frame write counts are exact:

| configuration | descriptor writes / frame | distinct images in compute | result |
|---|---|---|---|
| rung 3 (ray tracing only) | 8 | 0 | clean, 3400 frames |
| **rung 4** (adds one dispatch; both extra images are persistent, written once ever) | **8** | **4** | **AV at frame 110** |
| rung 3 + one dispatch writing an extra per-frame image | 9 | 2 | clean, 1600 frames |
| rung 3 + two dispatches over three distinct images | 10 | 3 | AV at frames 109, 106, 108 |

Rungs 3 and 4 issue **the same eight descriptor writes per frame over the same set of
images**, with identical allocation, lifetime and slot-recycling behaviour; they differ only
by one `vkCmdDispatch` referencing four heap slots. One survives 3400 frames, the other dies
at 110. Meanwhile the arm that adds *more* heap churn than either is clean. The outcome is
non-monotonic in our descriptor write rate, so descriptor writing is not the mechanism.

**Slot recycling is excluded directly.** In a diagnostic mode that never returns a heap index
to its free list and never destroys an image, view or sampler — heap indices strictly
increasing, no descriptor ever overwritten — the crash persists at the same fault offset.

**The trigger has no application-side counterpart.** Going from two to three distinct images
changes no code path in the renderer: same allocator, same barrier planner, same submission
structure, same object lifetimes. It changes the contents of one push-constant block and the
number of heap slots one dispatch reads.

**Serialization masking it is consistent with a driver-internal race, not with a deterministic
application error.** Blocking the CPU until each frame completes makes the crash disappear;
the GPU workload is byte-identical either way. What changes is whether application-thread
driver calls overlap the driver's own worker thread. Combined with the fault signature — a
worker thread dereferencing the NULL result of a hash lookup, with the splitmix64 mixing
constant live in registers — this points at an internal table read by the worker while it is
mutated elsewhere. The distinct-image count appears to control what is in that table; frame
overlap controls whether two threads reach it at once. Removing either hides the crash.

**Limits of this argument, stated plainly:**

* `VK_EXT_descriptor_heap` exists on no other vendor, so the clean AMD run is *not* a
  controlled comparison — that build takes an entirely different binding path. Heap mode is
  the one variable this investigation could never vary.
* Silent validation layers only cover what the layers check; there is no validation coverage
  for heap descriptor lifetime.
* If writing heap descriptors requires external synchronization against in-flight work that
  does not access those descriptors, that requirement is what we have violated — please say
  so and we will fix it. We could find no such statement in the extension documentation.

## Ruled out

Each of the following still faults at the identical offset, so none of them is the cause:

* **Memory aliasing.** The renderer can be told to give every transient resource its own
  dedicated allocation, with no aliasing and no aliasing barriers. Crashes identically,
  including at rung 4 specifically.
* **Object churn and descriptor-slot recycling.** A mode that leaks every image, view and
  sampler and never returns a descriptor-heap index to its free list — so the driver's
  object tables only ever grow — still crashes. It *is* a strong aggravator: surviving runs
  went an order of magnitude longer.
* **Cross-frame resources.** Replacing the ping-pong history images with per-frame images
  changes nothing.
* **Present mode.** `MAILBOX` and `FIFO` both crash.
* **GPU-assisted validation**, on or off.
* **Third-party implicit layers** — all force-disabled with
  `VK_LOADER_LAYERS_DISABLE=~implicit~` (ReShade, OBS, GOG Galaxy, Steam, NV Optimus).
* **NVIDIA Aftermath**, on or off.
* **Draining the graphics queue** (`vkQueueWaitIdle`) before each frame's CPU work.
* **Application-side synchronization.** The render graph's barrier planner was re-read end
  to end for this: write-after-read edges are emitted, resources entering a frame in an
  undefined state are treated as writes so their first user always gets a barrier, and the
  swapchain-resize path is `vkDeviceWaitIdle`-gated. Synchronization validation agrees.

## Notes for triage

* The trigger is a *count of distinct images*, evaluated across the frame's compute work
  rather than per dispatch, and it is not a resource-specific or shader-specific effect.
  Two is the largest stable value observed.
* The compute / ray-tracing asymmetry is the sharpest lead: ray tracing referencing five
  distinct images through the same heap, in the same frames, never faults.
* Presenting is required. The same renderer running headless with full frame overlap was
  soak-stable for 5000 frames, and a bare acquire/present loop with no compute work at all
  survives 54000 frames. The fault needs both a present in flight and the dispatch.
* Crash frequency scales with how much CPU-side frame work overlaps a busy queue. Fully
  serializing the frame loop hides it; it is not a fix.
