# SunPath (Sunray V2)

Rust hardware real time path-tracing library

This project was developed by [Riccardo-Finello](https://github.com/riccardoFinelloUniTn) supervised by Professor [Marco Patrignani](https://squera.github.io/) for the bachelor thesis at the University of Trento, Italy
<br>
It's based on the [sunray](https://github.com/kalsifer-742/sunray) project developed by [kalsifer-742](https://github.com/kalsifer-742) and [circled-square](https://github.com/circled-square)
## Environment variables

The library takes no command-line arguments. Every knob below is a debug /
diagnostic toggle read from the environment — anything that changes rendering
behaviour is part of the `Renderer` API instead. Names live in one place,
`src/utils.rs`.

Booleans accept `1`/`0`, `true`/`false`, `on`/`off` (case-insensitive);
an unrecognized value logs a warning and falls back to the default.

| Variable | Default | Effect |
|---|---|---|
| `SUNRAY_ENABLE_VALIDATION_LAYER` | on in debug builds, off in release | Vulkan validation layer |
| `SUNRAY_ENABLE_GPUAV` | off | GPU-assisted validation. No effect unless the validation layer is on |
| `SUNRAY_ENABLE_NSIGHT` | off | Debug-utils labels + object names for readable Nsight Graphics captures. Takes precedence over `SUNRAY_ENABLE_NVIDIA_AFTERMATH` |
| `SUNRAY_ENABLE_NVIDIA_AFTERMATH` | off | NVIDIA Aftermath crash dumps. The full user-space handler also needs the `nvidia-aftermath` feature |
| `SUNRAY_SERIALIZE_FRAMES` | **on** | Whole-frame serialization. Set to `0` to opt into frame overlap — known async use-after-free crash inside the NVIDIA driver, see `Renderer::render` |
| `SUNRAY_GRAPH_DUMP_DIR` | off | Per-frame render-graph dump (`.dot` + `.txt`). `1` writes into `<crate>/debug` (git-ignored); any other value is used as the destination directory |
| `SUNRAY_SHADER_DEBUG` | off | **Build-time.** Disables shader optimization and emits maximal SPIR-V debug info so GPU debuggers resolve Slang source. Changing it re-runs the build script |

Read once at `Renderer` construction, except `SUNRAY_GRAPH_DUMP_DIR` (per frame)
and `SUNRAY_SHADER_DEBUG` (build script).

`.cargo/config.toml` sets the common ones plus run aliases for the examples —
`cargo win` / `cargo png` / `cargo bevy` build release, and the `-dbg` variants
(`cargo win-dbg`, …) build the dev profile. The env values there are defaults
only: a variable already set in your shell always wins.

```sh
SUNRAY_ENABLE_NSIGHT=1 cargo win
SUNRAY_GRAPH_DUMP_DIR=1 cargo png   # dumps into ./debug
SUNRAY_SHADER_DEBUG=1 cargo build
```

### Cargo features

| Feature | Effect |
|---|---|
| *(default)* | none |
| `nvidia-aftermath` | Links the NVIDIA Aftermath SDK for the user-space crash-dump handler. Without it, `SUNRAY_ENABLE_NVIDIA_AFTERMATH` still wires up the Vulkan-side diagnostics/checkpoint extensions |
| `bevy` | Enables `sunray::bevy_integration` and the `bevy_app` example — see [docs/bevy_integration.md](docs/bevy_integration.md) |

## CI

`.github/workflows/ci.yml` runs two jobs on every push and PR.

**`check`** (GitHub-hosted, Ubuntu) — `cargo fmt --all --check`, `cargo clippy
--all-targets`, `cargo test`. Compile-time only: a hosted runner can build the crate
but can never run it, because the device requires `VK_EXT_descriptor_heap` and
`VK_KHR_shader_untyped_pointers`, which neither lavapipe nor SwiftShader implement.
Slang and the Vulkan loader both come from a pinned LunarG SDK — pinned, not `latest`,
because `build.rs` panics if the bundled Slang predates the `spvDescriptorHeapEXT`
capability.

**`gpu`** (self-hosted, this project's Windows box) — everything that actually
executes: `cargo test --release -- --include-ignored`, the offscreen `cargo png` render
compared byte-for-byte against a baseline, and 10-second liveness smoke tests of the
`window` and `bevy_app` examples. The runner must run interactively in a logged-in
session; a Windows service can neither reach the GPU nor open a window. The job is
gated to same-repo events, since a self-hosted runner reachable from a fork PR is
remote code execution.

The two render-graph tests that construct a `Core` are marked `#[ignore]`, so a plain
`cargo test` stays GPU-free; `--include-ignored` runs the full set.

### Re-baselining the render

`examples/png/render.sha256` pins the expected output of `cargo png`. After an
intentional visual change — or a driver update, which also shifts the hash — regenerate
it and commit:

```powershell
cargo png
(Get-FileHash examples/png/render.png -Algorithm SHA256).Hash | Set-Content -Encoding ascii examples/png/render.sha256
```

## Contribution

If you wish to contribute to the project you may check our issues, or if you found a bug or missing feature feel free to create one. 
You may also contact us at the e-mail addresses linked to our GitHub accounts.

If you're studying at University of Trento and are looking for a thesis subject you can ask Professor Marco Patrignani 
to be your supervisor to work on this project and we will be available if you need help or clarifications.

Thesis proposals:
- Library Integration https://github.com/kalsifer-742/sunray/issues/52

## Comparison

|                                                                      | Active project | Non-trivial | Real-time | Fully ray-traced | Hybrid | GPU | HW RT | Compute | SIMD | BVH | Mesh | Materials | Denoise | Rust | Crate |  Engine   |                                Notes |
|:---------------------------------------------------------------------|:--------------:|:-----------:|:---------:|:----------------:|:------:|:---:|:-----:|:-------:|:----:|:---:|:----:|:---------:|:-------:|:----:|:-----:|:---------:|-------------------------------------:|
| [Kajiya](https://github.com/EmbarkStudios/kajiya)                    |       ❌        |      ✅      |     ✅     |        ✅         |   ✅    |  ✅  |   ✅   |    ✅    |  ❌   |  ?  |  ✅   |     ✅     |    ✅    |  ✅   |   ❌   |     ❌     |                                      |
| [Cycles](https://projects.blender.org/blender/cycles)                |       ✅        |      ✅      |     ❌     |        ✅         |   ❌    |  ✅  |   ✅   |    ✅    |  ✅   |  ✅  |  ✅   |     ✅     |    ✅    |  ❌   |  N/A  | ✅ Blender |                                      |
| [manta-ray](https://github.com/ange-yaghi/manta-ray)                 |       ❌        |      ✅      |     ❌     |        ✅         |   ❌    |  ✅  |   ❌   |    ✅    |  ✅   |  ✅  |  ✅   |     ✅     |    ✅    |  ❌   |  N/A  | ✅ Blender |                                      |
| [luxcore](https://luxcorerender.org/)                                |       ✅        |      ✅      |     ❌     |        ?         |   ?    |  ✅  |   ❌   |    ✅    |  ?   |  ?  |  ✅   |     ✅     |    ?    |  ❌   |  N/A  | ✅ Blender |                                      |
| [akari_render](https://github.com/shiinamiyuki/akari_render)         |       ❌        |      ✅      |     ?     |        ?         |   ?    |  ✅  |   ❌   |    ✅    |  ?   |  ?  |  ✅   |     ✅     |    ?    |  ✅   |   ❌   | ✅ Blender |           Rebuild blender to install |
| [KaminariOS/rustracer](https://github.com/KaminariOS/rustracer)      |       ❌        |      ✅      |     ❌     |        ✅         |   ❌    |  ✅  |   ✅   |    ❌    |  ❌   |  ❌  |  ✅   |     ✅     |    ❌    |  ✅   |   ❌   |     ❌     |                             uses Nix |
| [RayTracingInVulkan](https://github.com/GPSnoopy/RayTracingInVulkan) |       ✅        |      ✅      |     ✅     |        ✅         |   ❌    |  ✅  |   ✅   |    ❌    |  ?   |  ✅  |  ✅   |  partial  |    ❌    |  ❌   |  N/A  |     ❌     |                                      |
| [referencePT](https://github.com/boksajak/referencePT)               |       ❌        |      ✅      |     ?     |        ?         |   ?    |  ✅  |   ✅   |    ❌    |  ❌   |  ?  |  ✅   |     ✅     |    ?    |  ❌   |  N/A  |     ❌     |                                      |
| [gbrt](https://github.com/giulianbiolo/gbrt)                         |       ❌        |      ❌      |     ❌     |        ❌         |   ❌    |  ❌  |   ❌   |    ❌    |  ✅   |  ✅  |  ✅   |     ❌     |    ❌    |  ✅   |   ❌   |     ❌     |                                      |
| [Godot4-RayTracing](https://github.com/bitegw/Godot4-Raytracing)     |       ❌        |      ❌      |     ✅     |        ✅         |   ❌    |  ✅  |   ❌   |    ✅    |  ❌   |  ❌  |  ❌   |  partial  |    ❌    |  ❌   |  N/A  |  ✅ Godot  |                                      |
| [Raytracing_Godot4](https://github.com/nekotogd/Raytracing_Godot4)   |       ❌        |      ❌      |     ✅     |        ✅         |   ❌    |  ✅  |   ❌   |    ✅    |  ❌   |  ❌  |  ❌   |     ❌     |    ❌    |  ❌   |  N/A  |  ✅ Godot  |                                      |
| [bevyray](https://github.com/GrandmasterB42/bevyray)                 |       ✅        |      ❌      |     ✅     |        ❌         |   ✅    |  ✅  |   ❌   |    ❌    |  ❌   |  ✅  |  ❌   |  partial  |    ❌    |  ✅   |   ❌   |  ✅ Bevy   |        raytracing in fragment shader |
| [hanamaru-renderer](https://github.com/gam0022/hanamaru-renderer)    |       ❌        |      ❌      |     ❌     |        ?         |   ?    |  ❌  |   ❌   |    ❌    |  ?   |  ✅  |  ✅   |     ✅     |    ✅    |  ✅   |   ❌   |     ❌     |                 docs are in japanese |
| [rtwlib](https://crates.io/crates/rtwlib)                            |       ✅        |      ❌      |     ❌     |        ✅         |   ❌    |  ❌  |   ❌   |    ❌    |  ❌   |  ❌  |  ❌   |     ❌     |    ❌    |  ✅   |   ✅   |     ❌     |                                      |
| [rustic-zen](https://crates.io/crates/rustic-zen)                    |       ❌        |      ❌      |     ✅     |        ?         |   ?    |  ❌  |   ❌   |    ❌    |  ?   |  ?  |  ?   |     ?     |    ?    |  ✅   |   ✅   |     ❌     |                                   2D |
| [andros21/rustracer](https://crates.io/crates/rustracer)             |       ❌        |      ❌      |     ❌     |        ✅         |   ❌    |  ❌  |   ❌   |    ❌    |  ❌   |  ❌  |  ❌   |     ❌     |    ❌    |  ✅   |   ✅   |     ❌     |                                      |
|                                                                      |                |             |           |                  |        |     |       |         |      |     |      |           |         |      |       |           |                                      |
| [Pbrt4](https://github.com/mmp/pbrt-v4)                              |       ✅        |      ✅      |     ❌     |        ✅         |   ❌    |  ✅  |   ✅   |    ❌    |  ❌   |  ✅  |  ✅   |     ✅     |    ✅    |  ❌   |   ❌   |     ❌     | Works on cpu, Gpu path requires cuda |
| [sunray](https://github.com/Kalsifer-742/sunray)                     |       ✅        |      ✅      |     ✅     |        ✅         |   ✅    |  ✅  |   ✅   |    ❌    |  ❌   |  ✅  |  ✅   |  partial  |    ✅    |  ✅   |   ✅   |  ✅ Bevy   |                                      |

## Resources

### General

- [Nvidia tutorial on vulkan KHR raytracing](https://nvpro-samples.github.io/vk_raytracing_tutorial_KHR/)
- [SaschaWillems basic ray tracing tutorial (C++)](https://github.com/SaschaWillems/Vulkan/blob/master/examples/raytracingbasic/raytracingbasic.cpp)
- [SaschaWillems vulkan tutorials (C++)](https://github.com/SaschaWillems/Vulkan)
- [Khronos vulkan samples (C++)](https://github.com/KhronosGroup/Vulkan-Samples/tree/main)
- [Ray Tracing in One Weekend - series](https://raytracing.github.io/)
- #### Other projects
  - [hatoo/ash-raytracing-example (Rust)](https://github.com/hatoo/ash-raytracing-example)
  - [adrien-ben/vulkan-examples-rs (Rust)](https://github.com/adrien-ben/vulkan-examples-rs)

### Rendering
- [Ray Tracing Gems II](https://developer.nvidia.com/ray-tracing-gems-ii)
- [pbrt](https://pbrt.org/)
  - [book](https://pbr-book.org/)
- [PBR for materials](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.pdf)
  - page 197 - appendix B: BRDF Implementation
- #### Shaders
  - https://www.gsn-lib.org/docs/nodes/raytracing.php
  - ##### Languages
    - [shader languages comparisons](https://alain.xyz/blog/a-review-of-shader-languages)
    - [slang](https://shader-slang.org/)
  - ##### Compilation
    - https://github.com/google/shaderc-rs

### Acceleration structure
- see [this nvidia blog](https://developer.nvidia.com/blog/best-practices-using-nvidia-rtx-ray-tracing/) for best practices for acceleration structures (and hit shading)

### glTF
- [2.0 reference guide pdf](https://www.khronos.org/files/gltf20-reference-guide.pdf)
- [2.0 spec](https://registry.khronos.org/glTF/specs/2.0/glTF-2.0.pdf)
- [khronos tutorials](https://github.com/KhronosGroup/glTF-Tutorials/tree/main)
- https://www.gltfeditor.com/
- #### Extensions
  - [KHR_lights_punctual](https://github.com/KhronosGroup/glTF/blob/main/extensions/2.0/Khronos/KHR_lights_punctual/README.md)
- #### Models
  - [Khronos sample assets](https://github.com/KhronosGroup/glTF-Sample-Assets/tree/main)
  - [Lantern](https://github.com/KhronosGroup/glTF-Sample-Assets/blob/main/Models/Lantern/README.md)

### Performance
- https://zeux.io/2020/02/27/writing-an-efficient-vulkan-renderer/
  - #### Syncronization
    - https://themaister.net/blog/2019/08/14/yet-another-blog-explaining-vulkan-synchronization/
    - https://xanderbert.github.io/2025/04/13/VulkanMemoryBarriers.html
    - https://cpp-rendering.io/barriers-vulkan-not-difficult/
    - [gpuopne - vulkan barriers explained](https://gpuopen.com/learn/vulkan-barriers-explained/)
    - [khr blog on image layout](https://www.khronos.org/blog/so-long-image-layouts-simplifying-vulkan-synchronisation)
  - #### Memory allocation
    - https://blog.io7m.com/2023/11/11/vulkan-memory-allocation.xhtml
    - https://github.com/Traverse-Research/gpu-allocator
    - https://github.com/gwihlidal/vk-mem-rs
    - https://docs.vulkan.org/guide/latest/memory_allocation.html
  - #### Queues
    - https://gpuopen.com/learn/concurrent-execution-asynchronous-queues/

### Miscelleaneus
- [graphics APIs](https://github.com/Vincent-Therrien/gpu-arena)
- [Semantic Versioning 2.0.0](https://semver.org/)
- #### Coordinate Systems
  - [nalgebra computer-graphics recipes](https://nalgebra.rs/docs/user_guide/cg_recipes)
  - https://learnopengl.com/Getting-started/Coordinate-Systems
- #### Rust
  - https://doc.rust-lang.org/book/
  - https://doc.rust-lang.org/rust-by-example/
- #### Vulkan
  - [docs](https://docs.vulkan.org/guide/latest/index.html)
  - [tutorial](https://docs.vulkan.org/tutorial/latest/00_Introduction.html)
  - [unofficaila tutorial](https://vulkan-tutorial.com/)
  - [paminerva tutorial](https://paminerva.github.io/docs/LearnVulkan/LearnVulkan)
