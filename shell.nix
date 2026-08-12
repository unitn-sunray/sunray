{
  pkgs ? import <nixpkgs> { },

  # Mesa/RADV providing the Vulkan ICD used at runtime, kept separate from `pkgs`
  # so the rest of the shell stays on the system channel. RADV exposes
  # VK_EXT_descriptor_heap by default from Mesa 26.2.0; 26.1.x has it behind
  # RADV_EXPERIMENTAL=heap (set automatically below). The ICD is self-contained
  # (own LLVM/libdrm in its closure), so it composes with the stable loader.
  # Override with e.g. `nix-shell --arg mesaPkgs 'import <nixos-unstable> { }'`.
  mesaPkgs ?
    import
      (builtins.fetchTarball {
        # nixos-unstable, 2026-08-09, mesa 26.2.0
        url = "https://github.com/NixOS/nixpkgs/archive/f13ff45afd1bb73e640eaa08a7066dbed07e3238.tar.gz";
        sha256 = "1bb21a1vjcmzs5fgrwjbx3x3b2n6ydww3gixk1czz8qyk9p98d6c";
      })
      {
        inherit (pkgs) config;
        inherit (pkgs.stdenv.hostPlatform) system;
      },
}:

let
  inherit (pkgs) lib;

  mesa = mesaPkgs.mesa;
  radvIcd = "${mesa}/share/vulkan/icd.d/radeon_icd.x86_64.json";

  x11Libs = with pkgs; [
    libX11
    libXcursor
    libXrandr
    libXi
  ];
  waylandLibs = with pkgs; [
    wayland
    libxkbcommon
  ];
  vulkanLibs = with pkgs; [
    vulkan-loader
    vulkan-headers
    vulkan-validation-layers
    vulkan-caps-viewer
    vulkan-tools # vulkaninfo, for checking which extensions the ICD exposes
  ];

  bpy-libs = with pkgs; [
    stdenv.cc.cc.lib # Standard C++ library
    zlib
    libGL
    libSM
    libICE
    libX11
    libXi
    libXxf86vm
    libXfixes
    libXrender
    wayland
    libxkbcommon
  ];
in
pkgs.mkShell (
  {
    nativeBuildInputs = with pkgs; [
      pkg-config
    ];

    buildInputs =
      with pkgs;
      [
        cmake
        clang
        llvmPackages.libclang
        glslang
        shaderc
        shader-slang
      ]
      ++ x11Libs
      ++ waylandLibs
      ++ vulkanLibs;

    # 1. Force shaderc-sys to use the pre-compiled Nix library (from previous step)
    SHADERC_LIB_DIR = "${pkgs.shaderc.lib}/lib";

    # 2. Tell the linker and runtime exactly where to find Vulkan and Windowing libraries
    #    Deliberately *not* including `mesa` here: only the ICD is overridden, the
    #    rest of the shell keeps the system GL/GBM libraries.
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath (
      bpy-libs ++ x11Libs ++ waylandLibs ++ vulkanLibs ++ [ pkgs.shader-slang ]
    );

    LIBCLANG_PATH = "${pkgs.llvmPackages.libclang.lib}/lib";

    # 3. Use the nixpkgs Slang compiler instead of the one bundled with the Vulkan SDK.
    #    Headers live in the `dev` output, the shared libs in the default output.
    SLANG_INCLUDE_DIR = "${pkgs.shader-slang.dev}/include";
    SLANG_LIB_DIR = "${pkgs.shader-slang}/lib";

    VULKAN_SDK = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";

    VK_LAYER_PATH = "${pkgs.vulkan-validation-layers}/share/vulkan/explicit_layer.d";

    # 4. Point the loader at the RADV ICD from `mesaPkgs` instead of the system one.
    #    `unset VK_DRIVER_FILES` inside the shell to fall back to /run/opengl-driver.
    VK_DRIVER_FILES = radvIcd;

    shellHook = ''
      echo "sunray: RADV ICD -> mesa ${mesa.version}"
    '';
  }
  # Mesa 26.1.x shipped VK_EXT_descriptor_heap gated behind this flag; it became
  # the default in 26.2.0, where setting it is unnecessary.
  // lib.optionalAttrs (lib.versionOlder mesa.version "26.2.0") {
    RADV_EXPERIMENTAL = "heap";
  }
)
