use ash::vk;
use image::{ExtendedColorType, ImageFormat};
use sunray::{Camera, Renderer, error::SrResult};

use nalgebra as na;

fn get_vk_format(image_format: ExtendedColorType) -> vk::Format {
    match image_format {
        ExtendedColorType::Rgba8 => vk::Format::R8G8B8A8_UNORM,
        _ => vk::Format::UNDEFINED,
    }
}

fn init_logging() {
    log4rs::config::init_file("examples/log4rs.yaml", log4rs::config::Deserializers::new()).unwrap();

    if cfg!(debug_assertions) {
        //stdlib unfortunately completely pollutes trace log level, TODO somehow config stdlib/log to fix this?
        log::set_max_level(log::LevelFilter::Debug);
    } else {
        log::set_max_level(log::LevelFilter::Warn);
    }
}

fn render_to_file(image_buf: &[u8], image_extent: (u32, u32), path: &str, format: ImageFormat) {
    let result = image::save_buffer_with_format(
        path,
        image_buf,
        image_extent.0,
        image_extent.1,
        ExtendedColorType::Rgba8,
        format,
    );

    match result {
        Ok(_) => println!("You can find your render here: {}", path),
        // Panics rather than logs: CI gates on this example's exit code, and a PNG
        // that failed to save must not report success.
        Err(e) => panic!("while saving '{path}': {e:?}"),
    }
}

fn render_and_save() -> SrResult<()> {
    let path = "examples/png/render.png";
    let image_extent = (1600, 1200);
    let image_format = get_vk_format(ExtendedColorType::Rgba8);
    let mut renderer: Renderer = Renderer::new(image_extent, image_format)?;

    // The instance list is owned by the caller and handed to the renderer per frame.
    let (_scene_group, scene_instances) = renderer.load_gltf("examples/assets/ReflectionRoom.glb")?;

    let camera = Camera::default()
        .set_position(na::Point3::new(13.0, 30.0, 25.0))
        .set_target(na::Point3::new(0.0, 13.0, 0.0))
        .set_fov_y(45.0);

    let image_buf = renderer.render_to_host_memory(&camera, &scene_instances)?;
    render_to_file(&image_buf, image_extent, path, ImageFormat::Png);

    Ok(())
}

// Returns the error so a failed render exits nonzero — CI gates on it.
fn main() -> SrResult<()> {
    init_logging();

    render_and_save().inspect_err(|e| log::error!("Sunray error: {e}"))
}
