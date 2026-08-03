//! egui integration, Option C from `docs/bevy_integration.md` §6: raw `egui` +
//! our own thin input layer built from `bevy_input`, our own (future) ash paint
//! backend. We do **not** use `bevy_egui` (its paint node needs the wgpu
//! `RenderDevice` we removed) nor `egui-ash-renderer` (descriptor-set + GLSL
//! raster pipeline — forbidden by this project's heap+Slang-only rule, and it
//! pulls a crates.io `ash` that clashes with our git `ash`).
//!
//! Data flow:
//! ```text
//! main world:  egui_begin (bevy_input -> RawInput, ctx.begin_pass)
//!              <user UI systems: Res<EguiContext>, egui::Window::show(...)>
//!              egui_end   (ctx.end_pass -> tessellate -> EguiFrameOutput)
//!   extract:   extract_egui (-> ExtractedEgui in the render world)
//! render world: paint_egui  (consumes ExtractedEgui — GPU paint TODO, see §6)
//! ```

use bevy_ecs::prelude::*;
use bevy_input::ButtonInput;
use bevy_input::keyboard::{Key, KeyCode, KeyboardInput};
use bevy_input::mouse::{MouseButton, MouseScrollUnit, MouseWheel};
use bevy_render::Extract;
use bevy_time::Time;
use bevy_window::{CursorMoved, PrimaryWindow, Window};

/// Holds the `egui::Context`. `egui::Context` is `Clone + Send + Sync` (it's an
/// `Arc` internally), so this is a normal resource. Add UI from your own
/// `Update` systems: `egui::Window::new("x").show(egui_ctx.ctx(), |ui| { .. })`.
#[derive(Resource)]
pub struct EguiContext {
    ctx: egui::Context,
}

impl Default for EguiContext {
    fn default() -> Self {
        Self {
            ctx: egui::Context::default(),
        }
    }
}

impl EguiContext {
    pub fn ctx(&self) -> &egui::Context {
        &self.ctx
    }
}

/// Cross-frame input state (egui needs the last pointer position for button
/// events that arrive without a fresh move).
#[derive(Resource, Default)]
struct EguiInputState {
    pointer_pos: egui::Pos2,
}

/// Main-world tessellated output for the current frame.
#[derive(Resource, Default)]
pub struct EguiFrameOutput {
    pub primitives: Vec<egui::ClippedPrimitive>,
    pub textures_delta: egui::TexturesDelta,
    pub pixels_per_point: f32,
}

/// Render-world copy of [`EguiFrameOutput`] (what the paint backend consumes).
#[derive(Resource, Default)]
pub struct ExtractedEgui {
    pub primitives: Vec<egui::ClippedPrimitive>,
    pub textures_delta: egui::TexturesDelta,
    pub pixels_per_point: f32,
}

fn build_modifiers(keys: &ButtonInput<KeyCode>) -> egui::Modifiers {
    let ctrl = keys.any_pressed([KeyCode::ControlLeft, KeyCode::ControlRight]);
    let shift = keys.any_pressed([KeyCode::ShiftLeft, KeyCode::ShiftRight]);
    let alt = keys.any_pressed([KeyCode::AltLeft, KeyCode::AltRight]);
    let mac_cmd = cfg!(target_os = "macos") && keys.any_pressed([KeyCode::SuperLeft, KeyCode::SuperRight]);
    egui::Modifiers {
        alt,
        ctrl,
        shift,
        mac_cmd,
        // egui's logical "command" is Cmd on macOS, Ctrl elsewhere.
        command: if cfg!(target_os = "macos") { mac_cmd } else { ctrl },
    }
}

fn map_named_key(key: &Key) -> Option<egui::Key> {
    Some(match key {
        Key::Enter => egui::Key::Enter,
        Key::Tab => egui::Key::Tab,
        Key::Space => egui::Key::Space,
        Key::ArrowDown => egui::Key::ArrowDown,
        Key::ArrowLeft => egui::Key::ArrowLeft,
        Key::ArrowRight => egui::Key::ArrowRight,
        Key::ArrowUp => egui::Key::ArrowUp,
        Key::End => egui::Key::End,
        Key::Home => egui::Key::Home,
        Key::Backspace => egui::Key::Backspace,
        Key::Delete => egui::Key::Delete,
        Key::Escape => egui::Key::Escape,
        _ => return None,
    })
}

/// Build `RawInput` from `bevy_input` and open the egui pass. Runs before user
/// UI systems. All coordinates are in egui *points* (Bevy's logical pixels);
/// `pixels_per_point` is set so tessellation can scale to physical pixels.
fn egui_begin(
    egui_ctx: Res<EguiContext>,
    mut input_state: ResMut<EguiInputState>,
    time: Res<Time>,
    windows: Query<&Window, With<PrimaryWindow>>,
    mut cursor_moved: MessageReader<CursorMoved>,
    mouse_buttons: Res<ButtonInput<MouseButton>>,
    mut wheel: MessageReader<MouseWheel>,
    keys: Res<ButtonInput<KeyCode>>,
    mut keyboard: MessageReader<KeyboardInput>,
) {
    let Ok(window) = windows.single() else {
        return;
    };
    let ppp = window.resolution.scale_factor();
    // Bevy's Window::width()/height() and cursor positions are already logical
    // (points), so no division by ppp here.
    let screen = egui::vec2(window.width(), window.height());
    let modifiers = build_modifiers(&keys);

    let mut events = Vec::new();

    for ev in cursor_moved.read() {
        input_state.pointer_pos = egui::pos2(ev.position.x, ev.position.y);
        events.push(egui::Event::PointerMoved(input_state.pointer_pos));
    }

    for (btn, egui_btn) in [
        (MouseButton::Left, egui::PointerButton::Primary),
        (MouseButton::Right, egui::PointerButton::Secondary),
        (MouseButton::Middle, egui::PointerButton::Middle),
    ] {
        if mouse_buttons.just_pressed(btn) || mouse_buttons.just_released(btn) {
            events.push(egui::Event::PointerButton {
                pos: input_state.pointer_pos,
                button: egui_btn,
                pressed: mouse_buttons.just_pressed(btn),
                modifiers,
            });
        }
    }

    for ev in wheel.read() {
        let (dx, dy) = match ev.unit {
            MouseScrollUnit::Line => (ev.x * 50.0, ev.y * 50.0),
            MouseScrollUnit::Pixel => (ev.x, ev.y),
        };
        events.push(egui::Event::MouseWheel {
            unit: egui::MouseWheelUnit::Point,
            delta: egui::vec2(dx, dy),
            phase: egui::TouchPhase::Move,
            modifiers,
        });
    }

    for ev in keyboard.read() {
        let pressed = ev.state.is_pressed();
        if let Some(key) = map_named_key(&ev.logical_key) {
            events.push(egui::Event::Key {
                key,
                physical_key: None,
                pressed,
                repeat: ev.repeat,
                modifiers,
            });
        }
        // Text input: emit the produced text on press, skipping control chars.
        if pressed {
            if let Some(text) = &ev.text {
                if !text.is_empty() && text.chars().all(|c| !c.is_control()) {
                    events.push(egui::Event::Text(text.to_string()));
                }
            }
        }
    }

    let raw_input = egui::RawInput {
        screen_rect: Some(egui::Rect::from_min_size(egui::Pos2::ZERO, screen)),
        time: Some(time.elapsed_secs_f64()),
        modifiers,
        events,
        ..Default::default()
    };

    egui_ctx.ctx.set_pixels_per_point(ppp);
    egui_ctx.ctx.begin_pass(raw_input);
}

/// Close the egui pass and tessellate. Runs after user UI systems.
fn egui_end(egui_ctx: Res<EguiContext>, mut out: ResMut<EguiFrameOutput>) {
    let ppp = egui_ctx.ctx.pixels_per_point();
    let full = egui_ctx.ctx.end_pass();
    // TODO §6: apply `full.platform_output` (cursor icon -> Window::cursor,
    // clipboard, opened URLs) back in the main world.
    out.primitives = egui_ctx.ctx.tessellate(full.shapes, ppp);
    out.textures_delta = full.textures_delta;
    out.pixels_per_point = ppp;
}

/// ExtractSchedule: move the tessellated frame into the render world.
fn extract_egui(mut dst: ResMut<ExtractedEgui>, src: Extract<Res<EguiFrameOutput>>) {
    dst.primitives = src.primitives.clone();
    dst.textures_delta = src.textures_delta.clone();
    dst.pixels_per_point = src.pixels_per_point;
}

/// Register main-world egui resources + the begin/end systems on the given app,
/// and the extract system on the render app. Called by
/// [`super::plugin::SunrayEguiPlugin`].
///
/// The GPU paint runs inside `SunrayRenderPlugin`'s `render_frame` (it needs the
/// per-frame swapchain image), consuming [`ExtractedEgui`] via
/// [`super::egui_paint::EguiPaint`].
pub(crate) fn register(app: &mut bevy_app::App) {
    use bevy_app::{PostUpdate, PreUpdate};
    use bevy_render::{ExtractSchedule, RenderApp};

    app.init_resource::<EguiContext>()
        .init_resource::<EguiInputState>()
        .init_resource::<EguiFrameOutput>();

    // Open the pass before user UI (PreUpdate), close + tessellate after it
    // (PostUpdate). User UI runs in Update in between.
    app.add_systems(PreUpdate, egui_begin);
    app.add_systems(PostUpdate, egui_end);

    let render_app = app.sub_app_mut(RenderApp);
    render_app.init_resource::<ExtractedEgui>();
    render_app.add_systems(ExtractSchedule, extract_egui);
}
