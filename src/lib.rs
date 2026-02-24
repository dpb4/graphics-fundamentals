use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use cgmath::{One, Rotation3, SquareMatrix};
use wgpu::util::DeviceExt;
use winit::{
    application::ApplicationHandler,
    event::*,
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::{KeyCode, PhysicalKey},
    window::Window,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;

use crate::model::{DrawModel, Vertex};

mod camera;
mod model;
mod obj_parse;
mod resources;
mod texture;
mod timing;

const ENABLE_DEBUG_TBN: bool = true;

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    position: [f32; 4],
    view_projection_matrix: [[f32; 4]; 4],
}

impl CameraUniform {
    fn new() -> Self {
        Self {
            position: [0.0; 4],
            view_projection_matrix: cgmath::Matrix4::identity().into(),
        }
    }

    fn update_view_proj(&mut self, camera: &camera::Camera, projection: &camera::Projection) {
        self.position = camera.position.to_homogeneous().into();
        self.view_projection_matrix =
            (projection.perspective_matrix() * camera.view_matrix()).into()
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
// padding fields are necessary because uniforms require 16 byte alignment
struct LightUniform {
    position: [f32; 3],
    _padding1: u32,
    ambient_color: [f32; 3],
    _padding2: u32,
    diffuse_color: [f32; 3],
    _padding3: u32,
    specular_color: [f32; 3],
    _padding4: u32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
struct TimestampUniform {
    time: u32,
}

// TODO:
// X clean up model loading
// X clean up debug pipelines
// X rewrite material loading and remove tobj dependence
// - generally just reconsider the mesh/model organization
// - add multiple lights
// - add proper material batching
// - add shadows
// - improve lighting

struct Pipelines {
    render: wgpu::RenderPipeline, // object which describes the various rendering phases to use
    render_alt: wgpu::RenderPipeline, // object which describes the various rendering phases to use
    light_debug: wgpu::RenderPipeline,
    geometry_debug: wgpu::RenderPipeline,
}

struct Uniforms {
    camera: CameraUniform,
    camera_buffer: wgpu::Buffer,

    light: LightUniform,
    light_buffer: wgpu::Buffer,

    timestamp: TimestampUniform,
    timestamp_buffer: wgpu::Buffer,

    model_transform_buffer: wgpu::Buffer,
}

struct Layouts {
    per_frame: wgpu::BindGroupLayout,
    per_pass: wgpu::BindGroupLayout,
    per_object: wgpu::BindGroupLayout,
}

struct Variables {
    is_mouse_pressed: bool,
    enable_geometry_debug: bool,
    swap_pipelines: bool,
    enable_light_rotation: bool,
}

struct Diagnostics {
    start_time: std::time::Instant,
    frame_count: u64,
    frame_time_avg: timing::RollingAverage,
    render_time_avg: timing::RollingAverage,
    update_time_avg: timing::RollingAverage,
}

pub struct State {
    window: Arc<Window>,                        // the actual window object
    device: wgpu::Device, // the 'gpu' which is being used (may not necessarily be a dedicated gpu)
    queue: wgpu::Queue,   // the command queue to send things to the device
    surface: wgpu::Surface<'static>, // the target of the rendering
    surface_config: wgpu::SurfaceConfiguration, // configuring the surface (size, colour format, etc)
    is_surface_configured: bool,

    camera: camera::Camera,
    projection: camera::Projection,
    model: model::Model,
    materials: Vec<model::Material>,
    material_map: HashMap<String, usize>,

    depth_texture: texture::Texture,
    debug_tbn_extras: Option<DebugTBNStateExtras>,
    debug_light_model: model::Model,

    camera_controller: camera::CameraController,

    layouts: Layouts,

    per_frame_bind_group: wgpu::BindGroup, // uniforms like camera, lights, etc
    per_object_bind_group: wgpu::BindGroup, // local things like model position or rotation, etc

    pipelines: Pipelines,
    uniforms: Uniforms,
    diagnostics: Diagnostics,
    variables: Variables,
}

struct DebugTBNStateExtras {
    tangent_bind_group: wgpu::BindGroup,
    bitangent_bind_group: wgpu::BindGroup,
    normal_bind_group: wgpu::BindGroup,
    debug_tbn_render_pipeline: wgpu::RenderPipeline,
    debug_tbn_uniforms: [Vec<model::VectorDebugUniform>; 3],
    debug_tangent_buffer: wgpu::Buffer,
    debug_bitangent_buffer: wgpu::Buffer,
    debug_normal_buffer: wgpu::Buffer,
    debug_vector_model: model::Model,
}

impl State {
    pub async fn new(window: Arc<Window>) -> anyhow::Result<Self> {
        let size = window.inner_size();

        // ---- DEVICE/SURFACE CONFIG ----

        // an 'instance' is a handle to the gpu which can get the device (adapter) or create surfaces
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::GL,
            ..Default::default()
        });

        let surface = instance.create_surface(window.clone()).unwrap();

        // adapter is the handle to the gpu
        // notably this basically returns a random adapter (eg could be either DX or Vulkan)
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface), // ensures that whatever adapter we get is compatible with our window's surface
                force_fallback_adapter: false, // this forces wgpu to pick an adapter that will work on ALL hardware; usually means that the rendering backend will be software instead of hardware
            })
            .await?;

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("main_device"),
                required_features: wgpu::Features::POLYGON_MODE_LINE, // allows use of specific extensions (eg float 64 support)
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                required_limits: if cfg!(target_arch = "wasm32") {
                    // sets resource limits for compatibility with different devices
                    wgpu::Limits::downlevel_webgl2_defaults()
                } else {
                    wgpu::Limits::default()
                },
                memory_hints: Default::default(), // you can prioritize performance, memory usage, or use some kind of custom allocater
                trace: wgpu::Trace::Off,          // TODO should probably turn this on
            })
            .await?;

        let surface_capabilities = surface.get_capabilities(&adapter);

        // find a usable srgb format, otherwise just fall back to the first format
        let surface_format = surface_capabilities
            .formats
            .iter()
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_capabilities.formats[0]);

        // configure the surface. this is also used later to get width/height of the screen
        let surface_config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_capabilities.present_modes[0], // this essentially controls vsync
            alpha_mode: surface_capabilities.alpha_modes[0],
            desired_maximum_frame_latency: 2,
            view_formats: vec![],
        };

        let camera_controller = camera::CameraController::new(10.0, 1.3);

        let (camera, projection, camera_uniform, camera_buffer) =
            Self::create_camera(&device, &surface_config);

        // ---- HIGH LEVEL RENDER CONFIG ----

        let light_uniform = LightUniform {
            position: [15.0, 15.0, 15.0],
            _padding1: 0,
            ambient_color: [0.01, 0.01, 0.01],
            _padding2: 0,
            diffuse_color: [0.5, 0.5, 0.5],
            _padding3: 0,
            specular_color: [1.0, 1.0, 1.0],
            _padding4: 0,
        };

        let timestamp_uniform = TimestampUniform { time: 0 };

        let depth_texture =
            texture::Texture::create_depth_texture(&device, &surface_config, "depth texture");

        // ---- BIND GROUP LAYOUTS ----

        // a BindGroup describes a set of resources and how they can be accessed by the shader(s)

        let (per_frame_bind_group_layout, per_pass_bind_group_layout, per_object_bind_group_layout) =
            Self::create_bind_group_layouts(&device);

        // ---- BUFFERS ----

        let light_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("light buffer"),
            contents: bytemuck::cast_slice(&[light_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let timestamp_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("timestamp buffer"),
            contents: bytemuck::cast_slice(&[timestamp_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let model_transform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("model transform buffer"),
            contents: bytemuck::cast_slice(&[model::ModelTransformationUniform::identity()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // ---- BIND GROUPS ----

        // bind group layouts can be be reused with various different bind groups to allow swapping the data on the fly
        let per_frame_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout: &per_frame_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: camera_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: light_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: timestamp_buffer.as_entire_binding(),
                },
            ],
            label: Some("camera_bind_group"),
        });

        // the per pass bind group is created by materials

        let per_object_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("per object bind group"),
            layout: &per_object_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: model_transform_buffer.as_entire_binding(),
            }],
        });

        // ---- MODEL LOADING ----

        let mut materials = Vec::new();
        let mut material_map = HashMap::new();

        resources::load_all_materials(
            "src/assets/materials/all_materials.mtl",
            &mut materials,
            &mut material_map,
            &device,
            &queue,
            &per_pass_bind_group_layout,
        );

        let model = resources::load_obj_model(
            "src/assets/models/sball3.obj",
            &mut materials,
            &mut material_map,
            &device,
            &queue,
            &per_pass_bind_group_layout,
        )
        .unwrap();
        // model.scale = 16.0;

        let debug_light_model = resources::load_obj_model(
            "src/assets/models/octahedron.obj",
            &mut materials,
            &mut material_map,
            &device,
            &queue,
            &per_pass_bind_group_layout,
        )
        .unwrap();

        // ---- RENDER PIPELINES ----

        let render_pipeline = {
            let render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("render pipeline layout"),
                    bind_group_layouts: &[
                        &per_frame_bind_group_layout,
                        &per_pass_bind_group_layout,
                        &per_object_bind_group_layout,
                    ],
                    immediate_size: 0,
                });

            let shader_descriptor = wgpu::include_wgsl!("shaders/shader.wgsl");

            Self::create_render_pipeline(
                &device,
                &render_pipeline_layout,
                surface_config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc()],
                shader_descriptor,
                wgpu::PolygonMode::Fill,
            )
        };

        let render_pipeline_alt = {
            let render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("render pipeline layout"),
                    bind_group_layouts: &[
                        &per_frame_bind_group_layout,
                        &per_pass_bind_group_layout,
                        &per_object_bind_group_layout,
                    ],
                    immediate_size: 0,
                });

            let shader_descriptor = wgpu::include_wgsl!("shaders/shader2.wgsl");

            Self::create_render_pipeline(
                &device,
                &render_pipeline_layout,
                surface_config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc()],
                shader_descriptor,
                wgpu::PolygonMode::Fill,
            )
        };

        let debug_light_render_pipeline = {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("debug light pipeline layout"),
                bind_group_layouts: &[&per_frame_bind_group_layout],
                immediate_size: 0,
            });
            let shader_descriptor = wgpu::include_wgsl!("shaders/debug_light.wgsl");

            Self::create_render_pipeline(
                &device,
                &layout,
                surface_config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc()],
                shader_descriptor,
                wgpu::PolygonMode::Fill,
            )
        };

        let debug_polygon_render_pipeline = {
            let render_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("debug polygon layout"),
                    bind_group_layouts: &[
                        &per_frame_bind_group_layout,
                        &per_pass_bind_group_layout,
                        &per_object_bind_group_layout,
                    ],
                    immediate_size: 0,
                });

            let shader_descriptor = wgpu::include_wgsl!("shaders/black.wgsl");

            Self::create_render_pipeline(
                &device,
                &render_pipeline_layout,
                surface_config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc()],
                shader_descriptor,
                wgpu::PolygonMode::Line,
            )
        };

        let mut state = Self {
            window,
            device,
            queue,
            surface,
            surface_config,
            is_surface_configured: true,
            pipelines: Pipelines {
                render: render_pipeline,
                render_alt: render_pipeline_alt,
                light_debug: debug_light_render_pipeline,
                geometry_debug: debug_polygon_render_pipeline,
            },
            camera,
            projection,
            model,
            debug_light_model,
            layouts: Layouts {
                per_frame: per_frame_bind_group_layout,
                per_pass: per_pass_bind_group_layout,
                per_object: per_object_bind_group_layout,
            },
            per_frame_bind_group,
            per_object_bind_group,
            camera_controller,
            uniforms: Uniforms {
                camera: camera_uniform,
                camera_buffer,
                light: light_uniform,
                light_buffer,
                timestamp: timestamp_uniform,
                timestamp_buffer,
                model_transform_buffer,
            },
            depth_texture,
            diagnostics: Diagnostics {
                start_time: std::time::Instant::now(),
                frame_count: 0,
                frame_time_avg: timing::RollingAverage::new(200),
                render_time_avg: timing::RollingAverage::new(200),
                update_time_avg: timing::RollingAverage::new(200),
            },
            variables: Variables {
                is_mouse_pressed: false,
                enable_geometry_debug: false,
                swap_pipelines: false,
                enable_light_rotation: false,
            },
            debug_tbn_extras: None,
            materials: materials,
            material_map: material_map,
        };

        if ENABLE_DEBUG_TBN {
            state.debug_tbn_extras = Some(Self::create_debug_extras(&mut state));
        }

        Ok(state)
    }

    fn create_camera(
        device: &wgpu::Device,
        surface_config: &wgpu::SurfaceConfiguration,
    ) -> (
        camera::Camera,
        camera::Projection,
        CameraUniform,
        wgpu::Buffer,
    ) {
        let camera = camera::Camera::new([0.0, 0.0, 10.0], cgmath::Deg(-90.0), cgmath::Deg(0.0));
        let projection = camera::Projection::new(
            surface_config.width,
            surface_config.height,
            80.0,
            0.1,
            100.0,
        );

        let mut camera_uniform = CameraUniform::new();
        camera_uniform.update_view_proj(&camera, &projection);

        let camera_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("camera buffer"),
            contents: bytemuck::cast_slice(&[camera_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        (camera, projection, camera_uniform, camera_buffer)
    }

    fn create_bind_group_layouts(
        device: &wgpu::Device,
    ) -> (
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
        wgpu::BindGroupLayout,
    ) {
        let per_frame = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                // camera uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // light uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                // timestamp uniform
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("per frame bind group layout"),
        });

        let per_pass = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            entries: &[
                // the diffuse texture data binding layout
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // the sampler binding layout
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // the normal texture data binding layout
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        multisampled: false,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                    },
                    count: None,
                },
                // the sampler binding layout
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                // the material info
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
            label: Some("per pass bind group layout"),
        });

        let per_object = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("per object bind group layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        (per_frame, per_pass, per_object)
    }

    fn create_debug_extras(state: &mut Self) -> DebugTBNStateExtras {
        let per_object_debug_bind_group_layout =
            state
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("debug TBN per object bind group layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::VERTEX,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let debug_tbn_uniforms = model::VectorDebugUniform::from_mesh_tbn(&state.model.meshes[0]);

        println!("t count: {}", debug_tbn_uniforms[0].len());
        println!("b count: {}", debug_tbn_uniforms[1].len());
        println!("n count: {}", debug_tbn_uniforms[2].len());

        println!("vertex count: {}", state.model.meshes[0].verts.len());

        let debug_tangent_buffer =
            state
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("debug TBN buffer"),
                    contents: bytemuck::cast_slice(&debug_tbn_uniforms[0][..]),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        let debug_bitangent_buffer =
            state
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("debug TBN buffer"),
                    contents: bytemuck::cast_slice(&debug_tbn_uniforms[1][..]),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        let debug_normal_buffer =
            state
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("debug TBN buffer"),
                    contents: bytemuck::cast_slice(&debug_tbn_uniforms[2][..]),
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                });

        let tangent_bind_group = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("debug tbn tangent bind group"),
            layout: &per_object_debug_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state.uniforms.model_transform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: debug_tangent_buffer.as_entire_binding(),
                },
            ],
        });

        let bitangent_bind_group = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("debug tbn bitangent bind group"),
            layout: &per_object_debug_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state.uniforms.model_transform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: debug_bitangent_buffer.as_entire_binding(),
                },
            ],
        });

        let normal_bind_group = state.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("debug tbn normal bind group"),
            layout: &per_object_debug_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: state.uniforms.model_transform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: debug_normal_buffer.as_entire_binding(),
                },
            ],
        });

        let debug_vector_model = resources::load_obj_model(
            "src/assets/models/arrow.obj",
            &mut state.materials,
            &mut state.material_map,
            &state.device,
            &state.queue,
            &state.layouts.per_pass,
        )
        .unwrap();

        let debug_tbn_render_pipeline = {
            let render_pipeline_layout =
                state
                    .device
                    .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                        label: Some("debug TBN layout"),
                        bind_group_layouts: &[
                            &state.layouts.per_frame,
                            &state.layouts.per_pass,
                            &per_object_debug_bind_group_layout,
                        ],
                        immediate_size: 0,
                    });

            let shader_descriptor = wgpu::include_wgsl!("shaders/debug_vector.wgsl");

            Self::create_render_pipeline(
                &state.device,
                &render_pipeline_layout,
                state.surface_config.format,
                Some(texture::Texture::DEPTH_FORMAT),
                &[model::ModelVertex::desc()],
                shader_descriptor,
                wgpu::PolygonMode::Line,
            )
        };

        DebugTBNStateExtras {
            tangent_bind_group,
            bitangent_bind_group,
            normal_bind_group,
            debug_tbn_render_pipeline,
            debug_tbn_uniforms,
            debug_tangent_buffer,
            debug_bitangent_buffer,
            debug_normal_buffer,
            debug_vector_model,
        }
    }

    pub fn update(&mut self, dt: Duration) {
        self.camera_controller.update_camera(&mut self.camera, dt);
        self.uniforms
            .camera
            .update_view_proj(&self.camera, &self.projection);
        self.queue.write_buffer(
            &self.uniforms.camera_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms.camera]),
        );

        if self.variables.enable_light_rotation {
            self.uniforms.light.position = (cgmath::Quaternion::from_angle_z(cgmath::Deg(0.1))
                * cgmath::Vector3::from(self.uniforms.light.position))
            .into();
        }
        self.queue.write_buffer(
            &self.uniforms.light_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms.light]),
        );

        self.uniforms.timestamp.time = self.diagnostics.start_time.elapsed().as_millis() as u32;
        self.queue.write_buffer(
            &self.uniforms.timestamp_buffer,
            0,
            bytemuck::cast_slice(&[self.uniforms.timestamp]),
        );
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.surface_config.width = width;
            self.surface_config.height = height;

            self.surface.configure(&self.device, &self.surface_config);
            self.is_surface_configured = true;

            self.depth_texture = texture::Texture::create_depth_texture(
                &self.device,
                &self.surface_config,
                "depth texture",
            );

            self.projection.resize(width, height);
        } else {
            log::warn!["resize was called with width 0 or height 0"]
        }
    }

    pub fn render(&mut self) -> Result<(), wgpu::SurfaceError> {
        self.window.request_redraw();

        if !self.is_surface_configured {
            log::warn!("render called while surface is not configured");
            return Ok(());
        }

        // wait for the surface to provide a new texture to which to render
        let target_surface = self.surface.get_current_texture()?;

        // TextureView controls how the rendering code interacts with the texture
        let target_view = target_surface
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());

        // create a command encoder to send commands to the gpu
        let mut command_encoder =
            self.device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("render command encoder"),
                });

        // encode the rendering pass:
        {
            let mut render_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("render pass"),
                color_attachments: &[
                    // location[0] refers to this color attachment
                    Some(wgpu::RenderPassColorAttachment {
                        view: &target_view,
                        resolve_target: None,
                        depth_slice: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(wgpu::Color {
                                r: 0.1,
                                g: 0.2,
                                b: 0.3,
                                a: 1.0,
                            }),
                            store: wgpu::StoreOp::Store,
                        },
                    }),
                ],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.depth_texture.view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                occlusion_query_set: None,
                timestamp_writes: None,
                multiview_mask: None,
            });

            if self.variables.swap_pipelines {
                render_pass.set_pipeline(&self.pipelines.render_alt);
            } else {
                render_pass.set_pipeline(&self.pipelines.render);
            }

            self.queue.write_buffer(
                &self.uniforms.model_transform_buffer,
                0,
                bytemuck::cast_slice(&[model::ModelTransformationUniform::from_model(&self.model)]),
            );

            render_pass.set_bind_group(0, &self.per_frame_bind_group, &[]);
            // render_pass.set_bind_group(1, &self.per_pass_bind_group, &[]);
            // render_pass.set_bind_group(2, &self.per_object_bind_group, &[]);

            render_pass.draw_model(&self.model, &self.materials, &self.per_object_bind_group);

            render_pass.set_pipeline(&self.pipelines.light_debug);

            // render_pass.set_bind_group(0, &self.per_frame_bind_group, &[]);
            // render_pass.set_bind_group(1, &self.per_pass_bind_group, &[]);
            // render_pass.set_bind_group(2, &self.per_object_bind_group, &[]);

            render_pass.draw_model(
                &self.debug_light_model,
                &self.materials,
                &self.per_frame_bind_group,
            );

            if self.variables.enable_geometry_debug {
                if let Some(debug_extras) = &self.debug_tbn_extras {
                    render_pass.set_pipeline(&self.pipelines.geometry_debug);
                    render_pass.draw_model(
                        &self.model,
                        &self.materials,
                        &self.per_object_bind_group,
                    );

                    render_pass.set_pipeline(&debug_extras.debug_tbn_render_pipeline);
                    render_pass.draw_mesh_instanced(
                        &debug_extras.debug_vector_model.meshes[0],
                        &self.materials[*self.material_map.get("blue").unwrap_or(&0)],
                        0..(debug_extras.debug_tbn_uniforms[0].len() as u32),
                        &debug_extras.tangent_bind_group,
                    );
                    render_pass.draw_mesh_instanced(
                        &debug_extras.debug_vector_model.meshes[0],
                        &self.materials[*self.material_map.get("green").unwrap_or(&0)],
                        0..(debug_extras.debug_tbn_uniforms[1].len() as u32),
                        &debug_extras.bitangent_bind_group,
                    );
                    render_pass.draw_mesh_instanced(
                        &debug_extras.debug_vector_model.meshes[0],
                        &self.materials[*self.material_map.get("red").unwrap_or(&0)],
                        0..(debug_extras.debug_tbn_uniforms[2].len() as u32),
                        &debug_extras.normal_bind_group,
                    );
                }
            }
        }

        // close the command encoder and submit the instructions to the gpu's render queue
        self.queue.submit(std::iter::once(command_encoder.finish()));

        self.diagnostics.frame_count += 1;

        // put the output from the rendering onto the window
        target_surface.present();
        Ok(())
    }

    pub fn handle_key(&mut self, event_loop: &ActiveEventLoop, code: KeyCode, is_pressed: bool) {
        match (code, is_pressed) {
            (KeyCode::Escape, true) => event_loop.exit(),
            (KeyCode::KeyG, true) => {
                self.variables.enable_geometry_debug = !self.variables.enable_geometry_debug
            }
            (KeyCode::KeyC, true) => {
                self.variables.swap_pipelines = !self.variables.swap_pipelines;
            }
            (KeyCode::KeyL, true) => {
                self.variables.enable_light_rotation = !self.variables.enable_light_rotation
            }
            (KeyCode::KeyR, true) => {
                self.model.rotation = cgmath::Quaternion::from_axis_angle(
                    cgmath::Vector3::unit_y(),
                    cgmath::Deg(self.diagnostics.frame_count as f32 * 0.1),
                )
            }
            _ => {
                self.camera_controller.handle_key(code, is_pressed);
            }
        }
    }

    fn handle_mouse_button(&mut self, button: MouseButton, pressed: bool) {
        match button {
            MouseButton::Left => self.variables.is_mouse_pressed = pressed,
            _ => {}
        }
    }

    fn handle_mouse_scroll(&mut self, delta: &MouseScrollDelta) {
        self.camera_controller.handle_scroll(delta);
    }

    fn create_render_pipeline(
        device: &wgpu::Device,
        layout: &wgpu::PipelineLayout,
        color_format: wgpu::TextureFormat,
        depth_format: Option<wgpu::TextureFormat>,
        vertex_layouts: &[wgpu::VertexBufferLayout],
        shader_descriptor: wgpu::ShaderModuleDescriptor,
        polygon_mode: wgpu::PolygonMode,
    ) -> wgpu::RenderPipeline {
        let shader = device.create_shader_module(shader_descriptor);

        device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("render pipeline"),
            layout: Some(layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: Some("vertex_main"),
                buffers: vertex_layouts,
                compilation_options: Default::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fragment_main"),
                targets: &[Some(wgpu::ColorTargetState {
                    format: color_format,
                    blend: Some(wgpu::BlendState {
                        alpha: wgpu::BlendComponent::REPLACE,
                        color: wgpu::BlendComponent::REPLACE,
                    }),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: Default::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: Some(wgpu::Face::Back),
                // Setting this to anything other than Fill requires Features::NON_FILL_POLYGON_MODE
                polygon_mode,
                // true requires Features::DEPTH_CLIP_CONTROL
                unclipped_depth: false,
                // true requires Features::CONSERVATIVE_RASTERIZATION
                conservative: false,
            },
            depth_stencil: depth_format.map(|format| wgpu::DepthStencilState {
                format,
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: wgpu::StencilState::default(),
                bias: wgpu::DepthBiasState::default(),
            }),
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview_mask: None,
            cache: None,
        })
    }
}

pub struct App {
    // if compiling for wasm, the callback needs this proxy
    #[cfg(target_arch = "wasm32")]
    proxy: Option<winit::event_loop::EventLoopProxy<State>>,
    state: Option<State>,
    last_instant: Instant,
}

impl App {
    pub fn new(#[cfg(target_arch = "wasm32")] event_loop: &EventLoop<State>) -> Self {
        #[cfg(target_arch = "wasm32")]
        let proxy = Some(event_loop.create_proxy());
        Self {
            state: None,
            #[cfg(target_arch = "wasm32")]
            proxy,
            last_instant: Instant::now(),
        }
    }
}

impl ApplicationHandler<State> for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        #[allow(unused_mut)]
        let mut window_attributes =
            winit::window::WindowAttributes::default().with_title("graphics fundamentals - dpb4");

        #[cfg(target_arch = "wasm32")]
        {
            use wasm_bindgen::JsCast;
            use winit::platform::web::WindowAttributesExtWebSys;

            const CANVAS_ID: &str = "canvas";

            let window = wgpu::web_sys::window().unwrap_throw();
            let document = window.document().unwrap_throw();
            let canvas = document.get_element_by_id(CANVAS_ID).unwrap_throw();
            let html_canvas_element = canvas.unchecked_into();
            window_attributes = window_attributes.with_canvas(Some(html_canvas_element));
        }

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        #[cfg(not(target_arch = "wasm32"))]
        {
            // If we are not on web we can use pollster to
            // await the
            self.state = Some(pollster::block_on(State::new(window)).unwrap());
        }

        #[cfg(target_arch = "wasm32")]
        {
            // Run the future asynchronously and use the
            // proxy to send the results to the event loop
            if let Some(proxy) = self.proxy.take() {
                wasm_bindgen_futures::spawn_local(async move {
                    assert!(
                        proxy
                            .send_event(
                                State::new(window)
                                    .await
                                    .expect("Unable to create canvas!!!")
                            )
                            .is_ok()
                    )
                });
            }
        }
    }

    #[allow(unused_mut)]
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, mut event: State) {
        #[cfg(target_arch = "wasm32")]
        {
            event.window.request_redraw();
            event.resize(
                event.window.inner_size().width,
                event.window.inner_size().height,
            );
        }
        self.state = Some(event);
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: DeviceId,
        event: DeviceEvent,
    ) {
        match &mut self.state {
            None => {
                return;
            }
            Some(state) => match event {
                DeviceEvent::MouseMotion {
                    delta: (mouse_dx, mouse_dy),
                } => {
                    if state.variables.is_mouse_pressed {
                        state.camera_controller.handle_mouse(mouse_dx, mouse_dy);
                    }
                }
                _ => {}
            },
        };
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: WindowEvent,
    ) {
        let state = match &mut self.state {
            Some(state) => state,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                let dt = self.last_instant.elapsed();
                self.last_instant = Instant::now();

                let before_update = Instant::now();
                state.update(dt);

                let update_time = before_update.elapsed();

                let before_render = Instant::now();
                match state.render() {
                    Ok(_) => {}
                    // reconfigure the surface if it's lost or outdated
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        let size = state.window.inner_size();
                        state.resize(size.width, size.height);
                    }
                    Err(e) => {
                        log::error!("[error] unable to render {}", e);
                    }
                };

                state
                    .diagnostics
                    .update_time_avg
                    .push(update_time.as_micros() as f32);
                state.diagnostics.frame_time_avg.push(dt.as_secs_f32());
                state
                    .diagnostics
                    .render_time_avg
                    .push(before_render.elapsed().as_micros() as f32);

                state.window.set_title(&format!(
                    "graphics fundamentals - dpb4        |  fps {: >3}   |   mspf {: >3} ms   |   rt {: >6} us   |   ru {: >3} %  |   ut {: >6} us   |   uu {: >3} %  |   {}",
                    (1.0 / state.diagnostics.frame_time_avg.get()) as u32,
                    (state.diagnostics.frame_time_avg.get() * 1000.0) as u32,

                    state.diagnostics.render_time_avg.get() as u32,
                    (state.diagnostics.render_time_avg.get() / (1.0 / 240.0 * 1000000.0)) as u32,

                    state.diagnostics.update_time_avg.get() as u32,
                    (state.diagnostics.update_time_avg.get() / (1.0 / 240.0 * 1000000.0)) as u32,

                    if state.variables.swap_pipelines { "[ALT PIPELINE]" } else {""}
                ));
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: key_state,
                        ..
                    },
                ..
            } => state.handle_key(event_loop, code, key_state.is_pressed()),
            WindowEvent::MouseInput {
                state: button_state,
                button,
                ..
            } => state.handle_mouse_button(button, button_state.is_pressed()),
            WindowEvent::MouseWheel { delta, .. } => {
                state.handle_mouse_scroll(&delta);
            }
            _ => {}
        }
    }
}

pub fn run() -> anyhow::Result<()> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        env_logger::init();
    }
    #[cfg(target_arch = "wasm32")]
    {
        console_log::init_with_level(log::Level::Info).unwrap_throw();
    }

    let event_loop = EventLoop::with_user_event().build()?;
    let mut app = App::new(
        #[cfg(target_arch = "wasm32")]
        &event_loop,
    );

    log::info!("yep logging is working");
    event_loop.run_app(&mut app)?;

    Ok(())
}

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
pub fn run_web() -> Result<(), wasm_bindgen::JsValue> {
    console_error_panic_hook::set_once();
    run().unwrap_throw();

    Ok(())
}
