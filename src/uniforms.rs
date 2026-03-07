use cgmath::SquareMatrix;

use crate::{DirectionalLight, PointLight, SpotLight, camera};

#[repr(C)]
#[derive(Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct CameraUniform {
    position: [f32; 4],
    view_projection_matrix: [[f32; 4]; 4],
}

impl CameraUniform {
    pub fn new() -> Self {
        Self {
            position: [0.0; 4],
            view_projection_matrix: cgmath::Matrix4::identity().into(),
        }
    }

    pub fn update_view_proj(&mut self, camera: &camera::Camera, projection: &camera::Projection) {
        self.position = camera.position.to_homogeneous().into();
        self.view_projection_matrix =
            (projection.perspective_matrix() * camera.view_matrix()).into()
    }
}

pub fn create_light_uniforms(
    point_lights: &Vec<PointLight>,
    directional_lights: &Vec<DirectionalLight>,
    spot_lights: &Vec<SpotLight>,
) -> (Vec<LightUniform>, LightMetadataUniform) {
    let mut light_uniforms: Vec<LightUniform> = Vec::new();

    let pl = point_lights.len() as u32;
    let dl = directional_lights.len() as u32;
    let sl = spot_lights.len() as u32;

    light_uniforms.extend(
        point_lights
            .clone()
            .into_iter()
            .map(|l| LightUniform::from(l)),
    );
    light_uniforms.extend(
        directional_lights
            .clone()
            .into_iter()
            .map(|l| LightUniform::from(l)),
    );
    light_uniforms.extend(
        spot_lights
            .clone()
            .into_iter()
            .map(|l| LightUniform::from(l)),
    );

    let light_metadata_uniform = LightMetadataUniform {
        point_count: pl,
        point_offset: 0,
        directional_count: dl,
        directional_offset: pl,
        spot_count: sl,
        spot_offset: pl + dl,
        _padding: [0; 2],
    };

    (light_uniforms, light_metadata_uniform)
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
// padding fields are necessary because uniforms require 16 byte alignment
pub struct LightUniform {
    position: [f32; 3],
    _padding1: u32,
    direction: [f32; 3],
    _padding2: u32,
    color: [f32; 3],
    _padding3: u32,
    params: [f32; 4],
}

impl From<PointLight> for LightUniform {
    fn from(value: PointLight) -> Self {
        Self {
            position: value.position,
            _padding1: 0,
            direction: [0.0; 3],
            _padding2: 0,
            color: value.color,
            _padding3: 0,
            params: [0.0; 4],
        }
    }
}

impl From<DirectionalLight> for LightUniform {
    fn from(value: DirectionalLight) -> Self {
        Self {
            position: [0.0; 3],
            _padding1: 0,
            direction: value.direction,
            _padding2: 0,
            color: value.color,
            _padding3: 0,
            params: [0.0; 4],
        }
    }
}

impl From<SpotLight> for LightUniform {
    fn from(value: SpotLight) -> Self {
        Self {
            position: value.position,
            _padding1: 0,
            direction: value.direction,
            _padding2: 0,
            color: value.color,
            _padding3: 0,
            params: [
                value.inner_angular_radius.cos(),
                value.outer_angular_radius.cos(),
                0.0,
                0.0,
            ],
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct LightMetadataUniform {
    point_count: u32,
    point_offset: u32,

    directional_count: u32,
    directional_offset: u32,

    spot_count: u32,
    spot_offset: u32,

    _padding: [u32; 2],
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct TimestampUniform {
    pub time: u32,
}
