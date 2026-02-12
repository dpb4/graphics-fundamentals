use wgpu::util::DeviceExt;

use crate::texture;
use std::ops::Range;
pub trait Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static>;
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelVertex {
    pub position: [f32; 3],
    pub tex_coords: [f32; 2],
    pub normal: [f32; 3],
    pub tangent: [f32; 3],
    pub bitangent: [f32; 3],
}

impl Vertex for ModelVertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<ModelVertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 3]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 5]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 8]>() as wgpu::BufferAddress,
                    shader_location: 3,
                    format: wgpu::VertexFormat::Float32x3,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 11]>() as wgpu::BufferAddress,
                    shader_location: 4,
                    format: wgpu::VertexFormat::Float32x3,
                },
            ],
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone, Debug, bytemuck::Pod, bytemuck::Zeroable)]
pub struct VertexDebugUniform {
    pub position: [f32; 4],
    pub normal: [f32; 4],
    pub tangent: [f32; 4],
    pub bitangent: [f32; 4],
}

impl VertexDebugUniform {
    pub fn from_model_vertex(mv: &ModelVertex) -> Self {
        Self {
            position: [mv.position[0], mv.position[1], mv.position[2], 0.0],
            normal: [mv.normal[0], mv.normal[1], mv.normal[2], 0.0],
            tangent: [mv.tangent[0], mv.tangent[1], mv.tangent[2], 0.0],
            bitangent: [mv.bitangent[0], mv.bitangent[1], mv.bitangent[2], 0.0],
        }
    }
}

pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub position: [f32; 3],
    pub rotation: cgmath::Quaternion<f32>,
    pub scale: f32,
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct ModelTransformationUniform {
    model_transformation_col0: [f32; 4],
    model_transformation_col1: [f32; 4],
    model_transformation_col2: [f32; 4],
    model_transformation_col3: [f32; 4],
}

impl ModelTransformationUniform {
    pub fn identity() -> Self {
        Self {
            model_transformation_col0: [1.0, 0.0, 0.0, 0.0],
            model_transformation_col1: [0.0, 1.0, 0.0, 0.0],
            model_transformation_col2: [0.0, 0.0, 1.0, 0.0],
            model_transformation_col3: [0.0, 0.0, 0.0, 1.0],
        }
    }

    pub fn from_model(model: &Model) -> Self {
        let matrix = cgmath::Matrix4::from_translation(model.position.into())
            * cgmath::Matrix4::from(model.rotation)
            * cgmath::Matrix4::from_scale(model.scale);
        Self {
            model_transformation_col0: matrix.x.into(),
            model_transformation_col1: matrix.y.into(),
            model_transformation_col2: matrix.z.into(),
            model_transformation_col3: matrix.w.into(),
        }
    }
}

pub struct Material {
    pub name: String,
    pub diffuse_texture: texture::Texture,
    pub normal_texture: texture::Texture,
    pub ambient_color: [f32; 3],
    pub diffuse_color: [f32; 3],
    pub specular_color: [f32; 3],
    pub bind_group: wgpu::BindGroup,
}

impl Material {
    pub fn new(
        device: &wgpu::Device,
        name: &str,
        diffuse_texture: Option<texture::Texture>,
        normal_texture: Option<texture::Texture>,
        ambient_color: [f32; 3],
        diffuse_color: [f32; 3],
        specular_color: [f32; 3],
        layout: &wgpu::BindGroupLayout,
    ) -> Self {
        let material_uniform = MaterialUniform::new(
            ambient_color,
            diffuse_color,
            specular_color,
            diffuse_texture.is_some(),
            normal_texture.is_some(),
        );
        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some(name),
            contents: bytemuck::cast_slice(&[material_uniform]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        let diffuse_texture = diffuse_texture.unwrap_or(texture::Texture::dummy(
            device,
            &(name.to_string() + " diffuse dummy"),
        ));
        let normal_texture = normal_texture.unwrap_or(texture::Texture::dummy(
            device,
            &(name.to_string() + " normal dummy"),
        ));

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&diffuse_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&diffuse_texture.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&normal_texture.view),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Sampler(&normal_texture.sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: material_buffer.as_entire_binding(),
                },
            ],
            label: Some(name),
        });

        Self {
            name: String::from(name),
            diffuse_texture,
            normal_texture,
            bind_group,
            ambient_color,
            diffuse_color,
            specular_color,
        }
    }
}

#[repr(C)]
#[derive(Debug, Copy, Clone, bytemuck::Pod, bytemuck::Zeroable)]
pub struct MaterialUniform {
    ambient_color: [f32; 3],
    _padding0: u32,
    diffuse_color: [f32; 3],
    _padding1: u32,
    specular_color: [f32; 3],
    _padding2: u32,
    has_diffuse_texture: u32,
    has_normal_texture: u32, // these are u32 to avoid any padding confusion while using bytemuck
    _padding3: [u32; 2],     // these are u32 to avoid any padding confusion while using bytemuck
}

impl MaterialUniform {
    fn new(
        ambient_color: [f32; 3],
        diffuse_color: [f32; 3],
        specular_color: [f32; 3],
        has_diffuse_texture: bool,
        has_normal_texture: bool,
    ) -> Self {
        Self {
            ambient_color,
            _padding0: 0,
            diffuse_color,
            _padding1: 0,
            specular_color,
            _padding2: 0,
            has_diffuse_texture: if has_diffuse_texture { 1 } else { 0 },
            has_normal_texture: if has_normal_texture { 1 } else { 0 },
            _padding3: [0, 0],
        }
    }
}

pub struct Mesh {
    pub name: String,
    pub vertex_buffer: wgpu::Buffer,
    pub index_buffer: wgpu::Buffer,
    pub index_count: u32,
    pub material: usize,
}

pub trait DrawModel<'a> {
    fn draw_mesh(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        per_object_bind_group: &'a wgpu::BindGroup,
    );
    fn draw_mesh_instanced(
        &mut self,
        mesh: &'a Mesh,
        material: &'a Material,
        instances: Range<u32>,
        per_object_bind_group: &'a wgpu::BindGroup,
    );

    fn draw_model(&mut self, model: &'a Model, per_object_bind_group: &'a wgpu::BindGroup);
    fn draw_model_instanced(
        &mut self,
        model: &'a Model,
        instances: Range<u32>,
        per_object_bind_group: &'a wgpu::BindGroup,
    );
}

impl<'a, 'b> DrawModel<'b> for wgpu::RenderPass<'a>
where
    'b: 'a,
{
    fn draw_mesh(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        per_object_bind_group: &'b wgpu::BindGroup,
    ) {
        self.draw_mesh_instanced(mesh, material, 0..1, per_object_bind_group);
    }

    fn draw_mesh_instanced(
        &mut self,
        mesh: &'b Mesh,
        material: &'b Material,
        instances: Range<u32>,
        per_object_bind_group: &'b wgpu::BindGroup,
    ) {
        self.set_vertex_buffer(0, mesh.vertex_buffer.slice(..));
        self.set_index_buffer(mesh.index_buffer.slice(..), wgpu::IndexFormat::Uint32);

        self.set_bind_group(1, &material.bind_group, &[]);
        self.set_bind_group(2, per_object_bind_group, &[]);

        self.draw_indexed(0..mesh.index_count, 0, instances);
    }

    fn draw_model(&mut self, model: &'b Model, per_object_bind_group: &'b wgpu::BindGroup) {
        self.draw_model_instanced(model, 0..1, per_object_bind_group);
    }

    fn draw_model_instanced(
        &mut self,
        model: &'b Model,
        instances: Range<u32>,
        per_object_bind_group: &'b wgpu::BindGroup,
    ) {
        for mesh in &model.meshes {
            let material = &model.materials[mesh.material];
            self.draw_mesh_instanced(mesh, material, instances.clone(), per_object_bind_group);
        }
    }
}
