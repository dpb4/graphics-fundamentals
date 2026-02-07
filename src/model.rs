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
            ],
        }
    }
}

pub struct Model {
    pub meshes: Vec<Mesh>,
    pub materials: Vec<Material>,
    pub position: [f32; 3],
    pub rotation: cgmath::Quaternion<f32>,
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
        let matrix = cgmath::Matrix4::from_translation(model.position.into()) * cgmath::Matrix4::from(model.rotation);
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
    pub bind_group: wgpu::BindGroup,
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
