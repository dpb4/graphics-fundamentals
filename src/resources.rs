use wgpu::util::DeviceExt;

use crate::{model, texture};

pub fn load_text(file_name: &String) -> anyhow::Result<String> {
    Ok(std::fs::read_to_string(std::path::Path::new(file_name))?)
}

pub fn load_binary(file_name: &str) -> anyhow::Result<Vec<u8>> {
    Ok(std::fs::read(std::path::Path::new(file_name))?)
}

pub fn load_texture(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
) -> anyhow::Result<texture::Texture> {
    let data = load_binary(file_name)?;
    texture::Texture::from_bytes(device, queue, &data, file_name)
}

pub fn load_model_from_memory(
    vertices: &[[f32; 3]],
    indices: &[u32],
    device: &wgpu::Device,
) -> model::Model {
    let model_verts = vertices
        .iter()
        .map(|v| model::ModelVertex {
            position: *v,
            tex_coords: [0.0; 2],
            normal: [0.0; 3],
        })
        .collect::<Vec<_>>();

    let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("memory obj vertex buffer"),
        contents: bytemuck::cast_slice(&model_verts),
        usage: wgpu::BufferUsages::VERTEX,
    });

    let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("memory obj index buffer"),
        contents: bytemuck::cast_slice(&indices),
        usage: wgpu::BufferUsages::INDEX,
    });

    model::Model {
        meshes: vec![model::Mesh {
            name: "placeholder memory mesh".to_string(),
            vertex_buffer,
            index_buffer,
            index_count: indices.len() as u32,
            material: 0,
        }],
        materials: Vec::new(),
    }
}

pub fn load_obj_model(
    file_name: &str,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<model::Model> {
    let (models, tobj_materials) = tobj::load_obj(
        file_name,
        &tobj::LoadOptions {
            single_index: true,
            triangulate: true,
            ..Default::default()
        },
    )?;

    let mut materials = tobj_materials?
        .iter()
        .map(|tm| {
            let diffuse_texture = load_texture(
                &format!("src/assets/{}", &tm.diffuse_texture.clone().unwrap()),
                device,
                queue,
            )
            .unwrap();

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
                ],
                label: None,
            });

            model::Material {
                name: tm.name.clone(),
                diffuse_texture,
                bind_group,
            }
        })
        .collect::<Vec<_>>();

    if materials.is_empty() {
        let diffuse_texture = load_texture("src/assets/cat.png", device, queue).unwrap();

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
            ],
            label: None,
        });
        materials.push(model::Material {
            name: "MISSING MATERIAL".to_string(),
            diffuse_texture,
            bind_group,
        })
    }

    let meshes = models
        .iter()
        .map(|m| {
            let vertices = (0..m.mesh.positions.len() / 3)
                .map(|i| {
                    if m.mesh.normals.is_empty() {
                        model::ModelVertex {
                            position: [
                                m.mesh.positions[i * 3],
                                m.mesh.positions[i * 3 + 1],
                                m.mesh.positions[i * 3 + 2],
                            ],
                            tex_coords: if m.mesh.texcoords.is_empty() {
                                [0.0; 2]
                            } else {
                                [m.mesh.texcoords[i * 2], 1.0 - m.mesh.texcoords[i * 2 + 1]]
                            },
                            normal: [0.0, 0.0, 0.0],
                        }
                    } else {
                        model::ModelVertex {
                            position: [
                                m.mesh.positions[i * 3],
                                m.mesh.positions[i * 3 + 1],
                                m.mesh.positions[i * 3 + 2],
                            ],
                            tex_coords: if m.mesh.texcoords.is_empty() {
                                [0.0; 2]
                            } else {
                                [m.mesh.texcoords[i * 2], 1.0 - m.mesh.texcoords[i * 2 + 1]]
                            },
                            normal: [
                                m.mesh.normals[i * 3],
                                m.mesh.normals[i * 3 + 1],
                                m.mesh.normals[i * 3 + 2],
                            ],
                        }
                    }
                })
                .collect::<Vec<_>>();

            let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&(m.name.clone() + " vertex buffer")),
                contents: bytemuck::cast_slice(&vertices),
                usage: wgpu::BufferUsages::VERTEX,
            });

            let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(&(m.name.clone() + " index buffer")),
                contents: bytemuck::cast_slice(&m.mesh.indices),
                usage: wgpu::BufferUsages::INDEX,
            });

            log::info!("loaded mesh: {}", m.name);
            model::Mesh {
                name: file_name.to_string(),
                vertex_buffer,
                index_buffer,
                index_count: m.mesh.indices.len() as u32,
                material: m.mesh.material_id.unwrap_or(0),
            }
        })
        .collect::<Vec<_>>();

    Ok(model::Model { meshes, materials })
}
