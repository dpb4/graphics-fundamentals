use cgmath::One;

use crate::{
    model::{self},
    texture,
};

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
    is_linear: bool,
) -> anyhow::Result<texture::Texture> {
    let data = load_binary(file_name)?;
    texture::Texture::from_bytes(device, queue, &data, file_name, is_linear)
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
            let diffuse_texture = tm.diffuse_texture.as_ref().and_then(|dtn| {
                load_texture(&format!("src/assets/{}", dtn), device, queue, false).ok()
            });

            let normal_texture = tm.normal_texture.as_ref().and_then(|dtn| {
                load_texture(&format!("src/assets/{}", dtn), device, queue, true).ok()
            });

            model::Material::new(
                device,
                &tm.name,
                diffuse_texture,
                normal_texture,
                tm.ambient.unwrap_or([0.0; 3]),
                tm.diffuse.unwrap_or([1.0, 0.0, 1.0]),
                tm.specular.unwrap_or([1.0; 3]),
                layout,
            )
        })
        .collect::<Vec<_>>();

    if materials.is_empty() {
        let diffuse_texture =
            load_texture("src/assets/debug_diffuse.png", device, queue, false).unwrap();
        let normal_texture =
            load_texture("src/assets/debug_normal.png", device, queue, true).unwrap();

        materials.push(model::Material::new(
            device,
            "DEBUG MATERIAL",
            Some(diffuse_texture),
            Some(normal_texture),
            [1.0; 3],
            [1.0; 3],
            [1.0; 3],
            layout,
        ))
    }

    let meshes = models
        .into_iter()
        .map(|m| {
            let vertices = (0..m.mesh.positions.len() / 3)
                .map(|i| model::ModelVertex {
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
                    normal: if m.mesh.normals.is_empty() {
                        [0.0, 0.0, 0.0]
                    } else {
                        [
                            m.mesh.normals[i * 3],
                            m.mesh.normals[i * 3 + 1],
                            m.mesh.normals[i * 3 + 2],
                        ]
                    },
                    tangent: [0.0, 0.0, 0.0],
                    bitangent: [0.0, 0.0, 0.0],
                })
                .collect::<Vec<_>>();

            model::Mesh::from_verts_inds(
                device,
                m.name,
                vertices,
                m.mesh.indices,
                m.mesh.material_id.unwrap_or(0),
            )
        })
        .collect::<Vec<_>>();

    Ok(model::Model {
        meshes,
        materials,
        position: [0.0; 3],
        rotation: cgmath::Quaternion::one(),
        scale: 1.0,
    })
}
