use cgmath::{InnerSpace, One};
use wgpu::util::DeviceExt;

use crate::{
    model::{self, VertexDebugUniform},
    texture,
};

const DET_EPSILON: f32 = 0.0001;

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

// pub fn load_model_from_memory(
//     vertices: &[[f32; 3]],
//     indices: &[u32],
//     device: &wgpu::Device,
// ) -> model::Model {
//     let model_verts = vertices
//         .iter()
//         .map(|v| model::ModelVertex {
//             position: *v,
//             tex_coords: [0.0; 2],
//             normal: [0.0; 3],
//         })
//         .collect::<Vec<_>>();

//     let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//         label: Some("memory obj vertex buffer"),
//         contents: bytemuck::cast_slice(&model_verts),
//         usage: wgpu::BufferUsages::VERTEX,
//     });

//     let index_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
//         label: Some("memory obj index buffer"),
//         contents: bytemuck::cast_slice(&indices),
//         usage: wgpu::BufferUsages::INDEX,
//     });

//     model::Model {
//         meshes: vec![model::Mesh {
//             name: "placeholder memory mesh".to_string(),
//             vertex_buffer,
//             index_buffer,
//             index_count: indices.len() as u32,
//             material: 0,
//         }],
//         materials: Vec::new(),
//     }
// }

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

            if diffuse_texture.is_some() {
                println!("material {} has diffuse texture", tm.name);
            } else {
                println!("material {} using diffuse color {:?}", tm.name, tm.diffuse);
            }

            let normal_texture = tm.normal_texture.as_ref().and_then(|dtn| {
                load_texture(&format!("src/assets/{}", dtn), device, queue, true).ok()
            });

            if normal_texture.is_some() {
                println!("material {} has normal map", tm.name);
            } else {
                println!("material {} has no normal map", tm.name);
            }

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
        .iter()
        .map(|m| {
            let mut vertices = (0..m.mesh.positions.len() / 3)
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
                            tangent: [0.0, 0.0, 0.0],
                            bitangent: [0.0, 0.0, 0.0],
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
                            tangent: [0.0, 0.0, 0.0],
                            bitangent: [0.0, 0.0, 0.0],
                        }
                    }
                })
                .collect::<Vec<_>>();

            calculate_tbs(&m.mesh, &mut vertices);

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

            // println!("{} normals: {:?}\n", m.name, vertices.iter().map(|v| v.normal).collect::<Vec<_>>());
            // println!("{} tangents: {:?}\n", m.name, vertices.iter().map(|v| v.tangent).collect::<Vec<_>>());
            // println!("{} bitangents: {:?}\n", m.name, vertices.iter().map(|v| v.bitangent).collect::<Vec<_>>());

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

    Ok(model::Model {
        meshes,
        materials,
        position: [0.0; 3],
        rotation: cgmath::Quaternion::one(),
        scale: 1.0,
    })
}

pub fn load_obj_model_for_buffer(
    file_name: &str,
) -> anyhow::Result<Vec<Vec<VertexDebugUniform>>> {
    let (models, _) = tobj::load_obj(
        file_name,
        &tobj::LoadOptions {
            single_index: true,
            triangulate: true,
            ..Default::default()
        },
    )?;

    let mut out = Vec::new();

    for m in models {
        let mut vertices = (0..m.mesh.positions.len() / 3)
            .map(|i| model::ModelVertex {
                position: [
                    m.mesh.positions[i * 3],
                    m.mesh.positions[i * 3 + 1],
                    m.mesh.positions[i * 3 + 2],
                ],
                tex_coords: [0.0; 2],
                normal: [0.0, 0.0, 0.0],
                tangent: [0.0, 0.0, 0.0],
                bitangent: [0.0, 0.0, 0.0],
            })
            .collect::<Vec<_>>();

        calculate_tbs(&m.mesh, &mut vertices);

        let buf_vec = vertices
            .into_iter()
            .map(|mv| VertexDebugUniform::from_model_vertex(&mv))
            .collect();

        out.push(buf_vec);
    }

    Ok(out)
}

fn calculate_tbs(mesh: &tobj::Mesh, model_verts: &mut [model::ModelVertex]) {
    let indices = &mesh.indices;
    let mut vertex_face_count = vec![0; model_verts.len()];

    for ti in indices.chunks(3) {
        let v0 = model_verts[ti[0] as usize];
        let v1 = model_verts[ti[1] as usize];
        let v2 = model_verts[ti[2] as usize];

        let pos0 = cgmath::Vector3::from(v0.position);
        let pos1 = cgmath::Vector3::from(v1.position);
        let pos2 = cgmath::Vector3::from(v2.position);

        let uv0 = cgmath::Vector2::from(v0.tex_coords);
        let uv1 = cgmath::Vector2::from(v1.tex_coords);
        let uv2 = cgmath::Vector2::from(v2.tex_coords);

        let delta_pos_0_1 = pos1 - pos0;
        let delta_pos_0_2 = pos2 - pos0;

        let delta_uv_0_1 = uv1 - uv0;
        let delta_uv_0_2 = uv2 - uv0;

        let det_denom = delta_uv_0_1.x * delta_uv_0_2.y - delta_uv_0_1.y * delta_uv_0_2.x;

        let tangent;
        let bitangent;

        if det_denom.abs() <= DET_EPSILON {
            // in this case the triangle is degenerate somehow; same UVs, 0 UVs, idk but it needs to be fixed
            // pick an arbitrary vector which isn't parallel to the normal
            let normal = cgmath::Vector3::from(v0.normal);
            let arb = if normal.z.abs() < 0.999 { cgmath::Vector3::unit_z() } else {cgmath::Vector3::unit_y() };
            
            tangent = arb.cross(normal).normalize();
            bitangent = normal.cross(tangent);
        } else {
            let r = 1.0 / det_denom;
            tangent = (delta_pos_0_1 * delta_uv_0_2.y - delta_pos_0_2 * delta_uv_0_1.y) * r;
            bitangent = (delta_pos_0_2 * delta_uv_0_1.x - delta_pos_0_1 * delta_uv_0_2.x) * r;
        }

        // each vertex in the triangle uses the same tangent/bitangent
        // note the addition instead of assignment, because multiple faces
        // could be calculating different T/Bs, hence the need for the average
        model_verts[ti[0] as usize].tangent =
            (tangent + cgmath::Vector3::from(model_verts[ti[0] as usize].tangent)).into();
        model_verts[ti[1] as usize].tangent =
            (tangent + cgmath::Vector3::from(model_verts[ti[1] as usize].tangent)).into();
        model_verts[ti[2] as usize].tangent =
            (tangent + cgmath::Vector3::from(model_verts[ti[2] as usize].tangent)).into();

        model_verts[ti[0] as usize].bitangent =
            (bitangent + cgmath::Vector3::from(model_verts[ti[0] as usize].bitangent)).into();
        model_verts[ti[1] as usize].bitangent =
            (bitangent + cgmath::Vector3::from(model_verts[ti[1] as usize].bitangent)).into();
        model_verts[ti[2] as usize].bitangent =
            (bitangent + cgmath::Vector3::from(model_verts[ti[2] as usize].bitangent)).into();

        // number of times a vertex gets used, to average the T/Bs
        vertex_face_count[ti[0] as usize] += 1;
        vertex_face_count[ti[1] as usize] += 1;
        vertex_face_count[ti[2] as usize] += 1;
    }

    // average out each vertex depending on how much it was used
    // for (i, n) in vertex_face_count.into_iter().enumerate() {
    //     if n == 0 {
    //         println!("N 0 - BAD!!!!");
    //     }
    //     let denom = 1.0 / n as f32;
    //     let v = &mut model_verts[i];
    //     v.tangent = (cgmath::Vector3::from(v.tangent) * denom).into();
    //     v.bitangent = (cgmath::Vector3::from(v.bitangent) * denom).into();
    // }

    for v in model_verts {
        let vn = cgmath::Vector3::from(v.normal);
        let vt = cgmath::Vector3::from(v.tangent);

        let tangent_gs = (vt - (vn * vn.dot(vt))).normalize();
        v.tangent = tangent_gs.into();
        v.bitangent = tangent_gs.cross(-vn).normalize().into();
    }
}
