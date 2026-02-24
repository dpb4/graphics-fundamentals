use std::collections::{HashMap, HashSet};

use cgmath::One;

use crate::{
    model::{self, Material},
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

pub fn load_material(
    filepath: &str,
    name: &str,
    device: &wgpu::Device,
    layout: &wgpu::BindGroupLayout,
    queue: &wgpu::Queue,
) -> Result<model::Material, crate::obj_parse::MTLLoadError> {
    let parsed_mtl = crate::obj_parse::parse_mtl(filepath, name)?;

    let diffuse_texture = parsed_mtl.map_kd.as_ref().and_then(|dtn| {
        load_texture(
            &format!("src/assets/materials/{}", dtn),
            device,
            queue,
            false,
        )
        .ok()
    });

    let normal_texture = parsed_mtl.map_bump.as_ref().and_then(|dtn| {
        load_texture(
            &format!("src/assets/materials/{}", dtn),
            device,
            queue,
            true,
        )
        .ok()
    });

    Ok(model::Material::new(
        device,
        name,
        diffuse_texture,
        normal_texture,
        parsed_mtl.ka.unwrap_or([0.0; 3]),
        parsed_mtl.kd.unwrap_or([1.0, 0.0, 1.0]),
        parsed_mtl.ks.unwrap_or([1.0; 3]),
        layout,
    ))
}

pub fn load_all_materials(
    filepath: &str,
    materials: &mut Vec<model::Material>,
    material_map: &mut HashMap<String, usize>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) {
    let parsed_mtls = crate::obj_parse::parse_all_mtls(filepath)
        .unwrap()
        .into_iter()
        .map(|pmtl| {
            let diffuse_texture = pmtl.map_kd.as_ref().and_then(|dtn| {
                load_texture(
                    &format!("src/assets/materials/{}", dtn),
                    device,
                    queue,
                    false,
                )
                .ok()
            });

            let normal_texture = pmtl.map_bump.as_ref().and_then(|dtn| {
                load_texture(
                    &format!("src/assets/materials/{}", dtn),
                    device,
                    queue,
                    true,
                )
                .ok()
            });

            model::Material::new(
                device,
                &pmtl.name.clone().unwrap_or("NONE".to_string()),
                diffuse_texture,
                normal_texture,
                pmtl.ka.unwrap_or([0.0; 3]),
                pmtl.kd.unwrap_or([1.0, 0.0, 1.0]),
                pmtl.ks.unwrap_or([1.0; 3]),
                layout,
            )
        });

    for m in parsed_mtls {
        println!("loaded mtl {}", &m.name);
        material_map.insert(m.name.clone(), materials.len());
        materials.push(m);
    }
}

pub fn load_obj_model(
    filepath: &str,
    materials: &mut Vec<model::Material>,
    material_map: &mut HashMap<String, usize>,
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    layout: &wgpu::BindGroupLayout,
) -> anyhow::Result<model::Model> {
    let pobj = crate::obj_parse::parse_obj(filepath).unwrap();

    let material = if let Some(mtl) = pobj.material {
        if material_map.contains_key(&mtl) {
            println!("material {} already loaded", &mtl);
            *material_map.get(&mtl).unwrap()
        } else {
            println!("loading material {}", &mtl);
            let new_index = materials.len();
            materials.push(
                load_material(&pobj.material_lib.unwrap(), &mtl, device, layout, queue).unwrap(),
            );
            material_map.insert(mtl, new_index);
            new_index
        }
    } else {
        0
    };

    let mesh = model::Mesh::from_verts_inds(
        &device,
        filepath.to_string(),
        pobj.model_verts,
        pobj.indices,
        material,
    );
    Ok(model::Model {
        meshes: vec![mesh],
        position: [0.0, 0.0, 0.0],
        rotation: cgmath::Quaternion::one(),
        scale: 1.0,
    })
}
